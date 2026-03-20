# Code Review Report — utc-chatbot-backend

**Date:** 2026-03-20
**Scope:** Full codebase review — core modules, storage layer, API layer, search/retrieval, QA/eval, and tests
**Reviewed by:** 5 parallel review agents (Claude Code)

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Critical Issues](#critical-issues)
- [Important Issues](#important-issues)
- [Security Issues](#security-issues)
- [Type Safety & Code Quality](#type-safety--code-quality)
- [Test Quality Issues](#test-quality-issues)
- [Test Coverage Gaps](#test-coverage-gaps)
- [Full Issue Index](#full-issue-index)

---

## Executive Summary

**57 files reviewed** across 5 areas. **47 issues identified**: 12 critical, 22 important, 5 security, 8 quality/test issues. The most impactful findings are:

1. **`asyncio.run()` crashes inside FastAPI's event loop** — affects embedding and HyDE sync wrappers
2. **Vector search parameter off-by-one** — all filtered vector searches return wrong results
3. **No thread safety on BM25 cache** — concurrent requests can corrupt the corpus
4. **Admin endpoints completely unauthenticated** — exposes raw user queries (FERPA risk)
5. **`identify_non_approved_articles` is a permanent no-op** — cleanup phase never works

---

## Critical Issues

### C-01. `asyncio.run()` crashes inside existing event loop
**Files:** `core/embedding.py:119,186` and `core/hyde_generator.py:232`
**Confidence:** 95

The sync wrappers `generate_embedding()`, `generate_embeddings_batch()`, and `generate_hypothetical_document_sync()` all call `asyncio.run()`. This raises `RuntimeError: This event loop is already running` when called from FastAPI (uvicorn runs its own event loop, and sync endpoints execute in a thread pool that shares the loop context).

This will crash on every HyDE request and any vector search path that invokes the sync embedding wrapper.

**Fix:** Use `asyncio.get_event_loop().run_until_complete()` with a loop check, use `nest_asyncio`, or make the FastAPI endpoints `async def` and `await` directly.

---

### C-02. Vector search `params` list has extra `query_vector` — off-by-one breaks all filtered searches
**File:** `core/storage_vector.py:431-452`
**Confidence:** 95

```python
where_conditions = ["1 - (e.embedding <=> %s::vector) >= %s"]
params = [query_vector, query_vector, min_similarity]  # 3 values for 2 placeholders
```

The opening condition has two `%s` placeholders but `params` starts with three values. When filters are added (status_names, category_names, etc.), every subsequent parameter is shifted one position right, producing wrong query results or psycopg type errors.

**Fix:** Change to `params = [query_vector, min_similarity]`.

---

### C-03. `identify_non_approved_articles` always returns empty list — cleanup is a no-op
**File:** `core/ingestion.py:336-347`
**Confidence:** 95

```python
all_articles = self.tdx_client.retrieve_all_articles()
non_approved_tdx_ids = [a.get("ID") for a in articles_list if a.get("StatusName") != "Approved"]
```

`retrieve_all_articles()` calls `list_article_ids()`, which already filters to approved-only articles. The list comprehension filtering for `!= "Approved"` will therefore always produce an empty list. Phase 0 of the pipeline is a permanent no-op.

---

### C-04. No thread safety on BM25 cache — concurrent requests corrupt corpus
**File:** `core/bm25_search.py:121-123, 258-276`
**Confidence:** 90

With 4 uvicorn workers and sync endpoints running in thread pools, `_chunks_cache`, `_bm25_model`, and `_tokenized_corpus` are plain instance attributes with no locking. Two concurrent requests that both find `_bm25_model is None` will interleave writes to `_tokenized_corpus`, producing a model built on a partially-populated corpus.

**Fix:** Add a `threading.Lock` around the build-and-assign block.

---

### C-05. `storage_query_log.log_query_with_results` is not atomic despite docstring claim
**File:** `core/storage_query_log.py:424-439`
**Confidence:** 92

The docstring promises "both the query and its results are logged together, or neither is logged." But the implementation calls `log_query()` (commits in its own `get_connection()` context) then `log_query_results()` (separate connection, separate commit). If `log_query_results()` fails, the query row is orphaned.

**Fix:** Refactor both inserts to share one `with self.get_connection() as conn:` block.

---

### C-06. `storage_base.close()` raises `AttributeError` in pool mode
**File:** `core/storage_base.py:151-156`
**Confidence:** 88

When `connection_pool` is provided (API mode), `self._conn` is never assigned. `close()` checks `self._conn`, which raises `AttributeError`. `__exit__` calls `close()` unconditionally.

**Fix:** Initialize `self._conn = None` unconditionally in `__init__`, or use `getattr(self, "_conn", None)`.

---

### C-07. `storage_raw.py` double `conn.commit()` in `delete_articles_by_tdx_ids`
**File:** `core/storage_raw.py:510`
**Confidence:** 100

Explicit `conn.commit()` at line 510 inside `with self.get_connection() as conn:` block. The context manager also commits on clean exit. psycopg3 raises `InvalidTransactionState` on the second commit.

**Fix:** Remove the manual `conn.commit()` at line 510.

---

### C-08. Crash on empty reranked results — unguarded `[0]` access
**File:** `core/reranker.py:200` and `api/utils/hybrid_search.py:241`
**Confidence:** 92

```python
f"top score: {reranked_results[0]['combined_score']:.3f}"
```

If the reranker returns an empty list (all indices out of range), this raises `IndexError`, crashing the request and bypassing graceful degradation.

**Fix:** Guard with `if reranked_results:` before accessing `[0]`.

---

### C-09. Timezone-naive vs timezone-aware datetime comparison
**File:** `core/ingestion.py:109-122`
**Confidence:** 85

`api_modified_date` is parsed as timezone-aware (UTC), but `db_metadata[article_id]` may be a naive datetime depending on the column type. The `>` comparison raises `TypeError`.

---

### C-10. RRF constant `k=1` is non-standard (paper uses `k=60`)
**File:** `api/utils/hybrid_search.py:28`
**Confidence:** 88

The canonical RRF paper (Cormack et al., 2009) uses `k=60`. At `k=1`, results are hyper-aggressive toward rank-1 documents. This may be intentional but differs substantially from the published algorithm's behavior and could cause unexpected relevance issues.

---

### C-11. `PromptStorageClient` opened without being closed — connection leak
**File:** `core/bm25_search.py:407-413`
**Confidence:** 85

```python
prompt_client = PromptStorageClient()
prompts = prompt_client.get_prompts_for_article_ids(article_ids)
```

No `with` statement, no `.close()`. Happens on every BM25 search with `include_system_prompts=True` (the default). Under load, this exhausts the database connection pool.

---

### C-12. `batch_search()` pre-builds BM25 model then rebuilds it per query
**File:** `core/bm25_search.py:490-504`
**Confidence:** 95

`batch_search()` pre-builds the model, then calls `self.search()` for each query, which calls `_build_bm25_model()` again. For filtered batch searches (`use_cache=False`), the model rebuilds on every iteration, making the pre-build wasted work and corrupting `_tokenized_corpus`.

---

## Important Issues

### I-01. Pipeline loads entire chunk table for incremental embedding
**File:** `core/pipeline.py:490-504`
**Confidence:** 90

`get_all_chunks()` loads the full corpus into memory (~500MB) even when only a handful of new articles need embedding. Should add a `get_chunks_by_ids()` method using `WHERE id = ANY(%s)`.

---

### I-02. TOCTOU race in `RateLimiter.acquire()`
**File:** `core/api_client.py:471-476`
**Confidence:** 80

Lock is released before `time.sleep()` and re-acquired after, allowing another thread to enter and potentially exceed the rate limit.

---

### I-03. Chunk storage duplicated in wrong module
**Files:** `core/storage_raw.py:299-432` and `core/storage_chunk.py`
**Confidence:** 88

`storage_raw.py` contains `store_chunks`, `get_chunk_count`, `get_all_chunks` — operations on the `article_chunks` table — duplicating `storage_chunk.py`. The `storage_raw.py` version uses `ON CONFLICT DO UPDATE` (upsert) while `storage_chunk.py` does not, creating silent behavioral divergence.

---

### I-04. `storage_chunk.get_existing_article_ids` return type `Set[int]` but returns UUIDs
**File:** `core/storage_chunk.py:47-64`
**Confidence:** 88

The method queries `articles.id` (UUID column) but annotates the return as `Set[int]`.

---

### I-05. `storage_prompt.get_prompts_for_article_ids` makes unnecessary second DB call
**File:** `core/storage_prompt.py:112-119`
**Confidence:** 88

After the main query (which already uses `COALESCE` with the default prompt), the code calls `get_default_prompt()` for missing article IDs — opening a second DB connection unnecessarily.

---

### I-06. `storage_reranker_log` avg_rank_change uses wrong correlation key
**File:** `core/storage_reranker_log.py:178-189`
**Confidence:** 85

The UPDATE subquery correlates on `query_log_id` rather than `reranker_log_id`. Works today due to UNIQUE constraint but is logically incorrect.

---

### I-07. `storage_prompt.update_prompt` has fragile parameter ordering
**File:** `core/storage_prompt.py:246-256`
**Confidence:** 82

`tag_name` is appended to `params` before the SET clause loop completes. Works only because `NOW()` doesn't consume a `%s`, but will break if that changes.

---

### I-08. `to_dict()` crashes on `None` `last_modified_date`
**Files:** `core/bm25_search.py:57` and `core/vector_search.py:54`
**Confidence:** 85

`TextChunk` constructed with `last_modified_date=result_dict.get("created_at")` may receive `None`, but `TextChunk.last_modified_date` is typed as non-optional `datetime`.

---

### I-09. Hardcoded `"cohere/"` prefix in reranker ignores config
**File:** `core/reranker.py:95, 185`
**Confidence:** 88

```python
response = litellm.rerank(model=f"cohere/{self._proxy_model}", ...)
```

If `RERANKER_MODEL` already contains a provider prefix or uses a non-Cohere model, this produces an invalid model string.

---

### I-10. `reranking_failed=True` when reranker is simply unavailable — poisons analytics
**File:** `api/utils/hybrid_search.py:228-234`
**Confidence:** 85

Every hybrid search when the reranker is not configured gets logged as a "failure," polluting failure analytics with expected "not configured" states.

---

### I-11. `bm25_validate` search method not in DB CHECK constraint
**File:** `api/routers/search.py:304`
**Confidence:** 88

The `query_results` table has a CHECK constraint on `search_method` (bm25/vector/hybrid/hyde). `bm25_validate` is not in this list and will cause a constraint violation.

---

### I-12. Pre-reranking results leaked into API response
**File:** `api/utils/hybrid_search.py:265` and `api/routers/search.py:673`
**Confidence:** 82

`reranking_metadata["rrf_results_before_reranking"]` contains the full pre-reranking result set (20 text documents), which is spread into the response metadata, potentially doubling response size.

---

### I-13. `processing.py` creates new `Tokenizer()` on every `text_to_chunks` call
**File:** `core/processing.py:120-125`
**Confidence:** 85

Should be an instance attribute initialized once in `__init__`.

---

### I-14. `pipeline.py` uses naive `datetime.now()` for timing
**File:** `core/pipeline.py:430, 533`
**Confidence:** 82

Rest of codebase uses `datetime.now(timezone.utc)`. Naive datetimes will mismatch if compared with UTC-aware values.

---

### I-15. `article.id` (Optional) passed to non-optional `parent_article_id`
**File:** `core/pipeline.py:260`
**Confidence:** 82

`TdxArticle.id` is `UUID | None`. `TextChunk.parent_article_id` is `UUID` (not Optional). Will raise `ValidationError` if called on articles not yet inserted.

---

### I-16. Eval config mutated in-place
**File:** `qa/eval_runner.py:95`
**Confidence:** 90

```python
cfg.max_top_k = max(cfg.max_top_k, max(cfg.k_values))
```

Mutates the caller-supplied mutable `@dataclass`. If reused across runs, the second run sees modified values.

---

### I-17. `evaluate` CLI missing `hybrid_reranked` and `hyde` choices
**File:** `main.py:166-170`
**Confidence:** 85

The `--methods` argparse choices only list `["bm25", "vector", "hybrid"]`, but `eval_runner.run()` supports `hybrid_reranked` and `hyde`.

---

### I-18. MRR `search_depth` uses wrong list for article-level evaluation
**File:** `qa/eval_metrics.py:116`
**Confidence:** 88

When `level="article"`, depth is computed from `retrieved_chunk_ids` instead of `retrieved_article_ids`.

---

### I-19. `param_sweep` conflates API errors with zero-result queries
**File:** `qa/param_sweep.py:399-444`
**Confidence:** 85

`zero_count` is incremented for both error responses and genuine zero-result queries, making optimization decisions unreliable.

---

### I-20. Asymmetric percentile index calculation in `param_sweep`
**File:** `qa/param_sweep.py:643-644`
**Confidence:** 85

Lower percentiles use `int(n * p) - 1` while upper percentiles use `int(n * p)`. Produces values one step too low for p10/p25.

---

### I-21. `admin_analytics.get_content_stats` bypasses dependency injection
**File:** `api/routers/admin_analytics.py:238-270`
**Confidence:** 85

Accesses the pool directly from `request.app.state` and constructs raw SQL inline, inconsistent with all other endpoints.

---

### I-22. `query_logs` `created_at` is synthesized, not DB-sourced
**File:** `api/routers/query_logs.py:103-110`
**Confidence:** 83

Returns `datetime.now(timezone.utc)` instead of the actual `created_at` from the database row.

---

## Security Issues

### S-01. CORS wildcard origin + `allow_credentials=True`
**File:** `api/main.py:249-255`
**Confidence:** 100

Violates the CORS spec. Browsers will refuse credentialed cross-origin requests with `*` origin.

**Fix:** Either remove `allow_credentials=True` or enumerate explicit allowed origins.

---

### S-02. Admin endpoints have no authentication
**Files:** `api/routers/admin_prompts.py:27-113` and `api/routers/admin_analytics.py:39-270`
**Confidence:** 95

Admin endpoints allow reading/writing system prompts and viewing raw user queries. The `popular-queries` endpoint returns verbatim user queries — FERPA-sensitive in a university environment. "Internal network only" is network-level intent, not application enforcement.

---

### S-03. SQL injection risk via `table_name` string interpolation
**File:** `core/storage_vector.py:56, 84, 152, 219, 268, 310, 334, 360`
**Confidence:** 88

`self.table_name` is f-string interpolated into SQL. Currently safe (hardcoded value), but `VectorStorageClient.__init__` accepts arbitrary `table_name: str`.

**Fix:** Add allowlist validation in `__init__`.

---

### S-04. Bearer token prefix logged at DEBUG level
**File:** `core/api_client.py:174`
**Confidence:** 85

```python
logger.debug(f"Bearer token: {self.bearer_token[:20]}...")
```

Even a 20-char JWT prefix is sensitive. Violates the project's own security principle.

---

### S-05. Duplicate request models invite divergence
**File:** `api/models/requests.py:181-284`
**Confidence:** 80

`HybridSearchRequest` and `HyDESearchRequest` are identical classes. Any future field addition must be manually duplicated. `HyDESearchRequest` should inherit from `HybridSearchRequest`.

---

## Type Safety & Code Quality

### Q-01. `embedding.py` uses `logging.getLogger` instead of project `get_logger`
**File:** `core/embedding.py:14`
**Confidence:** 90

Bypasses the project's rotating file handlers and colored console output. Same issue in `qa/dataset_generation.py:19`.

---

### Q-02. `Dict`/`List` from `typing` instead of built-in generics
**Files:** `core/schemas.py:3`, `core/ingestion.py:3`
**Confidence:** 80

CLAUDE.md specifies Python 3.11+. Use `dict[str, Any]`, `list[str]` instead.

---

### Q-03. Docstring placed after assignments — is dead code
**File:** `core/api_client.py:43-58`
**Confidence:** 100

`__init__` docstring appears after attribute assignments, making it an expression statement, not a discoverable docstring.

---

### Q-04. Tokenizer silently returns 0 for unrecognized model names
**File:** `core/tokenizer.py:13-15`
**Confidence:** 80

`litellm.token_counter()` returns 0 for unknown models without raising, leading to oversized chunks.

---

### Q-05. BM25 `_get_chunks()` branching logic is redundant
**File:** `core/bm25_search.py:198-207`
**Confidence:** 83

`has_filters or status_names is not None` includes a redundant check and routes default filters to `get_all_chunks_filtered` instead of `get_all_chunks`, making caching semantics confusing.

---

### Q-06. Duplicate key assignments in `search_hyde`
**File:** `api/routers/search.py:868-869, 892-893`
**Confidence:** 80

```python
reranking_metadata["reranker_status"] = "unavailable"
reranking_metadata["reranker_status"] = "unavailable"  # exact duplicate
```

Copy-paste dead assignments in both the unavailable and failed branches.

---

### Q-07. Naive `datetime.now()` for report filenames
**Files:** `qa/eval_runner.py:552`, `qa/param_sweep.py:819, 1779`
**Confidence:** 82

Report metadata uses `datetime.now(timezone.utc)` but filenames use naive local time. Timestamps will mismatch on UTC+N servers.

---

### Q-08. `dataset_generation.py` throughput average uses pre-filter chunk count
**File:** `qa/dataset_generation.py:569`
**Confidence:** 88

Divides total time by original chunk count, not actually-processed chunks, underreporting processing time.

---

## Test Quality Issues

### T-01. `test_logger.py` is flaky due to shared global logger state
**File:** `tests/test_logger.py:84-93`
**Confidence:** 85

`logging.getLogger("test_duplicate")` is a global singleton. Prior tests in the session may leave handlers attached. Tests also create log files in the repo root.

**Fix:** Use unique logger names, clean up handlers in teardown, or pass `file_output=False`.

---

### T-02. `conftest.py` `articles_with_missing_fields` fixture is dead code
**File:** `tests/conftest.py:97-128`
**Confidence:** 80

Defined but never requested by any test. The same data is rebuilt inline in `test_ingestion.py`.

---

## Test Coverage Gaps

| Gap | Description |
|-----|-------------|
| **`_evaluate_hybrid` and `_evaluate_hybrid_reranked`** | No test classes in `test_eval_runner.py` for hybrid evaluation paths |
| **`mrr()` with `max_k` parameter** | The `max_k` cutoff behavior is never exercised in `test_eval_metrics.py` |
| **`_parse_jsonl_record` with <3 QA pairs** | No test for 0, 1, or 2 item `qa_pairs` lists |
| **`param_sweep` percentile values** | `test_computes_percentiles` only checks `max` and `min`, not intermediate percentiles |
| **Admin endpoints** | No tests for `admin_prompts.py` or `admin_analytics.py` routers |
| **Connection pool lifecycle** | No tests for pool exhaustion, timeout, or cleanup under concurrent load |
| **Hybrid search RRF fusion** | No unit test for `reciprocal_rank_fusion()` or `weighted_score_fusion()` |

---

## Full Issue Index

| ID | Severity | Confidence | File | Summary |
|----|----------|------------|------|---------|
| C-01 | Critical | 95 | `core/embedding.py:119,186`, `core/hyde_generator.py:232` | `asyncio.run()` crashes inside event loop |
| C-02 | Critical | 95 | `core/storage_vector.py:431` | Vector search params off-by-one |
| C-03 | Critical | 95 | `core/ingestion.py:336` | Non-approved article cleanup is a no-op |
| C-04 | Critical | 90 | `core/bm25_search.py:121` | No thread safety on BM25 cache |
| C-05 | Critical | 92 | `core/storage_query_log.py:424` | False atomicity in `log_query_with_results` |
| C-06 | Critical | 88 | `core/storage_base.py:151` | `close()` raises `AttributeError` in pool mode |
| C-07 | Critical | 100 | `core/storage_raw.py:510` | Double `conn.commit()` |
| C-08 | Critical | 92 | `core/reranker.py:200` | Crash on empty reranked results |
| C-09 | Critical | 85 | `core/ingestion.py:109` | Naive/aware datetime comparison |
| C-10 | Critical | 88 | `api/utils/hybrid_search.py:28` | RRF k=1 vs standard k=60 |
| C-11 | Critical | 85 | `core/bm25_search.py:407` | PromptStorageClient connection leak |
| C-12 | Critical | 95 | `core/bm25_search.py:490` | batch_search rebuilds model per query |
| I-01 | Important | 90 | `core/pipeline.py:490` | Loads entire chunk table for incremental embed |
| I-02 | Important | 80 | `core/api_client.py:471` | TOCTOU race in RateLimiter |
| I-03 | Important | 88 | `core/storage_raw.py:299` | Chunk storage duplicated in wrong module |
| I-04 | Important | 88 | `core/storage_chunk.py:47` | Return type `Set[int]` but returns UUIDs |
| I-05 | Important | 88 | `core/storage_prompt.py:112` | Unnecessary second DB call |
| I-06 | Important | 85 | `core/storage_reranker_log.py:178` | Wrong correlation key in avg_rank_change |
| I-07 | Important | 82 | `core/storage_prompt.py:246` | Fragile parameter ordering |
| I-08 | Important | 85 | `core/vector_search.py:197` | `to_dict()` crashes on None date |
| I-09 | Important | 88 | `core/reranker.py:95` | Hardcoded `cohere/` prefix |
| I-10 | Important | 85 | `api/utils/hybrid_search.py:228` | `reranking_failed=True` poisons analytics |
| I-11 | Important | 88 | `api/routers/search.py:304` | `bm25_validate` not in DB constraint |
| I-12 | Important | 82 | `api/utils/hybrid_search.py:265` | Pre-reranking results leaked in response |
| I-13 | Important | 85 | `core/processing.py:120` | Tokenizer re-instantiated per call |
| I-14 | Important | 82 | `core/pipeline.py:430` | Naive `datetime.now()` for timing |
| I-15 | Important | 82 | `core/pipeline.py:260` | Optional UUID passed to non-optional field |
| I-16 | Important | 90 | `qa/eval_runner.py:95` | Eval config mutated in-place |
| I-17 | Important | 85 | `main.py:167` | CLI missing eval method choices |
| I-18 | Important | 88 | `qa/eval_metrics.py:116` | MRR uses wrong list for article-level |
| I-19 | Important | 85 | `qa/param_sweep.py:399` | Error and zero-result counts conflated |
| I-20 | Important | 85 | `qa/param_sweep.py:643` | Asymmetric percentile calculation |
| I-21 | Important | 85 | `api/routers/admin_analytics.py:238` | Bypasses dependency injection |
| I-22 | Important | 83 | `api/routers/query_logs.py:103` | Synthesized `created_at` |
| S-01 | Security | 100 | `api/main.py:249` | CORS wildcard + credentials |
| S-02 | Security | 95 | `api/routers/admin_*.py` | Admin endpoints unauthenticated |
| S-03 | Security | 88 | `core/storage_vector.py:56` | SQL injection via table_name |
| S-04 | Security | 85 | `core/api_client.py:174` | Bearer token logged |
| S-05 | Security | 80 | `api/models/requests.py:181` | Duplicate models invite divergence |
| Q-01 | Quality | 90 | `core/embedding.py:14` | Wrong logger used |
| Q-02 | Quality | 80 | `core/schemas.py:3` | Legacy typing imports |
| Q-03 | Quality | 100 | `core/api_client.py:43` | Docstring is dead code |
| Q-04 | Quality | 80 | `core/tokenizer.py:13` | Silent zero token count |
| Q-05 | Quality | 83 | `core/bm25_search.py:198` | Redundant branching logic |
| Q-06 | Quality | 80 | `api/routers/search.py:868` | Duplicate key assignments |
| Q-07 | Quality | 82 | `qa/eval_runner.py:552` | Naive datetime in filenames |
| Q-08 | Quality | 88 | `qa/dataset_generation.py:569` | Wrong throughput denominator |
| T-01 | Test | 85 | `tests/test_logger.py:84` | Flaky global logger state |
| T-02 | Test | 80 | `tests/conftest.py:97` | Dead fixture |
