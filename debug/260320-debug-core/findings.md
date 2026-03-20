# Debug Findings — core/**/*.* — 2026-03-20

## CRITICAL

### [CRITICAL] Bug #1: search_similar_vectors parameter binding off-by-one
- **Location:** [storage_vector.py:432](core/storage_vector.py#L432)
- **Hypothesis:** Extra query_vector in params misaligns all subsequent bindings
- **Evidence:** `params = [query_vector, query_vector, min_similarity]` for WHERE clause with 2 `%s` placeholders. The second `query_vector` binds to the `>= %s` threshold comparison (should be `min_similarity`). The `min_similarity` float then binds to the SELECT `similarity` calculation (should be `query_vector`).
- **Reproduction:** Any call to `search_similar_vectors()` with filters will produce a psycopg type error; without filters, the similarity threshold is non-functional.
- **Impact:** Similarity threshold comparison broken (comparing float result against vector). Reported similarity scores are wrong. Likely causes runtime DB type error when filters are applied.
- **Root cause:** Duplicate `query_vector` in initial params list
- **Suggested fix:** Change line 432 from:
  ```python
  params = [query_vector, query_vector, min_similarity]
  ```
  to:
  ```python
  params = [query_vector, min_similarity]
  ```

### [CRITICAL] Bug #2: process_text() destroys paragraph/list structure before chunking
- **Location:** [processing.py:54](core/processing.py#L54)
- **Hypothesis:** `" ".join(markdown_text.split())` collapses all whitespace including newlines
- **Evidence:** Python `.split()` without args splits on ALL whitespace (spaces, tabs, `\n`, `\r\n`). Joining with `" "` produces a single-line string. `RecursiveCharacterTextSplitter` default separators are `["\n\n", "\n", " ", ""]` — after flattening, only `" "` and `""` will ever match.
- **Impact:** Every chunk in the vector store loses semantic boundaries. Chunks split at arbitrary word boundaries instead of paragraph/section boundaries. This directly degrades RAG retrieval quality across the entire system.
- **Root cause:** Overly aggressive whitespace normalization
- **Suggested fix:**
  ```python
  import re
  cleaned_text = re.sub(r'\n{3,}', '\n\n', markdown_text).strip()
  # Or: cleaned_text = re.sub(r'[ \t]+', ' ', markdown_text).strip()  # only collapse horizontal whitespace
  ```

## HIGH

### [HIGH] Bug #3: Silently dropped articles not counted in sync statistics
- **Location:** [ingestion.py:114-116](core/ingestion.py#L114-L116)
- **Hypothesis:** Articles missing ID or ModifiedDate are silently dropped
- **Evidence:** `continue` at line 116 skips the article without adding to any output list or counter
- **Impact:** Caller statistics undercount skipped articles; data integrity issues go unnoticed
- **Root cause:** Missing tracking for skipped items
- **Suggested fix:** Add dropped articles to a separate list and return/log it

## MEDIUM

### [MEDIUM] Bug #4: asyncio.run() in sync wrappers — latent async incompatibility
- **Location:** [embedding.py:119,186](core/embedding.py#L119) and [hyde_generator.py:232](core/hyde_generator.py#L232)
- **Evidence:** All FastAPI endpoints are sync `def`, so uvicorn runs them in a threadpool (safe). But `asyncio.run()` will crash if endpoints are ever converted to `async def`.
- **Impact:** Latent time bomb — works today, breaks on async conversion
- **Suggested fix:** Use `nest_asyncio` or check for running loop before calling `asyncio.run()`

### [MEDIUM] Bug #5: RateLimiter race condition after sleep
- **Location:** [api_client.py:473-486](core/api_client.py#L473-L486)
- **Evidence:** After lock release, sleep, and re-acquire, capacity is NOT re-checked before appending
- **Impact:** Can exceed max_requests under concurrent access
- **Suggested fix:** Wrap in `while len(self.request_times) >= self.max_requests` loop

### [MEDIUM] Bug #6: log_query_with_results non-atomic
- **Location:** [storage_query_log.py:422-448](core/storage_query_log.py#L422-L448)
- **Evidence:** `log_query` and `log_query_results` each open separate connections/transactions
- **Impact:** Orphaned query_logs rows if results insertion fails
- **Suggested fix:** Consolidate into single connection context, or update docstring

### [MEDIUM] Bug #7: insert_query_logs omits command field
- **Location:** [storage_query_log.py:213-215](core/storage_query_log.py#L213-L215)
- **Evidence:** INSERT lists 6 columns; `command` is missing (present in single-row `log_query`)
- **Impact:** Bulk-inserted logs always have command=NULL
- **Suggested fix:** Add `command` to INSERT columns and values tuple

### [MEDIUM] Bug #8: Bearer token partially logged
- **Location:** [api_client.py:174](core/api_client.py#L174)
- **Evidence:** `logger.debug(f"Bearer token: {self.bearer_token[:20]}...")`
- **Impact:** Token fragment in debug logs — security antipattern in university environment
- **Suggested fix:** Remove or replace with `logger.debug("Bearer token received (length=%d)", len(self.bearer_token))`

### [MEDIUM] Bug #9: Inconsistent model prefix pattern
- **Location:** [reranker.py:95,185](core/reranker.py#L95)
- **Evidence:** Reranker applies `cohere/` at call site; embedding/HyDE apply `openai/` in `__init__`
- **Impact:** Fragile — config changes break differently per component
- **Suggested fix:** Standardize prefix application location

### [MEDIUM] Bug #10: Duplicate PostgresClient class name
- **Location:** [storage_raw.py:12](core/storage_raw.py#L12) and [storage_chunk.py:11](core/storage_chunk.py#L11)
- **Evidence:** Both define `class PostgresClient(BaseStorageClient)`
- **Impact:** Import collision risk; SoC violation (storage_raw has chunk methods)
- **Suggested fix:** Rename to `ArticleStorageClient` and `ChunkStorageClient`

### [MEDIUM] Bug #11: last_modified_date from wrong column
- **Location:** [vector_search.py:197-199](core/vector_search.py#L197-L199)
- **Evidence:** SQL selects `e.created_at` (embedding creation time); mapped to `last_modified_date`
- **Impact:** Downstream code gets wrong timestamp for content freshness
- **Suggested fix:** Select `c.last_modified_date` from `article_chunks` join

## LOW

### [LOW] Bug #12: Undefined name TextChunk in type annotation
- **Location:** [storage_raw.py:373](core/storage_raw.py#L373)
- **Evidence:** `TextChunk` not imported at module level (imported inside loop body at line 415)
- **Impact:** ruff F821; type checkers fail; runtime works
- **Suggested fix:** Add `TextChunk` to module-level imports

### [LOW] Bug #13: Naive datetime.now() in pipeline stats
- **Location:** [pipeline.py:430,533,569](core/pipeline.py#L430)
- **Evidence:** `datetime.now()` without timezone; all schemas use `datetime.now(timezone.utc)`
- **Impact:** Inconsistent timestamps in pipeline statistics
- **Suggested fix:** Replace with `datetime.now(timezone.utc)`

### [LOW] Bug #14: f-string SQL table_name interpolation
- **Location:** [storage_vector.py](core/storage_vector.py) (15+ sites)
- **Evidence:** `f"SELECT ... FROM {self.table_name}"` — currently safe (always "embeddings_openai")
- **Impact:** Defense-in-depth violation; architectural risk if table_name ever becomes dynamic
- **Suggested fix:** Use `psycopg.sql.Identifier()` or validate against allowlist

### [LOW] Bug #15: Em dash clean_text replacement is a no-op
- **Location:** [hyde_generator.py:98](core/hyde_generator.py#L98)
- **Evidence:** `"\u2014": "\u2014"` maps em dash to itself
- **Impact:** Cosmetic — em dashes pass through uncleaned
- **Suggested fix:** Change to `"\u2014": "--"`
