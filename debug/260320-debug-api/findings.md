# Debug Findings — API Layer (`api/**/*.*`)

**Date:** 2026-03-20
**Scope:** `api/**/*.*` (11 Python files, 2 HTML templates)
**Goal:** General bug finding
**Iterations:** 30

## Summary

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 2 |
| Medium | 5 |
| Low | 2 |
| Code Smells | 7 |
| **Total Bugs** | **9** |

---

## HIGH

### [HIGH] Bug #1: HTTPException swallowed by `except Exception` in query_logs.py
- **Location:** [api/routers/query_logs.py:80-83, 123](api/routers/query_logs.py#L80-L83)
- **Hypothesis:** HTTPException(404) raised inside try block is caught by generic `except Exception`
- **Evidence:** `HTTPException` inherits from `Exception`. The 404 raised at line 80 when a query log is not found is caught by `except Exception` at line 123, which converts it to a 500 error with message "Failed to log LLM response"
- **Impact:** Client receives HTTP 500 instead of HTTP 404 when query_log_id doesn't exist
- **Root cause:** Missing `except HTTPException: raise` before the generic `except Exception`
- **Suggested fix:**
  ```python
  except HTTPException:
      raise  # Re-raise HTTP exceptions as-is
  except ValueError as e:
      ...
  except Exception as e:
      ...
  ```

### [HIGH] Bug #2: Unauthenticated admin write endpoints + permissive CORS
- **Location:** [api/routers/admin_prompts.py](api/routers/admin_prompts.py) (all endpoints), [api/main.py:249-255](api/main.py#L249-L255)
- **Hypothesis:** Admin endpoints have no authentication while CORS allows all origins with credentials
- **Evidence:** No `Depends(verify_api_key)` on any admin endpoint. CORS is `allow_origins=["*"]` with `allow_credentials=True`. The `/api/v1/admin/prompts/bulk-save` endpoint can modify system prompts without auth.
- **Impact:** Any browser on any network visiting a malicious page can trigger CSRF attacks to modify system prompts, view analytics data, or delete prompts
- **Root cause:** Design decision "internal network only" combined with permissive CORS
- **Suggested fix:** Either (a) add API key auth to admin write endpoints, or (b) restrict CORS origins to specific admin UI domain

---

## MEDIUM

### [MEDIUM] Bug #3: CORS `allow_origins=["*"]` with `allow_credentials=True`
- **Location:** [api/main.py:249-255](api/main.py#L249-L255)
- **Hypothesis:** This combination violates CORS spec intent and allows any origin to make credentialed requests
- **Evidence:** FastAPI/Starlette works around the spec by reflecting the requesting origin, but this means ANY origin is effectively allowed with credentials
- **Impact:** Browsers from any origin can make authenticated cross-origin requests
- **Suggested fix:** Replace `["*"]` with specific allowed origins, or set `allow_credentials=False`

### [MEDIUM] Bug #4: IndexError when hybrid search returns 0 results
- **Location:** [api/utils/hybrid_search.py:241](api/utils/hybrid_search.py#L241)
- **Hypothesis:** `reranked_results[0]['combined_score']` crashes when result list is empty
- **Evidence:** If both BM25 and vector return empty results (empty corpus or all filtered out), `reranked_results` is `[]`, and `[0]` raises `IndexError`
- **Impact:** Hybrid search returns 500 instead of empty results when no matches found
- **Suggested fix:** Guard with `if reranked_results:` before logging top score

### [MEDIUM] Bug #5: BM25 validation queries never logged (`search_method="bm25_validate"` rejected)
- **Location:** [api/routers/search.py:304](api/routers/search.py#L304), [core/storage_query_log.py:287](core/storage_query_log.py#L287)
- **Hypothesis:** `search_method="bm25_validate"` is not in the allowed set
- **Evidence:** Validation at storage_query_log.py:287 only allows `("bm25", "vector", "hybrid", "hyde")`. The ValueError is caught by `except Exception` at search.py:311 and silently logged.
- **Impact:** BM25 validation queries are never recorded in analytics — missing data
- **Suggested fix:** Either add `"bm25_validate"` to the allowed methods or use `"bm25"` as the search_method

### [MEDIUM] Bug #6: `rrf_results_before_reranking` leaks into API response metadata
- **Location:** [api/routers/search.py:672-674](api/routers/search.py#L672-L674), [api/routers/search.py:1024-1028](api/routers/search.py#L1024-L1028)
- **Hypothesis:** Full RRF candidate list (with text content) is spread into response metadata
- **Evidence:** `**reranking_metadata` includes `rrf_results_before_reranking` (list of dicts with full TextChunk objects), bloating the response with all candidate text
- **Impact:** Response size bloated by 10-50x (40 chunks × full text), potential serialization issues
- **Suggested fix:** Pop or delete `rrf_results_before_reranking` from metadata before building response

### [MEDIUM] Bug #7: Latent Pydantic validation failure on NULL `last_modified_date`
- **Location:** [core/vector_search.py:197](core/vector_search.py#L197), [api/models/responses.py:50-52](api/models/responses.py#L50-L52)
- **Hypothesis:** `result_dict.get("last_modified_date")` returns None, passed to TextChunk non-Optional field
- **Evidence:** TextChunk.last_modified_date is `datetime` (not Optional) with default_factory. Explicitly passing `None` in Pydantic v2 causes ValidationError.
- **Impact:** If any DB row has NULL last_modified_date, all search endpoints crash with 500
- **Suggested fix:** Either make the field `Optional[datetime]` or use `result_dict.get("last_modified_date") or datetime.now(timezone.utc)`

---

## LOW

### [LOW] Bug #8: Duplicate dictionary key assignment (dead code)
- **Location:** [api/routers/search.py:868-869](api/routers/search.py#L868-L869), [api/routers/search.py:892-893](api/routers/search.py#L892-L893)
- **Evidence:** `reranking_metadata["reranker_status"]` assigned twice in succession (copy-paste error)
- **Impact:** No functional impact (dead code)
- **Suggested fix:** Remove duplicate lines

### [LOW] Bug #9: Health check reports "healthy" when BM25 corpus is empty
- **Location:** [api/routers/health.py:52-64](api/routers/health.py#L52-L64)
- **Evidence:** BM25 check sets status "healthy" even when `num_chunks=0`
- **Impact:** Misleading health status — system reports healthy but searches return empty
- **Suggested fix:** Set status to "degraded" when `num_chunks == 0`

---

## Code Smells (not bugs, but worth noting)

1. **No response model on GET LLM response** — [query_logs.py:141](api/routers/query_logs.py#L141): No `response_model`, undocumented API contract
2. **BM25 uses standalone connection** — [main.py:88-89](api/main.py#L88-L89): BM25Retriever creates own PostgresClient outside connection pool
3. **Latency truncation** — All search endpoints use `int()` truncation instead of `round()`, losing up to 0.999ms
4. **SQL construction anti-pattern** — [core/storage_prompt.py:251](core/storage_prompt.py#L251): f-string for SET clause (safe but fragile)
5. **BM25 validation hardcodes status filter** — [search.py:263](api/routers/search.py#L263): Hardcodes `["Approved"]` instead of inheriting from request
6. **Redundant `= ...` on DI params** — [admin_analytics.py:56](api/routers/admin_analytics.py#L56): Ellipsis default on Annotated+Depends params
7. **Inconsistent DI for content-stats** — [admin_analytics.py:238](api/routers/admin_analytics.py#L238): Direct app.state access instead of DI
