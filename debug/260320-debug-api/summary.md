# Debug Summary — API Layer

**Scope:** `api/**/*.*` | **Iterations:** 30 | **Date:** 2026-03-20

## Results

- **9 bugs confirmed** (2 High, 5 Medium, 2 Low)
- **7 code smells** identified
- **14 hypotheses disproven** (documented in eliminated.md)
- **Debug score:** 270.7

## Priority Fix Order

### Immediate (High)

1. **HTTPException swallowed in query_logs.py** — Add `except HTTPException: raise` before `except Exception`. One-line fix, prevents 404→500 conversion.

2. **Admin endpoints + CORS security** — Either add auth to admin write endpoints or restrict CORS origins. Most impactful security improvement.

### Soon (Medium)

3. **IndexError on empty hybrid results** — Guard `reranked_results[0]` with `if reranked_results:` check at hybrid_search.py:241.

4. **Strip `rrf_results_before_reranking` from response metadata** — Pop the key from metadata dict before building SearchResponse. Prevents response bloat.

5. **Fix `bm25_validate` search_method** — Either add to allowed methods in storage_query_log.py or use `"bm25"` as the method.

6. **CORS origins** — Replace `["*"]` with specific allowed origins.

7. **NULL last_modified_date handling** — Add fallback in vector_search.py where TextChunk is constructed.

### Low Priority

8. Remove duplicate dict key assignments in search.py (lines 868-869, 892-893).
9. Add degraded status to health check when BM25 corpus is empty.

## Files Modified (if fixes applied)

| File | Bugs | Fixes Needed |
|------|------|-------------|
| api/routers/query_logs.py | 1 | Add HTTPException re-raise |
| api/routers/search.py | 3 | Strip rrf metadata, fix search_method, remove duplicates |
| api/utils/hybrid_search.py | 1 | Guard empty results |
| api/main.py | 1 | Restrict CORS origins |
| api/routers/admin_prompts.py | 1 | Add auth (or decide to accept risk) |
| api/routers/health.py | 1 | Add degraded status for empty corpus |
| core/vector_search.py | 1 | Handle NULL last_modified_date |
