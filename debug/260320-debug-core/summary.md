# Debug Summary — core/**/*.* — 2026-03-20

## Executive Summary

20-iteration bounded debug sweep of `core/` (22 Python files). Found **15 bugs** across 12 files. Two are **CRITICAL** — one causes incorrect vector search results, the other degrades RAG chunk quality across the entire system.

## Statistics

| Metric | Value |
|--------|-------|
| Iterations | 20 |
| Hypotheses tested | 20 |
| Bugs confirmed | 15 |
| Hypotheses disproven | 4 |
| Reclassified | 2 |
| Files investigated | 14 / 22 |
| Techniques used | 4 / 7 (direct inspection, trace, pattern search, differential) |

## Severity Breakdown

| Severity | Count | IDs |
|----------|-------|-----|
| CRITICAL | 2 | #1 (vector search params), #2 (whitespace collapse) |
| HIGH | 1 | #3 (silently dropped articles) |
| MEDIUM | 7 | #4-#11 |
| LOW | 4 | #12-#15 |

## Priority Fix Order

1. **Bug #1 (CRITICAL):** Fix `search_similar_vectors` param binding — one-line fix, blocks correct vector search
2. **Bug #2 (CRITICAL):** Fix `process_text()` whitespace handling — requires re-processing all chunks after fix
3. **Bug #3 (HIGH):** Track dropped articles in `_categorize_articles()`
4. **Bug #8 (MEDIUM):** Remove bearer token from debug logs (security)
5. **Bug #7 (MEDIUM):** Add `command` to bulk `insert_query_logs`
6. **Bug #11 (MEDIUM):** Fix `last_modified_date` column selection in vector search SQL
7. Remaining MEDIUM/LOW bugs by effort

## Debug Score

```
debug_score = 15 * 15          = 225  (bugs found)
            + 20 * 3           = 60   (hypotheses tested)
            + (14/22) * 40     = 25.5 (file coverage)
            + (4/7) * 10       = 5.7  (technique diversity)
                               ------
            Total              = 316.2
```

## Files Not Investigated

- core/__init__.py (empty/minimal)
- core/storage_base.py (reviewed by agent, not directly verified beyond double-commit claim)
- core/storage_hyde_log.py (reviewed by agent, no high-severity claims)
- core/storage_cache_metrics.py (reviewed by agent, no claims to verify)
- Plus 4 more peripheral files

## Recommendations

1. **Immediate:** Fix bugs #1 and #2 before any further ingestion runs. Bug #2 means existing chunks may have degraded quality.
2. **Re-index:** After fixing bug #2, re-run the processing and embedding pipeline to regenerate all chunks with proper paragraph structure.
3. **Testing:** Add integration tests that verify vector search parameter binding with filters enabled (would have caught bug #1).
4. **Security:** Audit all DEBUG-level logging for credential leakage (bug #8 pattern).
