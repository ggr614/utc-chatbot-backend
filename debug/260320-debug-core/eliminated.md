# Eliminated Hypotheses — 2026-03-20

## Disproven

### Iteration 6: Timezone-naive vs aware datetime comparison
- **Hypothesis:** `_categorize_articles()` compares timezone-aware API dates with naive DB dates
- **Why disproven:** DB column `last_modified_date` is `TIMESTAMP WITH TIME ZONE`; psycopg v3 returns timezone-aware datetimes. API dates are normalized with `.replace("Z", "+00:00")`. Both sides are always aware.

### Iteration 18: Mutable default arguments
- **Hypothesis:** Python mutable default argument bug exists in core/
- **Why disproven:** Grep for `def \w+\(.*=\[\]` and `def \w+\(.*=\{\}` found zero matches. All defaults are immutable (None, str, int, bool).

### Iteration 19: Unhandled None returns
- **Hypothesis:** Functions return None on error paths without callers checking
- **Why disproven:** `get_prompt_by_tag` returns None intentionally; `get_default_prompt` raises ValueError if None. `get_query_by_id` returns None for missing IDs (documented). All callers handle None appropriately.

### Iteration 20: Additional SQL injection via string formatting
- **Hypothesis:** SQL injection risks beyond table_name f-string
- **Why disproven:** All f-string SQL in storage_*.py is isolated to `{self.table_name}` in storage_vector.py. All other SQL uses parameterized queries (`%s` placeholders) correctly.

## Reclassified

### Iteration 2: asyncio.run() in sync wrappers (CRITICAL -> MEDIUM)
- **Original claim:** Crashes inside existing event loops (FastAPI/uvicorn)
- **Why reclassified:** All FastAPI endpoints are sync `def`, so uvicorn runs them in a threadpool with no running event loop. `asyncio.run()` is safe in this configuration. Reclassified to MEDIUM as a latent risk.

### Iteration 9: get_prompts_for_article_ids fallback (HIGH -> LOW)
- **Original claim:** Logic error mapping non-existent article IDs to default prompt
- **Why reclassified:** This is intentional graceful degradation. The fallback ensures callers always get a prompt, even for edge cases. Not a bug — a design choice.
