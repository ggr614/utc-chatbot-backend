# Debug Summary — api/ Recent Commits

**Date:** 2026-03-21
**Scope:** `api/` directory (20 Python files, 2 HTML templates)
**Goal:** Hunt bugs introduced or present in recent commits
**Iterations:** 15

## Results

| Metric | Value |
|--------|-------|
| Bugs found | 7 (0 Critical, 1 High, 2 Medium, 4 Low) |
| Hypotheses tested | 15 (9 confirmed, 6 disproven) |
| Files investigated | 15 / 20 in scope |
| Techniques used | direct inspection, pattern search, differential (3/7) |

## Debug Score

```
debug_score = 7 * 15 + 15 * 3 + (15/20) * 40 + (3/7) * 10
            = 105 + 45 + 30 + 4.3
            = 184.3
```

## Priority Fix Order

1. **[HIGH] chat_service missing DI** — `openai_compat.py:54` — Simple fix, high impact. Use existing `get_chat_service` dependency.
2. **[MEDIUM] Cookie missing secure flag** — `auth.py:94` — Add `secure=True` to session cookie.
3. **[MEDIUM] Exception leak in SSE** — `openai_compat.py:116` — Replace `str(e)` with generic message.
4. **[LOW] User enumeration** — `auth.py:74` — Unify error messages.
5. **[LOW] Log injection** — `auth.py:65,71,75,87` — Sanitize username in logs.
6. **[LOW] Health endpoint info leak** — `health.py:77,102,143,204` — Genericize error messages.
7. **[LOW] Chat input validation** — `chat.py:18` — Add max_length bounds.

## Key Observations

- The recent auth commits (login/logout, JWT, admin protection) are well-structured with good patterns (timing-safe password comparison, open redirect prevention, router-level dependency injection).
- Prior bug fixes (commits 92cc35e, d4c9a44, 702018e, etc.) were correctly applied and verified.
- The `openai_compat.py` router is the newest and least polished — it bypasses the dependency injection pattern used everywhere else and leaks exceptions.
- No test failures or lint issues — codebase is clean from a CI perspective.
