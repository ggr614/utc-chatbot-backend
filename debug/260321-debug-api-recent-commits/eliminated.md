# Eliminated Hypotheses — api/ Recent Commits

Date: 2026-03-21

These hypotheses were tested and disproven. Recording them prevents re-investigation.

---

## Hypothesis 6: Open redirect via path-like URLs (DISPROVEN)
- **Location:** `api/routers/auth.py:90`
- **Why tested:** `next_url` redirect after login could allow external redirects
- **Evidence:** The check `not next_url.startswith("/") or next_url.startswith("//")` correctly blocks absolute URLs and protocol-relative URLs. Starlette's `RedirectResponse` handles URL encoding. Commit `5236306` already addressed this.
- **Learning:** The fix is adequate for the threat model.

## Hypothesis 8: rrf_results_before_reranking leaking in response (DISPROVEN)
- **Location:** `api/routers/search.py:672,1025`
- **Why tested:** Full RRF results stored in metadata dict could leak to response
- **Evidence:** `.pop("rrf_results_before_reranking", None)` is called before building the response in BOTH hybrid and HyDE endpoints. Fixed in commit `92cc35e`.
- **Learning:** Fix was applied consistently to both code paths.

## Hypothesis 9: Admin routes dependency ordering (DISPROVEN)
- **Location:** `api/main.py:269-282`
- **Why tested:** Auth router registered before admin routes — could cause ordering issues
- **Evidence:** Auth router has no `require_admin` dependency (correct for login/logout). Admin routers have it at registration level. No ordering conflict.

## Hypothesis 10: Admin routes accessible without auth (DISPROVEN)
- **Location:** `api/main.py:273-282`
- **Why tested:** Multiple route types (HTML + API) in admin routers
- **Evidence:** `dependencies=[Depends(require_admin)]` at `include_router()` level applies to ALL routes in the router.

## Hypothesis 13: CORS wildcard allows cross-origin attacks (DISPROVEN)
- **Location:** `api/main.py:299-305`
- **Why tested:** `allow_origins=["*"]` could enable cross-origin API abuse
- **Evidence:** `allow_credentials=False` makes wildcard safe. API key is header-based, not cookie-based. Fixed in commit `d4c9a44`.

## Hypothesis 14: prompt_resolution.py issues (DISPROVEN)
- **Location:** `api/utils/prompt_resolution.py`
- **Why tested:** New utility file in recent commits
- **Evidence:** Clean, minimal implementation. Returns first result's system_prompt or None. No issues.
