# Admin Authentication Design

**Date:** 2026-03-20
**Status:** Approved
**Approach:** Manual cookie-based JWT (Approach 2)

## Problem

Admin endpoints (`/admin/prompts`, `/admin/analytics`, `/api/v1/admin/*`) are completely unauthenticated. Anyone who can reach the API can modify system prompts that control LLM behavior or view query analytics data. The current "internal network only" assumption is fragile.

Search endpoints remain protected by the existing `X-API-Key` header mechanism, which is unaffected by this change.

## Requirements

- **Users:** Small team of named admins (static list, no self-registration)
- **Access method:** Browser-based HTML pages only
- **Network:** Internal UTC network / VPN only
- **Roles:** Single role - all admins get full access
- **Session length:** Short (2 hours), re-login required after expiry
- **Scope:** Admin endpoints only; search API auth is unchanged

## Architecture

### Authentication Flow

**Login:**
1. Admin navigates to `/admin/login` - served a simple HTML login form
2. Form POSTs username + password to `POST /auth/login`
3. Backend verifies credentials against `admin_users` table using argon2id
4. On success: creates a JWT (`sub=user_id`, `exp=now+2h`), signed with `AUTH_SECRET_KEY`
5. Sets JWT as an `HttpOnly`, `SameSite=Lax`, `Path=/` cookie named `admin_session`
6. Redirects to the originally requested admin page (or `/admin/prompts` by default)

**Protected requests:**
1. Admin hits any `/admin/*` or `/api/v1/admin/*` endpoint
2. `require_admin` FastAPI dependency reads the `admin_session` cookie
3. Decodes + validates the JWT (signature, expiry)
4. If invalid/expired: HTML pages redirect to `/admin/login`; API endpoints return 401
5. Injects the current admin user into the route handler for audit logging

**Logout:**
1. `POST /auth/logout` clears the `admin_session` cookie (`Max-Age=0`)
2. Redirects to `/admin/login`

### Security Properties

- `HttpOnly` - JavaScript cannot read the cookie (XSS protection)
- `SameSite=Lax` - cross-origin POST requests (forms, fetch) will not include the cookie, which protects mutating admin endpoints from CSRF. Cross-origin GET navigation will still include the cookie, but all admin GETs are read-only so this is acceptable.
- `Path=/` - cookie sent on all requests. Using `Path=/admin` would exclude `/api/v1/admin/*` endpoints. The `HttpOnly` and `SameSite=Lax` flags provide sufficient protection; the cookie is only validated by admin route dependencies.
- No `Secure` flag (internal network, HTTP) - easy to add later if HTTPS is enabled
- JWT is stateless - no DB hit on every request, just signature verification
- 2-hour expiry with no refresh token - admins re-login after timeout
- Timing-attack-safe password verification via argon2id

### CORS Note

The existing CORS middleware uses `allow_origins=["*"]` with `allow_credentials=False`. This is unchanged. The admin HTML pages are served from the same origin as the API, so `fetch()` calls from admin templates to `/api/v1/admin/*` are same-origin requests and do not go through CORS. Admin templates must NOT set `credentials: 'include'` with cross-origin mode.

### Future Hardening (Deferred)

- **Brute-force protection:** Add a failed-attempt counter per username with temporary lockout (e.g., 5 failures = 15-minute lock). Low priority given internal-network-only deployment.
- **`Secure` cookie flag:** Enable when HTTPS is configured.
- **Audit logging:** Log admin actions (prompt changes, etc.) with the authenticated user identity.

## Database Schema

New table managed via Alembic migration:

```sql
CREATE TABLE admin_users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username        TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    display_name    TEXT,
    is_active       BOOLEAN NOT NULL DEFAULT true,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at   TIMESTAMPTZ
);
```

No user registration endpoint. Admins are seeded via CLI command (`python main.py create-admin`).

## Configuration

New `AuthSettings` Pydantic class in `core/config.py`:

```
AUTH_SECRET_KEY          # Required. JWT signing key (64-char hex, generated with: python -c "import secrets; print(secrets.token_hex(32))")
AUTH_TOKEN_EXPIRE_MINUTES=120   # 2 hours default
AUTH_COOKIE_NAME=admin_session  # Cookie name
```

`AUTH_SECRET_KEY` is a required field (Pydantic `SecretStr`). If not set, the application will fail to start — same behavior as `DB_PASSWORD` or `TDX_WEBSERVICES_KEY`. This is intentional: admin auth should not silently degrade to unauthenticated.

Separate from existing `APISettings`. Search API key auth is completely unaffected.

## New Dependencies

- `PyJWT[crypto]` - JWT encode/decode (actively maintained; `python-jose` is unmaintained since 2022)
- `argon2-cffi` - argon2id password hashing (OWASP recommended)

## File Structure

### New Files

| File | Purpose |
|------|---------|
| `api/auth.py` | Core auth logic: password hashing, JWT create/verify, `require_admin` dependency (smart: redirects HTML requests, returns 401 for JSON) |
| `api/routers/auth.py` | Login/logout routes: `GET /admin/login`, `POST /auth/login`, `POST /auth/logout` |
| `api/templates/admin_login.html` | HTML login form (styled consistent with existing admin pages) |
| `core/storage_admin_user.py` | `AdminUserClient` - psycopg + connection pool, follows existing storage client pattern |
| `alembic/versions/xxx_add_admin_users.py` | Migration for `admin_users` table |

### Modified Files

| File | Change |
|------|--------|
| `core/config.py` | Add `AuthSettings` class + `get_auth_settings()` cached accessor |
| `api/main.py` | Register auth router; add `require_admin` dependency to admin router includes |
| `api/routers/admin_prompts.py` | No changes needed - protected via router-level `require_admin` dependency |
| `api/routers/admin_analytics.py` | No changes needed - protected via router-level `require_admin` dependency |
| `api/dependencies.py` | Add `get_admin_user_client()` dependency |
| `requirements.txt` | Add `PyJWT[crypto]`, `argon2-cffi` |
| `main.py` | Add `create-admin` CLI command |
| `api/templates/admin_prompts.html` | Add logout button/link |
| `api/templates/admin_analytics.html` | Add logout button/link |

### Unchanged

- Search endpoints, health checks, query logs - no auth changes
- Existing `verify_api_key` dependency - untouched
- All existing admin business logic - only auth gate added

## Route Protection Strategy

Admin routers are protected at the `include_router` level in `api/main.py`:

```python
app.include_router(
    admin_prompts.router,
    tags=["Admin - Prompts"],
    dependencies=[Depends(require_admin)],
)
```

This protects all routes in those routers automatically.

**Smart `require_admin` dependency:** A single dependency handles both HTML and JSON requests. It reads the `Accept` header to decide the error response:
- If `Accept` contains `text/html` (browser navigation) -> `RedirectResponse` to `/admin/login`
- Otherwise (JSON API call, e.g. `fetch()` from admin SPA) -> 401 Unauthorized JSON response

This avoids the complexity of two separate dependencies and the conflict that would arise from applying both a router-level and per-route dependency.

- `GET /admin/prompts` (browser) -> redirect to login if unauthenticated
- `GET /api/v1/admin/prompts` (fetch from JS) -> 401 Unauthorized if unauthenticated

The auth router (`/admin/login`, `/auth/login`, `/auth/logout`) has no auth dependency.

**Logout UX:** Admin HTML templates include a small `<form>` with a logout button that POSTs to `/auth/logout`. This keeps logout as a POST (no side effects on GET) while being simple for users.

## CLI: create-admin Command

```bash
python main.py create-admin --username david --display-name "David Wood"
```

- Prompts for password interactively (not passed as CLI arg for security)
- Hashes password with argon2id
- Inserts into `admin_users` table
- Idempotent: if username exists, offers to update the password
- `last_login_at` is updated by the login endpoint on each successful authentication

## Testing

- Unit tests for `api/auth.py`: JWT creation/validation, password hashing/verification, dependency behavior with valid/invalid/expired tokens
- Unit tests for `core/storage_admin_user.py`: CRUD operations (mocked DB)
- Unit tests for `api/routers/auth.py`: login success/failure, logout, redirect behavior
- Integration: existing admin endpoint tests updated to include auth cookie
- All tests use mocked DB connections (consistent with existing test patterns)
