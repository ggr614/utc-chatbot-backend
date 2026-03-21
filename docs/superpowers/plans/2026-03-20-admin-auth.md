# Admin Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cookie-based JWT authentication to all admin endpoints, protecting system prompt management and analytics from unauthorized access.

**Architecture:** Manual JWT auth layer using PyJWT + argon2-cffi. Admin credentials stored in `admin_users` PostgreSQL table. JWT set as HttpOnly cookie on login, validated by a single smart `require_admin` FastAPI dependency that redirects HTML requests and returns 401 for JSON requests. Follows existing psycopg + BaseStorageClient patterns.

**Tech Stack:** PyJWT[crypto], argon2-cffi, FastAPI dependencies, Alembic migrations, psycopg

**Spec:** `docs/superpowers/specs/2026-03-20-admin-auth-design.md`

---

### Task 1: Install Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Install PyJWT and argon2-cffi**

Run:
```bash
cd /c/Python/HelpdeskChatbot/utc-chatbot-backend && .venv/Scripts/pip install "PyJWT[crypto]" argon2-cffi
```

- [ ] **Step 2: Also install pytest-asyncio for async test support**

Run:
```bash
cd /c/Python/HelpdeskChatbot/utc-chatbot-backend && .venv/Scripts/pip install pytest-asyncio
```

- [ ] **Step 3: Manually add new dependencies to requirements.txt**

Add these lines to `requirements.txt` (alphabetically, matching existing format). Do NOT use `pip freeze` — the existing file is curated with direct dependencies only:

```
argon2-cffi==<installed version>
PyJWT==<installed version>
cryptography==<installed version>
pytest-asyncio==<installed version>
```

Get exact versions with:
```bash
.venv/Scripts/pip show PyJWT argon2-cffi cryptography pytest-asyncio | grep -E "^(Name|Version)"
```

- [ ] **Step 5: Verify imports work**

Run:
```bash
cd /c/Python/HelpdeskChatbot/utc-chatbot-backend && .venv/Scripts/python -c "import jwt; import argon2; print('PyJWT:', jwt.__version__); print('argon2-cffi:', argon2.__version__)"
```
Expected: Version numbers printed without errors.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt
git commit -m "feat(auth): add PyJWT and argon2-cffi dependencies"
```

---

### Task 2: Add AuthSettings Configuration

**Files:**
- Modify: `core/config.py:64-84` (after APISettings class)

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_auth.py`:

```python
"""Tests for AuthSettings configuration."""

import pytest
from unittest.mock import patch

from core.config import AuthSettings, get_auth_settings


class TestAuthSettings:
    """Test suite for AuthSettings."""

    def test_auth_settings_loads_from_env(self):
        """Test AuthSettings loads required and optional fields from env."""
        env_vars = {
            "AUTH_SECRET_KEY": "a" * 64,
            "AUTH_TOKEN_EXPIRE_MINUTES": "60",
            "AUTH_COOKIE_NAME": "test_session",
        }
        with patch.dict("os.environ", env_vars, clear=False):
            settings = AuthSettings()
            assert settings.SECRET_KEY.get_secret_value() == "a" * 64
            assert settings.TOKEN_EXPIRE_MINUTES == 60
            assert settings.COOKIE_NAME == "test_session"

    def test_auth_settings_defaults(self):
        """Test AuthSettings uses correct defaults."""
        env_vars = {
            "AUTH_SECRET_KEY": "b" * 64,
        }
        with patch.dict("os.environ", env_vars, clear=False):
            settings = AuthSettings()
            assert settings.TOKEN_EXPIRE_MINUTES == 120
            assert settings.COOKIE_NAME == "admin_session"

    def test_auth_settings_missing_secret_key_raises(self):
        """Test AuthSettings raises if AUTH_SECRET_KEY not set."""
        # Remove AUTH_SECRET_KEY if present, keep other env vars intact
        with patch.dict("os.environ", {"AUTH_SECRET_KEY": ""}, clear=False):
            with pytest.raises(Exception):
                AuthSettings()

    def test_get_auth_settings_cached(self):
        """Test get_auth_settings returns cached instance."""
        env_vars = {"AUTH_SECRET_KEY": "c" * 64}
        with patch.dict("os.environ", env_vars, clear=False):
            # Clear lru_cache before test
            get_auth_settings.cache_clear()
            s1 = get_auth_settings()
            s2 = get_auth_settings()
            assert s1 is s2
            get_auth_settings.cache_clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_config_auth.py -v`
Expected: FAIL — `AuthSettings` does not exist yet.

- [ ] **Step 3: Implement AuthSettings**

Add to `core/config.py` after the `APISettings` class (around line 83):

```python
class AuthSettings(BaseSettings):
    """Settings for admin authentication."""

    SECRET_KEY: SecretStr
    TOKEN_EXPIRE_MINUTES: int = 120
    COOKIE_NAME: str = "admin_session"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        env_prefix="AUTH_",
    )
```

Add the cached accessor after `get_api_settings()`:

```python
@lru_cache()
def get_auth_settings() -> AuthSettings:
    """Get cached auth settings for admin authentication."""
    return AuthSettings()  # type: ignore[call-arg]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/pytest tests/test_config_auth.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add core/config.py tests/test_config_auth.py
git commit -m "feat(auth): add AuthSettings configuration class"
```

---

### Task 3: Create Alembic Migration for admin_users Table

**Files:**
- Create: `alembic/versions/xxxx_add_admin_users.py` (generated by Alembic)

- [ ] **Step 1: Generate migration file**

Run:
```bash
cd /c/Python/HelpdeskChatbot/utc-chatbot-backend && .venv/Scripts/alembic revision -m "add_admin_users"
```

- [ ] **Step 2: Implement migration**

Edit the generated file in `alembic/versions/` with:

```python
def upgrade() -> None:
    """Add admin_users table for admin authentication."""
    op.execute("""
        CREATE TABLE admin_users (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            username        TEXT NOT NULL UNIQUE,
            password_hash   TEXT NOT NULL,
            display_name    TEXT,
            is_active       BOOLEAN NOT NULL DEFAULT true,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_login_at   TIMESTAMPTZ
        )
    """)

    op.create_index("idx_admin_users_username", "admin_users", ["username"])

    op.execute("""
        COMMENT ON TABLE admin_users IS
        'Admin users for authenticated access to admin endpoints'
    """)


def downgrade() -> None:
    """Remove admin_users table."""
    op.drop_index("idx_admin_users_username", table_name="admin_users")
    op.drop_table("admin_users")
```

- [ ] **Step 3: Verify migration applies**

Run:
```bash
cd /c/Python/HelpdeskChatbot/utc-chatbot-backend && .venv/Scripts/alembic upgrade head
```
Expected: Migration applies successfully.

- [ ] **Step 4: Verify rollback works**

Run:
```bash
cd /c/Python/HelpdeskChatbot/utc-chatbot-backend && .venv/Scripts/alembic downgrade -1 && .venv/Scripts/alembic upgrade head
```
Expected: Downgrade and re-upgrade both succeed.

- [ ] **Step 5: Commit**

```bash
git add alembic/versions/*add_admin_users*
git commit -m "feat(auth): add admin_users table migration"
```

---

### Task 4: Create AdminUserClient Storage

**Files:**
- Create: `core/storage_admin_user.py`
- Create: `tests/test_storage_admin_user.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_storage_admin_user.py`:

```python
"""Tests for AdminUserClient storage operations."""

import pytest
from unittest.mock import MagicMock, Mock, patch, call
from uuid import UUID

from core.storage_admin_user import AdminUserClient


class TestAdminUserClient:
    """Test suite for AdminUserClient."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for database connection."""
        with patch("core.storage_base.get_database_settings") as mock:
            settings = Mock()
            settings.HOST = "localhost"
            settings.USER = "test_user"
            settings.PASSWORD.get_secret_value.return_value = "test_password"
            settings.NAME = "test_db"
            mock.return_value = settings
            yield settings

    @pytest.fixture
    def client(self, mock_settings):
        """Create AdminUserClient with mocked settings."""
        return AdminUserClient()

    def test_get_user_by_username_found(self, client):
        """Test fetching an existing user by username."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            "550e8400-e29b-41d4-a716-446655440000",
            "david",
            "$argon2id$hash",
            "David Wood",
            True,
        )

        with patch.object(client, "get_connection") as mock_conn:
            conn = MagicMock()
            conn.__enter__ = Mock(return_value=conn)
            conn.__exit__ = Mock(return_value=None)
            conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            conn.cursor.return_value.__exit__ = Mock(return_value=None)
            mock_conn.return_value = conn

            result = client.get_user_by_username("david")

        assert result is not None
        assert result["username"] == "david"
        assert result["display_name"] == "David Wood"
        assert result["is_active"] is True

    def test_get_user_by_username_not_found(self, client):
        """Test fetching a non-existent user returns None."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        with patch.object(client, "get_connection") as mock_conn:
            conn = MagicMock()
            conn.__enter__ = Mock(return_value=conn)
            conn.__exit__ = Mock(return_value=None)
            conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            conn.cursor.return_value.__exit__ = Mock(return_value=None)
            mock_conn.return_value = conn

            result = client.get_user_by_username("nonexistent")

        assert result is None

    def test_create_user(self, client):
        """Test creating a new admin user."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("550e8400-e29b-41d4-a716-446655440000",)

        with patch.object(client, "get_connection") as mock_conn:
            conn = MagicMock()
            conn.__enter__ = Mock(return_value=conn)
            conn.__exit__ = Mock(return_value=None)
            conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            conn.cursor.return_value.__exit__ = Mock(return_value=None)
            mock_conn.return_value = conn

            user_id = client.create_user(
                username="david",
                password_hash="$argon2id$hash",
                display_name="David Wood",
            )

        assert user_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_update_last_login(self, client):
        """Test updating last_login_at timestamp."""
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1

        with patch.object(client, "get_connection") as mock_conn:
            conn = MagicMock()
            conn.__enter__ = Mock(return_value=conn)
            conn.__exit__ = Mock(return_value=None)
            conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            conn.cursor.return_value.__exit__ = Mock(return_value=None)
            mock_conn.return_value = conn

            result = client.update_last_login("550e8400-e29b-41d4-a716-446655440000")

        assert result is True

    def test_update_password(self, client):
        """Test updating a user's password hash."""
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1

        with patch.object(client, "get_connection") as mock_conn:
            conn = MagicMock()
            conn.__enter__ = Mock(return_value=conn)
            conn.__exit__ = Mock(return_value=None)
            conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            conn.cursor.return_value.__exit__ = Mock(return_value=None)
            mock_conn.return_value = conn

            result = client.update_password(
                "david", "$argon2id$newhash"
            )

        assert result is True

    def test_user_exists(self, client):
        """Test checking if a user exists."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (True,)

        with patch.object(client, "get_connection") as mock_conn:
            conn = MagicMock()
            conn.__enter__ = Mock(return_value=conn)
            conn.__exit__ = Mock(return_value=None)
            conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            conn.cursor.return_value.__exit__ = Mock(return_value=None)
            mock_conn.return_value = conn

            assert client.user_exists("david") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/pytest tests/test_storage_admin_user.py -v`
Expected: FAIL — `core.storage_admin_user` does not exist.

- [ ] **Step 3: Implement AdminUserClient**

Create `core/storage_admin_user.py`:

```python
"""Storage client for admin user management."""

from typing import Optional
from core.storage_base import BaseStorageClient
from utils.logger import get_logger

logger = get_logger(__name__)


class AdminUserClient(BaseStorageClient):
    """Storage client for admin_users table operations."""

    def get_user_by_username(self, username: str) -> Optional[dict]:
        """
        Fetch an admin user by username.

        Returns:
            Dict with id, username, password_hash, display_name, is_active
            or None if not found.
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, username, password_hash, display_name, is_active
                    FROM admin_users
                    WHERE username = %s
                    """,
                    (username,),
                )
                row = cur.fetchone()
                if row:
                    return {
                        "id": str(row[0]),
                        "username": row[1],
                        "password_hash": row[2],
                        "display_name": row[3],
                        "is_active": row[4],
                    }
                return None

    def create_user(
        self,
        username: str,
        password_hash: str,
        display_name: Optional[str] = None,
    ) -> str:
        """
        Create a new admin user.

        Returns:
            UUID string of created user.
        """
        logger.info(f"Creating admin user: {username}")
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO admin_users (username, password_hash, display_name)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (username, password_hash, display_name),
                )
                user_id = str(cur.fetchone()[0])
                logger.info(f"Created admin user {username} ({user_id})")
                return user_id

    def update_last_login(self, user_id: str) -> bool:
        """Update last_login_at to current timestamp."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE admin_users SET last_login_at = NOW() WHERE id = %s",
                    (user_id,),
                )
                return cur.rowcount > 0

    def update_password(self, username: str, password_hash: str) -> bool:
        """Update a user's password hash."""
        logger.info(f"Updating password for admin user: {username}")
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE admin_users SET password_hash = %s WHERE username = %s",
                    (password_hash, username),
                )
                return cur.rowcount > 0

    def user_exists(self, username: str) -> bool:
        """Check if a username already exists."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM admin_users WHERE username = %s)",
                    (username,),
                )
                return cur.fetchone()[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/test_storage_admin_user.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add core/storage_admin_user.py tests/test_storage_admin_user.py
git commit -m "feat(auth): add AdminUserClient storage layer"
```

---

### Task 5: Create Auth Core Module (Password Hashing + JWT + Dependencies)

**Files:**
- Create: `api/auth.py`
- Create: `tests/test_auth.py`

- [ ] **Step 1: Write failing tests for password hashing**

Create `tests/test_auth.py`:

```python
"""Tests for admin auth module (password hashing, JWT, dependencies)."""

import pytest
from unittest.mock import MagicMock, Mock, patch
from datetime import datetime, timedelta, timezone


class TestPasswordHashing:
    """Test password hashing and verification."""

    def test_hash_password_returns_argon2id_hash(self):
        from api.auth import hash_password
        hashed = hash_password("testpassword123")
        assert hashed.startswith("$argon2id$")

    def test_verify_password_correct(self):
        from api.auth import hash_password, verify_password
        hashed = hash_password("mypassword")
        assert verify_password("mypassword", hashed) is True

    def test_verify_password_incorrect(self):
        from api.auth import hash_password, verify_password
        hashed = hash_password("mypassword")
        assert verify_password("wrongpassword", hashed) is False

    def test_hash_password_unique_salts(self):
        from api.auth import hash_password
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2  # Different salts


class TestJWT:
    """Test JWT creation and validation."""

    @pytest.fixture(autouse=True)
    def mock_auth_settings(self):
        with patch("api.auth.get_auth_settings") as mock:
            settings = Mock()
            settings.SECRET_KEY.get_secret_value.return_value = "a" * 64
            settings.TOKEN_EXPIRE_MINUTES = 120
            settings.COOKIE_NAME = "admin_session"
            mock.return_value = settings
            yield settings

    def test_create_access_token(self):
        from api.auth import create_access_token
        token = create_access_token(user_id="user-123", username="david")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_and_decode_token_roundtrip(self):
        from api.auth import create_access_token, decode_access_token
        token = create_access_token(user_id="user-123", username="david")
        payload = decode_access_token(token)
        assert payload["sub"] == "user-123"
        assert payload["username"] == "david"

    def test_decode_expired_token_returns_none(self):
        from api.auth import create_access_token, decode_access_token
        with patch("api.auth.get_auth_settings") as mock:
            settings = Mock()
            settings.SECRET_KEY.get_secret_value.return_value = "a" * 64
            settings.TOKEN_EXPIRE_MINUTES = -1  # Already expired
            settings.COOKIE_NAME = "admin_session"
            mock.return_value = settings

            token = create_access_token(user_id="user-123", username="david")

        # Decode with valid settings (token is expired)
        result = decode_access_token(token)
        assert result is None

    def test_decode_invalid_token_returns_none(self):
        from api.auth import decode_access_token
        result = decode_access_token("not.a.valid.jwt")
        assert result is None

    def test_decode_token_wrong_secret_returns_none(self):
        from api.auth import create_access_token, decode_access_token
        token = create_access_token(user_id="user-123", username="david")

        with patch("api.auth.get_auth_settings") as mock:
            settings = Mock()
            settings.SECRET_KEY.get_secret_value.return_value = "b" * 64
            mock.return_value = settings
            result = decode_access_token(token)

        assert result is None


class TestRequireAdmin:
    """Test the require_admin FastAPI dependency."""

    @pytest.fixture(autouse=True)
    def mock_auth_settings(self):
        with patch("api.auth.get_auth_settings") as mock:
            settings = Mock()
            settings.SECRET_KEY.get_secret_value.return_value = "a" * 64
            settings.TOKEN_EXPIRE_MINUTES = 120
            settings.COOKIE_NAME = "admin_session"
            mock.return_value = settings
            yield settings

    @pytest.mark.asyncio
    async def test_require_admin_valid_cookie_returns_payload(self):
        from api.auth import create_access_token, require_admin
        from fastapi import Request

        token = create_access_token(user_id="user-123", username="david")

        request = MagicMock(spec=Request)
        request.cookies = {"admin_session": token}
        request.headers = {"accept": "application/json"}

        result = await require_admin(request)
        assert result["sub"] == "user-123"
        assert result["username"] == "david"

    @pytest.mark.asyncio
    async def test_require_admin_missing_cookie_json_returns_401(self):
        from api.auth import require_admin
        from fastapi import Request, HTTPException

        request = MagicMock(spec=Request)
        request.cookies = {}
        request.headers = {"accept": "application/json"}

        with pytest.raises(HTTPException) as exc_info:
            await require_admin(request)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_require_admin_missing_cookie_html_redirects(self):
        from api.auth import require_admin, AdminAuthRequired
        from fastapi import Request

        request = MagicMock(spec=Request)
        request.cookies = {}
        request.headers = {"accept": "text/html,application/xhtml+xml"}
        request.url = MagicMock()
        request.url.path = "/admin/prompts"

        with pytest.raises(AdminAuthRequired):
            await require_admin(request)

    @pytest.mark.asyncio
    async def test_require_admin_expired_cookie_returns_401(self):
        from api.auth import require_admin
        from fastapi import Request, HTTPException

        request = MagicMock(spec=Request)
        request.cookies = {"admin_session": "expired.jwt.token"}
        request.headers = {"accept": "application/json"}

        with pytest.raises(HTTPException) as exc_info:
            await require_admin(request)
        assert exc_info.value.status_code == 401
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/pytest tests/test_auth.py -v`
Expected: FAIL — `api.auth` does not exist.

- [ ] **Step 3: Implement api/auth.py**

Create `api/auth.py`:

```python
"""
Admin authentication: password hashing, JWT tokens, and route protection.

Uses argon2id for password hashing and PyJWT for stateless session tokens
stored as HttpOnly cookies.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote

import argon2
import jwt
from fastapi import HTTPException, Request, status
from fastapi.responses import RedirectResponse

from core.config import get_auth_settings
from utils.logger import get_logger

logger = get_logger(__name__)

_hasher = argon2.PasswordHasher()

ALGORITHM = "HS256"


# --- Custom exception for HTML redirect ---

class AdminAuthRequired(Exception):
    """Raised when admin auth fails on an HTML request. Handled by exception handler."""

    def __init__(self, next_url: str = "/admin/prompts"):
        self.next_url = next_url


async def admin_auth_exception_handler(request: Request, exc: AdminAuthRequired):
    """Convert AdminAuthRequired into a redirect to the login page."""
    return RedirectResponse(
        url=f"/admin/login?next={quote(exc.next_url)}",
        status_code=307,
    )


# --- Password hashing ---

def hash_password(password: str) -> str:
    """Hash a password using argon2id."""
    return _hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against an argon2id hash. Timing-safe."""
    try:
        return _hasher.verify(hashed_password, plain_password)
    except argon2.exceptions.VerifyMismatchError:
        return False
    except argon2.exceptions.VerificationError:
        return False


# --- JWT tokens ---

def create_access_token(user_id: str, username: str) -> str:
    """Create a JWT access token with user_id and username."""
    settings = get_auth_settings()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.TOKEN_EXPIRE_MINUTES
    )
    payload = {
        "sub": user_id,
        "username": username,
        "exp": expire,
    }
    return jwt.encode(
        payload,
        settings.SECRET_KEY.get_secret_value(),
        algorithm=ALGORITHM,
    )


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token. Returns payload dict or None."""
    settings = get_auth_settings()
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY.get_secret_value(),
            algorithms=[ALGORITHM],
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.debug("JWT token expired")
        return None
    except jwt.InvalidTokenError:
        logger.debug("Invalid JWT token")
        return None


# --- FastAPI dependency ---

async def require_admin(request: Request) -> dict:
    """
    FastAPI dependency that validates the admin session cookie.

    Smart response based on Accept header:
    - HTML requests (browser navigation) -> raises AdminAuthRequired
      (caught by exception handler, returns redirect to /admin/login)
    - JSON requests (fetch from SPA) -> 401 Unauthorized

    Returns:
        JWT payload dict with 'sub' (user_id) and 'username' keys.
    """
    settings = get_auth_settings()
    token = request.cookies.get(settings.COOKIE_NAME)

    accept = request.headers.get("accept", "")
    is_html = "text/html" in accept

    if not token:
        logger.debug("No admin session cookie found")
        if is_html:
            raise AdminAuthRequired(next_url=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    payload = decode_access_token(token)
    if payload is None:
        logger.debug("Invalid or expired admin session")
        if is_html:
            raise AdminAuthRequired(next_url=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid",
        )

    return payload
```

**Important:** The `AdminAuthRequired` exception handler must be registered in `api/main.py` (done in Task 9).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/test_auth.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api/auth.py tests/test_auth.py
git commit -m "feat(auth): add password hashing, JWT, and require_admin dependency"
```

---

### Task 6: Add AdminUserClient Dependency

**Files:**
- Modify: `api/dependencies.py:311-329` (after get_prompt_storage_client)

- [ ] **Step 1: Add get_admin_user_client dependency**

Add to `api/dependencies.py` after the existing dependency functions:

```python
from core.storage_admin_user import AdminUserClient
```

(Add to imports at top of file.)

```python
def get_admin_user_client(request: Request) -> AdminUserClient:
    """Dependency to create an AdminUserClient with shared connection pool."""
    try:
        connection_pool = request.app.state.connection_pool
        return AdminUserClient(connection_pool=connection_pool)
    except AttributeError:
        logger.error("Connection pool not found in app.state (not initialized)")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Connection pool not initialized",
        )
```

- [ ] **Step 2: Run existing tests to verify no regression**

Run: `.venv/Scripts/pytest tests/ -v`
Expected: All existing tests still PASS.

- [ ] **Step 3: Commit**

```bash
git add api/dependencies.py
git commit -m "feat(auth): add get_admin_user_client dependency"
```

---

### Task 7: Create Auth Router (Login/Logout Endpoints)

**Files:**
- Create: `api/routers/auth.py`
- Create: `tests/test_router_auth.py`

- [ ] **Step 1: Write failing tests for login/logout routes**

Create `tests/test_router_auth.py`:

```python
"""Tests for auth router (login, logout, login page)."""

import pytest
from unittest.mock import MagicMock, Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.routers.auth import router


@pytest.fixture
def app():
    """Create test FastAPI app with auth router."""
    app = FastAPI()
    app.include_router(router)

    # Mock app.state.connection_pool
    pool = MagicMock()
    app.state.connection_pool = pool

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestLoginPage:
    """Test GET /admin/login."""

    def test_login_page_returns_html(self, client):
        response = client.get("/admin/login")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Login" in response.text


class TestLogin:
    """Test POST /auth/login."""

    @patch("api.dependencies.get_admin_user_client")
    @patch("api.routers.auth.verify_password")
    @patch("api.routers.auth.create_access_token")
    def test_login_success_sets_cookie_and_redirects(
        self, mock_create_token, mock_verify, mock_get_client, client
    ):
        mock_user_client = MagicMock()
        mock_user_client.get_user_by_username.return_value = {
            "id": "user-123",
            "username": "david",
            "password_hash": "$argon2id$hash",
            "display_name": "David Wood",
            "is_active": True,
        }
        mock_get_client.return_value = mock_user_client
        mock_verify.return_value = True
        mock_create_token.return_value = "fake.jwt.token"

        response = client.post(
            "/auth/login",
            data={"username": "david", "password": "correct"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "admin_session" in response.cookies

    @patch("api.dependencies.get_admin_user_client")
    def test_login_invalid_username(self, mock_get_client, client):
        mock_user_client = MagicMock()
        mock_user_client.get_user_by_username.return_value = None
        mock_get_client.return_value = mock_user_client

        response = client.post(
            "/auth/login",
            data={"username": "nobody", "password": "wrong"},
            follow_redirects=False,
        )
        assert response.status_code == 200  # Re-render login page with error
        assert "Invalid" in response.text or "invalid" in response.text

    @patch("api.dependencies.get_admin_user_client")
    @patch("api.routers.auth.verify_password")
    def test_login_wrong_password(self, mock_verify, mock_get_client, client):
        mock_user_client = MagicMock()
        mock_user_client.get_user_by_username.return_value = {
            "id": "user-123",
            "username": "david",
            "password_hash": "$argon2id$hash",
            "display_name": "David Wood",
            "is_active": True,
        }
        mock_get_client.return_value = mock_user_client
        mock_verify.return_value = False

        response = client.post(
            "/auth/login",
            data={"username": "david", "password": "wrong"},
            follow_redirects=False,
        )
        assert response.status_code == 200
        assert "Invalid" in response.text or "invalid" in response.text

    @patch("api.dependencies.get_admin_user_client")
    @patch("api.routers.auth.verify_password")
    def test_login_inactive_user(self, mock_verify, mock_get_client, client):
        mock_user_client = MagicMock()
        mock_user_client.get_user_by_username.return_value = {
            "id": "user-123",
            "username": "david",
            "password_hash": "$argon2id$hash",
            "display_name": "David Wood",
            "is_active": False,
        }
        mock_get_client.return_value = mock_user_client
        mock_verify.return_value = True

        response = client.post(
            "/auth/login",
            data={"username": "david", "password": "correct"},
            follow_redirects=False,
        )
        assert response.status_code == 200
        assert "disabled" in response.text.lower() or "inactive" in response.text.lower()


class TestLogout:
    """Test POST /auth/logout."""

    def test_logout_clears_cookie_and_redirects(self, client):
        response = client.post("/auth/logout", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/admin/login"
        # Cookie should be cleared (max-age=0)
        cookie_header = response.headers.get("set-cookie", "")
        assert "admin_session" in cookie_header
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/pytest tests/test_router_auth.py -v`
Expected: FAIL — `api.routers.auth` does not exist.

- [ ] **Step 3: Implement auth router**

Create `api/routers/auth.py`:

```python
"""
Auth router for admin login/logout.

Provides:
- GET /admin/login: HTML login form
- POST /auth/login: Authenticate and set session cookie
- POST /auth/logout: Clear session cookie
"""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from api.auth import (
    create_access_token,
    hash_password,
    verify_password,
)
from api.dependencies import get_admin_user_client
from core.config import get_auth_settings
from core.storage_admin_user import AdminUserClient
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/admin/login", response_class=HTMLResponse, include_in_schema=False)
def login_page(error: str = "", next: str = Query(default="/admin/prompts", alias="next")):
    """Serve the admin login HTML page."""
    html_path = Path(__file__).parent.parent / "templates" / "admin_login.html"
    html = html_path.read_text(encoding="utf-8")
    if error:
        html = html.replace("<!--ERROR-->", f'<p class="error">{error}</p>')
    # Inject next_url as hidden field
    html = html.replace("<!--NEXT-->", f'<input type="hidden" name="next_url" value="{next}">')
    return HTMLResponse(content=html)


@router.post("/auth/login", include_in_schema=False)
def login(
    user_client: Annotated[AdminUserClient, Depends(get_admin_user_client)],
    username: str = Form(...),
    password: str = Form(...),
    next_url: str = Form(default="/admin/prompts"),
):
    """Authenticate admin user and set session cookie."""
    settings = get_auth_settings()

    user = user_client.get_user_by_username(username)

    if not user:
        logger.warning(f"Login attempt for non-existent user: {username}")
        # Still run password hash to prevent timing attacks
        hash_password("dummy")
        return _login_error("Invalid username or password")

    if not verify_password(password, user["password_hash"]):
        logger.warning(f"Failed login attempt for user: {username}")
        return _login_error("Invalid username or password")

    if not user["is_active"]:
        logger.warning(f"Login attempt for inactive user: {username}")
        return _login_error("Account is disabled. Contact an administrator.")

    # Success: create JWT, set cookie, update last_login, redirect
    token = create_access_token(user_id=user["id"], username=user["username"])

    # Update last_login_at (best-effort)
    try:
        user_client.update_last_login(user["id"])
    except Exception as e:
        logger.warning(f"Failed to update last_login_at: {e}")

    logger.info(f"Admin user '{username}' logged in successfully")

    response = RedirectResponse(url=next_url, status_code=303)
    response.set_cookie(
        key=settings.COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=settings.TOKEN_EXPIRE_MINUTES * 60,
    )
    return response


@router.post("/auth/logout", include_in_schema=False)
def logout():
    """Clear session cookie and redirect to login."""
    settings = get_auth_settings()
    response = RedirectResponse(url="/admin/login", status_code=303)
    response.delete_cookie(
        key=settings.COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
    )
    logger.info("Admin user logged out")
    return response


def _login_error(message: str) -> HTMLResponse:
    """Return login page with error message."""
    html_path = Path(__file__).parent.parent / "templates" / "admin_login.html"
    html = html_path.read_text(encoding="utf-8")
    html = html.replace("<!--ERROR-->", f'<p class="error">{message}</p>')
    return HTMLResponse(content=html)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/test_router_auth.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routers/auth.py tests/test_router_auth.py
git commit -m "feat(auth): add login/logout router with session cookie"
```

---

### Task 8: Create Login HTML Template

**Files:**
- Create: `api/templates/admin_login.html`

- [ ] **Step 1: Create admin_login.html**

Create `api/templates/admin_login.html` styled consistent with the existing admin pages (same font, colors, border-radius from `admin_prompts.html`):

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Admin Login</title>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0; padding: 0; background: #f5f6fa; color: #2d3436;
    display: flex; justify-content: center; align-items: center; min-height: 100vh;
  }
  .login-card {
    background: #fff; border-radius: 8px; padding: 40px; width: 100%;
    max-width: 400px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  h1 { margin: 0 0 8px; font-size: 1.4rem; font-weight: 600; }
  .subtitle { color: #636e72; font-size: 0.875rem; margin: 0 0 24px; }
  .form-group { margin-bottom: 16px; }
  .form-group label {
    display: block; margin-bottom: 6px; font-size: 0.8rem;
    font-weight: 600; text-transform: uppercase; color: #636e72;
  }
  input[type="text"], input[type="password"] {
    width: 100%; padding: 10px 12px; border: 1px solid #ddd; border-radius: 6px;
    font-size: 0.9rem;
  }
  input:focus { outline: none; border-color: #0984e3; box-shadow: 0 0 0 3px rgba(9,132,227,0.1); }
  button {
    width: 100%; padding: 10px 16px; border: none; border-radius: 6px;
    background: #0984e3; color: #fff; font-size: 0.9rem; font-weight: 500;
    cursor: pointer; transition: background 0.15s;
  }
  button:hover { background: #0770c2; }
  .error {
    background: #ffeaea; color: #d63031; padding: 10px 14px; border-radius: 6px;
    font-size: 0.85rem; margin-bottom: 16px;
  }
</style>
</head>
<body>
<div class="login-card">
  <h1>Admin Login</h1>
  <p class="subtitle">RAG Helpdesk Backend</p>
  <!--ERROR-->
  <form method="POST" action="/auth/login">
    <!--NEXT-->
    <div class="form-group">
      <label for="username">Username</label>
      <input type="text" id="username" name="username" required autocomplete="username" autofocus>
    </div>
    <div class="form-group">
      <label for="password">Password</label>
      <input type="password" id="password" name="password" required autocomplete="current-password">
    </div>
    <button type="submit">Sign In</button>
  </form>
</div>
</body>
</html>
```

- [ ] **Step 2: Verify login page renders**

Run: `.venv/Scripts/pytest tests/test_router_auth.py::TestLoginPage -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add api/templates/admin_login.html
git commit -m "feat(auth): add admin login page template"
```

---

### Task 9: Wire Auth Into api/main.py and Protect Admin Routes

**Files:**
- Modify: `api/main.py:27,217-239`

- [ ] **Step 1: Write failing integration test**

Create `tests/test_admin_auth_integration.py`:

```python
"""Integration tests: admin endpoints require authentication."""

import pytest
from unittest.mock import MagicMock, Mock, patch
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from api.auth import require_admin, AdminAuthRequired, admin_auth_exception_handler
from api.routers import admin_prompts, admin_analytics, auth as auth_router


@pytest.fixture
def mock_auth_settings():
    """Provide mocked auth settings for all tests."""
    settings = Mock()
    settings.SECRET_KEY.get_secret_value.return_value = "a" * 64
    settings.TOKEN_EXPIRE_MINUTES = 120
    settings.COOKIE_NAME = "admin_session"
    return settings


@pytest.fixture
def client(mock_auth_settings):
    """Build a fresh FastAPI app with admin routes protected by require_admin."""
    app = FastAPI()

    # Register exception handler for HTML redirects
    app.add_exception_handler(AdminAuthRequired, admin_auth_exception_handler)

    # Auth router (no auth required)
    app.include_router(auth_router.router, tags=["Auth"])

    # Admin routers (auth required)
    app.include_router(
        admin_prompts.router,
        tags=["Admin - Prompts"],
        dependencies=[Depends(require_admin)],
    )
    app.include_router(
        admin_analytics.router,
        tags=["Admin - Analytics"],
        dependencies=[Depends(require_admin)],
    )

    # Mock app.state
    app.state.connection_pool = MagicMock()
    app.state.bm25_retriever = MagicMock()
    app.state.vector_retriever = MagicMock()

    with patch("api.auth.get_auth_settings", return_value=mock_auth_settings):
        with TestClient(app) as c:
            yield c


class TestAdminEndpointsRequireAuth:
    """Verify admin endpoints return 401/redirect without auth."""

    def test_admin_prompts_page_redirects_to_login(self, client):
        response = client.get("/admin/prompts", follow_redirects=False)
        assert response.status_code == 307
        assert "/admin/login" in response.headers.get("location", "")

    def test_admin_api_returns_401(self, client):
        response = client.get(
            "/api/v1/admin/prompts",
            headers={"accept": "application/json"},
        )
        assert response.status_code == 401

    def test_login_page_accessible_without_auth(self, client):
        """Login page itself should not require auth."""
        response = client.get("/admin/login")
        assert response.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_admin_auth_integration.py -v`
Expected: FAIL — admin endpoints are not yet protected.

- [ ] **Step 3: Modify api/main.py to register auth router and protect admin routes**

Add imports at top of `api/main.py`:

```python
from api.routers import search, health, query_logs, admin_prompts, admin_analytics, auth
from api.auth import require_admin, AdminAuthRequired, admin_auth_exception_handler
from fastapi import Depends
```

Register the exception handler (after `app = FastAPI(...)` creation):

```python
# Register custom exception handler for admin auth redirects
app.add_exception_handler(AdminAuthRequired, admin_auth_exception_handler)
```

Register auth router (add before the admin routers):

```python
app.include_router(
    auth.router,
    tags=["Auth"],
)
```

Add `dependencies=[Depends(require_admin)]` to both admin router registrations:

```python
app.include_router(
    admin_prompts.router,
    tags=["Admin - Prompts"],
    dependencies=[Depends(require_admin)],
)
app.include_router(
    admin_analytics.router,
    tags=["Admin - Analytics"],
    dependencies=[Depends(require_admin)],
)
```

- [ ] **Step 4: Run integration tests to verify they pass**

Run: `.venv/Scripts/pytest tests/test_admin_auth_integration.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `.venv/Scripts/pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add api/main.py tests/test_admin_auth_integration.py
git commit -m "feat(auth): protect admin routes with require_admin dependency"
```

---

### Task 10: Add Logout Button to Admin Templates

**Files:**
- Modify: `api/templates/admin_prompts.html`
- Modify: `api/templates/admin_analytics.html`

- [ ] **Step 1: Add logout form to admin_prompts.html**

Find the `<h1>` tag in `admin_prompts.html` and wrap it in a header with a logout button:

```html
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:24px;">
  <h1 style="margin:0;">System Prompt Admin</h1>
  <form method="POST" action="/auth/logout" style="margin:0;">
    <button type="submit" style="background:none; color:#636e72; border:1px solid #ddd; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.8rem;">Logout</button>
  </form>
</div>
```

Remove the existing standalone `<h1>` tag.

- [ ] **Step 2: Add logout form to admin_analytics.html**

Apply the same header pattern to `admin_analytics.html`, replacing its `<h1>` with the same flex header + logout button.

- [ ] **Step 3: Verify pages still render**

Run: `.venv/Scripts/pytest tests/ -v -k "admin"`
Expected: All admin-related tests PASS.

- [ ] **Step 4: Commit**

```bash
git add api/templates/admin_prompts.html api/templates/admin_analytics.html
git commit -m "feat(auth): add logout button to admin templates"
```

---

### Task 11: Add create-admin CLI Command

**Files:**
- Modify: `main.py:77,603-662`
- Create: `tests/test_create_admin.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_create_admin.py`:

```python
"""Tests for create-admin CLI command."""

import pytest
from unittest.mock import MagicMock, Mock, patch
from io import StringIO


class TestCreateAdmin:
    """Test the create-admin CLI command."""

    @patch("builtins.input", return_value="y")
    @patch("getpass.getpass", side_effect=["securepass123", "securepass123"])
    @patch("core.storage_admin_user.AdminUserClient")
    @patch("core.storage_base.get_database_settings")
    def test_create_new_admin(
        self, mock_settings, mock_client_cls, mock_getpass, mock_input
    ):
        from main import command_create_admin
        import argparse

        # Setup mocks
        settings = Mock()
        settings.HOST = "localhost"
        settings.USER = "test"
        settings.PASSWORD.get_secret_value.return_value = "pw"
        settings.NAME = "testdb"
        mock_settings.return_value = settings

        mock_client = MagicMock()
        mock_client.user_exists.return_value = False
        mock_client.create_user.return_value = "new-uuid"
        mock_client_cls.return_value = mock_client

        args = argparse.Namespace(
            username="david",
            display_name="David Wood",
        )

        exit_code = command_create_admin(args)

        assert exit_code == 0
        mock_client.create_user.assert_called_once()

    @patch("getpass.getpass", side_effect=["password1", "password2"])
    @patch("core.storage_admin_user.AdminUserClient")
    @patch("core.storage_base.get_database_settings")
    def test_password_mismatch_returns_error(
        self, mock_settings, mock_client_cls, mock_getpass
    ):
        from main import command_create_admin
        import argparse

        settings = Mock()
        settings.HOST = "localhost"
        settings.USER = "test"
        settings.PASSWORD.get_secret_value.return_value = "pw"
        settings.NAME = "testdb"
        mock_settings.return_value = settings

        args = argparse.Namespace(
            username="david",
            display_name="David Wood",
        )

        exit_code = command_create_admin(args)

        assert exit_code == 1  # Should fail due to mismatch

    @patch("builtins.input", return_value="y")
    @patch("getpass.getpass", side_effect=["newpass123", "newpass123"])
    @patch("core.storage_admin_user.AdminUserClient")
    @patch("core.storage_base.get_database_settings")
    def test_update_existing_admin(
        self, mock_settings, mock_client_cls, mock_getpass, mock_input
    ):
        from main import command_create_admin
        import argparse

        settings = Mock()
        settings.HOST = "localhost"
        settings.USER = "test"
        settings.PASSWORD.get_secret_value.return_value = "pw"
        settings.NAME = "testdb"
        mock_settings.return_value = settings

        mock_client = MagicMock()
        mock_client.user_exists.return_value = True
        mock_client.update_password.return_value = True
        mock_client_cls.return_value = mock_client

        args = argparse.Namespace(
            username="david",
            display_name="David Wood",
        )

        exit_code = command_create_admin(args)

        assert exit_code == 0
        mock_client.update_password.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_create_admin.py -v`
Expected: FAIL — `command_create_admin` does not exist.

- [ ] **Step 3: Add create-admin subcommand to main.py**

In `setup_argparse()`, add the new subparser after the sweep parser:

```python
    # ========== CREATE-ADMIN COMMAND ==========
    create_admin_parser = subparsers.add_parser(
        "create-admin",
        help="Create or update an admin user",
        description="Create a new admin user or update an existing user's password.",
    )
    create_admin_parser.add_argument(
        "--username",
        required=True,
        help="Admin username for login",
    )
    create_admin_parser.add_argument(
        "--display-name",
        default=None,
        help="Display name (optional)",
    )
```

Add the command handler function:

```python
def command_create_admin(args: argparse.Namespace) -> int:
    """Create or update an admin user."""
    import getpass
    from api.auth import hash_password
    from core.storage_admin_user import AdminUserClient

    logger.info("=" * 80)
    logger.info("COMMAND: CREATE ADMIN USER")
    logger.info("=" * 80)

    try:
        client = AdminUserClient()

        password = getpass.getpass("Enter password: ")
        if not password or len(password) < 8:
            logger.error("Password must be at least 8 characters")
            return 1

        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            logger.error("Passwords do not match")
            return 1

        password_hash = hash_password(password)

        if client.user_exists(args.username):
            answer = input(
                f"User '{args.username}' already exists. Update password? (y/N): "
            )
            if answer.lower() != "y":
                logger.info("Cancelled")
                return 0
            client.update_password(args.username, password_hash)
            logger.info(f"Password updated for user '{args.username}'")
        else:
            user_id = client.create_user(
                username=args.username,
                password_hash=password_hash,
                display_name=args.display_name,
            )
            logger.info(f"Created admin user '{args.username}' (id: {user_id})")

        return 0

    except Exception as e:
        logger.error(f"Failed to create admin user: {e}", exc_info=True)
        return 1
```

Add the routing in the `main()` function's command router:

```python
        elif args.command == "create-admin":
            exit_code = command_create_admin(args)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/test_create_admin.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `.venv/Scripts/pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_create_admin.py
git commit -m "feat(auth): add create-admin CLI command"
```

---

### Task 12: Update .env Example and Documentation

**Files:**
- Modify: `example.env` (if it exists, otherwise `docker-compose.yml`)

- [ ] **Step 1: Check if example.env exists and add AUTH env vars**

Run: `ls -la /c/Python/HelpdeskChatbot/utc-chatbot-backend/example.env`

If it exists, add the following section. If not, skip to Step 2:

```env
# Admin Authentication
AUTH_SECRET_KEY=          # Required. Generate with: python -c "import secrets; print(secrets.token_hex(32))"
AUTH_TOKEN_EXPIRE_MINUTES=120
AUTH_COOKIE_NAME=admin_session
```

- [ ] **Step 2: Add AUTH_SECRET_KEY to docker-compose.yml api service**

Add to the `api` service `environment` section in `docker-compose.yml`:

```yaml
      # Admin Authentication
      AUTH_SECRET_KEY: ${AUTH_SECRET_KEY}
      AUTH_TOKEN_EXPIRE_MINUTES: ${AUTH_TOKEN_EXPIRE_MINUTES:-120}
      AUTH_COOKIE_NAME: ${AUTH_COOKIE_NAME:-admin_session}
```

- [ ] **Step 3: Commit**

```bash
git add example.env docker-compose.yml
git commit -m "feat(auth): add auth env vars to example.env and docker-compose"
```

---

### Task 13: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `.venv/Scripts/pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 2: Run linter**

Run: `.venv/Scripts/ruff check .`
Expected: No errors (or fix any that appear).

- [ ] **Step 3: Run formatter**

Run: `.venv/Scripts/ruff format .`

- [ ] **Step 4: Final commit if any formatting changes**

```bash
git add -A
git commit -m "style: format auth module code"
```
