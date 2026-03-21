"""Tests for admin auth module (password hashing, JWT, dependencies)."""

import pytest
from unittest.mock import MagicMock, Mock, patch


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
