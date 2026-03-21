"""Tests for auth router (login, logout, login page)."""

import pytest
from unittest.mock import MagicMock, Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.routers.auth import router


@pytest.fixture(autouse=True)
def mock_auth_settings():
    """Mock AuthSettings for all tests — avoids requiring AUTH_SECRET_KEY in env."""
    with patch("api.routers.auth.get_auth_settings") as mock:
        settings = MagicMock()
        settings.COOKIE_NAME = "admin_session"
        settings.TOKEN_EXPIRE_MINUTES = 120
        mock.return_value = settings
        yield settings


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

    @patch("api.routers.auth.get_admin_user_client")
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

    @patch("api.routers.auth.get_admin_user_client")
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

    @patch("api.routers.auth.get_admin_user_client")
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

    @patch("api.routers.auth.get_admin_user_client")
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
