"""Integration tests: admin endpoints require authentication."""

import pytest
from unittest.mock import MagicMock, Mock, patch
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from api.auth import require_admin, AdminAuthRequired, admin_auth_exception_handler
from api.routers import admin_prompts, admin_analytics
from api.routers import auth as auth_router


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
        # Simulate browser navigation with Accept: text/html so require_admin redirects
        response = client.get(
            "/admin/prompts",
            headers={
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            },
            follow_redirects=False,
        )
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
