"""
Auth router for admin login/logout.

Provides:
- GET /admin/login: HTML login form
- POST /auth/login: Authenticate and set session cookie
- POST /auth/logout: Clear session cookie
"""

from html import escape
from pathlib import Path

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from api.auth import (
    create_access_token,
    hash_password,
    verify_password,
)
from api.dependencies import get_admin_user_client
from core.config import get_auth_settings
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _login_error(message: str) -> HTMLResponse:
    """Return login page with error message."""
    html_path = Path(__file__).parent.parent / "templates" / "admin_login.html"
    html = html_path.read_text(encoding="utf-8")
    html = html.replace("<!--ERROR-->", f'<p class="error">{escape(message)}</p>')
    return HTMLResponse(content=html)


@router.get("/admin/login", response_class=HTMLResponse, include_in_schema=False)
def login_page(next: str = Query(default="/admin/prompts", alias="next")):
    """Serve the admin login HTML page."""
    html_path = Path(__file__).parent.parent / "templates" / "admin_login.html"
    html = html_path.read_text(encoding="utf-8")
    # Inject next_url as hidden field
    html = html.replace(
        "<!--NEXT-->",
        f'<input type="hidden" name="next_url" value="{escape(next)}">',
    )
    return HTMLResponse(content=html)


@router.post("/auth/login", include_in_schema=False)
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next_url: str = Form(default="/admin/prompts"),
):
    """Authenticate admin user and set session cookie."""
    settings = get_auth_settings()
    user_client = get_admin_user_client(request)

    # Sanitize username for safe logging (strip control chars/newlines)
    safe_username = username.replace("\n", "").replace("\r", "")[:100]

    user = user_client.get_user_by_username(username)

    if not user:
        logger.warning(f"Login attempt for non-existent user: {safe_username}")
        # Still run password hash to prevent timing attacks
        hash_password("dummy")
        return _login_error("Invalid username or password")

    if not user["is_active"]:
        logger.warning(f"Login attempt for inactive user: {safe_username}")
        # Run password hash to keep timing consistent with valid-user path
        hash_password("dummy")
        return _login_error("Invalid username or password")

    if not verify_password(password, user["password_hash"]):
        logger.warning(f"Failed login attempt for user: {safe_username}")
        return _login_error("Invalid username or password")

    # Success: create JWT, set cookie, update last_login, redirect
    token = create_access_token(user_id=user["id"], username=user["username"])

    # Update last_login_at (best-effort)
    try:
        user_client.update_last_login(user["id"])
    except Exception as e:
        logger.warning(f"Failed to update last_login_at: {e}")

    logger.info(f"Admin user '{safe_username}' logged in successfully")

    # Prevent open redirect — only allow local paths
    if not next_url.startswith("/") or next_url.startswith("//"):
        next_url = "/admin/prompts"

    response = RedirectResponse(url=next_url, status_code=303)
    response.set_cookie(
        key=settings.COOKIE_NAME,
        value=token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
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
