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
