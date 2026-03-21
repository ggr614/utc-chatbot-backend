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
