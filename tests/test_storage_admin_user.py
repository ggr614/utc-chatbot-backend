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
