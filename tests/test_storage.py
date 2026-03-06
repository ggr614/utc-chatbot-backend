"""
Tests for the storage module (PostgresClient).
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from pydantic import HttpUrl
from uuid import UUID

from core.storage_raw import PostgresClient
from core.schemas import TdxArticle


class TestPostgresClient:
    """Test suite for PostgresClient class."""

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
        """Create PostgresClient instance with mocked settings."""
        return PostgresClient()

    def test_init(self, client, mock_settings):
        """Test client initialization."""
        assert client.db_host == "localhost"
        assert client.db_user == "test_user"
        assert client.db_password == "test_password"
        assert client.db_name == "test_db"
        assert client.db_port == 5432

    def test_get_connection_success(self, client):
        """Test successful database connection."""
        with patch("core.storage_base.psycopg.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_conn.closed = False
            mock_connect.return_value = mock_conn

            with client.get_connection() as conn:
                assert conn == mock_conn

            mock_conn.commit.assert_called_once()
            # Close is called when closed=False is checked
            assert mock_conn.close.called

    def test_get_connection_rollback_on_error(self, client):
        """Test that connection rolls back on error."""
        with patch("core.storage_base.psycopg.connect") as mock_connect:
            with patch("core.storage_raw.psycopg.Error", Exception):
                mock_conn = MagicMock()
                mock_conn.closed = False
                mock_connect.return_value = mock_conn

                with pytest.raises(ConnectionError):
                    with client.get_connection() as conn:
                        # Simulate a database error
                        import psycopg

                        raise psycopg.Error("Test error")

                mock_conn.rollback.assert_called_once()
                assert mock_conn.close.called

    def test_get_article_metadata(self, client):
        """Test retrieving article metadata."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (123, datetime(2024, 1, 1, tzinfo=timezone.utc)),
            (456, datetime(2024, 1, 2, tzinfo=timezone.utc)),
        ]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            metadata = client.get_article_metadata()

            assert len(metadata) == 2
            assert metadata[123] == datetime(2024, 1, 1, tzinfo=timezone.utc)
            assert metadata[456] == datetime(2024, 1, 2, tzinfo=timezone.utc)
            mock_cursor.execute.assert_called_once_with(
                "SELECT tdx_article_id, last_modified_date FROM articles"
            )

    def test_get_existing_article_ids(self, client):
        """Test retrieving existing article IDs."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(123,), (456,), (789,)]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            article_ids = client.get_existing_article_ids()

            assert article_ids == {123, 456, 789}
            assert isinstance(article_ids, set)
            mock_cursor.execute.assert_called_once_with(
                "SELECT tdx_article_id FROM articles"
            )

    def test_insert_articles(self, client):
        """Test inserting new articles."""
        articles = [
            TdxArticle(
                tdx_article_id=123,
                title="Test Article",
                url=HttpUrl("https://example.com/123"),
                content_html="<p>Test content</p>",
                last_modified_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            TdxArticle(
                tdx_article_id=456,
                title="Another Article",
                url=HttpUrl("https://example.com/456"),
                content_html="<p>More content</p>",
                last_modified_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
        ]

        mock_cursor = MagicMock()
        # Mock fetchone to return UUIDs for RETURNING id clause
        test_uuid1 = UUID("12345678-1234-5678-1234-567812345678")
        test_uuid2 = UUID("87654321-4321-8765-4321-876543218765")
        mock_cursor.fetchone.side_effect = [(test_uuid1,), (test_uuid2,)]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            client.insert_articles(articles)

            assert mock_cursor.execute.call_count == 2
            # Verify the SQL query structure
            first_call = mock_cursor.execute.call_args_list[0]
            assert "INSERT INTO articles" in first_call[0][0]
            assert "tdx_article_id, title, url, content_html" in first_call[0][0]

    def test_update_articles(self, client):
        """Test updating existing articles."""
        test_uuid = UUID("12345678-1234-5678-1234-567812345678")
        articles = [
            TdxArticle(
                id=test_uuid,
                tdx_article_id=123,
                title="Updated Title",
                url=HttpUrl("https://example.com/123"),
                content_html="<p>Updated content</p>",
                last_modified_date=datetime(2024, 1, 3, tzinfo=timezone.utc),
            )
        ]

        mock_cursor = MagicMock()
        # Mock fetchone to return UUID for RETURNING id clause
        mock_cursor.fetchone.return_value = (test_uuid,)

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            client.update_articles(articles)

            mock_cursor.execute.assert_called_once()
            # Verify the SQL query structure
            call_args = mock_cursor.execute.call_args[0]
            assert "UPDATE articles" in call_args[0]
            assert "SET title = %s" in call_args[0]
            assert "WHERE tdx_article_id = %s" in call_args[0]
            # Verify parameters
            params = call_args[1]
            assert params[0] == "Updated Title"
            assert params[-1] == 123  # tdx_article_id should be last parameter

    def test_context_manager_enter_exit(self, client):
        """Test context manager protocol."""
        with client as c:
            assert c == client

        # No error should be raised

    def test_close(self, client):
        """Test closing connection."""
        mock_conn = MagicMock()
        mock_conn.closed = False
        client._conn = mock_conn

        client.close()

        mock_conn.close.assert_called_once()
        assert client._conn is None

    def test_close_when_already_closed(self, client):
        """Test closing when connection is already closed."""
        mock_conn = MagicMock()
        mock_conn.closed = True
        client._conn = mock_conn

        client.close()

        mock_conn.close.assert_not_called()

    def test_close_when_no_connection(self, client):
        """Test closing when there's no connection."""
        client._conn = None
        client.close()  # Should not raise an error

    def test_insert_articles_empty_list(self, client):
        """Test inserting empty list of articles."""
        # Should return early without attempting connection
        client.insert_articles([])
        # No error should be raised

    def test_update_articles_empty_list(self, client):
        """Test updating empty list of articles."""
        # Should return early without attempting connection
        client.update_articles([])
        # No error should be raised

    def test_init_validates_configuration(self):
        """Test that initialization validates configuration."""
        with patch("core.storage_base.get_database_settings") as mock_settings:
            # Test missing HOST
            settings = Mock()
            settings.HOST = None
            settings.USER = "user"
            settings.NAME = "db"
            settings.PASSWORD.get_secret_value.return_value = "pass"
            mock_settings.return_value = settings

            with pytest.raises(ValueError, match="HOST is not configured"):
                PostgresClient()

            # Test missing USER
            settings.HOST = "localhost"
            settings.USER = None
            with pytest.raises(ValueError, match="USER is not configured"):
                PostgresClient()

            # Test missing NAME
            settings.USER = "user"
            settings.NAME = None
            with pytest.raises(ValueError, match="NAME is not configured"):
                PostgresClient()
