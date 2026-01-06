"""
Tests for the storage_vector module (VectorStorageClient, OpenAIVectorStorage, CohereVectorStorage).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic import HttpUrl

from core.storage_vector import (
    VectorStorageClient,
    OpenAIVectorStorage,
    CohereVectorStorage,
)
from core.schemas import VectorRecord


class TestVectorStorageClient:
    """Test suite for VectorStorageClient base class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for database connection."""
        with patch("core.storage_vector.get_settings") as mock:
            settings = Mock()
            settings.DB_HOST = "localhost"
            settings.DB_USER = "test_user"
            settings.DB_PASSWORD.get_secret_value.return_value = "test_password"
            settings.DB_NAME = "test_db"
            mock.return_value = settings
            yield settings

    @pytest.fixture
    def client(self, mock_settings):
        """Create VectorStorageClient instance with mocked settings."""
        return VectorStorageClient(table_name="test_embeddings", embedding_dim=1536)

    def test_init(self, client):
        """Test client initialization."""
        assert client.db_host == "localhost"
        assert client.db_user == "test_user"
        assert client.db_password == "test_password"
        assert client.db_name == "test_db"
        assert client.db_port == 5432
        assert client.table_name == "test_embeddings"
        assert client.embedding_dim == 1536

    def test_init_validates_configuration(self):
        """Test that initialization validates configuration."""
        with patch("core.storage_vector.get_settings") as mock_settings:
            # Test missing DB_HOST
            settings = Mock()
            settings.DB_HOST = None
            settings.DB_USER = "user"
            settings.DB_NAME = "db"
            settings.DB_PASSWORD.get_secret_value.return_value = "pass"
            mock_settings.return_value = settings

            with pytest.raises(ValueError, match="DB_HOST is not configured"):
                VectorStorageClient(table_name="test", embedding_dim=1536)

    def test_get_connection_success(self, client):
        """Test successful database connection."""
        with patch("core.storage_vector.psycopg.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_conn.closed = False
            mock_connect.return_value = mock_conn

            with client.get_connection() as conn:
                assert conn == mock_conn

            mock_conn.commit.assert_called_once()
            assert mock_conn.close.called

    def test_get_connection_rollback_on_error(self, client):
        """Test that connection rolls back on error."""
        with patch("core.storage_vector.psycopg.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_conn.closed = False
            mock_connect.return_value = mock_conn

            with pytest.raises(ConnectionError):
                with client.get_connection() as conn:
                    import psycopg

                    raise psycopg.Error("Test error")

            mock_conn.rollback.assert_called_once()
            assert mock_conn.close.called

    def test_context_manager(self, client):
        """Test context manager protocol."""
        with client as c:
            assert c == client

    def test_close(self, client):
        """Test closing connection."""
        mock_conn = MagicMock()
        mock_conn.closed = False
        client._conn = mock_conn

        client.close()

        mock_conn.close.assert_called_once()
        assert client._conn is None

    def test_get_existing_chunk_ids(self, client):
        """Test retrieving existing chunk IDs."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("chunk_1",),
            ("chunk_2",),
            ("chunk_3",),
        ]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            chunk_ids = client.get_existing_chunk_ids()

            assert chunk_ids == {"chunk_1", "chunk_2", "chunk_3"}
            assert isinstance(chunk_ids, set)

    def test_get_chunks_by_article_id(self, client):
        """Test retrieving chunks for a specific article."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("chunk_1", 123, 0, "Text content 1", 100, "https://example.com", None),
            ("chunk_2", 123, 1, "Text content 2", 150, "https://example.com", None),
        ]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            chunks = client.get_chunks_by_article_id(123)

            assert len(chunks) == 2
            assert chunks[0]["chunk_id"] == "chunk_1"
            assert chunks[0]["parent_article_id"] == 123
            assert chunks[1]["chunk_sequence"] == 1

    def test_insert_embeddings_validates_dimensions(self, client):
        """Test that insert validates embedding dimensions."""
        record = VectorRecord(
            chunk_id="test_chunk",
            parent_article_id=123,
            chunk_sequence=0,
            text_content="Test content",
            token_count=10,
            source_url=HttpUrl("https://example.com"),
        )
        wrong_embedding = [0.1] * 768  # Wrong dimension

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            client.insert_embeddings([(record, wrong_embedding)])

    def test_insert_embeddings_success(self, client):
        """Test successful embedding insertion."""
        record = VectorRecord(
            chunk_id="test_chunk",
            parent_article_id=123,
            chunk_sequence=0,
            text_content="Test content",
            token_count=10,
            source_url=HttpUrl("https://example.com"),
        )
        embedding = [0.1] * 1536  # Correct dimension

        mock_cursor = MagicMock()

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            client.insert_embeddings([(record, embedding)])

            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args[0]
            assert "INSERT INTO test_embeddings" in call_args[0]

    def test_insert_embeddings_empty_list(self, client):
        """Test inserting empty list."""
        client.insert_embeddings([])
        # Should return early without error

    def test_update_embeddings_validates_dimensions(self, client):
        """Test that update validates embedding dimensions."""
        record = VectorRecord(
            chunk_id="test_chunk",
            parent_article_id=123,
            chunk_sequence=0,
            text_content="Test content",
            token_count=10,
            source_url=HttpUrl("https://example.com"),
        )
        wrong_embedding = [0.1] * 768  # Wrong dimension

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            client.update_embeddings([(record, wrong_embedding)])

    def test_update_embeddings_success(self, client):
        """Test successful embedding update."""
        record = VectorRecord(
            chunk_id="test_chunk",
            parent_article_id=123,
            chunk_sequence=0,
            text_content="Updated content",
            token_count=10,
            source_url=HttpUrl("https://example.com"),
        )
        embedding = [0.2] * 1536  # Correct dimension

        mock_cursor = MagicMock()

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            client.update_embeddings([(record, embedding)])

            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args[0]
            assert "UPDATE test_embeddings" in call_args[0]
            assert "WHERE chunk_id = %s" in call_args[0]

    def test_update_embeddings_empty_list(self, client):
        """Test updating empty list."""
        client.update_embeddings([])
        # Should return early without error

    def test_delete_embeddings_by_article_id(self, client):
        """Test deleting embeddings by article ID."""
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 5

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            deleted_count = client.delete_embeddings_by_article_id(123)

            assert deleted_count == 5
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args[0]
            assert "DELETE FROM test_embeddings" in call_args[0]
            assert "WHERE parent_article_id = %s" in call_args[0]

    def test_delete_embeddings_by_chunk_ids(self, client):
        """Test deleting embeddings by chunk IDs."""
        chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 3

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            deleted_count = client.delete_embeddings_by_chunk_ids(chunk_ids)

            assert deleted_count == 3
            mock_cursor.execute.assert_called_once()

    def test_delete_embeddings_by_chunk_ids_empty_list(self, client):
        """Test deleting with empty chunk ID list."""
        deleted_count = client.delete_embeddings_by_chunk_ids([])
        assert deleted_count == 0

    def test_search_similar_vectors_validates_dimension(self, client):
        """Test that search validates query vector dimension."""
        wrong_query = [0.1] * 768  # Wrong dimension

        with pytest.raises(ValueError, match="Query vector dimension mismatch"):
            client.search_similar_vectors(wrong_query)

    def test_search_similar_vectors_success(self, client):
        """Test successful vector similarity search."""
        query_vector = [0.1] * 1536  # Correct dimension
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("chunk_1", 123, 0, "Text 1", 100, "https://example.com", None, 0.95),
            ("chunk_2", 124, 0, "Text 2", 150, "https://example.com", None, 0.85),
        ]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            results = client.search_similar_vectors(
                query_vector, limit=10, min_similarity=0.7
            )

            assert len(results) == 2
            assert results[0]["chunk_id"] == "chunk_1"
            assert results[0]["similarity"] == 0.95
            assert results[1]["similarity"] == 0.85
            mock_cursor.execute.assert_called_once()


class TestOpenAIVectorStorage:
    """Test suite for OpenAIVectorStorage class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for database connection."""
        with patch("core.storage_vector.get_settings") as mock:
            settings = Mock()
            settings.DB_HOST = "localhost"
            settings.DB_USER = "test_user"
            settings.DB_PASSWORD.get_secret_value.return_value = "test_password"
            settings.DB_NAME = "test_db"
            mock.return_value = settings
            yield settings

    def test_init(self, mock_settings):
        """Test OpenAI client initialization."""
        client = OpenAIVectorStorage()

        assert client.table_name == "embeddings_openai"
        assert client.embedding_dim == 3072
        assert client.db_host == "localhost"

    def test_correct_dimension(self, mock_settings):
        """Test that OpenAI client uses correct embedding dimension."""
        client = OpenAIVectorStorage()

        # Should accept 3072-dimensional vectors
        record = VectorRecord(
            chunk_id="test_chunk",
            parent_article_id=123,
            chunk_sequence=0,
            text_content="Test content",
            token_count=10,
            source_url=HttpUrl("https://example.com"),
        )
        embedding = [0.1] * 3072

        # Validate dimensions (should not raise)
        for idx, (rec, emb) in enumerate([(record, embedding)]):
            if len(emb) != client.embedding_dim:
                pytest.fail(f"Dimension validation failed for OpenAI (expected 3072)")


class TestCohereVectorStorage:
    """Test suite for CohereVectorStorage class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for database connection."""
        with patch("core.storage_vector.get_settings") as mock:
            settings = Mock()
            settings.DB_HOST = "localhost"
            settings.DB_USER = "test_user"
            settings.DB_PASSWORD.get_secret_value.return_value = "test_password"
            settings.DB_NAME = "test_db"
            mock.return_value = settings
            yield settings

    def test_init(self, mock_settings):
        """Test Cohere client initialization."""
        client = CohereVectorStorage()

        assert client.table_name == "embeddings_cohere"
        assert client.embedding_dim == 1536
        assert client.db_host == "localhost"

    def test_correct_dimension(self, mock_settings):
        """Test that Cohere client uses correct embedding dimension."""
        client = CohereVectorStorage()

        # Should accept 1536-dimensional vectors
        record = VectorRecord(
            chunk_id="test_chunk",
            parent_article_id=123,
            chunk_sequence=0,
            text_content="Test content",
            token_count=10,
            source_url=HttpUrl("https://example.com"),
        )
        embedding = [0.1] * 1536

        # Validate dimensions (should not raise)
        for idx, (rec, emb) in enumerate([(record, embedding)]):
            if len(emb) != client.embedding_dim:
                pytest.fail(f"Dimension validation failed for Cohere (expected 1536)")
