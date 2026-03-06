"""
Tests for the embedding module (EmbeddingGenerator).
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from litellm.exceptions import RateLimitError, Timeout, AuthenticationError, APIError

from core.embedding import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for LiteLLM proxy."""
        with patch("core.embedding.get_litellm_settings") as mock:
            settings = Mock()
            settings.EMBEDDING_MODEL = "text-embedding-large-3"
            settings.PROXY_BASE_URL = "http://localhost:4000"
            settings.EMBED_MAX_TOKENS = 8191
            settings.EMBED_DIM = 3072
            settings.PROXY_API_KEY.get_secret_value.return_value = "test-key"
            mock.return_value = settings
            yield settings

    @pytest.fixture
    def embeddings_client(self, mock_settings):
        """Create EmbeddingGenerator instance with mocked settings."""
        with patch("core.embedding.Tokenizer"):
            return EmbeddingGenerator()

    def test_init_validates_empty_api_key(self):
        """Test that initialization validates API key is not empty."""
        with patch("core.embedding.get_litellm_settings") as mock_settings:
            settings = Mock()
            settings.EMBEDDING_MODEL = "text-embedding-large-3"
            settings.PROXY_BASE_URL = "http://localhost:4000"
            settings.EMBED_MAX_TOKENS = 8191
            settings.EMBED_DIM = 3072
            settings.PROXY_API_KEY.get_secret_value.return_value = ""
            mock_settings.return_value = settings

            with patch("core.embedding.Tokenizer"):
                with pytest.raises(ValueError, match="LITELLM_PROXY_API_KEY is empty"):
                    EmbeddingGenerator()

    def test_generate_embedding_validates_empty_chunk(self, embeddings_client):
        """Test that empty chunk raises ValueError."""
        with pytest.raises(ValueError, match="Chunk cannot be empty"):
            embeddings_client.generate_embedding("")

        with pytest.raises(ValueError, match="Chunk cannot be empty"):
            embeddings_client.generate_embedding("   ")

    def test_generate_embedding_validates_token_count(self, embeddings_client):
        """Test that chunks exceeding max_tokens raise ValueError."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=10000)

        with pytest.raises(ValueError, match="Chunk is too long"):
            embeddings_client.generate_embedding("This is a very long chunk")

    @patch("core.embedding.litellm.aembedding", new_callable=AsyncMock)
    def test_generate_embedding_success(self, mock_aembedding, embeddings_client):
        """Test successful embedding generation."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        mock_response = Mock()
        mock_response.data = [{"index": 0, "embedding": [0.1] * 3072}]
        mock_aembedding.return_value = mock_response

        result = embeddings_client.generate_embedding("Test chunk")

        assert isinstance(result, list)
        assert len(result) == 3072
        assert all(isinstance(x, float) for x in result)

    @patch("core.embedding.litellm.aembedding", new_callable=AsyncMock)
    def test_generate_embedding_validates_dimension(
        self, mock_aembedding, embeddings_client
    ):
        """Test that wrong embedding dimension raises ValueError."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        mock_response = Mock()
        mock_response.data = [{"index": 0, "embedding": [0.1] * 1536}]
        mock_aembedding.return_value = mock_response

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            embeddings_client.generate_embedding("Test chunk")

    @patch("core.embedding.litellm.aembedding", new_callable=AsyncMock)
    def test_generate_embedding_handles_rate_limit(
        self, mock_aembedding, embeddings_client
    ):
        """Test that rate limit errors are wrapped in RuntimeError."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        mock_aembedding.side_effect = RateLimitError(
            "Rate limit", llm_provider="litellm_proxy", model="text-embedding-large-3"
        )

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            embeddings_client.generate_embedding("Test chunk")

    @patch("core.embedding.litellm.aembedding", new_callable=AsyncMock)
    def test_generate_embedding_handles_authentication_error(
        self, mock_aembedding, embeddings_client
    ):
        """Test that authentication errors are handled properly."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        mock_aembedding.side_effect = AuthenticationError(
            "Auth failed", llm_provider="litellm_proxy", model="text-embedding-large-3"
        )

        with pytest.raises(RuntimeError, match="LiteLLM authentication failed"):
            embeddings_client.generate_embedding("Test chunk")

    @patch("core.embedding.litellm.aembedding", new_callable=AsyncMock)
    def test_generate_embedding_handles_timeout(
        self, mock_aembedding, embeddings_client
    ):
        """Test that timeout errors are handled properly."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        mock_aembedding.side_effect = Timeout(
            "Timeout", model="text-embedding-large-3", llm_provider="litellm_proxy"
        )

        with pytest.raises(RuntimeError, match="API timeout"):
            embeddings_client.generate_embedding("Test chunk")

    @patch("core.embedding.litellm.aembedding", new_callable=AsyncMock)
    def test_generate_embedding_handles_api_error(
        self, mock_aembedding, embeddings_client
    ):
        """Test that generic API errors are handled properly."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        mock_aembedding.side_effect = APIError(
            status_code=500,
            message="Server error",
            llm_provider="litellm_proxy",
            model="text-embedding-large-3",
        )

        with pytest.raises(RuntimeError, match="LiteLLM API error"):
            embeddings_client.generate_embedding("Test chunk")


class TestGenerateEmbeddingsBatch:
    """Test suite for generate_embeddings_batch method."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for LiteLLM proxy."""
        with patch("core.embedding.get_litellm_settings") as mock:
            settings = Mock()
            settings.EMBEDDING_MODEL = "text-embedding-large-3"
            settings.PROXY_BASE_URL = "http://localhost:4000"
            settings.EMBED_MAX_TOKENS = 8191
            settings.EMBED_DIM = 3072
            settings.PROXY_API_KEY.get_secret_value.return_value = "test-key"
            mock.return_value = settings
            yield settings

    @pytest.fixture
    def embeddings_client(self, mock_settings):
        """Create EmbeddingGenerator instance with mocked settings."""
        with patch("core.embedding.Tokenizer"):
            return EmbeddingGenerator()

    def test_batch_validates_empty_list(self, embeddings_client):
        """Test that empty chunks list raises ValueError."""
        with pytest.raises(ValueError, match="Chunks list cannot be empty"):
            embeddings_client.generate_embeddings_batch([])

    def test_batch_validates_empty_chunk_in_list(self, embeddings_client):
        """Test that an empty chunk within the list raises ValueError."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        with pytest.raises(ValueError, match="Chunk at index 1 is empty"):
            embeddings_client.generate_embeddings_batch(["valid chunk", "", "another"])

    @patch("core.embedding.litellm.aembedding", new_callable=AsyncMock)
    def test_batch_success(self, mock_aembedding, embeddings_client):
        """Test successful batch embedding generation."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        mock_response = Mock()
        mock_response.data = [
            {"index": 0, "embedding": [0.1] * 3072},
            {"index": 1, "embedding": [0.2] * 3072},
            {"index": 2, "embedding": [0.3] * 3072},
        ]
        mock_aembedding.return_value = mock_response

        result = embeddings_client.generate_embeddings_batch(
            ["chunk1", "chunk2", "chunk3"]
        )

        assert len(result) == 3
        assert all(len(emb) == 3072 for emb in result)

        # Verify the API was called with all chunks at once
        mock_aembedding.assert_called_once()
        call_kwargs = mock_aembedding.call_args[1]
        assert call_kwargs["input"] == ["chunk1", "chunk2", "chunk3"]
        assert call_kwargs["model"] == "openai/text-embedding-large-3"

    @patch("core.embedding.litellm.aembedding", new_callable=AsyncMock)
    def test_batch_preserves_order(self, mock_aembedding, embeddings_client):
        """Test that batch results are returned in the correct order."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        # Return items in reverse order to test sorting by index
        mock_response = Mock()
        mock_response.data = [
            {"index": 2, "embedding": [2.0] * 3072},
            {"index": 0, "embedding": [0.0] * 3072},
            {"index": 1, "embedding": [1.0] * 3072},
        ]
        mock_aembedding.return_value = mock_response

        result = embeddings_client.generate_embeddings_batch(
            ["chunk0", "chunk1", "chunk2"]
        )

        # Verify order matches index, not response order
        assert result[0][0] == 0.0
        assert result[1][0] == 1.0
        assert result[2][0] == 2.0

    @patch("core.embedding.litellm.aembedding", new_callable=AsyncMock)
    def test_batch_validates_dimension(self, mock_aembedding, embeddings_client):
        """Test that wrong embedding dimension raises ValueError."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        mock_response = Mock()
        mock_response.data = [
            {"index": 0, "embedding": [0.1] * 1536},  # Wrong dimension
        ]
        mock_aembedding.return_value = mock_response

        with pytest.raises(ValueError, match="Embedding dimension mismatch at index 0"):
            embeddings_client.generate_embeddings_batch(["test chunk"])

    @patch("core.embedding.litellm.aembedding", new_callable=AsyncMock)
    def test_batch_handles_rate_limit(self, mock_aembedding, embeddings_client):
        """Test that rate limit errors are wrapped in RuntimeError for batch."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        mock_aembedding.side_effect = RateLimitError(
            "Rate limit", llm_provider="litellm_proxy", model="text-embedding-large-3"
        )

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            embeddings_client.generate_embeddings_batch(["test chunk"])

    @patch("core.embedding.litellm.aembedding", new_callable=AsyncMock)
    def test_batch_handles_authentication_error(
        self, mock_aembedding, embeddings_client
    ):
        """Test that authentication errors are handled for batch."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        mock_aembedding.side_effect = AuthenticationError(
            "Auth failed", llm_provider="litellm_proxy", model="text-embedding-large-3"
        )

        with pytest.raises(RuntimeError, match="LiteLLM authentication failed"):
            embeddings_client.generate_embeddings_batch(["test chunk"])
