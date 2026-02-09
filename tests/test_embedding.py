"""
Tests for the embedding module (GenerateEmbeddingsOpenAI).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from openai import RateLimitError, APITimeoutError, AuthenticationError, APIError

from core.embedding import GenerateEmbeddingsOpenAI


class TestGenerateEmbeddingsOpenAI:
    """Test suite for GenerateEmbeddingsOpenAI class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for Azure OpenAI."""
        with patch("core.embedding.get_embedding_settings") as mock:
            settings = Mock()
            settings.DEPLOYMENT_NAME = "text-embedding-3-large"
            settings.ENDPOINT = "https://test.openai.azure.com"
            settings.API_VERSION = "2023-05-15"
            settings.MAX_TOKENS = 8000
            settings.EMBED_DIM = 3072
            settings.API_KEY.get_secret_value.return_value = "test-key"
            mock.return_value = settings
            yield settings

    @pytest.fixture
    def embeddings_client(self, mock_settings):
        """Create GenerateEmbeddingsOpenAI instance with mocked settings."""
        with patch("core.embedding.AzureOpenAI"):
            return GenerateEmbeddingsOpenAI()

    def test_init_validates_configuration(self):
        """Test that initialization validates required configuration."""
        with patch("core.embedding.get_embedding_settings") as mock_settings:
            settings = Mock()
            settings.DEPLOYMENT_NAME = None
            settings.ENDPOINT = "https://test.openai.azure.com"
            settings.API_VERSION = "2023-05-15"
            settings.MAX_TOKENS = 8000
            settings.EMBED_DIM = 3072
            settings.API_KEY.get_secret_value.return_value = "test-key"
            mock_settings.return_value = settings

            with pytest.raises(ValueError, match="DEPLOYMENT_NAME"):
                GenerateEmbeddingsOpenAI()

    def test_generate_embedding_validates_empty_chunk(self, embeddings_client):
        """Test that empty chunk raises ValueError."""
        with pytest.raises(ValueError, match="Chunk cannot be empty"):
            embeddings_client.generate_embedding("")

        with pytest.raises(ValueError, match="Chunk cannot be empty"):
            embeddings_client.generate_embedding("   ")

    def test_generate_embedding_validates_token_count(self, embeddings_client):
        """Test that chunks exceeding max_tokens raise RuntimeError."""
        with patch.object(
            embeddings_client.tokenizer, "num_tokens_from_string", return_value=10000
        ):
            with pytest.raises(RuntimeError, match="Failed to count tokens"):
                embeddings_client.generate_embedding("This is a very long chunk")

    def test_generate_embedding_success(self, embeddings_client):
        """Test successful embedding generation."""
        # Mock tokenizer
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1] * 3072  # Correct dimension

        embeddings_client.client.embeddings.create = Mock(return_value=mock_response)

        result = embeddings_client.generate_embedding("Test chunk")

        assert isinstance(result, list)
        assert len(result) == 3072
        assert all(isinstance(x, float) for x in result)

    def test_generate_embedding_validates_dimension(self, embeddings_client):
        """Test that wrong embedding dimension raises ValueError."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        # Mock response with wrong dimension
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1] * 1536  # Wrong dimension

        embeddings_client.client.embeddings.create = Mock(return_value=mock_response)

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            embeddings_client.generate_embedding("Test chunk")

    def test_generate_embedding_retries_on_rate_limit(self, embeddings_client):
        """Test that rate limit errors trigger retries."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        # First call raises RateLimitError, second succeeds
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1] * 3072

        # Create proper error with response and body
        mock_error_response = Mock()
        mock_error_response.status_code = 429
        rate_limit_error = RateLimitError(
            "Rate limit", response=mock_error_response, body=None
        )

        embeddings_client.client.embeddings.create = Mock(
            side_effect=[rate_limit_error, mock_response]
        )

        with patch("core.embedding.time.sleep"):  # Don't actually sleep in tests
            result = embeddings_client.generate_embedding("Test chunk")

        assert len(result) == 3072

    def test_generate_embedding_fails_after_max_retries(self, embeddings_client):
        """Test that failure after max retries raises RuntimeError."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        mock_error_response = Mock()
        mock_error_response.status_code = 429
        rate_limit_error = RateLimitError(
            "Rate limit", response=mock_error_response, body=None
        )

        embeddings_client.client.embeddings.create = Mock(side_effect=rate_limit_error)

        with patch("core.embedding.time.sleep"):
            with pytest.raises(RuntimeError, match="Rate limit exceeded"):
                embeddings_client.generate_embedding("Test chunk")

    def test_generate_embedding_handles_authentication_error(self, embeddings_client):
        """Test that authentication errors are handled properly."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        mock_error_response = Mock()
        mock_error_response.status_code = 401
        auth_error = AuthenticationError(
            "Auth failed", response=mock_error_response, body=None
        )

        embeddings_client.client.embeddings.create = Mock(side_effect=auth_error)

        with pytest.raises(RuntimeError, match="Azure OpenAI authentication failed"):
            embeddings_client.generate_embedding("Test chunk")


class TestGenerateEmbeddingsBatch:
    """Test suite for generate_embeddings_batch method."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for Azure OpenAI."""
        with patch("core.embedding.get_embedding_settings") as mock:
            settings = Mock()
            settings.DEPLOYMENT_NAME = "text-embedding-3-large"
            settings.ENDPOINT = "https://test.openai.azure.com"
            settings.API_VERSION = "2023-05-15"
            settings.MAX_TOKENS = 8000
            settings.EMBED_DIM = 3072
            settings.API_KEY.get_secret_value.return_value = "test-key"
            mock.return_value = settings
            yield settings

    @pytest.fixture
    def embeddings_client(self, mock_settings):
        """Create GenerateEmbeddingsOpenAI instance with mocked settings."""
        with patch("core.embedding.AzureOpenAI"):
            return GenerateEmbeddingsOpenAI()

    def test_batch_validates_empty_list(self, embeddings_client):
        """Test that empty chunks list raises ValueError."""
        with pytest.raises(ValueError, match="Chunks list cannot be empty"):
            embeddings_client.generate_embeddings_batch([])

    def test_batch_validates_empty_chunk_in_list(self, embeddings_client):
        """Test that an empty chunk within the list raises ValueError."""
        with pytest.raises(ValueError, match="Chunk at index 1 is empty"):
            embeddings_client.generate_embeddings_batch(["valid chunk", "", "another"])

    def test_batch_success(self, embeddings_client):
        """Test successful batch embedding generation."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        # Mock response with 3 embeddings
        mock_items = []
        for i in range(3):
            item = Mock()
            item.index = i
            item.embedding = [0.1 * (i + 1)] * 3072
            mock_items.append(item)

        mock_response = Mock()
        mock_response.data = mock_items

        embeddings_client.client.embeddings.create = Mock(return_value=mock_response)

        result = embeddings_client.generate_embeddings_batch(
            ["chunk1", "chunk2", "chunk3"]
        )

        assert len(result) == 3
        assert all(len(emb) == 3072 for emb in result)
        # Verify the API was called with all chunks at once
        embeddings_client.client.embeddings.create.assert_called_once_with(
            input=["chunk1", "chunk2", "chunk3"], model="text-embedding-3-large"
        )

    def test_batch_preserves_order(self, embeddings_client):
        """Test that batch results are returned in the correct order."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        # Return items in reverse order to test sorting by index
        mock_items = []
        for i in [2, 0, 1]:  # Intentionally out of order
            item = Mock()
            item.index = i
            item.embedding = [float(i)] * 3072
            mock_items.append(item)

        mock_response = Mock()
        mock_response.data = mock_items

        embeddings_client.client.embeddings.create = Mock(return_value=mock_response)

        result = embeddings_client.generate_embeddings_batch(
            ["chunk0", "chunk1", "chunk2"]
        )

        # Verify order matches index, not response order
        assert result[0][0] == 0.0
        assert result[1][0] == 1.0
        assert result[2][0] == 2.0

    def test_batch_validates_dimension(self, embeddings_client):
        """Test that wrong embedding dimension raises ValueError."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        mock_items = []
        item = Mock()
        item.index = 0
        item.embedding = [0.1] * 1536  # Wrong dimension
        mock_items.append(item)

        mock_response = Mock()
        mock_response.data = mock_items

        embeddings_client.client.embeddings.create = Mock(return_value=mock_response)

        with pytest.raises(ValueError, match="Embedding dimension mismatch at index 0"):
            embeddings_client.generate_embeddings_batch(["test chunk"])

    def test_batch_validates_response_count(self, embeddings_client):
        """Test that mismatched response count raises ValueError."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        # Return only 1 embedding for 2 chunks
        item = Mock()
        item.index = 0
        item.embedding = [0.1] * 3072

        mock_response = Mock()
        mock_response.data = [item]

        embeddings_client.client.embeddings.create = Mock(return_value=mock_response)

        with pytest.raises(ValueError, match="Response count mismatch"):
            embeddings_client.generate_embeddings_batch(["chunk1", "chunk2"])

    def test_batch_retries_on_rate_limit(self, embeddings_client):
        """Test that rate limit errors trigger retries for batch."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        item = Mock()
        item.index = 0
        item.embedding = [0.1] * 3072

        mock_response = Mock()
        mock_response.data = [item]

        mock_error_response = Mock()
        mock_error_response.status_code = 429
        rate_limit_error = RateLimitError(
            "Rate limit", response=mock_error_response, body=None
        )

        embeddings_client.client.embeddings.create = Mock(
            side_effect=[rate_limit_error, mock_response]
        )

        with patch("core.embedding.time.sleep"):
            result = embeddings_client.generate_embeddings_batch(["test chunk"])

        assert len(result) == 1
        assert len(result[0]) == 3072

    def test_batch_fails_after_max_retries(self, embeddings_client):
        """Test that failure after max retries raises RuntimeError."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        mock_error_response = Mock()
        mock_error_response.status_code = 429
        rate_limit_error = RateLimitError(
            "Rate limit", response=mock_error_response, body=None
        )

        embeddings_client.client.embeddings.create = Mock(side_effect=rate_limit_error)

        with patch("core.embedding.time.sleep"):
            with pytest.raises(RuntimeError, match="Rate limit exceeded"):
                embeddings_client.generate_embeddings_batch(["test chunk"])
