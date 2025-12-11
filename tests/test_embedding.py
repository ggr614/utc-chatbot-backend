"""
Tests for the embedding module (GenerateEmbeddingsOpenAI and GenerateEmbeddingsAWS).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from openai import RateLimitError, APITimeoutError, AuthenticationError, APIError
from botocore.exceptions import ClientError

from core.embedding import GenerateEmbeddingsOpenAI, GenerateEmbeddingsAWS


class TestGenerateEmbeddingsOpenAI:
    """Test suite for GenerateEmbeddingsOpenAI class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for Azure OpenAI."""
        with patch("core.embedding.get_settings") as mock:
            settings = Mock()
            settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-3-large"
            settings.AZURE_OPENAI_EMBED_ENDPOINT = "https://test.openai.azure.com"
            settings.AZURE_OPENAI_API_VERSION = "2023-05-15"
            settings.AZURE_MAX_TOKENS = 8000
            settings.AZURE_EMBED_DIM = 3072
            settings.AZURE_OPENAI_API_KEY.get_secret_value.return_value = "test-key"
            mock.return_value = settings
            yield settings

    @pytest.fixture
    def embeddings_client(self, mock_settings):
        """Create GenerateEmbeddingsOpenAI instance with mocked settings."""
        with patch("core.embedding.AzureOpenAI"):
            return GenerateEmbeddingsOpenAI()

    def test_init_validates_configuration(self):
        """Test that initialization validates required configuration."""
        with patch("core.embedding.get_settings") as mock_settings:
            settings = Mock()
            settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = None
            settings.AZURE_OPENAI_EMBED_ENDPOINT = "https://test.openai.azure.com"
            settings.AZURE_OPENAI_API_VERSION = "2023-05-15"
            settings.AZURE_MAX_TOKENS = 8000
            settings.AZURE_EMBED_DIM = 3072
            settings.AZURE_OPENAI_API_KEY.get_secret_value.return_value = "test-key"
            mock_settings.return_value = settings

            with pytest.raises(ValueError, match="AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"):
                GenerateEmbeddingsOpenAI()

    def test_generate_embedding_validates_empty_chunk(self, embeddings_client):
        """Test that empty chunk raises ValueError."""
        with pytest.raises(ValueError, match="Chunk cannot be empty"):
            embeddings_client.generate_embedding("")

        with pytest.raises(ValueError, match="Chunk cannot be empty"):
            embeddings_client.generate_embedding("   ")

    def test_generate_embedding_validates_token_count(self, embeddings_client):
        """Test that chunks exceeding max_tokens raise RuntimeError."""
        with patch.object(embeddings_client.tokenizer, 'num_tokens_from_string', return_value=10000):
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
        rate_limit_error = RateLimitError("Rate limit", response=mock_error_response, body=None)

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
        rate_limit_error = RateLimitError("Rate limit", response=mock_error_response, body=None)

        embeddings_client.client.embeddings.create = Mock(
            side_effect=rate_limit_error
        )

        with patch("core.embedding.time.sleep"):
            with pytest.raises(RuntimeError, match="Rate limit exceeded"):
                embeddings_client.generate_embedding("Test chunk")

    def test_generate_embedding_handles_authentication_error(self, embeddings_client):
        """Test that authentication errors are handled properly."""
        embeddings_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        mock_error_response = Mock()
        mock_error_response.status_code = 401
        auth_error = AuthenticationError("Auth failed", response=mock_error_response, body=None)

        embeddings_client.client.embeddings.create = Mock(
            side_effect=auth_error
        )

        with pytest.raises(RuntimeError, match="Azure OpenAI authentication failed"):
            embeddings_client.generate_embedding("Test chunk")


class TestGenerateEmbeddingsAWS:
    """Test suite for GenerateEmbeddingsAWS class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for AWS Bedrock."""
        with patch("core.embedding.get_settings") as mock:
            settings = Mock()
            settings.AWS_EMBED_MODEL_ID = "cohere.embed-english-v3"
            settings.AWS_REGION = "us-east-1"
            settings.AWS_MAX_TOKENS = 512
            settings.AWS_EMBED_DIM = 1536
            settings.AWS_ACCESS_KEY_ID.get_secret_value.return_value = "test-access-key"
            settings.AWS_SECRET_ACCESS_KEY.get_secret_value.return_value = "test-secret-key"
            mock.return_value = settings
            yield settings

    @pytest.fixture
    def aws_client(self, mock_settings):
        """Create GenerateEmbeddingsAWS instance with mocked settings."""
        with patch("core.embedding.boto3.client"):
            return GenerateEmbeddingsAWS()

    def test_init_validates_configuration(self):
        """Test that initialization validates required configuration."""
        with patch("core.embedding.get_settings") as mock_settings:
            settings = Mock()
            settings.AWS_EMBED_MODEL_ID = None
            settings.AWS_REGION = "us-east-1"
            settings.AWS_MAX_TOKENS = 512
            settings.AWS_EMBED_DIM = 1536
            settings.AWS_ACCESS_KEY_ID.get_secret_value.return_value = "test-key"
            settings.AWS_SECRET_ACCESS_KEY.get_secret_value.return_value = "test-secret"
            mock_settings.return_value = settings

            with pytest.raises(ValueError, match="AWS_EMBED_MODEL_ID"):
                GenerateEmbeddingsAWS()

    def test_generate_embedding_validates_empty_chunk(self, aws_client):
        """Test that empty chunk raises ValueError."""
        with pytest.raises(ValueError, match="Chunk cannot be empty"):
            aws_client.generate_embedding("")

    def test_generate_embedding_success(self, aws_client):
        """Test successful embedding generation."""
        aws_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        # Mock AWS response
        mock_response = {
            "body": Mock()
        }
        mock_response["body"].read = Mock(return_value=b'{"embeddings": {"float": [[0.1, 0.2]]}}')
        # Create embeddings with correct dimension
        embedding_data = {"embeddings": {"float": [[0.1] * 1536]}}
        mock_response["body"].read = Mock(return_value=str(embedding_data).replace("'", '"').encode())

        aws_client.client.invoke_model = Mock(return_value=mock_response)

        result = aws_client.generate_embedding("Test chunk")

        assert isinstance(result, list)
        assert len(result) == 1536

    def test_generate_embedding_handles_throttling(self, aws_client):
        """Test that throttling exceptions trigger retries."""
        aws_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        # First call raises throttling error, second succeeds
        error_response = {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}
        throttling_error = ClientError(error_response, "invoke_model")

        embedding_data = {"embeddings": {"float": [[0.1] * 1536]}}
        mock_success_response = {"body": Mock()}
        mock_success_response["body"].read = Mock(return_value=str(embedding_data).replace("'", '"').encode())

        aws_client.client.invoke_model = Mock(
            side_effect=[throttling_error, mock_success_response]
        )

        with patch("core.embedding.time.sleep"):
            result = aws_client.generate_embedding("Test chunk")

        assert len(result) == 1536

    def test_generate_embedding_handles_authentication_error(self, aws_client):
        """Test that authentication errors are handled properly."""
        aws_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        error_response = {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}}
        auth_error = ClientError(error_response, "invoke_model")

        aws_client.client.invoke_model = Mock(side_effect=auth_error)

        with pytest.raises(RuntimeError, match="AWS authentication failed"):
            aws_client.generate_embedding("Test chunk")

    def test_generate_embedding_validates_dimension(self, aws_client):
        """Test that wrong embedding dimension raises ValueError."""
        aws_client.tokenizer.num_tokens_from_string = Mock(return_value=100)

        # Mock response with wrong dimension
        embedding_data = {"embeddings": {"float": [[0.1] * 768]}}  # Wrong dimension
        mock_response = {"body": Mock()}
        mock_response["body"].read = Mock(return_value=str(embedding_data).replace("'", '"').encode())

        aws_client.client.invoke_model = Mock(return_value=mock_response)

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            aws_client.generate_embedding("Test chunk")
