from openai import (
    AzureOpenAI,
    OpenAIError,
    APIError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
)
from typing import List
from core.config import get_embedding_settings
from core.tokenizer import Tokenizer
import time
import logging

logger = logging.getLogger(__name__)


class GenerateEmbeddingsOpenAI:
    def __init__(self):
        try:
            settings = get_embedding_settings()

            # Validate required configuration
            if not settings.DEPLOYMENT_NAME:
                raise ValueError("EMBEDDING_DEPLOYMENT_NAME is not configured")
            if not settings.ENDPOINT:
                raise ValueError("EMBEDDING_ENDPOINT is not configured")
            if not settings.API_VERSION:
                raise ValueError("EMBEDDING_API_VERSION is not configured")
            if not settings.MAX_TOKENS or settings.MAX_TOKENS <= 0:
                raise ValueError("EMBEDDING_MAX_TOKENS must be a positive integer")
            if not settings.EMBED_DIM or settings.EMBED_DIM <= 0:
                raise ValueError("EMBEDDING_EMBED_DIM must be a positive integer")

            self.deployment_name = settings.DEPLOYMENT_NAME
            self.max_tokens = settings.MAX_TOKENS
            self.expected_dim = settings.EMBED_DIM
            self.tokenizer = Tokenizer()

            # Initialize client with error handling
            try:
                api_key = settings.API_KEY.get_secret_value()
                if not api_key:
                    raise ValueError("AZURE_OPENAI_API_KEY is empty")

                self.client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=settings.ENDPOINT,
                    api_version=settings.API_VERSION,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize Azure OpenAI client: {str(e)}"
                ) from e

        except Exception as e:
            logger.error(f"Failed to initialize GenerateEmbeddingsOpenAI: {str(e)}")
            raise

    def generate_embedding(self, chunk: str) -> List[float]:
        # Input validation
        if not chunk or not chunk.strip():
            raise ValueError("Chunk cannot be empty.")

        # Token count validation with error handling
        try:
            token_count = self.tokenizer.num_tokens_from_string(chunk)
            if token_count > self.max_tokens:
                raise ValueError(
                    f"Chunk is too long. Max tokens allowed: {self.max_tokens} "
                    f"but chunk had {token_count} tokens."
                )
        except Exception as e:
            logger.error(f"Token counting failed: {str(e)}")
            raise RuntimeError(f"Failed to count tokens: {str(e)}") from e

        # Retry logic for API calls
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=chunk, model=self.deployment_name
                )

                # Validate response structure
                if not response or not hasattr(response, "data"):
                    raise ValueError(
                        "Invalid response structure: missing 'data' attribute"
                    )

                if not response.data or len(response.data) == 0:
                    raise ValueError("Response data is empty")

                if not hasattr(response.data[0], "embedding"):
                    raise ValueError("Response data[0] missing 'embedding' attribute")

                embeddings = response.data[0].embedding

                if not embeddings:
                    raise ValueError("Embeddings list is empty")

                # Validate embedding type and dimension
                if not isinstance(embeddings, list):
                    raise ValueError(
                        f"Expected embeddings to be a list, got {type(embeddings)}"
                    )

                if not all(isinstance(x, (int, float)) for x in embeddings):
                    raise ValueError("Embeddings contain non-numeric values")

                if len(embeddings) != self.expected_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch. Expected {self.expected_dim}, "
                        f"got {len(embeddings)}"
                    )

                return embeddings

            except AuthenticationError as e:
                logger.error(f"Authentication failed: {str(e)}")
                raise RuntimeError(
                    f"Azure OpenAI authentication failed: {str(e)}"
                ) from e

            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"Rate limit hit, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts")
                    raise RuntimeError(f"Rate limit exceeded: {str(e)}") from e

            except APITimeoutError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    logger.warning(
                        f"API timeout, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"API timeout after {max_retries} attempts")
                    raise RuntimeError(f"API timeout: {str(e)}") from e

            except APIError as e:
                # For server errors (5xx), retry
                status_code = getattr(e, "status_code", None)
                if status_code and 500 <= status_code < 600:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2**attempt)
                        logger.warning(
                            f"Server error {status_code}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                logger.error(f"API error: {str(e)}")
                raise RuntimeError(f"Azure OpenAI API error: {str(e)}") from e

            except OpenAIError as e:
                logger.error(f"OpenAI client error: {str(e)}")
                raise RuntimeError(f"OpenAI client error: {str(e)}") from e

            except ValueError as e:
                # Don't retry on validation errors
                logger.error(f"Validation error: {str(e)}")
                raise

            except Exception as e:
                logger.error(f"Unexpected error generating embeddings: {str(e)}")
                raise RuntimeError(f"Unexpected error: {str(e)}") from e

        raise RuntimeError(
            f"Failed to generate embeddings after {max_retries} attempts"
        )
