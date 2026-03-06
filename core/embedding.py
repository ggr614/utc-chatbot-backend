import litellm
from litellm.exceptions import (
    RateLimitError,
    Timeout,
    AuthenticationError,
    APIError,
)
from typing import List
from core.config import get_litellm_settings
from core.tokenizer import Tokenizer
import asyncio
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Provider-agnostic embedding generator using LiteLLM proxy."""

    def __init__(self):
        try:
            settings = get_litellm_settings()

            self.model = settings.EMBEDDING_MODEL
            self._proxy_model = f"openai/{self.model}"
            self.api_base = settings.PROXY_BASE_URL
            self.api_key = settings.PROXY_API_KEY.get_secret_value()
            self.max_tokens = settings.EMBED_MAX_TOKENS
            self.expected_dim = settings.EMBED_DIM
            self.deployment_name = self.model  # backward compat for get_stats()
            self.tokenizer = Tokenizer()

            if not self.api_key:
                raise ValueError("LITELLM_PROXY_API_KEY is empty")

            logger.info(f"EmbeddingGenerator initialized: model={self.model}")

        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingGenerator: {str(e)}")
            raise

    async def agenerate_embeddings_batch(
        self, chunks: List[str]
    ) -> List[List[float]]:
        """Generate embeddings for multiple text chunks in a single API call (async).

        Args:
            chunks: List of text strings to embed

        Returns:
            List of embedding vectors in the same order as input chunks

        Raises:
            ValueError: If chunks list is empty or contains invalid entries
            RuntimeError: If API call fails after retries
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty.")

        for idx, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                raise ValueError(f"Chunk at index {idx} is empty.")

            token_count = self.tokenizer.num_tokens_from_string(chunk)
            if token_count > self.max_tokens:
                raise ValueError(
                    f"Chunk at index {idx} is too long. Max tokens allowed: "
                    f"{self.max_tokens} but chunk had {token_count} tokens."
                )

        try:
            response = await litellm.aembedding(
                model=self._proxy_model,
                input=chunks,
                api_base=self.api_base,
                api_key=self.api_key,
                num_retries=3,
                timeout=60.0,
            )

            sorted_data = sorted(response.data, key=lambda x: x["index"])

            all_embeddings = []
            for idx, item in enumerate(sorted_data):
                embedding = item["embedding"]

                if len(embedding) != self.expected_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch at index {idx}. "
                        f"Expected {self.expected_dim}, got {len(embedding)}"
                    )

                all_embeddings.append(embedding)

            return all_embeddings

        except AuthenticationError as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise RuntimeError(
                f"LiteLLM authentication failed: {str(e)}"
            ) from e
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {str(e)}")
            raise RuntimeError(f"Rate limit exceeded: {str(e)}") from e
        except Timeout as e:
            logger.error(f"API timeout: {str(e)}")
            raise RuntimeError(f"API timeout: {str(e)}") from e
        except APIError as e:
            logger.error(f"API error: {str(e)}")
            raise RuntimeError(f"LiteLLM API error: {str(e)}") from e
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating batch embeddings: {str(e)}")
            raise RuntimeError(f"Unexpected error: {str(e)}") from e

    def generate_embeddings_batch(self, chunks: List[str]) -> List[List[float]]:
        """Synchronous wrapper for backward compatibility."""
        return asyncio.run(self.agenerate_embeddings_batch(chunks))

    async def agenerate_embedding(self, chunk: str) -> List[float]:
        """Generate embedding for a single text chunk (async).

        Args:
            chunk: Text string to embed

        Returns:
            Embedding vector

        Raises:
            ValueError: If chunk is empty or too long
            RuntimeError: If API call fails after retries
        """
        if not chunk or not chunk.strip():
            raise ValueError("Chunk cannot be empty.")

        token_count = self.tokenizer.num_tokens_from_string(chunk)
        if token_count > self.max_tokens:
            raise ValueError(
                f"Chunk is too long. Max tokens allowed: {self.max_tokens} "
                f"but chunk had {token_count} tokens."
            )

        try:
            response = await litellm.aembedding(
                model=self._proxy_model,
                input=[chunk],
                api_base=self.api_base,
                api_key=self.api_key,
                num_retries=3,
                timeout=30.0,
            )

            embedding = response.data[0]["embedding"]

            if len(embedding) != self.expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch. Expected {self.expected_dim}, "
                    f"got {len(embedding)}"
                )

            return embedding

        except AuthenticationError as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise RuntimeError(
                f"LiteLLM authentication failed: {str(e)}"
            ) from e
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {str(e)}")
            raise RuntimeError(f"Rate limit exceeded: {str(e)}") from e
        except Timeout as e:
            logger.error(f"API timeout: {str(e)}")
            raise RuntimeError(f"API timeout: {str(e)}") from e
        except APIError as e:
            logger.error(f"API error: {str(e)}")
            raise RuntimeError(f"LiteLLM API error: {str(e)}") from e
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {str(e)}")
            raise RuntimeError(f"Unexpected error: {str(e)}") from e

    def generate_embedding(self, chunk: str) -> List[float]:
        """Synchronous wrapper for backward compatibility."""
        return asyncio.run(self.agenerate_embedding(chunk))
