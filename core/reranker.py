"""
Neural reranker for improving hybrid search results.

Uses LiteLLM proxy to rerank search results based on semantic relevance
to the query. Provides superior ranking compared to score-based fusion methods.

Example:
    >>> from core.reranker import Reranker
    >>> reranker = Reranker()
    >>> results = [
    ...     {"rank": 1, "combined_score": 0.045, "chunk": chunk1},
    ...     {"rank": 2, "combined_score": 0.038, "chunk": chunk2},
    ... ]
    >>> reranked = reranker.rerank(query="password reset", results=results)
    >>> print(f"Top result score: {reranked[0]['combined_score']}")
"""

import litellm
from litellm.exceptions import (
    RateLimitError,
    Timeout,
    AuthenticationError,
    APIError,
)
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from uuid import UUID
import time

from utils.logger import get_logger, PerformanceLogger
from core.config import get_litellm_settings

logger = get_logger(__name__)


@dataclass
class RerankerResult:
    """
    Represents a single reranked result.

    Attributes:
        chunk_id: UUID of the chunk
        rank: Position after reranking (1-indexed)
        relevance_score: Relevance score (0-1, higher is better)
        original_rank: Original rank from fusion (for debugging)
        original_score: Original fusion score (for debugging)
    """

    chunk_id: UUID
    rank: int
    relevance_score: float
    original_rank: int
    original_score: float


class Reranker:
    """
    Neural reranker using LiteLLM proxy.

    Takes a query and a list of search results, sends the text content to
    the reranking model via LiteLLM proxy, and returns results sorted by
    relevance score.

    Example:
        >>> reranker = Reranker()
        >>> results = hybrid_search(query, bm25, vector, rrf_k=60)
        >>> reranked = reranker.rerank(query=query, results=results)
        >>> print(f"Top result: {reranked[0]['combined_score']:.3f}")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """
        Initialize reranker with LiteLLM proxy settings.

        Args:
            model: LiteLLM model alias (default: from settings)
            max_retries: Number of retry attempts (default: 3)
            timeout: API call timeout in seconds (default: 30.0)

        Raises:
            ValueError: If required configuration is missing
            RuntimeError: If initialization fails
        """
        logger.info("Initializing Reranker")

        try:
            settings = get_litellm_settings()

            self.model = model or settings.RERANKER_MODEL
            self._proxy_model =self.model
            self.api_base = settings.PROXY_BASE_URL
            self.api_key = settings.PROXY_API_KEY.get_secret_value()
            self.max_retries = max_retries
            self.timeout = timeout
            self._last_rerank_latency_ms = 0

            if not self.api_key:
                raise ValueError("LITELLM_PROXY_API_KEY is empty")

            logger.info(
                f"Reranker initialized: model={self.model}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Reranker: {str(e)}")
            raise

    @property
    def last_rerank_latency_ms(self) -> int:
        """Get the latency of the last reranking operation in milliseconds."""
        return self._last_rerank_latency_ms

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using LiteLLM rerank API.

        Takes a list of search results from hybrid search (with RRF scores)
        and reranks them using the reranking model. The input results should
        have the structure from hybrid_search() with keys: 'rank',
        'combined_score', and 'chunk' (TextChunk object).

        Args:
            query: Search query string
            results: List of result dicts from hybrid_search
            top_n: Optional limit on results returned (default: return all)

        Returns:
            List of result dicts with updated ranks and scores, sorted by relevance

        Raises:
            ValueError: If query is empty or results list is invalid
            RuntimeError: If API call fails after retries
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Handle empty results
        if not results:
            logger.warning("Empty results list provided, returning empty list")
            return []

        # Validate results structure
        if not all(isinstance(r, dict) for r in results):
            raise ValueError("Results must be a list of dicts")
        if not all("chunk" in r for r in results):
            raise ValueError("Each result must have a 'chunk' field")

        # Limit to max 1000 documents
        if len(results) > 1000:
            logger.warning(
                f"Results exceed limit of 1000, truncating from {len(results)} to 1000"
            )
            results = results[:1000]

        logger.info(f"Reranking {len(results)} results for query: '{query[:50]}...'")

        start_time = time.time()

        # Extract text content from chunks
        documents = []
        for r in results:
            chunk = r.get("chunk")
            if chunk and hasattr(chunk, "text_content"):
                documents.append(chunk.text_content)
            else:
                logger.warning(f"Result missing valid chunk with text_content: {r}")
                documents.append("")

        logger.debug(f"Extracted {len(documents)} documents for reranking")

        try:
            with PerformanceLogger(logger, "LiteLLM rerank API call"):
                response = litellm.rerank(
                    model=f"cohere/{self._proxy_model}",
                    query=query,
                    documents=documents,
                    top_n=top_n if top_n is not None else len(documents),
                    api_base=self.api_base,
                    api_key=self.api_key,
                )

            # Parse response and map back to original result structure
            reranked_results = self._parse_response(response, results)

            latency_ms = int((time.time() - start_time) * 1000)
            self._last_rerank_latency_ms = latency_ms

            logger.info(
                f"Reranking completed: {len(reranked_results)} results, "
                f"top score: {reranked_results[0]['combined_score']:.3f}, "
                f"latency: {latency_ms}ms"
            )

            return reranked_results

        except AuthenticationError as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise RuntimeError(f"Reranking authentication failed: {str(e)}") from e
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {str(e)}")
            raise RuntimeError(f"Reranking rate limit exceeded: {str(e)}") from e
        except Timeout as e:
            logger.error(f"API timeout: {str(e)}")
            raise RuntimeError(f"Reranking API timeout: {str(e)}") from e
        except APIError as e:
            logger.error(f"API error: {str(e)}")
            raise RuntimeError(f"Reranking API error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected reranking error: {str(e)}")
            raise RuntimeError(f"Reranking failed: {str(e)}") from e

    def _parse_response(
        self, response, original_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Parse LiteLLM rerank response and map back to original result structure.

        Args:
            response: LiteLLM RerankResponse object
            original_results: Original results list from hybrid search

        Returns:
            Reranked results with updated scores and ranks

        Raises:
            RuntimeError: If response format is invalid
        """
        try:
            # Handle both object and dict responses from LiteLLM
            if isinstance(response, dict):
                rerank_results = response["results"]
            else:
                rerank_results = response.results

            if not rerank_results:
                logger.warning(
                    "No results returned from reranker, returning original results"
                )
                return original_results

            reranked = []
            for new_rank, rerank_result in enumerate(rerank_results, start=1):
                # Handle both object attributes and dict responses from LiteLLM
                if isinstance(rerank_result, dict):
                    index = rerank_result["index"]
                    relevance_score = rerank_result["relevance_score"]
                else:
                    index = rerank_result.index
                    relevance_score = rerank_result.relevance_score

                if index < 0 or index >= len(original_results):
                    logger.warning(
                        f"Index out of range: {index} (max: {len(original_results) - 1})"
                    )
                    continue

                original = original_results[index]

                reranked_result = {
                    "rank": new_rank,
                    "combined_score": relevance_score,
                    "chunk": original["chunk"],
                }

                # Preserve system_prompt from original result
                if "system_prompt" in original:
                    reranked_result["system_prompt"] = original["system_prompt"]

                # Store original scores in metadata for debugging
                reranked_result["metadata"] = {
                    "original_rank": original.get("rank", 0),
                    "original_score": original.get("combined_score", 0.0),
                }

                reranked.append(reranked_result)

            logger.debug(
                f"Successfully mapped {len(reranked)} reranked results "
                f"(original: {len(original_results)})"
            )

            return reranked

        except Exception as e:
            logger.error(f"Failed to parse rerank response: {str(e)}")
            raise RuntimeError(f"Failed to parse reranking response: {e}") from e
