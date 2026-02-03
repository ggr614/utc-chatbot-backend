"""
AWS Bedrock Cohere reranker for improving hybrid search results.

Uses Cohere Rerank v3.5 model via AWS Bedrock to rerank search results
based on semantic relevance to the query. Provides superior ranking
compared to score-based fusion methods.

Example:
    >>> from core.reranker import CohereReranker
    >>> reranker = CohereReranker()
    >>> results = [
    ...     {"rank": 1, "combined_score": 0.045, "chunk": chunk1},
    ...     {"rank": 2, "combined_score": 0.038, "chunk": chunk2},
    ... ]
    >>> reranked = reranker.rerank(query="password reset", results=results)
    >>> print(f"Top result score: {reranked[0]['combined_score']}")
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from uuid import UUID
import json
import time

from utils.logger import get_logger, PerformanceLogger
from core.config import get_aws_reranker_settings

logger = get_logger(__name__)


@dataclass
class RerankerResult:
    """
    Represents a single reranked result.

    Attributes:
        chunk_id: UUID of the chunk
        rank: Position after reranking (1-indexed)
        relevance_score: Cohere relevance score (0-1, higher is better)
        original_rank: Original rank from fusion (for debugging)
        original_score: Original fusion score (for debugging)
    """

    chunk_id: UUID
    rank: int
    relevance_score: float
    original_rank: int
    original_score: float


class CohereReranker:
    """
    AWS Bedrock Cohere reranker for improving hybrid search results.

    Uses Cohere Rerank v3.5 model via AWS Bedrock to rerank search results
    based on semantic relevance to the query.

    Model: cohere.rerank-v3-5:0
    Region: us-east-1 (configurable)
    Max documents: 1000
    API version: 2

    The reranker takes a query and a list of search results, sends the text
    content to Cohere's reranking API, and returns results sorted by
    relevance score. Includes retry logic with exponential backoff for
    handling rate limits and transient errors.

    Example:
        >>> reranker = CohereReranker()
        >>> results = hybrid_search(query, bm25, vector, rrf_k=60)
        >>> reranked = reranker.rerank(query=query, results=results)
        >>> print(f"Top result: {reranked[0]['combined_score']:.3f}")
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """
        Initialize Cohere reranker with AWS Bedrock.

        Args:
            model_id: Bedrock model ID (default: from settings or cohere.rerank-v3-5:0)
            region_name: AWS region (default: from settings)
            max_retries: Number of retry attempts (default: 3)
            timeout: API call timeout in seconds (default: 30.0)

        Raises:
            ValueError: If required configuration is missing
            RuntimeError: If boto3 client initialization fails
        """
        logger.info("Initializing CohereReranker")

        try:
            # Load settings
            settings = get_aws_reranker_settings()

            # Validate required configuration
            if not settings.ACCESS_KEY_ID:
                raise ValueError("AWS_ACCESS_KEY_ID is not configured")
            if not settings.SECRET_ACCESS_KEY:
                raise ValueError("AWS_SECRET_ACCESS_KEY is not configured")
            if not settings.RERANKER_ARN:
                raise ValueError("AWS_RERANKER_ARN is not configured")

            # Store configuration
            self.model_id = model_id or settings.RERANKER_ARN
            self.region_name = region_name or settings.REGION_NAME
            self.max_retries = max_retries
            self.timeout = timeout
            self._last_rerank_latency_ms = 0  # Track last reranking latency

            logger.debug(
                f"Configuration: model_id={self.model_id}, "
                f"region={self.region_name}, max_retries={self.max_retries}"
            )

            # Initialize boto3 client
            try:
                import boto3
                from botocore.exceptions import NoCredentialsError

                self.bedrock_client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=self.region_name,
                    aws_access_key_id=settings.ACCESS_KEY_ID.get_secret_value(),
                    aws_secret_access_key=settings.SECRET_ACCESS_KEY.get_secret_value(),
                )

                logger.info("CohereReranker initialized successfully")
                logger.debug(
                    f"Using model: {self.model_id} in region {self.region_name}"
                )

            except ImportError as e:
                raise RuntimeError(
                    "boto3 is not installed. Install it with: pip install boto3"
                ) from e
            except NoCredentialsError as e:
                raise RuntimeError("AWS credentials not found or invalid") from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize boto3 Bedrock client: {str(e)}"
                ) from e

        except Exception as e:
            logger.error(f"Failed to initialize CohereReranker: {str(e)}")
            raise

    @property
    def last_rerank_latency_ms(self) -> int:
        """
        Get the latency of the last reranking operation in milliseconds.

        Returns:
            int: Latency in milliseconds (0 if no reranking has been performed)
        """
        return self._last_rerank_latency_ms

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using Cohere Rerank API.

        Takes a list of search results from hybrid search (with RRF scores)
        and reranks them using Cohere's neural reranking model. The input
        results should have the structure from hybrid_search() with keys:
        'rank', 'combined_score', and 'chunk' (TextChunk object).

        The reranked results maintain the same structure but with updated
        ranks and scores (Cohere relevance scores replace combined_score).

        Args:
            query: Search query string
            results: List of result dicts from hybrid_search
                Structure: [{'rank': int, 'combined_score': float, 'chunk': TextChunk}, ...]
            top_n: Optional limit on results returned (default: return all)

        Returns:
            List of result dicts with updated ranks and scores, sorted by relevance
            Structure: [{'rank': int, 'combined_score': float, 'chunk': TextChunk}, ...]

        Raises:
            ValueError: If query is empty or results list is invalid
            RuntimeError: If API call fails after retries

        Example:
            >>> results = [
            ...     {"rank": 1, "combined_score": 0.045, "chunk": chunk1},
            ...     {"rank": 2, "combined_score": 0.038, "chunk": chunk2},
            ... ]
            >>> reranked = reranker.rerank(query="password reset", results=results)
            >>> print(reranked[0]["combined_score"])  # Cohere score (0-1)
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

        # Limit to Cohere API max (1000 documents)
        if len(results) > 1000:
            logger.warning(
                f"Results exceed Cohere limit of 1000, truncating from {len(results)} to 1000"
            )
            results = results[:1000]

        logger.info(f"Reranking {len(results)} results for query: '{query[:50]}...'")

        # Track reranking latency
        start_time = time.time()

        # Extract text content from chunks
        documents = []
        for r in results:
            chunk = r.get("chunk")
            if chunk and hasattr(chunk, "text_content"):
                documents.append(chunk.text_content)
            else:
                logger.warning(f"Result missing valid chunk with text_content: {r}")
                documents.append("")  # Empty placeholder

        logger.debug(f"Extracted {len(documents)} documents for reranking")

        # Prepare request body
        request_body = {
            "query": query,
            "documents": documents,
            "top_n": top_n if top_n is not None else len(documents),
            "api_version": 2,  # Required by AWS Bedrock
        }

        # Invoke Cohere reranker with retry logic
        response_body = self._invoke_with_retry(request_body)

        # Parse and map results back to original structure
        reranked_results = self._parse_response(response_body, results)

        # Calculate and store reranking latency
        latency_ms = int((time.time() - start_time) * 1000)
        self._last_rerank_latency_ms = latency_ms

        logger.info(
            f"Reranking completed: {len(reranked_results)} results, "
            f"top score: {reranked_results[0]['combined_score']:.3f}, "
            f"latency: {latency_ms}ms"
        )

        return reranked_results

    def _invoke_with_retry(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke Bedrock API with exponential backoff retry logic.

        Args:
            request_body: Request payload for Cohere API

        Returns:
            Parsed response body as dict

        Raises:
            RuntimeError: If all retry attempts fail
        """
        from botocore.exceptions import ClientError

        retry_delay = 1.0  # Base delay in seconds

        for attempt in range(self.max_retries):
            try:
                with PerformanceLogger(
                    logger, f"Cohere rerank API call (attempt {attempt + 1})"
                ):
                    response = self.bedrock_client.invoke_model(
                        modelId=self.model_id,
                        body=json.dumps(request_body),
                        contentType="application/json",
                        accept="application/json",
                    )

                    # Parse response
                    response_body = json.loads(response["body"].read())
                    logger.debug(
                        f"Received {len(response_body.get('results', []))} results from API"
                    )
                    return response_body

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                error_message = e.response.get("Error", {}).get("Message", str(e))

                # Log error details
                logger.warning(
                    f"AWS Bedrock error on attempt {attempt + 1}/{self.max_retries}: "
                    f"{error_code} - {error_message}"
                )

                # Check if retryable
                retryable_codes = [
                    "ThrottlingException",
                    "ServiceUnavailable",
                    "InternalServerError",
                ]

                if error_code in retryable_codes and attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = retry_delay * (2**attempt)
                    logger.info(
                        f"Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-retryable error or exhausted retries
                    logger.error(
                        f"Cohere reranking failed: {error_code} - {error_message}"
                    )
                    raise RuntimeError(
                        f"Cohere reranking failed after {attempt + 1} attempts: {error_message}"
                    ) from e

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Cohere API response: {str(e)}")
                raise RuntimeError(f"Invalid JSON response from Cohere API: {e}") from e

            except Exception as e:
                logger.error(f"Unexpected error invoking Cohere API: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    logger.info(f"Retrying in {wait_time:.1f}s due to unexpected error")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"Cohere reranking failed: {str(e)}") from e

        # Should not reach here
        raise RuntimeError("Exhausted retry attempts without successful response")

    def _parse_response(
        self, response_body: Dict[str, Any], original_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Parse Cohere API response and map back to original result structure.

        Args:
            response_body: Parsed JSON response from Cohere API
            original_results: Original results list from hybrid search

        Returns:
            Reranked results with updated scores and ranks

        Raises:
            RuntimeError: If response format is invalid
        """
        try:
            rerank_results = response_body.get("results", [])

            if not rerank_results:
                logger.warning(
                    "No results returned from Cohere API, returning original results"
                )
                return original_results

            # Map Cohere results back to original structure
            reranked = []
            for new_rank, cohere_result in enumerate(rerank_results, start=1):
                index = cohere_result.get("index")
                relevance_score = cohere_result.get("relevance_score", 0.0)

                # Validate index
                if index is None or not isinstance(index, int):
                    logger.warning(f"Invalid index in Cohere result: {cohere_result}")
                    continue

                if index < 0 or index >= len(original_results):
                    logger.warning(
                        f"Index out of range: {index} (max: {len(original_results) - 1})"
                    )
                    continue

                # Get original result
                original = original_results[index]

                # Create reranked result with updated score and rank
                reranked_result = {
                    "rank": new_rank,
                    "combined_score": relevance_score,  # Replace with Cohere score
                    "chunk": original["chunk"],  # Keep original chunk
                }

                # Store original scores in metadata for debugging
                if "metadata" not in reranked_result:
                    reranked_result["metadata"] = {}
                reranked_result["metadata"]["original_rank"] = original.get("rank", 0)
                reranked_result["metadata"]["original_score"] = original.get(
                    "combined_score", 0.0
                )

                reranked.append(reranked_result)

            logger.debug(
                f"Successfully mapped {len(reranked)} reranked results "
                f"(original: {len(original_results)})"
            )

            return reranked

        except Exception as e:
            logger.error(f"Failed to parse Cohere response: {str(e)}")
            raise RuntimeError(f"Failed to parse reranking response: {e}") from e
