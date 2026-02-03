"""
Dependency injection for FastAPI endpoints.

Provides reusable dependencies for:
- API key authentication
- Retriever access (BM25, vector, reranker)
- Query logging client
"""

from fastapi import Header, HTTPException, status, Request
from typing import Annotated, Optional
from core.config import get_api_settings
from core.bm25_search import BM25Retriever
from core.vector_search import VectorRetriever
from core.reranker import CohereReranker
from core.storage_query_log import QueryLogClient
from core.storage_reranker_log import RerankerLogClient
from utils.logger import get_logger

logger = get_logger(__name__)


async def verify_api_key(
    x_api_key: Annotated[
        str, Header(description="API authentication key", alias="X-API-Key")
    ],
) -> str:
    """
    Dependency to verify API key from X-API-Key header.

    Supports multiple keys via API_ALLOWED_API_KEYS environment variable
    (comma-separated list).

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        The validated API key (for logging purposes)

    Raises:
        HTTPException: 401 Unauthorized if API key is invalid or missing
        HTTPException: 500 Internal Server Error if API keys not configured

    Example:
        >>> @router.post("/endpoint")
        >>> async def endpoint(api_key: Annotated[str, Depends(verify_api_key)]):
        >>>     # API key has been validated
        >>>     pass
    """
    try:
        settings = get_api_settings()
    except Exception as e:
        logger.error(f"Failed to load API settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API configuration error",
        )

    # Build list of valid API keys
    valid_keys = set()

    # Add primary API key
    if settings.API_KEY:
        valid_keys.add(settings.API_KEY.get_secret_value())

    # Add additional allowed keys (comma-separated)
    if settings.ALLOWED_API_KEYS:
        additional_keys = [
            k.strip() for k in settings.ALLOWED_API_KEYS.split(",") if k.strip()
        ]
        valid_keys.update(additional_keys)

    # Verify at least one key is configured
    if not valid_keys:
        logger.error("No API keys configured (API_API_KEY is empty)")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API authentication not properly configured",
        )

    # Validate provided key
    if x_api_key not in valid_keys:
        # Log attempt without exposing full key
        logger.warning(
            f"Invalid API key attempt: {x_api_key[:8]}*** (length: {len(x_api_key)})"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Key is valid
    logger.debug(f"API key validated successfully: {x_api_key[:8]}***")
    return x_api_key


def get_bm25_retriever(request: Request) -> BM25Retriever:
    """
    Dependency to access the shared BM25 retriever.

    The retriever is initialized once at application startup and stored
    in app.state for reuse across requests.

    Args:
        request: FastAPI request object

    Returns:
        Initialized BM25Retriever instance

    Raises:
        HTTPException: 500 Internal Server Error if retriever not initialized

    Example:
        >>> @router.post("/search/bm25")
        >>> def search(retriever: Annotated[BM25Retriever, Depends(get_bm25_retriever)]):
        >>>     results = retriever.search("query")
    """
    try:
        return request.app.state.bm25_retriever
    except AttributeError:
        logger.error("BM25 retriever not found in app.state (not initialized)")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BM25 retriever not initialized",
        )


def get_vector_retriever(request: Request) -> VectorRetriever:
    """
    Dependency to access the shared vector retriever.

    The retriever is initialized once at application startup and stored
    in app.state for reuse across requests.

    Args:
        request: FastAPI request object

    Returns:
        Initialized VectorRetriever instance

    Raises:
        HTTPException: 500 Internal Server Error if retriever not initialized

    Example:
        >>> @router.post("/search/vector")
        >>> def search(retriever: Annotated[VectorRetriever, Depends(get_vector_retriever)]):
        >>>     results = retriever.search("query")
    """
    try:
        return request.app.state.vector_retriever
    except AttributeError:
        logger.error("Vector retriever not found in app.state (not initialized)")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector retriever not initialized",
        )


def get_reranker(request: Request) -> Optional[CohereReranker]:
    """
    Dependency to access the shared Cohere reranker.

    The reranker is initialized once at application startup and stored
    in app.state for reuse across requests. Returns None if reranker
    failed to initialize (graceful degradation).

    Args:
        request: FastAPI request object

    Returns:
        Initialized CohereReranker instance, or None if unavailable

    Example:
        >>> @router.post("/search/hybrid")
        >>> def search(reranker: Annotated[Optional[CohereReranker], Depends(get_reranker)]):
        >>>     if reranker:
        >>>         results = reranker.rerank("query", candidates)
    """
    try:
        reranker = request.app.state.reranker

        # Check if reranker is None (initialization failed)
        if reranker is None:
            logger.warning(
                "Cohere reranker is None (initialization failed at startup). "
                "Hybrid search will fall back to RRF."
            )

        return reranker  # May be None, which is OK (graceful degradation)

    except AttributeError:
        logger.warning("Cohere reranker not found in app.state. Falling back to RRF.")
        return None  # Graceful degradation


def get_query_log_client(request: Request) -> QueryLogClient:
    """
    Dependency to create a query log client with shared connection pool.

    Creates a new QueryLogClient per request using the shared connection pool
    from app.state.

    Args:
        request: FastAPI request object

    Returns:
        QueryLogClient instance with connection pool

    Raises:
        HTTPException: 500 Internal Server Error if connection pool not initialized

    Example:
        >>> @router.post("/endpoint")
        >>> def endpoint(log_client: Annotated[QueryLogClient, Depends(get_query_log_client)]):
        >>>     log_client.log_query("query", "miss", 100, "user123")
    """
    try:
        connection_pool = request.app.state.connection_pool
        return QueryLogClient(connection_pool=connection_pool)
    except AttributeError:
        logger.error("Connection pool not found in app.state (not initialized)")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Connection pool not initialized",
        )


def get_reranker_log_client(request: Request) -> RerankerLogClient:
    """
    Dependency for RerankerLogClient with connection pooling.

    Uses the shared connection pool from app.state for efficient
    concurrent database access. Used for logging reranker performance
    (initial RRF rankings vs final Cohere reranked results).

    Args:
        request: FastAPI Request object

    Returns:
        RerankerLogClient: Client for logging reranker operations

    Raises:
        HTTPException: 500 Internal Server Error if connection pool not initialized

    Example:
        >>> @router.post("/endpoint")
        >>> def endpoint(reranker_log_client: Annotated[RerankerLogClient, Depends(get_reranker_log_client)]):
        >>>     reranker_log_client.log_reranking(...)
    """
    try:
        connection_pool = request.app.state.connection_pool
        return RerankerLogClient(connection_pool=connection_pool)
    except AttributeError:
        logger.error("Connection pool not found in app.state (not initialized)")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Connection pool not initialized",
        )
