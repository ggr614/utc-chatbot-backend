"""
Chat API routes for RAG retrieval.
"""
import logging
import time
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.middleware.auth import verify_api_key
from core.storage_vector import OpenAIVectorStorage
from core.embedding import GenerateEmbeddingsOpenAI

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class RetrieveRequest(BaseModel):
    """Request model for RAG retrieval."""
    query: str = Field(..., description="User's question")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of results")
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum similarity")
    user_id: Optional[str] = Field(default=None, description="User identifier for logging")


class RetrieveResult(BaseModel):
    """Single retrieval result."""
    chunk_id: str
    text_content: str
    source_url: str
    similarity: float
    token_count: int


class RetrieveResponse(BaseModel):
    """Response model for RAG retrieval."""
    query: str
    results: List[RetrieveResult]
    retrieval_time_ms: int


class LogRequest(BaseModel):
    """Request model for query logging."""
    query: str
    response: Optional[str] = None
    user_id: Optional[str] = None
    cache_result: str = "miss"
    latency_ms: Optional[int] = None


# Initialize clients (lazy loading)
_embed_client = None
_vector_store = None


def get_embed_client():
    global _embed_client
    if _embed_client is None:
        _embed_client = GenerateEmbeddingsOpenAI()
    return _embed_client


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = OpenAIVectorStorage()
    return _vector_store


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_context(
    request: RetrieveRequest,
    _: str = Depends(verify_api_key),
):
    """
    Retrieve relevant KB articles for a user query.

    Uses vector similarity search to find the most relevant chunks.
    """
    start_time = time.time()

    try:
        # Generate embedding for query
        embed_client = get_embed_client()
        query_embedding = embed_client.generate_embedding(request.query)

        # Search for similar vectors
        vector_store = get_vector_store()
        results = vector_store.search_similar_vectors(
            query_vector=query_embedding,
            limit=request.top_k,
            min_similarity=request.min_similarity,
        )

        # Format results
        formatted_results = [
            RetrieveResult(
                chunk_id=str(r["chunk_id"]),
                text_content=r["text_content"],
                source_url=r["source_url"],
                similarity=r["similarity"],
                token_count=r["token_count"],
            )
            for r in results
        ]

        retrieval_time = int((time.time() - start_time) * 1000)

        logger.info(
            f"Retrieved {len(formatted_results)} results for query "
            f"(similarity >= {request.min_similarity}) in {retrieval_time}ms"
        )

        return RetrieveResponse(
            query=request.query,
            results=formatted_results,
            retrieval_time_ms=retrieval_time,
        )

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/log")
async def log_query(
    request: LogRequest,
    _: str = Depends(verify_api_key),
):
    """
    Log a query for analytics.
    """
    try:
        from api.services.analytics import AnalyticsService
        import psycopg

        analytics = AnalyticsService()

        with analytics.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_logs (raw_query, cache_result, latency_ms, user_id, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        request.query,
                        request.cache_result,
                        request.latency_ms,
                        request.user_id,
                        datetime.utcnow(),
                    ),
                )
                conn.commit()

        return {"status": "logged"}

    except Exception as e:
        logger.error(f"Logging error: {e}")
        # Don't fail the request if logging fails
        return {"status": "error", "detail": str(e)}
