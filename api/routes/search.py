"""
Search API routes - Compatible with David's RAG Helpdesk Filter.

Provides the /api/v1/search/hybrid endpoint that the Open-webui filter calls.
"""
import logging
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.middleware.auth import verify_api_key
from core.storage_vector import OpenAIVectorStorage
from core.embedding import GenerateEmbeddingsOpenAI

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["search"])


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search - matches David's filter expectations."""
    query: str = Field(..., description="User's search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return after reranking")
    fetch_top_k: int = Field(default=20, ge=1, le=100, description="Candidates to fetch before fusion")
    rrf_k: int = Field(default=1, ge=1, description="RRF constant for score fusion")
    min_bm25_score: Optional[float] = Field(default=None, description="Minimum BM25 score threshold")
    min_vector_similarity: Optional[float] = Field(default=0.0, ge=0.0, le=1.0, description="Minimum vector similarity")
    email: Optional[str] = Field(default=None, description="User email for logging")
    command: Optional[str] = Field(default=None, description="Command mode (search, follow_up)")


class SearchResult(BaseModel):
    """Single search result."""
    chunk_id: str
    parent_article_id: str
    text_content: str
    source_url: str
    similarity: float
    token_count: int
    # For RRF scoring
    rrf_score: Optional[float] = None
    bm25_score: Optional[float] = None
    vector_similarity: Optional[float] = None


class WarmCacheHit(BaseModel):
    """Warm cache hit result."""
    cache_entry_id: str
    canonical_question: str
    verified_answer: str
    similarity: float
    source_urls: List[str] = []


class HybridSearchResponse(BaseModel):
    """Response model for hybrid search."""
    query: str
    results: List[SearchResult]
    query_log_id: Optional[int] = None
    latency_ms: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Warm cache fields
    cache_hit: bool = False
    warm_cache: Optional[WarmCacheHit] = None


class LogResponseRequest(BaseModel):
    """Request model for logging LLM response."""
    response_text: str
    model_name: Optional[str] = None
    llm_latency_ms: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    citations: Optional[Dict[str, Any]] = None


# Lazy-loaded clients
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


def log_query_to_db(
    query: str,
    email: Optional[str] = None,
    cache_result: str = "miss",
    latency_ms: int = 0,
    num_results: int = 0,
) -> Optional[int]:
    """Log query to database and return the query_log_id."""
    try:
        from api.services.analytics import AnalyticsService

        analytics = AnalyticsService()

        with analytics.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_logs (raw_query, user_id, cache_result, latency_ms, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    RETURNING id
                    """,
                    (query, email, cache_result, latency_ms),
                )
                result = cur.fetchone()
                conn.commit()
                return result[0] if result else None
    except Exception as e:
        logger.error(f"Failed to log query: {e}")
        return None


def check_warm_cache(query_embedding: List[float], threshold: float = 0.80) -> Optional[Dict[str, Any]]:
    """
    Check warm cache for a matching canonical question.

    Returns cached entry if similarity > threshold, else None.
    """
    try:
        from api.services.analytics import AnalyticsService

        analytics = AnalyticsService()

        with analytics.get_connection() as conn:
            with conn.cursor() as cur:
                # Search for similar cached questions
                cur.execute(
                    """
                    SELECT
                        id,
                        canonical_question,
                        verified_answer,
                        article_id,
                        source_urls,
                        1 - (query_embedding <=> %s::vector) as similarity
                    FROM warm_cache_entries
                    WHERE is_active = true
                    AND query_embedding IS NOT NULL
                    ORDER BY query_embedding <=> %s::vector
                    LIMIT 1
                    """,
                    (query_embedding, query_embedding),
                )
                result = cur.fetchone()

                if result and result[5] >= threshold:
                    logger.info(f"Warm cache HIT: similarity={result[5]:.3f}")
                    return {
                        "id": result[0],
                        "canonical_question": result[1],
                        "verified_answer": result[2],
                        "article_id": result[3],
                        "source_urls": result[4] or [],
                        "similarity": result[5],
                    }
                elif result:
                    logger.info(f"Warm cache miss: best similarity={result[5]:.3f} < threshold={threshold}")

                return None
    except Exception as e:
        logger.error(f"Warm cache check error: {e}")
        return None


@router.post("/search/hybrid", response_model=HybridSearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    _: str = Depends(verify_api_key),
):
    """
    Hybrid search endpoint - compatible with David's RAG Helpdesk Filter.

    First checks warm cache for pre-verified answers, then falls back to
    vector similarity search.
    """
    start_time = time.time()

    logger.info(f"Hybrid search request: query='{request.query[:50]}...', email={request.email}")

    try:
        # Generate embedding for query
        embed_client = get_embed_client()
        query_embedding = embed_client.generate_embedding(request.query)

        # Check warm cache first
        cache_entry = check_warm_cache(query_embedding)

        if cache_entry:
            # Cache HIT - return verified answer
            latency_ms = int((time.time() - start_time) * 1000)

            query_log_id = log_query_to_db(
                query=request.query,
                email=request.email,
                cache_result="hit",
                latency_ms=latency_ms,
                num_results=1,
            )

            logger.info(f"Warm cache HIT in {latency_ms}ms (query_log_id={query_log_id})")

            return HybridSearchResponse(
                query=request.query,
                results=[],  # No search results needed - using cached answer
                query_log_id=query_log_id,
                latency_ms=latency_ms,
                cache_hit=True,
                warm_cache=WarmCacheHit(
                    cache_entry_id=str(cache_entry["id"]),
                    canonical_question=cache_entry["canonical_question"],
                    verified_answer=cache_entry["verified_answer"],
                    similarity=cache_entry["similarity"],
                    source_urls=cache_entry.get("source_urls", []),
                ),
                metadata={
                    "search_type": "warm_cache",
                    "system_prompts": {},
                    "source_urls": cache_entry.get("source_urls", []),
                },
            )

        # Cache MISS - do normal vector search
        vector_store = get_vector_store()
        results = vector_store.search_similar_vectors(
            query_vector=query_embedding,
            limit=request.top_k,
            min_similarity=request.min_vector_similarity or 0.0,
        )

        # Format results
        formatted_results = []
        for r in results:
            formatted_results.append(
                SearchResult(
                    chunk_id=str(r["chunk_id"]),
                    parent_article_id=str(r.get("parent_article_id", "")),
                    text_content=r["text_content"],
                    source_url=r["source_url"],
                    similarity=r["similarity"],
                    token_count=r["token_count"],
                    vector_similarity=r["similarity"],
                    rrf_score=r["similarity"],
                )
            )

        latency_ms = int((time.time() - start_time) * 1000)

        # Log the query as cache miss
        query_log_id = log_query_to_db(
            query=request.query,
            email=request.email,
            cache_result="miss",
            latency_ms=latency_ms,
            num_results=len(formatted_results),
        )

        logger.info(
            f"Hybrid search (cache miss): {len(formatted_results)} results in {latency_ms}ms "
            f"(query_log_id={query_log_id})"
        )

        return HybridSearchResponse(
            query=request.query,
            results=formatted_results,
            query_log_id=query_log_id,
            latency_ms=latency_ms,
            cache_hit=False,
            metadata={
                "search_type": "vector",
                "system_prompts": {},
            },
        )

    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query-logs/{query_log_id}/response")
async def log_llm_response(
    query_log_id: int,
    request: LogResponseRequest,
    _: str = Depends(verify_api_key),
):
    """
    Log LLM response for a query - called by David's filter outlet.

    Associates the LLM's response with the original query for analytics.
    """
    logger.info(f"Logging LLM response for query_log_id={query_log_id}")

    try:
        from api.services.analytics import AnalyticsService

        analytics = AnalyticsService()

        with analytics.get_connection() as conn:
            with conn.cursor() as cur:
                # Check if response already logged (idempotency)
                cur.execute(
                    "SELECT id FROM llm_responses WHERE query_log_id = %s",
                    (query_log_id,)
                )
                existing = cur.fetchone()
                if existing:
                    return {"id": existing[0], "status": "already_logged"}

                # Insert the LLM response
                cur.execute(
                    """
                    INSERT INTO llm_responses (
                        query_log_id, response_text, model_name,
                        llm_latency_ms, prompt_tokens, completion_tokens,
                        citations, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    RETURNING id
                    """,
                    (
                        query_log_id,
                        request.response_text,
                        request.model_name,
                        request.llm_latency_ms,
                        request.prompt_tokens,
                        request.completion_tokens,
                        request.citations,
                    ),
                )
                result = cur.fetchone()
                conn.commit()

                response_id = result[0] if result else None
                logger.info(f"LLM response logged: id={response_id}")

                return {"id": response_id, "status": "logged"}

    except Exception as e:
        logger.error(f"Failed to log LLM response: {e}")
        raise HTTPException(status_code=500, detail=str(e))
