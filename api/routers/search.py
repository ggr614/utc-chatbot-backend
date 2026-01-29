"""
Search endpoints for BM25, vector, and hybrid retrieval.

Provides three search methods:
1. BM25 - Fast keyword-based sparse retrieval
2. Vector - Semantic dense retrieval via embeddings
3. Hybrid - Combined BM25 + vector with fusion algorithms

All endpoints include query logging for analytics.
"""

from fastapi import APIRouter, Depends, Request, HTTPException, status
from typing import Annotated
import time

from api.dependencies import (
    verify_api_key,
    get_bm25_retriever,
    get_vector_retriever,
    get_query_log_client,
)
from api.models.requests import (
    BM25SearchRequest,
    VectorSearchRequest,
    HybridSearchRequest,
)
from api.models.responses import SearchResponse, SearchResultChunk
from api.utils.hybrid_search import hybrid_search
from core.bm25_search import BM25Retriever
from core.vector_search import VectorRetriever
from core.storage_query_log import QueryLogClient
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/bm25",
    response_model=SearchResponse,
    summary="BM25 Sparse Keyword Search",
    description="Perform BM25 keyword-based sparse retrieval. Best for exact keyword matching, technical terms, and identifiers. Fast (<100ms after corpus cached).",
    tags=["Search"],
)
def search_bm25(
    request: BM25SearchRequest,
    api_key: Annotated[str, Depends(verify_api_key)],
    retriever: Annotated[BM25Retriever, Depends(get_bm25_retriever)],
    query_log_client: Annotated[QueryLogClient, Depends(get_query_log_client)],
) -> SearchResponse:
    """
    Perform BM25 keyword-based sparse retrieval.

    **Best for:**
    - Exact keyword matching
    - Technical terms and acronyms
    - Identifiers (IDs, codes, specific names)

    **Characteristics:**
    - Fast response (~50-100ms after corpus cached)
    - No API calls required (uses in-memory index)
    - Scores are unbounded (typically 0-20 range)

    **Request:**
    - `query`: Search query text (1-1000 chars)
    - `top_k`: Number of results to return (1-100, default 10)
    - `min_score`: Optional minimum BM25 score threshold
    - `user_id`: Optional user identifier for analytics

    **Response:**
    - Ranked results with BM25 scores
    - Latency in milliseconds
    - Metadata (min_score if provided)
    """
    start_time = time.time()

    logger.info(
        f"BM25 search: query='{request.query[:50]}', top_k={request.top_k}, "
        f"min_score={request.min_score}, user_id={request.user_id}"
    )

    try:
        # Perform BM25 search
        results = retriever.search(
            query=request.query, top_k=request.top_k, min_score=request.min_score
        )

        # Convert BM25SearchResult to SearchResultChunk
        result_chunks = [
            SearchResultChunk(
                rank=r.rank,
                score=r.score,
                chunk_id=r.chunk.chunk_id,
                parent_article_id=r.chunk.parent_article_id,
                chunk_sequence=r.chunk.chunk_sequence,
                text_content=r.chunk.text_content,
                token_count=r.chunk.token_count,
                source_url=r.chunk.source_url,
                last_modified_date=r.chunk.last_modified_date,
            )
            for r in results
        ]

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"BM25 search completed: {len(results)} results, {latency_ms}ms latency"
        )

        # Prepare results for logging (extract minimal data)
        results_for_logging = [
            {
                "rank": r.rank,
                "score": r.score,
                "chunk_id": r.chunk_id,
                "parent_article_id": r.parent_article_id,
            }
            for r in result_chunks
        ]

        # Log query and results to database (best-effort, don't fail request if logging fails)
        try:
            query_log_client.log_query_with_results(
                raw_query=request.query,
                cache_result="miss",  # No cache implemented yet
                search_method="bm25",
                results=results_for_logging,
                latency_ms=latency_ms,
                user_id=request.user_id,
                query_embedding=None,
            )
        except Exception as e:
            logger.error(f"Query and result logging failed: {e}")
            # Don't propagate logging errors to client

        # Build response
        return SearchResponse(
            query=request.query,
            method="bm25",
            results=result_chunks,
            total_results=len(results),
            latency_ms=latency_ms,
            metadata={"min_score": request.min_score}
            if request.min_score is not None
            else {},
        )

    except ValueError as e:
        # Validation errors from retriever
        logger.warning(f"BM25 search validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except Exception as e:
        # Unexpected errors
        logger.error(f"BM25 search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Search failed"
        )


@router.post(
    "/vector",
    response_model=SearchResponse,
    summary="Vector Semantic Search",
    description="Perform vector-based semantic similarity search using embeddings. Best for natural language queries and conceptual similarity. Makes API call (~500ms-1s).",
    tags=["Search"],
)
def search_vector(
    request: VectorSearchRequest,
    api_key: Annotated[str, Depends(verify_api_key)],
    retriever: Annotated[VectorRetriever, Depends(get_vector_retriever)],
    query_log_client: Annotated[QueryLogClient, Depends(get_query_log_client)],
) -> SearchResponse:
    """
    Perform vector-based semantic similarity search.

    **Best for:**
    - Natural language queries
    - Conceptual similarity
    - Synonym and paraphrase matching

    **Characteristics:**
    - Semantic understanding (matches meaning, not just keywords)
    - Slower response (~500ms-1s due to embedding API call)
    - Similarity scores in 0-1 range (cosine similarity)

    **Request:**
    - `query`: Search query text (1-1000 chars)
    - `top_k`: Number of results to return (1-100, default 10)
    - `min_similarity`: Optional minimum similarity threshold (0.0-1.0)
    - `user_id`: Optional user identifier for analytics

    **Response:**
    - Ranked results with similarity scores (0-1)
    - Latency in milliseconds (includes embedding generation)
    - Metadata (min_similarity if provided)

    **Note:** Makes API call to Azure OpenAI for query embedding generation.
    """
    start_time = time.time()

    logger.info(
        f"Vector search: query='{request.query[:50]}', top_k={request.top_k}, "
        f"min_similarity={request.min_similarity}, user_id={request.user_id}"
    )

    try:
        # Perform vector search (includes embedding generation)
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
        )

        # Convert VectorSearchResult to SearchResultChunk
        result_chunks = [
            SearchResultChunk(
                rank=r.rank,
                score=r.similarity,  # Use similarity as score
                chunk_id=r.chunk.chunk_id,
                parent_article_id=r.chunk.parent_article_id,
                chunk_sequence=r.chunk.chunk_sequence,
                text_content=r.chunk.text_content,
                token_count=r.chunk.token_count,
                source_url=r.chunk.source_url,
                last_modified_date=r.chunk.last_modified_date,
            )
            for r in results
        ]

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Vector search completed: {len(results)} results, {latency_ms}ms latency"
        )

        # Prepare results for logging (extract minimal data)
        results_for_logging = [
            {
                "rank": r.rank,
                "score": r.score,
                "chunk_id": r.chunk_id,
                "parent_article_id": r.parent_article_id,
            }
            for r in result_chunks
        ]

        # Log query and results to database (best-effort)
        try:
            query_log_client.log_query_with_results(
                raw_query=request.query,
                cache_result="miss",
                search_method="vector",
                results=results_for_logging,
                latency_ms=latency_ms,
                user_id=request.user_id,
                query_embedding=None,  # Could store embedding here if needed
            )
        except Exception as e:
            logger.error(f"Query and result logging failed: {e}")

        # Build response
        return SearchResponse(
            query=request.query,
            method="vector",
            results=result_chunks,
            total_results=len(results),
            latency_ms=latency_ms,
            metadata={"min_similarity": request.min_similarity}
            if request.min_similarity is not None
            else {},
        )

    except ValueError as e:
        # Validation errors from retriever
        logger.warning(f"Vector search validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except RuntimeError as e:
        # Embedding API failures
        logger.error(f"Vector search runtime error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service unavailable",
        )

    except Exception as e:
        # Unexpected errors
        logger.error(f"Vector search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Search failed"
        )


@router.post(
    "/hybrid",
    response_model=SearchResponse,
    summary="Hybrid Search (BM25 + Vector)",
    description="Perform hybrid search combining BM25 and vector retrieval using Reciprocal Rank Fusion or weighted scoring. Best overall performance.",
    tags=["Search"],
)
def search_hybrid(
    request: HybridSearchRequest,
    api_key: Annotated[str, Depends(verify_api_key)],
    bm25_retriever: Annotated[BM25Retriever, Depends(get_bm25_retriever)],
    vector_retriever: Annotated[VectorRetriever, Depends(get_vector_retriever)],
    query_log_client: Annotated[QueryLogClient, Depends(get_query_log_client)],
) -> SearchResponse:
    """
    Perform hybrid search combining BM25 and vector retrieval.

    **Best for:**
    - General-purpose search (works well for most queries)
    - Combining precision and recall
    - Balancing keyword matching and semantic understanding

    **Fusion Methods:**

    1. **RRF (Reciprocal Rank Fusion)** - Default, recommended
       - Rank-based fusion (robust to score scales)
       - Formula: score = Σ(1 / (k + rank))
       - Parameter: `rrf_k` (default 60)

    2. **Weighted** - Score-based fusion
       - Combines normalized scores with weights
       - Formula: score = (w_bm25 × norm_bm25) + (w_vec × similarity)
       - Parameter: `bm25_weight` (default 0.5)

    **Characteristics:**
    - Combines best of both methods
    - Latency: ~500ms-1s (dominated by vector embedding API)
    - More robust than single method

    **Request:**
    - `query`: Search query text (1-1000 chars)
    - `top_k`: Number of results to return (1-100, default 10)
    - `fusion_method`: "rrf" or "weighted" (default "rrf")
    - `bm25_weight`: Weight for BM25 (0-1, only for weighted fusion)
    - `rrf_k`: RRF constant (default 60, only for RRF fusion)
    - `min_bm25_score`: Optional BM25 score threshold
    - `min_vector_similarity`: Optional vector similarity threshold
    - `user_id`: Optional user identifier for analytics

    **Response:**
    - Ranked results with combined scores
    - Latency in milliseconds
    - Metadata (fusion_method, bm25_weight or rrf_k)
    """
    start_time = time.time()

    logger.info(
        f"Hybrid search: query='{request.query[:50]}', top_k={request.top_k}, "
        f"fusion={request.fusion_method}, bm25_weight={request.bm25_weight}, "
        f"rrf_k={request.rrf_k}, user_id={request.user_id}"
    )

    try:
        # Perform hybrid search
        results = hybrid_search(
            query=request.query,
            bm25_retriever=bm25_retriever,
            vector_retriever=vector_retriever,
            top_k=request.top_k,
            bm25_weight=request.bm25_weight,
            fusion_method=request.fusion_method,
            rrf_k=request.rrf_k,
            min_bm25_score=request.min_bm25_score,
            min_vector_similarity=request.min_vector_similarity,
        )

        # Convert hybrid results to SearchResultChunk
        result_chunks = [
            SearchResultChunk(
                rank=r["rank"],
                score=r["combined_score"],
                chunk_id=r["chunk"].chunk_id,
                parent_article_id=r["chunk"].parent_article_id,
                chunk_sequence=r["chunk"].chunk_sequence,
                text_content=r["chunk"].text_content,
                token_count=r["chunk"].token_count,
                source_url=r["chunk"].source_url,
                last_modified_date=r["chunk"].last_modified_date,
            )
            for r in results
        ]

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Hybrid search completed: {len(results)} results, {latency_ms}ms latency"
        )

        # Prepare results for logging (extract minimal data)
        results_for_logging = [
            {
                "rank": r.rank,
                "score": r.score,
                "chunk_id": r.chunk_id,
                "parent_article_id": r.parent_article_id,
            }
            for r in result_chunks
        ]

        # Log query and results to database (best-effort)
        try:
            query_log_client.log_query_with_results(
                raw_query=request.query,
                cache_result="miss",
                search_method="hybrid",
                results=results_for_logging,
                latency_ms=latency_ms,
                user_id=request.user_id,
                query_embedding=None,
            )
        except Exception as e:
            logger.error(f"Query and result logging failed: {e}")

        # Build metadata with fusion-specific info
        metadata = {"fusion_method": request.fusion_method}
        if request.fusion_method == "rrf":
            metadata["rrf_k"] = request.rrf_k
        else:  # weighted
            metadata["bm25_weight"] = request.bm25_weight
            metadata["vector_weight"] = 1.0 - request.bm25_weight

        # Build response
        return SearchResponse(
            query=request.query,
            method="hybrid",
            results=result_chunks,
            total_results=len(results),
            latency_ms=latency_ms,
            metadata=metadata,
        )

    except ValueError as e:
        # Validation errors from hybrid_search or retrievers
        logger.warning(f"Hybrid search validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except RuntimeError as e:
        # Search method failures (likely embedding API)
        logger.error(f"Hybrid search runtime error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service degraded",
        )

    except Exception as e:
        # Unexpected errors
        logger.error(f"Hybrid search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Search failed"
        )
