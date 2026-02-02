"""
Hybrid search implementation combining BM25 and vector retrieval with reranking.

Workflow:
1. BM25 Search - Fast keyword-based retrieval
2. Vector Search - Semantic similarity retrieval
3. RRF Fusion - Reciprocal Rank Fusion combines and deduplicates
4. Cohere Rerank - Neural reranking refines relevance

Reference:
- RRF: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
  "Reciprocal rank fusion outperforms condorcet and individual rank learning methods."
"""

from typing import List, Dict, Any, Tuple, Optional
from core.bm25_search import BM25Retriever, BM25SearchResult
from core.vector_search import VectorRetriever, VectorSearchResult
from core.reranker import CohereReranker
from core.schemas import TextChunk
from utils.logger import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    bm25_results: List[BM25SearchResult],
    vector_results: List[VectorSearchResult],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Combine search results using Reciprocal Rank Fusion (RRF).

    RRF formula: score(chunk) = Σ(1 / (k + rank_i)) across all methods
    where rank_i is the rank of the chunk in method i (1-indexed).

    This is a rank-based fusion method that doesn't require score normalization,
    making it robust to different score scales between BM25 and vector search.

    Args:
        bm25_results: Results from BM25 search
        vector_results: Results from vector search
        k: RRF constant (typically 60). Higher k gives less weight to ranks.

    Returns:
        List of dicts with keys: rank, combined_score, chunk
        Sorted by RRF score descending (highest score first)

    Example:
        >>> rrf_results = reciprocal_rank_fusion(bm25, vector, k=60)
        >>> top_result = rrf_results[0]
        >>> print(f"Rank: {top_result['rank']}, Score: {top_result['combined_score']}")
    """
    # Build chunk_id → RRF score mapping
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, TextChunk] = {}

    # Add BM25 contributions
    for result in bm25_results:
        chunk_id = str(result.chunk.chunk_id)
        rrf_contribution = 1.0 / (k + result.rank)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_contribution
        chunk_map[chunk_id] = result.chunk
        logger.debug(
            f"BM25 contribution for chunk {chunk_id[:8]}: "
            f"rank={result.rank}, rrf={rrf_contribution:.4f}"
        )

    # Add vector contributions
    for result in vector_results:
        chunk_id = str(result.chunk.chunk_id)
        rrf_contribution = 1.0 / (k + result.rank)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_contribution
        chunk_map[chunk_id] = result.chunk
        logger.debug(
            f"Vector contribution for chunk {chunk_id[:8]}: "
            f"rank={result.rank}, rrf={rrf_contribution:.4f}"
        )

    # Sort by RRF score descending
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Build result list with re-ranked positions
    results = []
    for rank, (chunk_id, score) in enumerate(sorted_chunks, start=1):
        results.append(
            {"rank": rank, "combined_score": score, "chunk": chunk_map[chunk_id]}
        )

    logger.info(
        f"RRF fusion combined {len(bm25_results)} BM25 + {len(vector_results)} vector "
        f"results into {len(results)} unique chunks (k={k})"
    )

    return results


def hybrid_search(
    query: str,
    bm25_retriever: BM25Retriever,
    vector_retriever: VectorRetriever,
    reranker: Optional[CohereReranker],
    top_k: int = 10,
    rrf_k: int = 60,
    min_bm25_score: Optional[float] = None,
    min_vector_similarity: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Perform hybrid search with Cohere reranking.

    Workflow:
    1. Fetch 2×top_k results from BM25 and vector search
    2. Apply Reciprocal Rank Fusion (RRF) to combine and deduplicate
    3. Rerank fused results using Cohere Rerank v3.5 via AWS Bedrock
    4. Return top_k reranked results with metadata

    Args:
        query: Search query string
        bm25_retriever: Initialized BM25 retriever
        vector_retriever: Initialized vector retriever
        reranker: Initialized Cohere reranker
        top_k: Number of final results to return (default: 10)
        rrf_k: RRF constant (default: 60, controls rank-based weighting)
        min_bm25_score: Minimum BM25 score threshold (filters BM25 results)
        min_vector_similarity: Minimum vector similarity threshold (filters vector results)

    Returns:
        Tuple of (results, metadata):
        - results: List of dicts with keys: rank, combined_score (rerank score), chunk
        - metadata: Dict with reranking status and diagnostics

    Raises:
        RuntimeError: If BM25 or vector search fails (reranking failures are caught)

    Example:
        >>> results, metadata = hybrid_search(
        ...     query="password reset",
        ...     bm25_retriever=bm25,
        ...     vector_retriever=vector,
        ...     reranker=reranker,
        ...     top_k=10
        ... )
        >>> print(f"Reranked: {metadata['reranked']}")
        >>> for result in results:
        ...     print(f"{result['rank']}: {result['combined_score']:.3f}")
    """
    logger.info(f"Hybrid search with reranking: query='{query}', top_k={top_k}")

    # Fetch more results from each method to improve fusion quality
    # Request 2× top_k, capped at 100 max
    fetch_k = min(top_k * 2, 100)

    try:
        # Perform BM25 search
        logger.debug(f"Fetching BM25 results (top_k={fetch_k})...")
        bm25_results = bm25_retriever.search(
            query=query, top_k=fetch_k, min_score=min_bm25_score
        )
        logger.info(f"BM25 search returned {len(bm25_results)} results")

    except Exception as e:
        logger.error(f"BM25 search failed: {str(e)}")
        raise RuntimeError(f"BM25 search failed: {e}") from e

    try:
        # Perform vector search
        logger.debug(f"Fetching vector results (top_k={fetch_k})...")
        vector_results = vector_retriever.search(
            query=query, top_k=fetch_k, min_similarity=min_vector_similarity
        )
        logger.info(f"Vector search returned {len(vector_results)} results")

    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        raise RuntimeError(f"Vector search failed: {e}") from e

    # Apply RRF fusion
    logger.debug(f"Applying RRF fusion (k={rrf_k})...")
    combined_results = reciprocal_rank_fusion(
        bm25_results=bm25_results, vector_results=vector_results, k=rrf_k
    )

    # Apply Cohere reranking with fallback
    if reranker is None:
        # Reranker not available (initialization failed)
        logger.warning("Reranker not available, using RRF results")
        reranked_results = combined_results
        reranking_metadata = {
            "reranked": False,
            "reranking_failed": True,
            "fallback_method": "rrf",
            "error": "Reranker not initialized",
        }
    else:
        try:
            logger.debug(f"Reranking {len(combined_results)} fused results...")
            reranked_results = reranker.rerank(query=query, results=combined_results)
            logger.info(
                f"Reranking completed: {len(reranked_results)} results, "
                f"top score: {reranked_results[0]['combined_score']:.3f}"
            )
            reranking_metadata = {
                "reranked": True,
                "num_candidates_reranked": len(combined_results),
            }
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}", exc_info=True)
            # Fallback to RRF results
            reranked_results = combined_results
            reranking_metadata = {
                "reranked": False,
                "reranking_failed": True,
                "fallback_method": "rrf",
                "error": str(e),
            }

    # Return top_k final results
    final_results = reranked_results[:top_k]

    logger.info(
        f"Hybrid search completed: {len(final_results)} results returned "
        f"(reranked={reranking_metadata.get('reranked', False)})"
    )

    return final_results, reranking_metadata
