"""
Hybrid search implementation combining BM25 and vector retrieval.

Provides two fusion methods:
1. Reciprocal Rank Fusion (RRF) - Rank-based, robust, default
2. Weighted Score Fusion - Score-based with normalization

Reference:
- RRF: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
  "Reciprocal rank fusion outperforms condorcet and individual rank learning methods."
"""

from typing import List, Dict, Any, Literal, Optional
from core.bm25_search import BM25Retriever, BM25SearchResult
from core.vector_search import VectorRetriever, VectorSearchResult
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


def weighted_score_fusion(
    bm25_results: List[BM25SearchResult],
    vector_results: List[VectorSearchResult],
    bm25_weight: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Combine search results using weighted score fusion.

    BM25 scores are normalized to [0, 1] using min-max normalization.
    Vector similarities are already in [0, 1] range.

    Combined score = (bm25_weight * norm_bm25_score) + ((1 - bm25_weight) * vector_similarity)

    Args:
        bm25_results: Results from BM25 search
        vector_results: Results from vector search
        bm25_weight: Weight for BM25 results (0-1). Vector weight = 1 - bm25_weight.

    Returns:
        List of dicts with keys: rank, combined_score, chunk
        Sorted by weighted score descending (highest score first)

    Example:
        >>> weighted_results = weighted_score_fusion(bm25, vector, bm25_weight=0.4)
        >>> # Gives 40% weight to BM25, 60% to vector
    """
    vector_weight = 1.0 - bm25_weight

    # Normalize BM25 scores to [0, 1] using min-max normalization
    bm25_scores_raw = [r.score for r in bm25_results]
    if bm25_scores_raw:
        max_bm25 = max(bm25_scores_raw)
        min_bm25 = min(bm25_scores_raw)
        bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0
    else:
        bm25_range = 1.0
        min_bm25 = 0.0

    logger.debug(
        f"BM25 score normalization: min={min_bm25:.2f}, max={max_bm25 if bm25_scores_raw else 0:.2f}, range={bm25_range:.2f}"
    )

    # Build chunk_id → weighted score mapping
    weighted_scores: Dict[str, float] = {}
    chunk_map: Dict[str, TextChunk] = {}

    # Add BM25 contributions (normalized)
    for result in bm25_results:
        chunk_id = str(result.chunk.chunk_id)
        norm_score = (result.score - min_bm25) / bm25_range
        weighted_contribution = bm25_weight * norm_score
        weighted_scores[chunk_id] = weighted_contribution
        chunk_map[chunk_id] = result.chunk
        logger.debug(
            f"BM25 contribution for chunk {chunk_id[:8]}: "
            f"raw={result.score:.2f}, norm={norm_score:.4f}, weighted={weighted_contribution:.4f}"
        )

    # Add vector contributions (already 0-1)
    for result in vector_results:
        chunk_id = str(result.chunk.chunk_id)
        weighted_contribution = vector_weight * result.similarity
        weighted_scores[chunk_id] = (
            weighted_scores.get(chunk_id, 0.0) + weighted_contribution
        )
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = result.chunk
        logger.debug(
            f"Vector contribution for chunk {chunk_id[:8]}: "
            f"similarity={result.similarity:.4f}, weighted={weighted_contribution:.4f}"
        )

    # Sort by weighted score descending
    sorted_chunks = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)

    # Build result list with re-ranked positions
    results = []
    for rank, (chunk_id, score) in enumerate(sorted_chunks, start=1):
        results.append(
            {"rank": rank, "combined_score": score, "chunk": chunk_map[chunk_id]}
        )

    logger.info(
        f"Weighted fusion combined {len(bm25_results)} BM25 + {len(vector_results)} vector "
        f"results into {len(results)} unique chunks (bm25_weight={bm25_weight:.2f})"
    )

    return results


def hybrid_search(
    query: str,
    bm25_retriever: BM25Retriever,
    vector_retriever: VectorRetriever,
    top_k: int = 10,
    bm25_weight: float = 0.5,
    fusion_method: Literal["rrf", "weighted"] = "rrf",
    rrf_k: int = 60,
    min_bm25_score: Optional[float] = None,
    min_vector_similarity: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining BM25 and vector retrieval.

    Fetches results from both methods and combines them using the specified
    fusion algorithm. Fetches 2× top_k from each method to improve fusion quality.

    Args:
        query: Search query string
        bm25_retriever: Initialized BM25 retriever
        vector_retriever: Initialized vector retriever
        top_k: Number of final results to return
        bm25_weight: Weight for BM25 results (0-1), only used with 'weighted' fusion
        fusion_method: 'rrf' (Reciprocal Rank Fusion) or 'weighted' (score-based)
        rrf_k: RRF constant, only used with 'rrf' fusion
        min_bm25_score: Minimum BM25 score threshold (filters BM25 results)
        min_vector_similarity: Minimum vector similarity threshold (filters vector results)

    Returns:
        List of dicts with keys: rank, combined_score, chunk
        Top top_k results sorted by combined score descending

    Raises:
        ValueError: If fusion_method is invalid
        RuntimeError: If either search method fails

    Example:
        >>> results = hybrid_search(
        ...     query="password reset",
        ...     bm25_retriever=bm25,
        ...     vector_retriever=vector,
        ...     top_k=10,
        ...     fusion_method="rrf"
        ... )
        >>> for result in results:
        ...     print(f"{result['rank']}: {result['chunk'].text_content[:100]}")
    """
    logger.info(
        f"Hybrid search: query='{query}', method={fusion_method}, top_k={top_k}"
    )

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

    # Apply fusion method
    if fusion_method == "rrf":
        logger.debug(f"Applying RRF fusion (k={rrf_k})...")
        combined_results = reciprocal_rank_fusion(
            bm25_results=bm25_results, vector_results=vector_results, k=rrf_k
        )
    elif fusion_method == "weighted":
        logger.debug(f"Applying weighted fusion (bm25_weight={bm25_weight:.2f})...")
        combined_results = weighted_score_fusion(
            bm25_results=bm25_results,
            vector_results=vector_results,
            bm25_weight=bm25_weight,
        )
    else:
        raise ValueError(
            f"Invalid fusion method: {fusion_method}. Must be 'rrf' or 'weighted'."
        )

    # Return top_k final results
    final_results = combined_results[:top_k]

    logger.info(
        f"Hybrid search completed: {len(final_results)} results returned (method={fusion_method})"
    )

    return final_results
