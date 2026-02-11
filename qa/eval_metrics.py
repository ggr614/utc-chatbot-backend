"""
IR Metric Computation for Retrieval Evaluation

Pure stateless functions for computing standard Information Retrieval metrics.
All functions operate on lists of RetrievalResult objects.

Metrics:
- Hit Rate @ K (chunk and article level)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG @ K)
- Precision @ K
"""

import math
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class RetrievalResult(BaseModel):
    """Result of a single retrieval query with ground truth."""

    query: str
    expected_chunk_id: UUID
    expected_article_id: UUID
    retrieved_chunk_ids: list[UUID]  # ordered by rank (index 0 = rank 1)
    retrieved_article_ids: list[UUID]  # parallel to chunk_ids
    retrieval_method: str  # "bm25", "vector", "hybrid"
    latency_ms: Optional[float] = None
    error: Optional[str] = None


def _find_rank(result: RetrievalResult, k: int, level: str = "chunk") -> Optional[int]:
    """
    Find the 1-indexed rank of the first relevant item in top-K results.

    Args:
        result: Single retrieval result with ground truth
        k: Number of top results to consider
        level: "chunk" matches on chunk_id, "article" matches on parent_article_id

    Returns:
        1-indexed rank if found in top-K, None otherwise
    """
    if level == "chunk":
        expected = result.expected_chunk_id
        retrieved = result.retrieved_chunk_ids[:k]
        for i, rid in enumerate(retrieved):
            if rid == expected:
                return i + 1
    elif level == "article":
        expected = result.expected_article_id
        retrieved = result.retrieved_article_ids[:k]
        for i, rid in enumerate(retrieved):
            if rid == expected:
                return i + 1
    else:
        raise ValueError(f"level must be 'chunk' or 'article', got '{level}'")
    return None


def hit_rate_at_k(
    results: list[RetrievalResult], k: int, level: str = "chunk"
) -> float:
    """
    Compute Hit Rate (Recall@K for single-relevant-document) at rank K.

    For each query, checks if the expected ID appears in the top-K results.

    Args:
        results: List of retrieval results
        k: Number of top results to consider
        level: "chunk" or "article"

    Returns:
        Float between 0.0 and 1.0
    """
    if not results:
        return 0.0

    valid = [r for r in results if r.error is None]
    if not valid:
        return 0.0

    hits = sum(1 for r in valid if _find_rank(r, k, level) is not None)
    return hits / len(valid)


def mrr(
    results: list[RetrievalResult],
    level: str = "chunk",
    max_k: Optional[int] = None,
) -> float:
    """
    Compute Mean Reciprocal Rank.

    MRR = (1/|Q|) * sum(1/rank_i) where rank_i is the rank of the first
    relevant result for query i (0 contribution if not found).

    Args:
        results: List of retrieval results
        level: "chunk" or "article"
        max_k: If set, only consider results up to this rank

    Returns:
        Float between 0.0 and 1.0
    """
    if not results:
        return 0.0

    valid = [r for r in results if r.error is None]
    if not valid:
        return 0.0

    search_depth = max_k or max(len(r.retrieved_chunk_ids) for r in valid)
    reciprocal_sum = 0.0
    for r in valid:
        rank = _find_rank(r, search_depth, level)
        if rank is not None:
            reciprocal_sum += 1.0 / rank

    return reciprocal_sum / len(valid)


def ndcg_at_k(results: list[RetrievalResult], k: int, level: str = "chunk") -> float:
    """
    Compute NDCG@K with binary relevance (single relevant document).

    With single-relevant-document ground truth:
    - DCG@K = 1/log2(rank+1) if relevant doc is in top-K, else 0
    - IDCG@K = 1/log2(2) = 1.0 (ideal: relevant doc at rank 1)
    - NDCG@K = DCG@K / IDCG@K = 1/log2(rank+1)

    Args:
        results: List of retrieval results
        k: Number of top results to consider
        level: "chunk" or "article"

    Returns:
        Float between 0.0 and 1.0
    """
    if not results:
        return 0.0

    valid = [r for r in results if r.error is None]
    if not valid:
        return 0.0

    idcg = 1.0 / math.log2(2)  # = 1.0 (relevant doc at rank 1)
    ndcg_sum = 0.0

    for r in valid:
        rank = _find_rank(r, k, level)
        if rank is not None:
            dcg = 1.0 / math.log2(rank + 1)
            ndcg_sum += dcg / idcg

    return ndcg_sum / len(valid)


def precision_at_k(
    results: list[RetrievalResult], k: int, level: str = "chunk"
) -> float:
    """
    Compute Precision@K.

    For single-relevant-document: 1/K if found in top-K, else 0/K per query.
    At article level: count of chunks from expected article in top-K / K.

    Args:
        results: List of retrieval results
        k: Number of top results to consider
        level: "chunk" or "article"

    Returns:
        Float between 0.0 and 1.0
    """
    if not results:
        return 0.0

    valid = [r for r in results if r.error is None]
    if not valid:
        return 0.0

    precision_sum = 0.0
    for r in valid:
        if level == "chunk":
            top_k_ids = r.retrieved_chunk_ids[:k]
            relevant_count = sum(1 for cid in top_k_ids if cid == r.expected_chunk_id)
        elif level == "article":
            top_k_article_ids = r.retrieved_article_ids[:k]
            relevant_count = sum(
                1 for aid in top_k_article_ids if aid == r.expected_article_id
            )
        else:
            raise ValueError(f"level must be 'chunk' or 'article', got '{level}'")
        precision_sum += relevant_count / k

    return precision_sum / len(valid)


def compute_all_metrics(
    results: list[RetrievalResult],
    k_values: list[int],
) -> dict:
    """
    Compute all IR metrics for a set of retrieval results at multiple K values.

    Args:
        results: List of retrieval results
        k_values: List of K values to evaluate at (e.g., [1, 3, 5, 10, 20])

    Returns:
        {
            "chunk_level": {
                "hit_rate": {"@1": 0.45, "@3": 0.67, ...},
                "mrr": 0.52,
                "ndcg": {"@1": 0.45, "@3": 0.58, ...},
                "precision": {"@1": 0.45, "@3": 0.22, ...},
            },
            "article_level": { ... same structure ... },
            "total_queries": 500,
            "failed_queries": 3,
            "avg_latency_ms": 487.3,
        }
    """
    total = len(results)
    failed = sum(1 for r in results if r.error is not None)

    latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    output = {
        "total_queries": total,
        "failed_queries": failed,
        "avg_latency_ms": round(avg_latency, 1) if avg_latency is not None else None,
    }

    for level in ("chunk_level", "article_level"):
        level_key = "chunk" if level == "chunk_level" else "article"
        level_metrics = {
            "hit_rate": {},
            "mrr": round(mrr(results, level=level_key), 4),
            "ndcg": {},
            "precision": {},
        }
        for k in k_values:
            k_label = f"@{k}"
            level_metrics["hit_rate"][k_label] = round(
                hit_rate_at_k(results, k, level=level_key), 4
            )
            level_metrics["ndcg"][k_label] = round(
                ndcg_at_k(results, k, level=level_key), 4
            )
            level_metrics["precision"][k_label] = round(
                precision_at_k(results, k, level=level_key), 4
            )
        output[level] = level_metrics

    return output
