"""Tests for qa/eval_metrics.py — IR metric computation functions."""

import pytest
from uuid import UUID

from qa.eval_metrics import (
    RetrievalResult,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    compute_all_metrics,
    _find_rank,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

CHUNK_A = UUID("00000000-0000-0000-0000-000000000001")
CHUNK_B = UUID("00000000-0000-0000-0000-000000000002")
CHUNK_C = UUID("00000000-0000-0000-0000-000000000003")
CHUNK_D = UUID("00000000-0000-0000-0000-000000000004")
CHUNK_E = UUID("00000000-0000-0000-0000-000000000005")

ARTICLE_X = UUID("10000000-0000-0000-0000-000000000001")
ARTICLE_Y = UUID("10000000-0000-0000-0000-000000000002")
ARTICLE_Z = UUID("10000000-0000-0000-0000-000000000003")


def _make_result(
    expected_chunk: UUID,
    expected_article: UUID,
    retrieved_chunks: list[UUID],
    retrieved_articles: list[UUID],
    error: str | None = None,
    latency_ms: float | None = 10.0,
) -> RetrievalResult:
    return RetrievalResult(
        query="test query",
        expected_chunk_id=expected_chunk,
        expected_article_id=expected_article,
        retrieved_chunk_ids=retrieved_chunks,
        retrieved_article_ids=retrieved_articles,
        retrieval_method="bm25",
        latency_ms=latency_ms,
        error=error,
    )


# Result where expected chunk is at rank 1
RESULT_HIT_RANK1 = _make_result(
    CHUNK_A,
    ARTICLE_X,
    [CHUNK_A, CHUNK_B, CHUNK_C],
    [ARTICLE_X, ARTICLE_Y, ARTICLE_Z],
)

# Result where expected chunk is at rank 3
RESULT_HIT_RANK3 = _make_result(
    CHUNK_C,
    ARTICLE_Z,
    [CHUNK_A, CHUNK_B, CHUNK_C, CHUNK_D],
    [ARTICLE_X, ARTICLE_Y, ARTICLE_Z, ARTICLE_X],
)

# Result where expected chunk is not found
RESULT_MISS = _make_result(
    CHUNK_E,
    ARTICLE_Z,
    [CHUNK_A, CHUNK_B, CHUNK_C],
    [ARTICLE_X, ARTICLE_Y, ARTICLE_X],
)

# Result where chunk missed but article found at rank 1
RESULT_ARTICLE_HIT = _make_result(
    CHUNK_E,
    ARTICLE_X,
    [CHUNK_A, CHUNK_B, CHUNK_C],
    [ARTICLE_X, ARTICLE_Y, ARTICLE_Z],
)

# Failed result
RESULT_ERROR = _make_result(
    CHUNK_A, ARTICLE_X, [], [], error="API timeout", latency_ms=5000.0
)


# ── _find_rank tests ─────────────────────────────────────────────────────────


class TestFindRank:
    def test_chunk_found_at_rank1(self):
        assert _find_rank(RESULT_HIT_RANK1, k=3, level="chunk") == 1

    def test_chunk_found_at_rank3(self):
        assert _find_rank(RESULT_HIT_RANK3, k=5, level="chunk") == 3

    def test_chunk_not_found_within_k(self):
        assert _find_rank(RESULT_HIT_RANK3, k=2, level="chunk") is None

    def test_chunk_not_in_results(self):
        assert _find_rank(RESULT_MISS, k=3, level="chunk") is None

    def test_article_found(self):
        assert _find_rank(RESULT_ARTICLE_HIT, k=3, level="article") == 1

    def test_article_not_found(self):
        assert _find_rank(RESULT_MISS, k=3, level="article") is None

    def test_invalid_level(self):
        with pytest.raises(ValueError, match="level must be"):
            _find_rank(RESULT_HIT_RANK1, k=3, level="invalid")


# ── hit_rate_at_k tests ──────────────────────────────────────────────────────


class TestHitRateAtK:
    def test_all_hits(self):
        results = [RESULT_HIT_RANK1, RESULT_HIT_RANK3]
        assert hit_rate_at_k(results, k=5, level="chunk") == 1.0

    def test_no_hits(self):
        results = [RESULT_MISS, RESULT_MISS]
        assert hit_rate_at_k(results, k=3, level="chunk") == 0.0

    def test_partial_hits(self):
        results = [RESULT_HIT_RANK1, RESULT_MISS]
        assert hit_rate_at_k(results, k=3, level="chunk") == 0.5

    def test_k_filters_results(self):
        # RESULT_HIT_RANK3 has expected at rank 3, so k=2 should miss
        results = [RESULT_HIT_RANK1, RESULT_HIT_RANK3]
        assert hit_rate_at_k(results, k=2, level="chunk") == 0.5

    def test_article_level(self):
        # RESULT_ARTICLE_HIT has chunk miss but article hit at rank 1
        results = [RESULT_ARTICLE_HIT, RESULT_MISS]
        assert hit_rate_at_k(results, k=3, level="chunk") == 0.0
        assert hit_rate_at_k(results, k=3, level="article") == 0.5

    def test_empty_results(self):
        assert hit_rate_at_k([], k=5, level="chunk") == 0.0

    def test_error_results_excluded(self):
        results = [RESULT_HIT_RANK1, RESULT_ERROR]
        # Only 1 valid result (the hit), error excluded
        assert hit_rate_at_k(results, k=3, level="chunk") == 1.0

    def test_all_errors(self):
        results = [RESULT_ERROR, RESULT_ERROR]
        assert hit_rate_at_k(results, k=3, level="chunk") == 0.0


# ── mrr tests ────────────────────────────────────────────────────────────────


class TestMRR:
    def test_single_hit_rank1(self):
        results = [RESULT_HIT_RANK1]
        assert mrr(results, level="chunk") == pytest.approx(1.0)

    def test_single_hit_rank3(self):
        results = [RESULT_HIT_RANK3]
        assert mrr(results, level="chunk") == pytest.approx(1 / 3)

    def test_mixed_results(self):
        # rank 1 (1/1) + rank 3 (1/3) + miss (0) = 1.333.../3 ≈ 0.4444
        results = [RESULT_HIT_RANK1, RESULT_HIT_RANK3, RESULT_MISS]
        expected = (1.0 + 1 / 3 + 0) / 3
        assert mrr(results, level="chunk") == pytest.approx(expected)

    def test_no_hits(self):
        results = [RESULT_MISS, RESULT_MISS]
        assert mrr(results, level="chunk") == 0.0

    def test_empty(self):
        assert mrr([], level="chunk") == 0.0

    def test_errors_excluded(self):
        results = [RESULT_HIT_RANK1, RESULT_ERROR]
        assert mrr(results, level="chunk") == pytest.approx(1.0)


# ── ndcg_at_k tests ──────────────────────────────────────────────────────────


class TestNDCGAtK:
    def test_perfect_ranking(self):
        # Relevant doc at rank 1: DCG = 1/log2(2) = 1.0, IDCG = 1.0
        results = [RESULT_HIT_RANK1]
        assert ndcg_at_k(results, k=5, level="chunk") == pytest.approx(1.0)

    def test_rank3(self):
        # Relevant doc at rank 3: DCG = 1/log2(4) = 0.5, NDCG = 0.5
        import math

        results = [RESULT_HIT_RANK3]
        expected = (1.0 / math.log2(4)) / 1.0  # 0.5
        assert ndcg_at_k(results, k=5, level="chunk") == pytest.approx(expected)

    def test_miss(self):
        results = [RESULT_MISS]
        assert ndcg_at_k(results, k=5, level="chunk") == pytest.approx(0.0)

    def test_empty(self):
        assert ndcg_at_k([], k=5, level="chunk") == 0.0


# ── precision_at_k tests ─────────────────────────────────────────────────────


class TestPrecisionAtK:
    def test_hit_at_k1(self):
        results = [RESULT_HIT_RANK1]
        assert precision_at_k(results, k=1, level="chunk") == pytest.approx(1.0)

    def test_hit_at_k3(self):
        # 1 relevant doc in top-3, so precision = 1/3
        results = [RESULT_HIT_RANK1]
        assert precision_at_k(results, k=3, level="chunk") == pytest.approx(1 / 3)

    def test_miss(self):
        results = [RESULT_MISS]
        assert precision_at_k(results, k=3, level="chunk") == 0.0

    def test_article_level_multiple_hits(self):
        # Article X appears at ranks 1 and 3 in RESULT_ARTICLE_HIT
        # retrieved articles: [X, Y, Z], expected X -> 1 match in top-3
        results = [RESULT_ARTICLE_HIT]
        assert precision_at_k(results, k=3, level="article") == pytest.approx(1 / 3)

    def test_empty(self):
        assert precision_at_k([], k=5, level="chunk") == 0.0


# ── compute_all_metrics tests ────────────────────────────────────────────────


class TestComputeAllMetrics:
    def test_structure(self):
        results = [RESULT_HIT_RANK1, RESULT_MISS]
        output = compute_all_metrics(results, k_values=[1, 5])

        assert "chunk_level" in output
        assert "article_level" in output
        assert "total_queries" in output
        assert "failed_queries" in output
        assert "avg_latency_ms" in output

        chunk = output["chunk_level"]
        assert "hit_rate" in chunk
        assert "mrr" in chunk
        assert "ndcg" in chunk
        assert "precision" in chunk

        assert "@1" in chunk["hit_rate"]
        assert "@5" in chunk["hit_rate"]

    def test_total_queries(self):
        results = [RESULT_HIT_RANK1, RESULT_MISS, RESULT_ERROR]
        output = compute_all_metrics(results, k_values=[5])
        assert output["total_queries"] == 3
        assert output["failed_queries"] == 1

    def test_latency(self):
        results = [RESULT_HIT_RANK1, RESULT_MISS]  # both have latency 10.0ms
        output = compute_all_metrics(results, k_values=[5])
        assert output["avg_latency_ms"] == 10.0

    def test_values_match_individual_functions(self):
        results = [RESULT_HIT_RANK1, RESULT_HIT_RANK3, RESULT_MISS]
        output = compute_all_metrics(results, k_values=[1, 3, 5])

        assert output["chunk_level"]["hit_rate"]["@5"] == round(
            hit_rate_at_k(results, 5, "chunk"), 4
        )
        assert output["chunk_level"]["mrr"] == round(mrr(results, "chunk"), 4)
        assert output["article_level"]["hit_rate"]["@3"] == round(
            hit_rate_at_k(results, 3, "article"), 4
        )
