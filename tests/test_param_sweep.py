"""Tests for qa/param_sweep.py — vector search parameter sweep."""

import json
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID
from datetime import datetime, timezone

import pytest

from core.schemas import TextChunk
from core.vector_search import VectorSearchResult
from qa.eval_dataset import EvalQuestion, EvalDataset
from qa.eval_metrics import RetrievalResult
from qa.param_sweep import (
    CachedVectorResult,
    RankedItem,
    SweepConfig,
    SweepResult,
    VectorParamSweep,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

CHUNK_A_ID = UUID("00000000-0000-0000-0000-000000000001")
CHUNK_B_ID = UUID("00000000-0000-0000-0000-000000000002")
CHUNK_C_ID = UUID("00000000-0000-0000-0000-000000000003")
CHUNK_D_ID = UUID("00000000-0000-0000-0000-000000000004")
CHUNK_E_ID = UUID("00000000-0000-0000-0000-000000000005")

ARTICLE_X_ID = UUID("10000000-0000-0000-0000-000000000001")
ARTICLE_Y_ID = UUID("10000000-0000-0000-0000-000000000002")


def _make_text_chunk(chunk_id: UUID, article_id: UUID) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        parent_article_id=article_id,
        chunk_sequence=0,
        text_content="test content",
        token_count=50,
        source_url="https://example.com/article/1",
        last_modified_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def _make_eval_question(
    question: str = "How do I reset my password?",
    chunk_id: UUID = CHUNK_A_ID,
    article_id: UUID = ARTICLE_X_ID,
    quality: str = "high",
) -> EvalQuestion:
    return EvalQuestion(
        question=question,
        answer="Go to the password reset page.",
        sufficient_context=True,
        chunk_id=chunk_id,
        parent_article_id=article_id,
        chunk_sequence=0,
        source_url="https://example.com",
        token_count=200,
        overall_quality=quality,
        chunk_summary="Password reset instructions.",
        qa_pair_index=0,
    )


def _make_cached_result(
    query: str = "test query",
    expected_chunk: UUID = CHUNK_A_ID,
    expected_article: UUID = ARTICLE_X_ID,
    items: list[tuple[UUID, UUID, float]] | None = None,
    latency_ms: float = 500.0,
    error: str | None = None,
) -> CachedVectorResult:
    """Build a CachedVectorResult for testing.

    items: list of (chunk_id, article_id, similarity) tuples, sorted desc.
    """
    if items is None:
        items = [
            (CHUNK_A_ID, ARTICLE_X_ID, 0.9),
            (CHUNK_B_ID, ARTICLE_Y_ID, 0.8),
            (CHUNK_C_ID, ARTICLE_X_ID, 0.6),
            (CHUNK_D_ID, ARTICLE_Y_ID, 0.4),
            (CHUNK_E_ID, ARTICLE_X_ID, 0.2),
        ]
    return CachedVectorResult(
        query=query,
        expected_chunk_id=expected_chunk,
        expected_article_id=expected_article,
        results=[
            RankedItem(chunk_id=cid, article_id=aid, similarity=sim)
            for cid, aid, sim in items
        ],
        latency_ms=latency_ms,
        error=error,
    )


def _make_sweep(
    config: SweepConfig | None = None,
) -> VectorParamSweep:
    """Create a VectorParamSweep with a mocked retriever."""
    mock_retriever = MagicMock()
    cfg = config or SweepConfig(
        top_k_values=[1, 3, 5],
        min_similarity_values=[0.0, 0.5, 0.8],
        fetch_top_k=10,
    )
    return VectorParamSweep(vector_retriever=mock_retriever, config=cfg)


# ── TestSweepConfig ──────────────────────────────────────────────────────────


class TestSweepConfig:
    def test_default_values(self):
        cfg = SweepConfig()
        assert cfg.top_k_values == [1, 3, 5, 7, 10, 15, 20]
        assert cfg.min_similarity_values == [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        assert cfg.fetch_top_k == 50
        assert cfg.sample_size == 200
        assert cfg.random_seed == 42
        assert cfg.primary_metric == "mrr"
        assert cfg.primary_level == "chunk"

    def test_fetch_top_k_must_cover_top_k_values(self):
        with pytest.raises(ValueError, match="fetch_top_k"):
            VectorParamSweep(
                vector_retriever=MagicMock(),
                config=SweepConfig(top_k_values=[10, 20], fetch_top_k=15),
            )

    def test_fetch_top_k_equal_to_max_is_valid(self):
        sweep = VectorParamSweep(
            vector_retriever=MagicMock(),
            config=SweepConfig(top_k_values=[5, 10], fetch_top_k=10),
        )
        assert sweep.config.fetch_top_k == 10


# ── TestApplyConfig ──────────────────────────────────────────────────────────


class TestApplyConfig:
    def test_no_filtering_returns_all(self):
        """min_similarity=0.0 and large top_k returns all cached results."""
        sweep = _make_sweep()
        cached = [_make_cached_result()]

        results, zero_count, avg_results = sweep._apply_config(cached, top_k=10, min_similarity=0.0)

        assert len(results) == 1
        assert len(results[0].retrieved_chunk_ids) == 5
        assert zero_count == 0
        assert avg_results == 5.0

    def test_top_k_truncates(self):
        """top_k=3 on 5 cached results returns first 3."""
        sweep = _make_sweep()
        cached = [_make_cached_result()]

        results, zero_count, avg_results = sweep._apply_config(cached, top_k=3, min_similarity=0.0)

        assert len(results[0].retrieved_chunk_ids) == 3
        assert results[0].retrieved_chunk_ids[0] == CHUNK_A_ID
        assert results[0].retrieved_chunk_ids[1] == CHUNK_B_ID
        assert results[0].retrieved_chunk_ids[2] == CHUNK_C_ID
        assert avg_results == 3.0

    def test_min_similarity_filters(self):
        """min_similarity=0.5 removes items below threshold."""
        sweep = _make_sweep()
        cached = [_make_cached_result()]
        # Items: 0.9, 0.8, 0.6, 0.4, 0.2
        # After filter >= 0.5: 0.9, 0.8, 0.6

        results, zero_count, avg_results = sweep._apply_config(cached, top_k=10, min_similarity=0.5)

        assert len(results[0].retrieved_chunk_ids) == 3
        assert results[0].retrieved_chunk_ids == [CHUNK_A_ID, CHUNK_B_ID, CHUNK_C_ID]

    def test_filter_and_truncate_combined(self):
        """Filtering happens before truncation."""
        sweep = _make_sweep()
        cached = [_make_cached_result()]
        # Items: 0.9, 0.8, 0.6, 0.4, 0.2
        # Filter >= 0.5: [0.9, 0.8, 0.6], then truncate to top_k=2: [0.9, 0.8]

        results, _, avg_results = sweep._apply_config(cached, top_k=2, min_similarity=0.5)

        assert len(results[0].retrieved_chunk_ids) == 2
        assert results[0].retrieved_chunk_ids == [CHUNK_A_ID, CHUNK_B_ID]
        assert avg_results == 2.0

    def test_all_filtered_returns_empty(self):
        """min_similarity=0.99 when all results < 0.99 returns empty list."""
        sweep = _make_sweep()
        cached = [_make_cached_result()]

        results, zero_count, avg_results = sweep._apply_config(cached, top_k=10, min_similarity=0.99)

        assert len(results[0].retrieved_chunk_ids) == 0
        assert zero_count == 1
        assert avg_results == 0.0

    def test_error_propagated(self):
        """Error results are propagated as errors with empty retrieved lists."""
        sweep = _make_sweep()
        cached = [_make_cached_result(error="API failed")]

        results, zero_count, _ = sweep._apply_config(cached, top_k=10, min_similarity=0.0)

        assert results[0].error == "API failed"
        assert results[0].retrieved_chunk_ids == []
        assert zero_count == 1

    def test_retrieval_method_encodes_config(self):
        """The retrieval_method string includes top_k and min_similarity."""
        sweep = _make_sweep()
        cached = [_make_cached_result()]

        results, _, _ = sweep._apply_config(cached, top_k=5, min_similarity=0.3)

        assert "k5" in results[0].retrieval_method
        assert "0.3" in results[0].retrieval_method

    def test_multiple_queries(self):
        """Handles multiple queries correctly."""
        sweep = _make_sweep()
        cached = [
            _make_cached_result(
                query="Q1",
                expected_chunk=CHUNK_A_ID,
                items=[(CHUNK_A_ID, ARTICLE_X_ID, 0.95)],
            ),
            _make_cached_result(
                query="Q2",
                expected_chunk=CHUNK_B_ID,
                items=[(CHUNK_C_ID, ARTICLE_X_ID, 0.3)],
            ),
        ]

        results, zero_count, avg_results = sweep._apply_config(cached, top_k=5, min_similarity=0.5)

        assert len(results) == 2
        assert len(results[0].retrieved_chunk_ids) == 1  # 0.95 passes
        assert len(results[1].retrieved_chunk_ids) == 0  # 0.3 filtered out
        assert zero_count == 1
        assert avg_results == 0.5

    def test_exact_threshold_included(self):
        """Items with similarity exactly equal to min_similarity are included."""
        sweep = _make_sweep()
        cached = [
            _make_cached_result(
                items=[(CHUNK_A_ID, ARTICLE_X_ID, 0.5)],
            )
        ]

        results, zero_count, _ = sweep._apply_config(cached, top_k=10, min_similarity=0.5)

        assert len(results[0].retrieved_chunk_ids) == 1
        assert zero_count == 0


# ── TestSweepAllConfigs ──────────────────────────────────────────────────────


class TestSweepAllConfigs:
    def test_correct_number_of_results(self):
        """3 top_k x 3 min_sim = 9 SweepResult objects."""
        sweep = _make_sweep()
        cached = [_make_cached_result()]

        results = sweep._sweep_all_configs(cached)

        assert len(results) == 9  # 3 x 3

    def test_each_config_has_metrics(self):
        """Every SweepResult has a metrics dict with expected structure."""
        sweep = _make_sweep()
        cached = [_make_cached_result()]

        results = sweep._sweep_all_configs(cached)

        for sr in results:
            assert "chunk_level" in sr.metrics
            assert "article_level" in sr.metrics
            assert "total_queries" in sr.metrics
            assert sr.metrics["total_queries"] == 1

    def test_zero_result_queries_counted(self):
        """Queries with 0 results after filtering are counted."""
        sweep = _make_sweep()
        # All items have similarity 0.1 — min_sim=0.8 should filter everything
        cached = [
            _make_cached_result(
                items=[(CHUNK_A_ID, ARTICLE_X_ID, 0.1)],
            )
        ]

        results = sweep._sweep_all_configs(cached)

        # Find result for min_sim=0.8
        high_sim_results = [sr for sr in results if sr.min_similarity == 0.8]
        assert all(sr.zero_result_queries == 1 for sr in high_sim_results)

    def test_avg_results_per_query(self):
        """Average number of results per query is correctly computed."""
        sweep = _make_sweep()
        cached = [
            _make_cached_result(
                items=[
                    (CHUNK_A_ID, ARTICLE_X_ID, 0.9),
                    (CHUNK_B_ID, ARTICLE_Y_ID, 0.7),
                ],
            )
        ]

        results = sweep._sweep_all_configs(cached)

        # For top_k=1, min_sim=0.0: should have 1 result
        r = next(sr for sr in results if sr.top_k == 1 and sr.min_similarity == 0.0)
        assert r.avg_results_per_query == 1.0

        # For top_k=5, min_sim=0.0: should have 2 results (only 2 available)
        r = next(sr for sr in results if sr.top_k == 5 and sr.min_similarity == 0.0)
        assert r.avg_results_per_query == 2.0


# ── TestFindBestConfigs ──────────────────────────────────────────────────────


class TestFindBestConfigs:
    def test_finds_best_mrr(self):
        """Correctly identifies the config with highest MRR."""
        sweep = _make_sweep()
        cached = [
            _make_cached_result(
                expected_chunk=CHUNK_A_ID,
                items=[
                    (CHUNK_A_ID, ARTICLE_X_ID, 0.9),
                    (CHUNK_B_ID, ARTICLE_Y_ID, 0.7),
                    (CHUNK_C_ID, ARTICLE_X_ID, 0.5),
                ],
            )
        ]

        sweep_results = sweep._sweep_all_configs(cached)
        best = sweep._find_best_configs(sweep_results)

        assert "chunk_mrr" in best
        # With expected=CHUNK_A at rank 1, MRR=1.0 for configs that include it
        assert best["chunk_mrr"]["value"] == 1.0

    def test_finds_best_for_each_metric(self):
        """Returns best config for each tracked metric."""
        sweep = _make_sweep()
        cached = [_make_cached_result()]

        sweep_results = sweep._sweep_all_configs(cached)
        best = sweep._find_best_configs(sweep_results)

        expected_keys = [
            "chunk_mrr",
            "chunk_hit_rate_at_5",
            "chunk_ndcg_at_5",
            "article_mrr",
            "article_hit_rate_at_5",
            "article_ndcg_at_5",
        ]
        for key in expected_keys:
            assert key in best
            assert "top_k" in best[key]
            assert "min_similarity" in best[key]
            assert "value" in best[key]

    def test_high_threshold_degrades_metrics(self):
        """Very high min_similarity produces worse metrics than no threshold."""
        sweep = _make_sweep()
        # Expected chunk is at similarity 0.6
        cached = [
            _make_cached_result(
                expected_chunk=CHUNK_C_ID,
                items=[
                    (CHUNK_A_ID, ARTICLE_X_ID, 0.9),
                    (CHUNK_B_ID, ARTICLE_Y_ID, 0.8),
                    (CHUNK_C_ID, ARTICLE_X_ID, 0.6),
                ],
            )
        ]

        sweep_results = sweep._sweep_all_configs(cached)

        # min_sim=0.8 should filter out CHUNK_C (0.6), causing a miss
        high_sim = next(
            sr for sr in sweep_results if sr.min_similarity == 0.8 and sr.top_k == 5
        )
        no_filter = next(
            sr for sr in sweep_results if sr.min_similarity == 0.0 and sr.top_k == 5
        )

        high_mrr = high_sim.metrics["chunk_level"]["mrr"]
        no_filter_mrr = no_filter.metrics["chunk_level"]["mrr"]
        assert high_mrr < no_filter_mrr


# ── TestRunVectorSearch ──────────────────────────────────────────────────────


class TestRunVectorSearch:
    def test_calls_search_per_query(self):
        """Calls vector_retriever.search() once per question."""
        mock_retriever = MagicMock()
        chunk = _make_text_chunk(CHUNK_A_ID, ARTICLE_X_ID)
        mock_retriever.search.return_value = [
            VectorSearchResult(chunk=chunk, similarity=0.85, rank=1)
        ]

        sweep = VectorParamSweep(
            vector_retriever=mock_retriever,
            config=SweepConfig(top_k_values=[5], min_similarity_values=[0.0], fetch_top_k=10),
        )

        questions = [
            _make_eval_question("Q1?"),
            _make_eval_question("Q2?"),
        ]
        cached = sweep._run_vector_search(questions)

        assert mock_retriever.search.call_count == 2
        assert len(cached) == 2

    def test_captures_similarity_scores(self):
        """Similarity scores from VectorSearchResult are captured."""
        mock_retriever = MagicMock()
        chunk_a = _make_text_chunk(CHUNK_A_ID, ARTICLE_X_ID)
        chunk_b = _make_text_chunk(CHUNK_B_ID, ARTICLE_Y_ID)
        mock_retriever.search.return_value = [
            VectorSearchResult(chunk=chunk_a, similarity=0.92, rank=1),
            VectorSearchResult(chunk=chunk_b, similarity=0.75, rank=2),
        ]

        sweep = VectorParamSweep(
            vector_retriever=mock_retriever,
            config=SweepConfig(top_k_values=[5], min_similarity_values=[0.0], fetch_top_k=10),
        )

        cached = sweep._run_vector_search([_make_eval_question()])

        assert len(cached) == 1
        assert len(cached[0].results) == 2
        assert cached[0].results[0].similarity == 0.92
        assert cached[0].results[0].chunk_id == CHUNK_A_ID
        assert cached[0].results[1].similarity == 0.75

    def test_uses_fetch_top_k(self):
        """Uses config.fetch_top_k as the top_k parameter."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        sweep = VectorParamSweep(
            vector_retriever=mock_retriever,
            config=SweepConfig(
                top_k_values=[5],
                min_similarity_values=[0.0],
                fetch_top_k=42,
            ),
        )

        sweep._run_vector_search([_make_eval_question()])

        call_kwargs = mock_retriever.search.call_args[1]
        assert call_kwargs["top_k"] == 42

    def test_handles_search_failure(self):
        """Failed searches produce CachedVectorResult with error."""
        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = RuntimeError("API timeout")

        sweep = VectorParamSweep(
            vector_retriever=mock_retriever,
            config=SweepConfig(top_k_values=[5], min_similarity_values=[0.0], fetch_top_k=10),
        )

        cached = sweep._run_vector_search([_make_eval_question()])

        assert len(cached) == 1
        assert cached[0].error == "API timeout"
        assert cached[0].results == []
        assert cached[0].latency_ms >= 0


# ── TestGenerateReport ───────────────────────────────────────────────────────


class TestGenerateReport:
    def _make_dataset(self) -> EvalDataset:
        return EvalDataset(
            questions=[_make_eval_question()],
            total_chunks_in_source=10,
            total_qa_pairs_in_source=30,
            filtered_count=1,
            quality_distribution={"high": 1},
            source_file="test.jsonl",
        )

    def test_report_structure(self):
        """Report has required top-level keys."""
        sweep = _make_sweep()
        cached = [_make_cached_result()]
        sweep_results = sweep._sweep_all_configs(cached)
        best = sweep._find_best_configs(sweep_results)

        report = sweep._generate_report(
            self._make_dataset(), cached, sweep_results, best
        )

        assert "metadata" in report
        assert "best_configs" in report
        assert "recommendation" in report
        assert "similarity_distribution" in report
        assert "all_configurations" in report
        assert "heatmaps" in report

    def test_metadata_fields(self):
        """Metadata contains expected fields."""
        sweep = _make_sweep()
        cached = [_make_cached_result()]
        sweep_results = sweep._sweep_all_configs(cached)
        best = sweep._find_best_configs(sweep_results)

        report = sweep._generate_report(
            self._make_dataset(), cached, sweep_results, best
        )

        meta = report["metadata"]
        assert meta["tool"] == "vector_param_sweep"
        assert meta["total_configurations"] == 9
        assert "search_latency" in meta

    def test_all_configurations_included(self):
        """All 9 configs (3x3) appear in all_configurations list."""
        sweep = _make_sweep()
        cached = [_make_cached_result()]
        sweep_results = sweep._sweep_all_configs(cached)
        best = sweep._find_best_configs(sweep_results)

        report = sweep._generate_report(
            self._make_dataset(), cached, sweep_results, best
        )

        assert len(report["all_configurations"]) == 9

    def test_recommendation_present(self):
        """Report includes a recommendation with required fields."""
        sweep = _make_sweep()
        cached = [_make_cached_result()]
        sweep_results = sweep._sweep_all_configs(cached)
        best = sweep._find_best_configs(sweep_results)

        report = sweep._generate_report(
            self._make_dataset(), cached, sweep_results, best
        )

        rec = report["recommendation"]
        assert "top_k" in rec
        assert "min_similarity" in rec
        assert "primary_metric" in rec
        assert "metrics" in rec


# ── TestSaveReport ───────────────────────────────────────────────────────────


class TestSaveReport:
    def test_saves_json(self, tmp_path):
        """Report is written as valid JSON."""
        sweep = _make_sweep()
        report = {"metadata": {"test": True}, "results": []}

        json_path, latest_path = sweep.save_report(report, str(tmp_path))

        with open(json_path, "r") as f:
            loaded = json.load(f)
        assert loaded["metadata"]["test"] is True

    def test_saves_latest_copy(self, tmp_path):
        """A 'latest' copy is also written."""
        sweep = _make_sweep()
        report = {"metadata": {"test": True}, "results": []}

        _, latest_path = sweep.save_report(report, str(tmp_path))

        latest_file = tmp_path / "sweep_report_latest.json"
        assert latest_file.exists()
        with open(latest_file, "r") as f:
            loaded = json.load(f)
        assert loaded["metadata"]["test"] is True

    def test_filename_contains_timestamp(self, tmp_path):
        """Timestamped filename is used."""
        sweep = _make_sweep()
        report = {"metadata": {}}

        json_path, _ = sweep.save_report(report, str(tmp_path))

        assert "sweep_report_" in json_path
        assert json_path.endswith(".json")


# ── TestSimilarityDistribution ───────────────────────────────────────────────


class TestSimilarityDistribution:
    def test_computes_percentiles(self):
        """Percentile computation works with valid data."""
        sweep = _make_sweep()
        cached = [
            _make_cached_result(
                items=[
                    (CHUNK_A_ID, ARTICLE_X_ID, 0.95),
                    (CHUNK_B_ID, ARTICLE_Y_ID, 0.85),
                    (CHUNK_C_ID, ARTICLE_X_ID, 0.75),
                    (CHUNK_D_ID, ARTICLE_Y_ID, 0.65),
                    (CHUNK_E_ID, ARTICLE_X_ID, 0.55),
                ],
            )
        ]

        dist = sweep._compute_similarity_distribution(cached)

        assert "top_1_result" in dist
        assert "top_5_result" in dist
        assert dist["top_1_result"]["max"] == 0.95
        assert dist["top_5_result"]["min"] == 0.55

    def test_handles_errors(self):
        """Skips errored results gracefully."""
        sweep = _make_sweep()
        cached = [_make_cached_result(error="failed")]

        dist = sweep._compute_similarity_distribution(cached)

        assert dist["top_1_result"] == {}
        assert dist["top_5_result"] == {}
