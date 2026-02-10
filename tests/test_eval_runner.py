"""Tests for qa/eval_runner.py — evaluation orchestrator."""

import json
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID
from datetime import datetime, timezone

from core.schemas import TextChunk
from core.bm25_search import BM25SearchResult
from core.vector_search import VectorSearchResult
from qa.eval_dataset import EvalQuestion, EvalDataset
from qa.eval_metrics import RetrievalResult
from qa.eval_runner import RetrievalEvaluator, EvalConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

CHUNK_A_ID = UUID("00000000-0000-0000-0000-000000000001")
CHUNK_B_ID = UUID("00000000-0000-0000-0000-000000000002")
CHUNK_C_ID = UUID("00000000-0000-0000-0000-000000000003")

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


def _make_eval_dataset(questions: list[EvalQuestion] | None = None) -> EvalDataset:
    if questions is None:
        questions = [
            _make_eval_question("Q1?", CHUNK_A_ID, ARTICLE_X_ID, "high"),
            _make_eval_question("Q2?", CHUNK_B_ID, ARTICLE_Y_ID, "high"),
            _make_eval_question("Q3?", CHUNK_C_ID, ARTICLE_X_ID, "medium"),
        ]
    return EvalDataset(
        questions=questions,
        total_chunks_in_source=3,
        total_qa_pairs_in_source=9,
        filtered_count=len(questions),
        quality_distribution={"high": 2, "medium": 1},
        source_file="test.jsonl",
    )


CHUNK_A = _make_text_chunk(CHUNK_A_ID, ARTICLE_X_ID)
CHUNK_B = _make_text_chunk(CHUNK_B_ID, ARTICLE_Y_ID)
CHUNK_C = _make_text_chunk(CHUNK_C_ID, ARTICLE_X_ID)


# ── _build_retrieval_result tests ─────────────────────────────────────────────


class TestBuildRetrievalResult:
    def setup_method(self):
        self.bm25 = MagicMock()
        self.evaluator = RetrievalEvaluator(
            bm25_retriever=self.bm25,
            config=EvalConfig(k_values=[1, 3], max_top_k=3),
        )

    def test_bm25_results(self):
        bm25_results = [
            BM25SearchResult(chunk=CHUNK_A, score=5.0, rank=1),
            BM25SearchResult(chunk=CHUNK_B, score=3.0, rank=2),
        ]
        question = _make_eval_question("Q?", CHUNK_A_ID, ARTICLE_X_ID)

        result = self.evaluator._build_retrieval_result(
            question=question,
            bm25_results=bm25_results,
            method="bm25",
            latency_ms=5.0,
        )

        assert result.retrieval_method == "bm25"
        assert result.retrieved_chunk_ids == [CHUNK_A_ID, CHUNK_B_ID]
        assert result.retrieved_article_ids == [ARTICLE_X_ID, ARTICLE_Y_ID]
        assert result.expected_chunk_id == CHUNK_A_ID

    def test_vector_results(self):
        vector_results = [
            VectorSearchResult(chunk=CHUNK_B, similarity=0.95, rank=1),
            VectorSearchResult(chunk=CHUNK_A, similarity=0.80, rank=2),
        ]
        question = _make_eval_question("Q?", CHUNK_A_ID, ARTICLE_X_ID)

        result = self.evaluator._build_retrieval_result(
            question=question,
            vector_results=vector_results,
            method="vector",
            latency_ms=500.0,
        )

        assert result.retrieval_method == "vector"
        assert result.retrieved_chunk_ids == [CHUNK_B_ID, CHUNK_A_ID]
        assert result.latency_ms == 500.0

    def test_hybrid_results(self):
        hybrid_results = [
            {"rank": 1, "combined_score": 0.03, "chunk": CHUNK_C},
            {"rank": 2, "combined_score": 0.02, "chunk": CHUNK_A},
        ]
        question = _make_eval_question("Q?", CHUNK_A_ID, ARTICLE_X_ID)

        result = self.evaluator._build_retrieval_result(
            question=question,
            hybrid_results=hybrid_results,
            method="hybrid",
            latency_ms=600.0,
        )

        assert result.retrieval_method == "hybrid"
        assert result.retrieved_chunk_ids == [CHUNK_C_ID, CHUNK_A_ID]
        assert result.retrieved_article_ids == [ARTICLE_X_ID, ARTICLE_X_ID]

    def test_hyde_results_use_hybrid_format(self):
        """HyDE results use the same hybrid_results dict format."""
        hybrid_results = [
            {"rank": 1, "combined_score": 0.03, "chunk": CHUNK_A},
            {"rank": 2, "combined_score": 0.02, "chunk": CHUNK_B},
        ]
        question = _make_eval_question("Q?", CHUNK_A_ID, ARTICLE_X_ID)

        result = self.evaluator._build_retrieval_result(
            question=question,
            hybrid_results=hybrid_results,
            method="hyde",
            latency_ms=1500.0,
        )

        assert result.retrieval_method == "hyde"
        assert result.retrieved_chunk_ids == [CHUNK_A_ID, CHUNK_B_ID]
        assert result.latency_ms == 1500.0

    def test_empty_results(self):
        question = _make_eval_question("Q?", CHUNK_A_ID, ARTICLE_X_ID)

        result = self.evaluator._build_retrieval_result(
            question=question,
            bm25_results=[],
            method="bm25",
            latency_ms=1.0,
        )

        assert result.retrieved_chunk_ids == []
        assert result.retrieved_article_ids == []


# ── _evaluate_bm25 tests ─────────────────────────────────────────────────────


class TestEvaluateBM25:
    def test_calls_batch_search(self):
        bm25 = MagicMock()
        bm25.batch_search.return_value = {
            "Q1?": [BM25SearchResult(chunk=CHUNK_A, score=5.0, rank=1)],
            "Q2?": [BM25SearchResult(chunk=CHUNK_B, score=3.0, rank=1)],
        }

        evaluator = RetrievalEvaluator(
            bm25_retriever=bm25,
            config=EvalConfig(max_top_k=5, k_values=[1, 3]),
        )

        questions = [
            _make_eval_question("Q1?", CHUNK_A_ID, ARTICLE_X_ID),
            _make_eval_question("Q2?", CHUNK_B_ID, ARTICLE_Y_ID),
        ]

        results = evaluator._evaluate_bm25(questions)

        assert len(results) == 2
        assert results[0].retrieval_method == "bm25"
        bm25.batch_search.assert_called_once_with(
            queries=["Q1?", "Q2?"],
            top_k=5,
        )

    def test_missing_query_in_batch(self):
        """If batch_search doesn't return a query, result should be empty."""
        bm25 = MagicMock()
        bm25.batch_search.return_value = {
            "Q1?": [BM25SearchResult(chunk=CHUNK_A, score=5.0, rank=1)],
            # Q2? is missing
        }

        evaluator = RetrievalEvaluator(
            bm25_retriever=bm25,
            config=EvalConfig(max_top_k=5, k_values=[1]),
        )

        questions = [
            _make_eval_question("Q1?", CHUNK_A_ID, ARTICLE_X_ID),
            _make_eval_question("Q2?", CHUNK_B_ID, ARTICLE_Y_ID),
        ]

        results = evaluator._evaluate_bm25(questions)
        assert len(results) == 2
        assert results[1].retrieved_chunk_ids == []


# ── _evaluate_vector tests ───────────────────────────────────────────────────


class TestEvaluateVector:
    def test_calls_search_per_query(self):
        vector = MagicMock()
        vector.search.return_value = [
            VectorSearchResult(chunk=CHUNK_A, similarity=0.9, rank=1),
        ]

        evaluator = RetrievalEvaluator(
            bm25_retriever=MagicMock(),
            vector_retriever=vector,
            config=EvalConfig(max_top_k=5, k_values=[1]),
        )

        questions = [
            _make_eval_question("Q1?", CHUNK_A_ID, ARTICLE_X_ID),
            _make_eval_question("Q2?", CHUNK_B_ID, ARTICLE_Y_ID),
        ]

        results = evaluator._evaluate_vector(questions)

        assert len(results) == 2
        assert vector.search.call_count == 2
        assert results[0].retrieval_method == "vector"

    def test_handles_search_failure(self):
        vector = MagicMock()
        vector.search.side_effect = RuntimeError("API timeout")

        evaluator = RetrievalEvaluator(
            bm25_retriever=MagicMock(),
            vector_retriever=vector,
            config=EvalConfig(max_top_k=5, k_values=[1]),
        )

        questions = [_make_eval_question("Q1?", CHUNK_A_ID, ARTICLE_X_ID)]
        results = evaluator._evaluate_vector(questions)

        assert len(results) == 1
        assert results[0].error == "API timeout"
        assert results[0].retrieved_chunk_ids == []


# ── _evaluate_hyde tests ─────────────────────────────────────────────────────


class TestEvaluateHyDE:
    def test_calls_hyde_generator_and_searches(self):
        """Verify HyDE generates hypothetical docs and uses them for vector search."""
        bm25 = MagicMock()
        bm25.search.return_value = [
            BM25SearchResult(chunk=CHUNK_A, score=5.0, rank=1),
        ]

        vector = MagicMock()
        vector.search.return_value = [
            VectorSearchResult(chunk=CHUNK_B, similarity=0.9, rank=1),
        ]

        hyde = MagicMock()
        hyde.generate_hypothetical_document_sync.return_value = (
            "Hypothetical answer about password reset.",
            {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
        )

        evaluator = RetrievalEvaluator(
            bm25_retriever=bm25,
            vector_retriever=vector,
            hyde_generator=hyde,
            config=EvalConfig(max_top_k=5, k_values=[1, 3], rrf_k=60),
        )

        questions = [
            _make_eval_question("Q1?", CHUNK_A_ID, ARTICLE_X_ID),
        ]

        results = evaluator._evaluate_hyde(questions)

        assert len(results) == 1
        assert results[0].retrieval_method == "hyde"

        # Verify HyDE generator was called with original query
        hyde.generate_hypothetical_document_sync.assert_called_once_with(
            query="Q1?",
        )

        # Verify BM25 was called with original query
        bm25.search.assert_called_once_with(
            query="Q1?",
            top_k=5,
        )

        # Verify vector was called with hypothetical document
        vector.search.assert_called_once_with(
            query="Hypothetical answer about password reset.",
            top_k=5,
            include_system_prompts=False,
        )

        # Verify results contain chunks from both BM25 and vector (fused)
        assert CHUNK_A_ID in results[0].retrieved_chunk_ids
        assert CHUNK_B_ID in results[0].retrieved_chunk_ids

    def test_hyde_generation_failure_falls_back(self):
        """When HyDE generation fails, vector search uses original query."""
        bm25 = MagicMock()
        bm25.search.return_value = [
            BM25SearchResult(chunk=CHUNK_A, score=5.0, rank=1),
        ]

        vector = MagicMock()
        vector.search.return_value = [
            VectorSearchResult(chunk=CHUNK_B, similarity=0.9, rank=1),
        ]

        hyde = MagicMock()
        hyde.generate_hypothetical_document_sync.side_effect = RuntimeError(
            "LLM API timeout"
        )

        evaluator = RetrievalEvaluator(
            bm25_retriever=bm25,
            vector_retriever=vector,
            hyde_generator=hyde,
            config=EvalConfig(max_top_k=5, k_values=[1], rrf_k=60),
        )

        questions = [
            _make_eval_question("Q1?", CHUNK_A_ID, ARTICLE_X_ID),
        ]

        results = evaluator._evaluate_hyde(questions)

        assert len(results) == 1
        assert results[0].error is None
        assert results[0].retrieval_method == "hyde"

        # Vector search should have been called with the original query (fallback)
        vector.search.assert_called_once_with(
            query="Q1?",
            top_k=5,
            include_system_prompts=False,
        )

    def test_search_failure_produces_error_result(self):
        """When BM25 or vector search fails entirely, produce error RetrievalResult."""
        bm25 = MagicMock()
        bm25.search.side_effect = RuntimeError("Database connection lost")

        vector = MagicMock()

        hyde = MagicMock()
        hyde.generate_hypothetical_document_sync.return_value = (
            "Hypothetical answer.",
            None,
        )

        evaluator = RetrievalEvaluator(
            bm25_retriever=bm25,
            vector_retriever=vector,
            hyde_generator=hyde,
            config=EvalConfig(max_top_k=5, k_values=[1], rrf_k=60),
        )

        questions = [
            _make_eval_question("Q1?", CHUNK_A_ID, ARTICLE_X_ID),
        ]

        results = evaluator._evaluate_hyde(questions)

        assert len(results) == 1
        assert results[0].error == "Database connection lost"
        assert results[0].retrieved_chunk_ids == []
        assert results[0].retrieval_method == "hyde"


# ── Run dispatch HyDE tests ─────────────────────────────────────────────────


class TestRunDispatchHyde:
    def test_skips_hyde_without_vector(self, mocker):
        """run() skips hyde when no VectorRetriever is provided."""
        bm25 = MagicMock()
        hyde = MagicMock()

        evaluator = RetrievalEvaluator(
            bm25_retriever=bm25,
            hyde_generator=hyde,
            config=EvalConfig(
                methods=["hyde"],
                k_values=[1],
                dataset_path="data/qa_pairs.jsonl",
            ),
        )

        mocker.patch(
            "qa.eval_runner.load_eval_dataset",
            return_value=_make_eval_dataset(),
        )

        report = evaluator.run()

        assert "hyde" not in report.get("results", {})

    def test_skips_hyde_without_generator(self, mocker):
        """run() skips hyde when no HyDEGenerator is provided."""
        bm25 = MagicMock()
        vector = MagicMock()

        evaluator = RetrievalEvaluator(
            bm25_retriever=bm25,
            vector_retriever=vector,
            config=EvalConfig(
                methods=["hyde"],
                k_values=[1],
                dataset_path="data/qa_pairs.jsonl",
            ),
        )

        mocker.patch(
            "qa.eval_runner.load_eval_dataset",
            return_value=_make_eval_dataset(),
        )

        report = evaluator.run()

        assert "hyde" not in report.get("results", {})


# ── Report generation tests ──────────────────────────────────────────────────


class TestGenerateReport:
    def test_report_structure(self):
        bm25 = MagicMock()
        evaluator = RetrievalEvaluator(
            bm25_retriever=bm25,
            config=EvalConfig(k_values=[1, 5]),
        )

        dataset = _make_eval_dataset()
        results = [
            RetrievalResult(
                query="Q?",
                expected_chunk_id=CHUNK_A_ID,
                expected_article_id=ARTICLE_X_ID,
                retrieved_chunk_ids=[CHUNK_A_ID],
                retrieved_article_ids=[ARTICLE_X_ID],
                retrieval_method="bm25",
                latency_ms=5.0,
            )
        ]
        metrics = {
            "bm25": {
                "chunk_level": {},
                "article_level": {},
                "total_queries": 1,
                "failed_queries": 0,
            }
        }
        quality_breakdown = {}

        report = evaluator._generate_report(
            dataset, {"bm25": results}, metrics, quality_breakdown
        )

        assert "metadata" in report
        assert "results" in report
        assert "quality_breakdown" in report
        assert report["metadata"]["questions_evaluated"] == 3
        assert report["metadata"]["k_values"] == [1, 5]


# ── Quality breakdown tests ──────────────────────────────────────────────────


class TestQualityBreakdown:
    def test_groups_by_quality(self):
        bm25 = MagicMock()
        evaluator = RetrievalEvaluator(
            bm25_retriever=bm25,
            config=EvalConfig(k_values=[1]),
        )

        questions = [
            _make_eval_question("Q1?", CHUNK_A_ID, ARTICLE_X_ID, "high"),
            _make_eval_question("Q2?", CHUNK_B_ID, ARTICLE_Y_ID, "medium"),
        ]

        results_by_method = {
            "bm25": [
                RetrievalResult(
                    query="Q1?",
                    expected_chunk_id=CHUNK_A_ID,
                    expected_article_id=ARTICLE_X_ID,
                    retrieved_chunk_ids=[CHUNK_A_ID],
                    retrieved_article_ids=[ARTICLE_X_ID],
                    retrieval_method="bm25",
                ),
                RetrievalResult(
                    query="Q2?",
                    expected_chunk_id=CHUNK_B_ID,
                    expected_article_id=ARTICLE_Y_ID,
                    retrieved_chunk_ids=[],
                    retrieved_article_ids=[],
                    retrieval_method="bm25",
                ),
            ]
        }

        breakdown = evaluator._compute_quality_breakdown(questions, results_by_method)

        assert "high" in breakdown
        assert "medium" in breakdown
        assert breakdown["high"]["question_count"] == 1
        assert breakdown["medium"]["question_count"] == 1
        assert "bm25" in breakdown["high"]


# ── Save report tests ────────────────────────────────────────────────────────


class TestSaveReport:
    def test_saves_json(self, tmp_path):
        evaluator = RetrievalEvaluator(
            bm25_retriever=MagicMock(),
            config=EvalConfig(output_dir=str(tmp_path)),
        )

        report = {"metadata": {"test": True}, "results": {}}
        json_path, latest_path = evaluator.save_report(report, str(tmp_path))

        assert Path(json_path).exists()
        assert Path(latest_path).exists()

        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["metadata"]["test"] is True
