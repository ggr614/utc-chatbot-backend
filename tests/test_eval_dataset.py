"""Tests for qa/eval_dataset.py — dataset loading, parsing, and sampling."""

import json
import pytest
from pathlib import Path
from uuid import UUID

from qa.eval_dataset import (
    load_eval_dataset,
    _parse_jsonl_record,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_record(
    chunk_id: str = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    article_id: str = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    quality: str = "high",
    sufficient_context: list[bool] | None = None,
) -> dict:
    """Build a single JSONL record matching the dataset format."""
    if sufficient_context is None:
        sufficient_context = [True, True, False]

    response = {
        "qa_pairs": [
            {
                "question": f"Question {i + 1}?",
                "answer": f"Answer {i + 1}.",
                "sufficient_context": sufficient_context[i],
            }
            for i in range(3)
        ],
        "chunk_summary": "A test chunk about something.",
        "overall_quality": quality,
    }

    return {
        "chunk_id": chunk_id,
        "parent_article_id": article_id,
        "chunk_sequence": 0,
        "source_url": "https://example.com/article/123",
        "token_count": 200,
        "last_modified_date": "2025-01-01T00:00:00+00:00",
        "generated_at": "2025-01-02T00:00:00+00:00",
        "response": json.dumps(response),
    }


def _write_jsonl(tmp_path: Path, records: list[dict]) -> Path:
    """Write records to a JSONL file and return the path."""
    filepath = tmp_path / "test_qa_pairs.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return filepath


# ── _parse_jsonl_record tests ─────────────────────────────────────────────────


class TestParseJsonlRecord:
    def test_parses_three_questions(self):
        record = _make_record()
        questions = _parse_jsonl_record(record)
        assert len(questions) == 3

    def test_question_fields(self):
        record = _make_record()
        questions = _parse_jsonl_record(record)
        q = questions[0]

        assert q.question == "Question 1?"
        assert q.answer == "Answer 1."
        assert q.sufficient_context is True
        assert q.chunk_id == UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        assert q.parent_article_id == UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
        assert q.chunk_sequence == 0
        assert q.token_count == 200
        assert q.overall_quality == "high"
        assert q.qa_pair_index == 0

    def test_sufficient_context_values(self):
        record = _make_record(sufficient_context=[True, True, False])
        questions = _parse_jsonl_record(record)
        assert questions[0].sufficient_context is True
        assert questions[1].sufficient_context is True
        assert questions[2].sufficient_context is False

    def test_qa_pair_indices(self):
        record = _make_record()
        questions = _parse_jsonl_record(record)
        assert [q.qa_pair_index for q in questions] == [0, 1, 2]

    def test_empty_response_raises(self):
        record = _make_record()
        record["response"] = ""
        with pytest.raises(ValueError, match="Empty response"):
            _parse_jsonl_record(record)

    def test_malformed_json_response_raises(self):
        record = _make_record()
        record["response"] = "not valid json {"
        with pytest.raises(json.JSONDecodeError):
            _parse_jsonl_record(record)


# ── load_eval_dataset tests ───────────────────────────────────────────────────


class TestLoadEvalDataset:
    def test_loads_basic_dataset(self, tmp_path):
        records = [_make_record()]
        filepath = _write_jsonl(tmp_path, records)

        dataset = load_eval_dataset(
            filepath=str(filepath),
            filter_sufficient_context=False,
        )

        assert dataset.total_chunks_in_source == 1
        assert dataset.total_qa_pairs_in_source == 3
        assert len(dataset.questions) == 3

    def test_filter_sufficient_context(self, tmp_path):
        # Each record has 2 True and 1 False
        records = [_make_record()]
        filepath = _write_jsonl(tmp_path, records)

        dataset = load_eval_dataset(
            filepath=str(filepath),
            filter_sufficient_context=True,
        )

        assert len(dataset.questions) == 2
        assert all(q.sufficient_context for q in dataset.questions)
        assert dataset.filtered_count == 2

    def test_quality_filter(self, tmp_path):
        records = [
            _make_record(
                chunk_id="aaaaaaaa-aaaa-aaaa-aaaa-000000000001",
                quality="high",
            ),
            _make_record(
                chunk_id="aaaaaaaa-aaaa-aaaa-aaaa-000000000002",
                quality="medium",
            ),
            _make_record(
                chunk_id="aaaaaaaa-aaaa-aaaa-aaaa-000000000003",
                quality="low",
            ),
        ]
        filepath = _write_jsonl(tmp_path, records)

        dataset = load_eval_dataset(
            filepath=str(filepath),
            filter_sufficient_context=False,
            quality_filter="high",
        )

        assert all(q.overall_quality == "high" for q in dataset.questions)
        assert len(dataset.questions) == 3  # 3 QA pairs from 1 high-quality chunk

    def test_sampling(self, tmp_path):
        records = [
            _make_record(
                chunk_id=f"aaaaaaaa-aaaa-aaaa-aaaa-{i:012d}",
                sufficient_context=[True, True, True],
            )
            for i in range(10)
        ]
        filepath = _write_jsonl(tmp_path, records)

        dataset = load_eval_dataset(
            filepath=str(filepath),
            filter_sufficient_context=False,
            sample_size=5,
            random_seed=42,
        )

        assert len(dataset.questions) == 5

    def test_sampling_deterministic(self, tmp_path):
        records = [
            _make_record(
                chunk_id=f"aaaaaaaa-aaaa-aaaa-aaaa-{i:012d}",
                sufficient_context=[True, True, True],
            )
            for i in range(10)
        ]
        filepath = _write_jsonl(tmp_path, records)

        d1 = load_eval_dataset(
            str(filepath),
            filter_sufficient_context=False,
            sample_size=5,
            random_seed=42,
        )
        d2 = load_eval_dataset(
            str(filepath),
            filter_sufficient_context=False,
            sample_size=5,
            random_seed=42,
        )

        q1 = [q.question for q in d1.questions]
        q2 = [q.question for q in d2.questions]
        assert q1 == q2

    def test_sample_larger_than_dataset(self, tmp_path):
        records = [_make_record(sufficient_context=[True, True, True])]
        filepath = _write_jsonl(tmp_path, records)

        dataset = load_eval_dataset(
            filepath=str(filepath),
            filter_sufficient_context=False,
            sample_size=100,
        )

        # Should return all 3 questions, not error
        assert len(dataset.questions) == 3

    def test_quality_distribution(self, tmp_path):
        records = [
            _make_record(
                chunk_id="aaaaaaaa-aaaa-aaaa-aaaa-000000000001",
                quality="high",
                sufficient_context=[True, True, True],
            ),
            _make_record(
                chunk_id="aaaaaaaa-aaaa-aaaa-aaaa-000000000002",
                quality="medium",
                sufficient_context=[True, True, True],
            ),
        ]
        filepath = _write_jsonl(tmp_path, records)

        dataset = load_eval_dataset(
            filepath=str(filepath),
            filter_sufficient_context=False,
        )

        assert dataset.quality_distribution == {"high": 3, "medium": 3}

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_eval_dataset(filepath="nonexistent.jsonl")

    def test_malformed_records_skipped(self, tmp_path):
        filepath = tmp_path / "test.jsonl"
        good = _make_record()
        with open(filepath, "w") as f:
            f.write(json.dumps(good) + "\n")
            f.write('{"chunk_id": "bad", "response": "not json {"}\n')  # malformed
            f.write(json.dumps(good) + "\n")

        dataset = load_eval_dataset(
            filepath=str(filepath),
            filter_sufficient_context=False,
        )

        # Should have loaded 2 good records (6 questions), skipped 1
        assert len(dataset.questions) == 6
        assert dataset.total_chunks_in_source == 3

    def test_empty_lines_skipped(self, tmp_path):
        filepath = tmp_path / "test.jsonl"
        good = _make_record()
        with open(filepath, "w") as f:
            f.write(json.dumps(good) + "\n")
            f.write("\n")
            f.write("  \n")

        dataset = load_eval_dataset(
            filepath=str(filepath),
            filter_sufficient_context=False,
        )

        assert dataset.total_chunks_in_source == 1
        assert len(dataset.questions) == 3
