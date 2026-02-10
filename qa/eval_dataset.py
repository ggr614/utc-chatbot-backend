"""
Evaluation Dataset Loader

Loads, parses, filters, and samples the QA pairs JSONL file
into structured EvalQuestion records for retrieval evaluation.

The QA pairs JSONL contains records where each record has a `response` field
that is a JSON string containing 3 QA pairs generated from a specific chunk.
Each question's ground truth is the chunk_id it was generated from.
"""

import json
import random
from pathlib import Path
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from utils.logger import get_logger

logger = get_logger(__name__)


class EvalQuestion(BaseModel):
    """A single evaluation question with its ground truth."""

    question: str
    answer: str
    sufficient_context: bool
    chunk_id: UUID
    parent_article_id: UUID
    chunk_sequence: int
    source_url: str
    token_count: int
    overall_quality: str  # "high", "medium", "low"
    chunk_summary: str
    qa_pair_index: int  # 0, 1, or 2 within the chunk's QA pairs


class EvalDataset(BaseModel):
    """Container for the evaluation dataset with metadata."""

    questions: list[EvalQuestion]
    total_chunks_in_source: int
    total_qa_pairs_in_source: int
    filtered_count: int
    quality_distribution: dict[str, int]
    source_file: str


def _parse_jsonl_record(record: dict) -> list[EvalQuestion]:
    """
    Parse a single JSONL record into individual EvalQuestion objects.

    Each record has a `response` field that is a JSON string containing:
    - qa_pairs: list of {question, answer, sufficient_context}
    - chunk_summary: str
    - overall_quality: "high"/"medium"/"low"

    Args:
        record: Parsed JSON object from one JSONL line

    Returns:
        List of EvalQuestion objects (up to 3 per record)

    Raises:
        ValueError: If response field is missing or unparseable
    """
    response_str = record.get("response", "")
    if not response_str:
        raise ValueError("Empty response field")

    parsed = json.loads(response_str)
    qa_pairs = parsed.get("qa_pairs", [])
    overall_quality = parsed.get("overall_quality", "unknown")
    chunk_summary = parsed.get("chunk_summary", "")

    questions = []
    for idx, pair in enumerate(qa_pairs):
        questions.append(
            EvalQuestion(
                question=pair["question"],
                answer=pair["answer"],
                sufficient_context=pair.get("sufficient_context", True),
                chunk_id=UUID(record["chunk_id"]),
                parent_article_id=UUID(record["parent_article_id"]),
                chunk_sequence=record["chunk_sequence"],
                source_url=record["source_url"],
                token_count=record["token_count"],
                overall_quality=overall_quality,
                chunk_summary=chunk_summary,
                qa_pair_index=idx,
            )
        )

    return questions


def load_eval_dataset(
    filepath: str = "data/qa_pairs.jsonl",
    filter_sufficient_context: bool = True,
    quality_filter: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> EvalDataset:
    """
    Load and parse QA pairs dataset for evaluation.

    Args:
        filepath: Path to the JSONL file
        filter_sufficient_context: If True, only keep questions where
            sufficient_context=True (default: True)
        quality_filter: Filter by quality tier ("high", "medium", "low", or None for all)
        sample_size: Number of questions to randomly sample (None = all)
        random_seed: Seed for reproducible sampling (default: 42)

    Returns:
        EvalDataset with filtered/sampled questions and metadata

    Raises:
        FileNotFoundError: If filepath does not exist
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    all_questions: list[EvalQuestion] = []
    total_chunks = 0
    total_qa_pairs = 0
    parse_errors = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total_chunks += 1
            try:
                record = json.loads(line)
                questions = _parse_jsonl_record(record)
                total_qa_pairs += len(questions)
                all_questions.extend(questions)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                parse_errors += 1
                logger.warning(f"Skipping line {line_num}: {e}")
                continue

    if parse_errors > 0:
        logger.warning(
            f"Skipped {parse_errors}/{total_chunks} records due to parse errors"
        )

    logger.info(
        f"Loaded {total_qa_pairs} QA pairs from {total_chunks} chunks "
        f"({parse_errors} parse errors)"
    )

    # Apply filters
    filtered = all_questions

    if filter_sufficient_context:
        before = len(filtered)
        filtered = [q for q in filtered if q.sufficient_context]
        logger.info(f"Filtered sufficient_context=True: {before} -> {len(filtered)}")

    if quality_filter:
        before = len(filtered)
        filtered = [q for q in filtered if q.overall_quality == quality_filter]
        logger.info(f"Filtered quality='{quality_filter}': {before} -> {len(filtered)}")

    # Compute quality distribution after filtering
    quality_dist: dict[str, int] = {}
    for q in filtered:
        quality_dist[q.overall_quality] = quality_dist.get(q.overall_quality, 0) + 1

    filtered_count = len(filtered)

    # Sample if requested
    if sample_size is not None and sample_size < len(filtered):
        rng = random.Random(random_seed)
        filtered = rng.sample(filtered, sample_size)
        logger.info(f"Sampled {sample_size} questions (seed={random_seed})")

    return EvalDataset(
        questions=filtered,
        total_chunks_in_source=total_chunks,
        total_qa_pairs_in_source=total_qa_pairs,
        filtered_count=filtered_count,
        quality_distribution=quality_dist,
        source_file=str(filepath),
    )
