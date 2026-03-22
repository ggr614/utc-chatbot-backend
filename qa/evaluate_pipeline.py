"""
QA Pipeline Quality Evaluator

Scores the QA pipeline on three dimensions:
1. Metric Completeness — Does the pipeline output all expected metric categories?
2. Comparison Depth — Does the run log show deltas, trends, and regression flags?
3. Actionability — Does the output surface the worst-performing items with context?

Usage:
    python qa/evaluate_pipeline.py --score-only
    python qa/evaluate_pipeline.py --verbose
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

# ---------------------------------------------------------------------------
# Expected metrics for a retrieval QA pipeline
# ---------------------------------------------------------------------------

EXPECTED_METRICS = [
    "hit_rate",       # accuracy / recall for single-doc retrieval
    "precision",      # precision@k
    "mrr",            # mean reciprocal rank
    "ndcg",           # normalized discounted cumulative gain
    "recall",         # explicit recall metric (may equal hit_rate)
    "f1",             # F1 score (harmonic mean of precision and recall)
    "latency",        # avg latency per query
    "failure_rate",   # proportion of failed queries
]

# ---------------------------------------------------------------------------
# Synthetic test data for pipeline probing
# ---------------------------------------------------------------------------

_ARTICLE_ID = UUID("00000000-0000-0000-0000-000000000001")

SYNTHETIC_RESULTS = []
for i in range(10):
    chunk_id = UUID(f"00000000-0000-0000-0000-{i:012d}")
    # For odd-indexed results, the expected chunk is at rank 1 (hit)
    # For even-indexed results, the expected chunk is not retrieved (miss)
    if i % 2 == 1:
        retrieved = [chunk_id, UUID(f"00000000-0000-0000-0000-{99:012d}")]
    else:
        retrieved = [UUID(f"00000000-0000-0000-0000-{99:012d}")]

    SYNTHETIC_RESULTS.append({
        "query": f"test question {i}",
        "expected_chunk_id": str(chunk_id),
        "expected_article_id": str(_ARTICLE_ID),
        "retrieved_chunk_ids": [str(c) for c in retrieved],
        "retrieved_article_ids": [str(_ARTICLE_ID)] * len(retrieved),
        "retrieval_method": "bm25",
        "latency_ms": 50.0 + i * 10,
        "error": None,
    })


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def score_metric_completeness(verbose: bool = False) -> float:
    """
    Score = (metrics present / metrics expected) x 100.

    Probes the pipeline by importing eval_metrics and running compute_all_metrics
    on synthetic data, then checking which metric keys appear in the output.
    """
    from qa.eval_metrics import RetrievalResult, compute_all_metrics

    results = [
        RetrievalResult(
            query=r["query"],
            expected_chunk_id=UUID(r["expected_chunk_id"]),
            expected_article_id=UUID(r["expected_article_id"]),
            retrieved_chunk_ids=[UUID(c) for c in r["retrieved_chunk_ids"]],
            retrieved_article_ids=[UUID(a) for a in r["retrieved_article_ids"]],
            retrieval_method=r["retrieval_method"],
            latency_ms=r["latency_ms"],
            error=r["error"],
        )
        for r in SYNTHETIC_RESULTS
    ]

    metrics = compute_all_metrics(results, k_values=[1, 3, 5])

    present = set()

    # Check chunk_level metrics
    chunk = metrics.get("chunk_level", {})
    if "hit_rate" in chunk:
        present.add("hit_rate")
    if "precision" in chunk:
        present.add("precision")
    if "mrr" in chunk:
        present.add("mrr")
    if "ndcg" in chunk:
        present.add("ndcg")
    if "recall" in chunk:
        present.add("recall")
    if "f1" in chunk:
        present.add("f1")

    # Check top-level metrics
    if metrics.get("avg_latency_ms") is not None:
        present.add("latency")
    if "failed_queries" in metrics and "total_queries" in metrics:
        present.add("failure_rate")

    score = (len(present) / len(EXPECTED_METRICS)) * 100

    if verbose:
        print(f"\n--- Metric Completeness ---")
        for m in EXPECTED_METRICS:
            status = "PRESENT" if m in present else "MISSING"
            print(f"  {m}: {status}")
        print(f"  Score: {len(present)}/{len(EXPECTED_METRICS)} = {score:.1f}")

    return round(score, 1)


def score_comparison_depth(verbose: bool = False) -> float:
    """
    Score = (comparison features present / comparison features expected) x 100.

    Expected features:
    1. Delta from previous run (numeric difference)
    2. Trend direction (up/down/stable arrow or indicator)
    3. Regression flagging (automatic alert when metric worsens)
    """
    expected_features = ["delta_from_previous", "trend_direction", "regression_flag"]
    present = set()

    # Check if eval_runner has comparison capabilities
    try:
        import qa.eval_runner as runner
        source = Path(runner.__file__).read_text(encoding="utf-8")

        # Check for delta computation
        if any(kw in source for kw in ["delta", "previous_run", "prior_run", "last_run", "run_log"]):
            present.add("delta_from_previous")

        # Check for trend direction indicators
        if any(kw in source for kw in ["trend", "arrow", "direction", "\u2191", "\u2193", "\u2192"]):
            present.add("trend_direction")

        # Check for regression flagging
        if any(kw in source for kw in ["regression", "regress", "worse", "degraded", "decline"]):
            present.add("regression_flag")
    except Exception:
        pass

    # Also check eval_metrics for comparison features
    try:
        import qa.eval_metrics as metrics_mod
        source = Path(metrics_mod.__file__).read_text(encoding="utf-8")

        if any(kw in source for kw in ["delta", "previous", "compare", "diff"]):
            present.add("delta_from_previous")
        if any(kw in source for kw in ["trend", "arrow", "\u2191", "\u2193"]):
            present.add("trend_direction")
        if any(kw in source for kw in ["regression", "regress", "worse"]):
            present.add("regression_flag")
    except Exception:
        pass

    score = (len(present) / len(expected_features)) * 100

    if verbose:
        print(f"\n--- Comparison Depth ---")
        for f in expected_features:
            status = "PRESENT" if f in present else "MISSING"
            print(f"  {f}: {status}")
        print(f"  Score: {len(present)}/{len(expected_features)} = {score:.1f}")

    return round(score, 1)


def score_actionability(verbose: bool = False) -> float:
    """
    Score actionability of pipeline output.
    0 = missing, 50 = present but vague, 100 = specific and ranked.

    Checks if the pipeline surfaces worst-performing items with context.
    """
    score = 0.0

    try:
        import qa.eval_runner as runner
        source = Path(runner.__file__).read_text(encoding="utf-8")

        # Check for worst-performer identification
        has_worst = any(kw in source for kw in [
            "worst", "bottom", "lowest", "poorest",
            "worst_performing", "failed_queries_detail",
            "low_score", "miss", "zero_hit",
        ])

        # Check for ranked list of problems
        has_ranking = any(kw in source for kw in [
            "rank", "sorted", "top_n", "top_3", "top_k_worst",
            "worst_queries", "hardest_queries",
        ])

        # Check for contextual detail (question + expected + actual)
        has_context = any(kw in source for kw in [
            "expected_chunk", "retrieved_chunk", "query.*score",
            "per_query", "question_detail", "result_detail",
        ])

        if has_worst and has_ranking and has_context:
            score = 100.0
        elif has_worst and (has_ranking or has_context):
            score = 50.0
        elif has_worst:
            score = 25.0

    except Exception:
        pass

    if verbose:
        print(f"\n--- Actionability ---")
        print(f"  Score: {score:.1f}")

    return round(score, 1)


# ---------------------------------------------------------------------------
# Run log management
# ---------------------------------------------------------------------------


RUN_LOG_PATH = Path(__file__).parent / "run_log.tsv"
RUN_LOG_HEADER = "timestamp\tgit_sha\tcomposite_score\tcompleteness\tcomparison_depth\tactionability\tnotes"


def get_git_sha() -> str:
    """Get short git SHA of current commit."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_previous_run() -> dict | None:
    """Load the last row from run_log.tsv, if it exists."""
    if not RUN_LOG_PATH.exists():
        return None

    lines = RUN_LOG_PATH.read_text(encoding="utf-8").strip().split("\n")
    # Skip header
    data_lines = [l for l in lines if not l.startswith("timestamp") and l.strip()]
    if not data_lines:
        return None

    last = data_lines[-1].split("\t")
    if len(last) < 7:
        return None

    return {
        "timestamp": last[0],
        "git_sha": last[1],
        "composite_score": float(last[2]),
        "completeness": float(last[3]),
        "comparison_depth": float(last[4]),
        "actionability": float(last[5]),
        "notes": last[6] if len(last) > 6 else "",
    }


def append_run_log(
    composite: float,
    completeness: float,
    comparison_depth: float,
    actionability: float,
    notes: str,
) -> None:
    """Append a row to qa/run_log.tsv."""
    if not RUN_LOG_PATH.exists():
        RUN_LOG_PATH.write_text(RUN_LOG_HEADER + "\n", encoding="utf-8")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    git_sha = get_git_sha()
    row = f"{timestamp}\t{git_sha}\t{composite:.1f}\t{completeness:.1f}\t{comparison_depth:.1f}\t{actionability:.1f}\t{notes}"

    with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(row + "\n")


def print_delta_summary(current: dict, previous: dict) -> None:
    """Print comparison between current and previous run."""
    fields = ["composite_score", "completeness", "comparison_depth", "actionability"]
    print("\n--- Delta from previous run ---")
    for field in fields:
        old = previous[field]
        new = current[field]
        delta = new - old
        sign = "+" if delta > 0 else ""
        print(f"  {field}: {old:.1f} -> {new:.1f}, {sign}{delta:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QA pipeline quality")
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Print only the composite score number",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed breakdown",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Notes to log with this run",
    )
    args = parser.parse_args()

    verbose = args.verbose and not args.score_only

    # Compute sub-scores
    completeness = score_metric_completeness(verbose=verbose)
    comparison_depth = score_comparison_depth(verbose=verbose)
    actionability = score_actionability(verbose=verbose)

    composite = round((completeness + comparison_depth + actionability) / 3, 1)

    # Load previous run for comparison
    previous = load_previous_run()

    # Log the run
    notes = args.notes or "evaluation run"
    append_run_log(composite, completeness, comparison_depth, actionability, notes)

    if args.score_only:
        print(composite)
        return

    # Print results
    print(f"\n{'=' * 50}")
    print(f"  QA Pipeline Quality Score: {composite}")
    print(f"{'=' * 50}")
    print(f"  Completeness:      {completeness}")
    print(f"  Comparison Depth:  {comparison_depth}")
    print(f"  Actionability:     {actionability}")
    print(f"{'=' * 50}")

    if previous:
        current = {
            "composite_score": composite,
            "completeness": completeness,
            "comparison_depth": comparison_depth,
            "actionability": actionability,
        }
        print_delta_summary(current, previous)

    print(f"\nLogged to: {RUN_LOG_PATH}")


if __name__ == "__main__":
    main()
