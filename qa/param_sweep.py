"""
Vector Search Parameter Sweep

Sweeps over (top_k, min_similarity) configurations to find optimal vector
search parameters. Runs vector search once with a wide net, caches results
with similarity scores, then applies configurations post-hoc.
"""

import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID

from rich.console import Console
from rich.table import Table

from core.vector_search import VectorRetriever
from qa.eval_dataset import EvalDataset, EvalQuestion, load_eval_dataset
from qa.eval_metrics import RetrievalResult, compute_all_metrics
from utils.logger import get_logger

logger = get_logger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RankedItem:
    """A single retrieved item with its similarity score."""

    chunk_id: UUID
    article_id: UUID
    similarity: float


@dataclass
class CachedVectorResult:
    """
    Cached vector search result for one query.

    Stores the full ranked list WITH similarity scores so that
    (top_k, min_similarity) configurations can be applied post-hoc.
    """

    query: str
    expected_chunk_id: UUID
    expected_article_id: UUID
    results: list[RankedItem]  # sorted by similarity descending
    latency_ms: float
    error: Optional[str] = None


@dataclass
class SweepConfig:
    """Configuration for a vector search parameter sweep."""

    dataset_path: str = "data/qa_pairs.jsonl"
    top_k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 7, 10, 15, 20])
    min_similarity_values: list[float] = field(
        default_factory=lambda: [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    k_values_for_metrics: list[int] = field(
        default_factory=lambda: [1, 3, 5, 10, 20]
    )
    sample_size: Optional[int] = 200
    quality_filter: Optional[str] = None
    random_seed: int = 42
    fetch_top_k: int = 50
    output_dir: str = "data"
    primary_metric: str = "mrr"
    primary_level: str = "chunk"


@dataclass
class SweepResult:
    """Result of a single (top_k, min_similarity) configuration evaluation."""

    top_k: int
    min_similarity: float
    metrics: dict
    avg_results_per_query: float
    zero_result_queries: int


# ---------------------------------------------------------------------------
# Sweep orchestrator
# ---------------------------------------------------------------------------


class VectorParamSweep:
    """
    Orchestrates a parameter sweep over vector search configurations.

    Strategy:
    1. Load eval dataset
    2. Run vector search ONCE with wide net (high top_k, no min_similarity)
    3. Cache full results with similarity scores
    4. For each (top_k, min_similarity) combination, post-hoc filter
       and compute metrics
    5. Rank configurations and generate report
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        config: Optional[SweepConfig] = None,
    ):
        self.vector = vector_retriever
        self.config = config or SweepConfig()

        if self.config.fetch_top_k < max(self.config.top_k_values):
            raise ValueError(
                f"fetch_top_k ({self.config.fetch_top_k}) must be >= "
                f"max(top_k_values) ({max(self.config.top_k_values)})"
            )

    def run(self) -> dict:
        """Execute the full parameter sweep. Returns the sweep report."""
        cfg = self.config

        # Load dataset
        logger.info("Loading evaluation dataset...")
        dataset = load_eval_dataset(
            filepath=cfg.dataset_path,
            filter_sufficient_context=True,
            quality_filter=cfg.quality_filter,
            sample_size=cfg.sample_size,
            random_seed=cfg.random_seed,
        )
        console.print(
            f"Loaded {len(dataset.questions)} evaluation questions "
            f"(from {dataset.total_chunks_in_source} chunks)"
        )

        if not dataset.questions:
            logger.error("No questions in dataset after filtering")
            return {"error": "No questions in dataset after filtering"}

        total_configs = len(cfg.top_k_values) * len(cfg.min_similarity_values)
        console.print(
            f"Sweeping {len(cfg.top_k_values)} top_k x "
            f"{len(cfg.min_similarity_values)} min_similarity = "
            f"{total_configs} configurations"
        )

        # Single-pass vector search
        cached = self._run_vector_search(dataset.questions)

        # Sweep all configurations
        console.print("\n[bold]Evaluating configurations...[/bold]")
        sweep_results = self._sweep_all_configs(cached)

        # Find best configs and generate report
        best_configs = self._find_best_configs(sweep_results)
        report = self._generate_report(dataset, cached, sweep_results, best_configs)

        # Save and display
        json_path, _ = self.save_report(report, cfg.output_dir)
        console.print(f"\nReport saved to: {json_path}")
        self.print_summary(report)

        return report

    # ------------------------------------------------------------------
    # Vector search (single pass)
    # ------------------------------------------------------------------

    def _run_vector_search(
        self, questions: list[EvalQuestion]
    ) -> list[CachedVectorResult]:
        """Run vector search once for all questions with wide net."""
        console.print(
            f"\n[bold]Running vector search[/bold] "
            f"({len(questions)} queries, fetch_top_k={self.config.fetch_top_k})..."
        )

        cached: list[CachedVectorResult] = []
        total = len(questions)

        for idx, q in enumerate(questions, 1):
            start = time.time()
            try:
                search_results = self.vector.search(
                    query=q.question,
                    top_k=self.config.fetch_top_k,
                    include_system_prompts=False,
                )
                latency = (time.time() - start) * 1000

                items = [
                    RankedItem(
                        chunk_id=r.chunk.chunk_id,
                        article_id=r.chunk.parent_article_id,
                        similarity=r.similarity,
                    )
                    for r in search_results
                ]

                cached.append(
                    CachedVectorResult(
                        query=q.question,
                        expected_chunk_id=q.chunk_id,
                        expected_article_id=q.parent_article_id,
                        results=items,
                        latency_ms=latency,
                    )
                )
            except Exception as e:
                latency = (time.time() - start) * 1000
                logger.error(f"Vector search failed for query {idx}: {e}")
                cached.append(
                    CachedVectorResult(
                        query=q.question,
                        expected_chunk_id=q.chunk_id,
                        expected_article_id=q.parent_article_id,
                        results=[],
                        latency_ms=latency,
                        error=str(e),
                    )
                )

            if idx % max(1, total // 10) == 0 or idx == total:
                elapsed = sum(c.latency_ms for c in cached)
                avg = elapsed / idx
                remaining = avg * (total - idx)
                console.print(
                    f"  [{idx}/{total}] "
                    f"avg={avg:.0f}ms/query, "
                    f"~{remaining / 1000:.0f}s remaining"
                )

        return cached

    # ------------------------------------------------------------------
    # Post-hoc filtering
    # ------------------------------------------------------------------

    def _apply_config(
        self,
        cached: list[CachedVectorResult],
        top_k: int,
        min_similarity: float,
    ) -> tuple[list[RetrievalResult], int, float]:
        """
        Post-hoc filter cached results for a specific (top_k, min_similarity).

        Returns:
            (retrieval_results, zero_result_count, avg_results_per_query)
        """
        results: list[RetrievalResult] = []
        total_result_count = 0
        zero_count = 0

        for cr in cached:
            if cr.error is not None:
                results.append(
                    RetrievalResult(
                        query=cr.query,
                        expected_chunk_id=cr.expected_chunk_id,
                        expected_article_id=cr.expected_article_id,
                        retrieved_chunk_ids=[],
                        retrieved_article_ids=[],
                        retrieval_method=f"vector_k{top_k}_sim{min_similarity}",
                        latency_ms=cr.latency_ms,
                        error=cr.error,
                    )
                )
                zero_count += 1
                continue

            # Filter by similarity threshold, then truncate to top_k
            filtered = [
                item for item in cr.results if item.similarity >= min_similarity
            ]
            truncated = filtered[:top_k]

            n = len(truncated)
            total_result_count += n
            if n == 0:
                zero_count += 1

            results.append(
                RetrievalResult(
                    query=cr.query,
                    expected_chunk_id=cr.expected_chunk_id,
                    expected_article_id=cr.expected_article_id,
                    retrieved_chunk_ids=[item.chunk_id for item in truncated],
                    retrieved_article_ids=[item.article_id for item in truncated],
                    retrieval_method=f"vector_k{top_k}_sim{min_similarity}",
                    latency_ms=cr.latency_ms,
                )
            )

        valid_count = len(cached)
        avg_results = total_result_count / valid_count if valid_count else 0.0

        return results, zero_count, avg_results

    def _sweep_all_configs(
        self, cached: list[CachedVectorResult]
    ) -> list[SweepResult]:
        """Iterate over all (top_k, min_similarity) combinations."""
        cfg = self.config
        sweep_results: list[SweepResult] = []

        total = len(cfg.top_k_values) * len(cfg.min_similarity_values)
        count = 0

        for min_sim in cfg.min_similarity_values:
            for top_k in cfg.top_k_values:
                retrieval_results, zero_count, avg_results = self._apply_config(
                    cached, top_k, min_sim
                )
                metrics = compute_all_metrics(
                    retrieval_results, cfg.k_values_for_metrics
                )

                sweep_results.append(
                    SweepResult(
                        top_k=top_k,
                        min_similarity=min_sim,
                        metrics=metrics,
                        avg_results_per_query=round(avg_results, 2),
                        zero_result_queries=zero_count,
                    )
                )

                count += 1
                if count % max(1, total // 5) == 0 or count == total:
                    console.print(f"  [{count}/{total}] configurations evaluated")

        return sweep_results

    # ------------------------------------------------------------------
    # Best config identification
    # ------------------------------------------------------------------

    def _find_best_configs(self, sweep_results: list[SweepResult]) -> dict:
        """Find the best configuration for each tracked metric."""
        tracked_metrics = [
            ("chunk_mrr", "chunk_level", "mrr"),
            ("chunk_hit_rate_at_5", "chunk_level", "hit_rate", "@5"),
            ("chunk_ndcg_at_5", "chunk_level", "ndcg", "@5"),
            ("article_mrr", "article_level", "mrr"),
            ("article_hit_rate_at_5", "article_level", "hit_rate", "@5"),
            ("article_ndcg_at_5", "article_level", "ndcg", "@5"),
        ]

        best: dict[str, dict] = {}

        for entry in tracked_metrics:
            metric_key = entry[0]
            level = entry[1]
            metric_name = entry[2]
            k_label = entry[3] if len(entry) > 3 else None

            best_val = -1.0
            best_sr: Optional[SweepResult] = None

            for sr in sweep_results:
                if k_label is not None:
                    val = sr.metrics[level][metric_name].get(k_label, 0.0)
                else:
                    val = sr.metrics[level][metric_name]

                if val > best_val:
                    best_val = val
                    best_sr = sr

            if best_sr is not None:
                best[metric_key] = {
                    "top_k": best_sr.top_k,
                    "min_similarity": best_sr.min_similarity,
                    "value": round(best_val, 4),
                    "zero_result_queries": best_sr.zero_result_queries,
                }

        return best

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _compute_similarity_distribution(
        self, cached: list[CachedVectorResult]
    ) -> dict:
        """Compute similarity score percentile statistics."""
        top1_sims: list[float] = []
        top5_sims: list[float] = []

        for cr in cached:
            if cr.error or not cr.results:
                continue
            top1_sims.append(cr.results[0].similarity)
            if len(cr.results) >= 5:
                top5_sims.append(cr.results[4].similarity)

        def percentiles(values: list[float]) -> dict:
            if not values:
                return {}
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            return {
                "p10": round(sorted_vals[max(0, int(n * 0.1) - 1)], 4),
                "p25": round(sorted_vals[max(0, int(n * 0.25) - 1)], 4),
                "p50": round(statistics.median(sorted_vals), 4),
                "p75": round(sorted_vals[min(n - 1, int(n * 0.75))], 4),
                "p90": round(sorted_vals[min(n - 1, int(n * 0.9))], 4),
                "p95": round(sorted_vals[min(n - 1, int(n * 0.95))], 4),
                "min": round(sorted_vals[0], 4),
                "max": round(sorted_vals[-1], 4),
            }

        return {
            "top_1_result": percentiles(top1_sims),
            "top_5_result": percentiles(top5_sims),
        }

    def _generate_report(
        self,
        dataset: EvalDataset,
        cached: list[CachedVectorResult],
        sweep_results: list[SweepResult],
        best_configs: dict,
    ) -> dict:
        """Generate the complete sweep report."""
        cfg = self.config

        latencies = [c.latency_ms for c in cached if c.latency_ms is not None]
        total_latency = sum(latencies)
        avg_latency = total_latency / len(latencies) if latencies else 0.0

        # Build recommendation from primary metric
        primary_key = f"{cfg.primary_level}_{cfg.primary_metric}"
        # Fall back to chunk_mrr if primary key not found
        rec_source = best_configs.get(primary_key, best_configs.get("chunk_mrr", {}))

        # Find the full SweepResult for the recommendation
        rec_metrics = {}
        for sr in sweep_results:
            if (
                sr.top_k == rec_source.get("top_k")
                and sr.min_similarity == rec_source.get("min_similarity")
            ):
                rec_metrics = sr.metrics
                break

        recommendation = {
            "top_k": rec_source.get("top_k"),
            "min_similarity": rec_source.get("min_similarity"),
            "primary_metric": primary_key,
            "primary_value": rec_source.get("value"),
            "metrics": rec_metrics,
        }

        # Serialize all configurations
        all_configs = []
        for sr in sweep_results:
            all_configs.append(
                {
                    "top_k": sr.top_k,
                    "min_similarity": sr.min_similarity,
                    "avg_results_per_query": sr.avg_results_per_query,
                    "zero_result_queries": sr.zero_result_queries,
                    "metrics": sr.metrics,
                }
            )

        # Build heatmap data for key metrics
        heatmaps = self._build_heatmap_data(sweep_results)

        return {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": "vector_param_sweep",
                "dataset_file": dataset.source_file,
                "total_chunks_in_source": dataset.total_chunks_in_source,
                "total_qa_pairs_in_source": dataset.total_qa_pairs_in_source,
                "questions_evaluated": len(dataset.questions),
                "quality_distribution": dataset.quality_distribution,
                "sample_size": cfg.sample_size,
                "random_seed": cfg.random_seed,
                "fetch_top_k": cfg.fetch_top_k,
                "top_k_values": cfg.top_k_values,
                "min_similarity_values": cfg.min_similarity_values,
                "total_configurations": len(sweep_results),
                "k_values_for_metrics": cfg.k_values_for_metrics,
                "primary_metric": primary_key,
                "search_latency": {
                    "total_ms": round(total_latency, 1),
                    "avg_ms_per_query": round(avg_latency, 1),
                },
            },
            "best_configs": best_configs,
            "recommendation": recommendation,
            "similarity_distribution": self._compute_similarity_distribution(cached),
            "all_configurations": all_configs,
            "heatmaps": heatmaps,
        }

    def _build_heatmap_data(self, sweep_results: list[SweepResult]) -> dict:
        """Build heatmap grids for key metrics."""
        cfg = self.config

        # Index results for fast lookup
        lookup: dict[tuple[int, float], SweepResult] = {}
        for sr in sweep_results:
            lookup[(sr.top_k, sr.min_similarity)] = sr

        heatmap_defs = [
            ("chunk_mrr", "chunk_level", "mrr", None),
            ("chunk_hit_rate_at_5", "chunk_level", "hit_rate", "@5"),
            ("chunk_ndcg_at_5", "chunk_level", "ndcg", "@5"),
            ("chunk_hit_rate_at_1", "chunk_level", "hit_rate", "@1"),
        ]

        heatmaps = {}
        for hm_key, level, metric_name, k_label in heatmap_defs:
            rows = []
            for min_sim in cfg.min_similarity_values:
                row = []
                for top_k in cfg.top_k_values:
                    sr = lookup.get((top_k, min_sim))
                    if sr is None:
                        row.append(None)
                    elif k_label is not None:
                        row.append(sr.metrics[level][metric_name].get(k_label, 0.0))
                    else:
                        row.append(sr.metrics[level][metric_name])
                rows.append(row)

            heatmaps[hm_key] = {
                "rows_label": "min_similarity",
                "rows": cfg.min_similarity_values,
                "cols_label": "top_k",
                "cols": cfg.top_k_values,
                "values": rows,
            }

        return heatmaps

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_report(self, report: dict, output_dir: str) -> tuple[str, str]:
        """Save sweep report to JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_path / f"sweep_report_{timestamp}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        latest_path = output_path / "sweep_report_latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        return str(json_path), str(latest_path)

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def print_summary(self, report: dict) -> None:
        """Print formatted summary tables to console using rich."""
        console.print("\n[bold]=" * 60)
        console.print("[bold]VECTOR PARAMETER SWEEP RESULTS[/bold]")
        console.print("[bold]=" * 60)

        meta = report["metadata"]
        console.print(f"Questions evaluated: {meta['questions_evaluated']}")
        console.print(f"Configurations tested: {meta['total_configurations']}")
        console.print(
            f"Vector search time: {meta['search_latency']['total_ms'] / 1000:.1f}s "
            f"({meta['search_latency']['avg_ms_per_query']:.1f}ms avg)"
        )
        console.print(f"Primary metric: {meta['primary_metric']}")
        console.print()

        # Print heatmaps
        heatmaps = report.get("heatmaps", {})
        for hm_key, hm_data in heatmaps.items():
            self._print_heatmap_table(hm_key, hm_data)

        # Print best configs
        self._print_best_configs_table(report.get("best_configs", {}))

        # Print similarity distribution
        self._print_similarity_distribution(report.get("similarity_distribution", {}))

        # Print recommendation
        self._print_recommendation(report.get("recommendation", {}))

    def _print_heatmap_table(self, title: str, hm_data: dict) -> None:
        """Print a heatmap as a rich table."""
        rows_vals = hm_data["rows"]
        cols_vals = hm_data["cols"]
        values = hm_data["values"]

        # Find the max value for highlighting
        all_vals = [v for row in values for v in row if v is not None]
        max_val = max(all_vals) if all_vals else 0.0

        table = Table(title=title.replace("_", " ").title())
        table.add_column("min_sim", style="cyan", justify="right")
        for c in cols_vals:
            table.add_column(f"k={c}", justify="right")

        for i, min_sim in enumerate(rows_vals):
            row_cells = [f"{min_sim:.1f}"]
            for j, _ in enumerate(cols_vals):
                val = values[i][j]
                if val is None:
                    row_cells.append("-")
                elif val >= max_val - 0.001:
                    row_cells.append(f"[bold green]{val:.4f}[/bold green]")
                elif val >= max_val - 0.005:
                    row_cells.append(f"[green]{val:.4f}[/green]")
                else:
                    row_cells.append(f"{val:.4f}")
            table.add_row(*row_cells)

        console.print(table)
        console.print()

    def _print_best_configs_table(self, best_configs: dict) -> None:
        """Print the best configuration per metric."""
        table = Table(title="Best Configuration Per Metric")
        table.add_column("Metric", style="cyan")
        table.add_column("top_k", justify="right")
        table.add_column("min_sim", justify="right")
        table.add_column("Value", justify="right")
        table.add_column("Zero-Result Queries", justify="right")

        for metric_key, cfg in best_configs.items():
            table.add_row(
                metric_key,
                str(cfg["top_k"]),
                f"{cfg['min_similarity']:.1f}",
                f"{cfg['value']:.4f}",
                str(cfg["zero_result_queries"]),
            )

        console.print(table)
        console.print()

    def _print_similarity_distribution(self, dist: dict) -> None:
        """Print similarity score distribution."""
        if not dist:
            return

        table = Table(title="Similarity Score Distribution")
        table.add_column("Position", style="cyan")
        pct_keys = ["min", "p10", "p25", "p50", "p75", "p90", "p95", "max"]
        for key in pct_keys:
            table.add_column(key, justify="right")

        for label, data in dist.items():
            if not data:
                continue
            row = [label.replace("_", " ").title()]
            for key in pct_keys:
                val = data.get(key)
                row.append(f"{val:.4f}" if val is not None else "-")
            table.add_row(*row)

        console.print(table)
        console.print()

    def _print_recommendation(self, rec: dict) -> None:
        """Print the recommendation."""
        if not rec or rec.get("top_k") is None:
            return

        console.print("[bold]RECOMMENDATION[/bold]")
        console.print("=" * 60)
        console.print(f"  top_k          = {rec['top_k']}")
        console.print(f"  min_similarity = {rec['min_similarity']}")
        console.print(f"  {rec['primary_metric']}: {rec['primary_value']:.4f}")

        metrics = rec.get("metrics", {})
        chunk = metrics.get("chunk_level", {})
        if chunk:
            hr5 = chunk.get("hit_rate", {}).get("@5")
            ndcg5 = chunk.get("ndcg", {}).get("@5")
            if hr5 is not None:
                console.print(f"  Chunk Hit@5:   {hr5:.4f}")
            if ndcg5 is not None:
                console.print(f"  Chunk NDCG@5:  {ndcg5:.4f}")

        console.print("=" * 60)
        console.print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """Standalone entry point for running vector parameter sweep."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Vector search parameter sweep for optimal configuration"
    )
    parser.add_argument(
        "--dataset",
        default="data/qa_pairs.jsonl",
        help="Path to QA pairs JSONL file",
    )
    parser.add_argument(
        "--top-k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7, 10, 15, 20],
        help="top_k values to sweep",
    )
    parser.add_argument(
        "--min-similarity-values",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help="min_similarity thresholds to sweep",
    )
    parser.add_argument(
        "--fetch-top-k",
        type=int,
        default=50,
        help="Number of results to fetch per query (default: 50)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of questions to sample (0 for all)",
    )
    parser.add_argument(
        "--quality-filter",
        choices=["high", "medium", "low"],
        default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--primary-metric",
        choices=["mrr", "hit_rate_at_5", "ndcg_at_5"],
        default="mrr",
        help="Primary metric to rank configurations by",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20],
        help="K values for computing metrics at each configuration",
    )

    args = parser.parse_args()

    config = SweepConfig(
        dataset_path=args.dataset,
        top_k_values=args.top_k_values,
        min_similarity_values=args.min_similarity_values,
        k_values_for_metrics=args.k_values,
        sample_size=args.sample_size if args.sample_size > 0 else None,
        quality_filter=args.quality_filter,
        random_seed=args.seed,
        fetch_top_k=args.fetch_top_k,
        output_dir=args.output_dir,
        primary_metric=args.primary_metric,
    )

    vector = VectorRetriever()

    sweep = VectorParamSweep(vector_retriever=vector, config=config)
    sweep.run()


if __name__ == "__main__":
    main()
