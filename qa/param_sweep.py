"""
Parameter Sweep for Search Configuration Optimization

Contains two sweep orchestrators:
- VectorParamSweep: Sweeps (top_k, min_similarity) for vector search.
- HybridParamSweep: Sweeps (rrf_k, top_k) for hybrid search, with optional
  Cohere reranker comparison.

Both use a single-pass caching strategy: run expensive searches once, then
apply configurations post-hoc.
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

from core.bm25_search import BM25Retriever, BM25SearchResult
from core.reranker import Reranker
from core.vector_search import VectorRetriever, VectorSearchResult
from api.utils.hybrid_search import reciprocal_rank_fusion
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
    raw_results: list = field(default_factory=list)


@dataclass
class SweepConfig:
    """Configuration for a vector search parameter sweep."""

    dataset_path: str = "data/qa_pairs.jsonl"
    top_k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 7, 10, 15, 20])
    min_similarity_values: list[float] = field(
        default_factory=lambda: [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    k_values_for_metrics: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    sample_size: Optional[int] = 200
    quality_filter: Optional[str] = None
    random_seed: int = 42
    fetch_top_k: int = 50
    output_dir: str = "data"
    primary_metric: str = "mrr"
    primary_level: str = "chunk"
    include_reranker: bool = False


@dataclass
class SweepResult:
    """Result of a single (top_k, min_similarity) configuration evaluation."""

    top_k: int
    min_similarity: float
    metrics: dict
    avg_results_per_query: float
    zero_result_queries: int
    reranked: bool = False


@dataclass
class HybridSweepConfig:
    """Configuration for a hybrid search parameter sweep."""

    dataset_path: str = "data/qa_pairs.jsonl"
    rrf_k_values: list[int] = field(default_factory=lambda: [1, 10, 30, 60, 100])
    top_k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    k_values_for_metrics: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    fetch_top_k: int = 50
    sample_size: Optional[int] = 200
    quality_filter: Optional[str] = None
    random_seed: int = 42
    output_dir: str = "data"
    primary_metric: str = "mrr"
    primary_level: str = "chunk"
    include_reranker: bool = True


@dataclass
class CachedHybridResult:
    """
    Cached raw BM25 + vector results for one query.

    Stores the full result lists so that RRF fusion can be applied
    post-hoc with different rrf_k values.
    """

    query: str
    expected_chunk_id: UUID
    expected_article_id: UUID
    bm25_results: list[BM25SearchResult]
    vector_results: list[VectorSearchResult]
    search_latency_ms: float
    error: Optional[str] = None


@dataclass
class HybridSweepResult:
    """Result of a single (rrf_k, top_k, reranked) configuration evaluation."""

    rrf_k: int
    top_k: int
    reranked: bool
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
        reranker: Optional[Reranker] = None,
    ):
        self.vector = vector_retriever
        self.reranker = reranker
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

        reranker_active = self.reranker is not None and cfg.include_reranker
        multiplier = 2 if reranker_active else 1
        total_configs = len(cfg.top_k_values) * len(cfg.min_similarity_values) * multiplier
        console.print(
            f"Sweeping {len(cfg.top_k_values)} top_k x "
            f"{len(cfg.min_similarity_values)} min_similarity"
            f"{' x 2 (reranked/not)' if reranker_active else ''}"
            f" = {total_configs} configurations"
        )

        # Single-pass vector search
        cached = self._run_vector_search(dataset.questions)

        # Optional reranking
        reranked = None
        reranker_latency = 0.0
        if reranker_active:
            rerank_start = time.time()
            reranked = self._run_reranking(cached)
            reranker_latency = (time.time() - rerank_start) * 1000

        # Sweep all configurations
        console.print("\n[bold]Evaluating configurations...[/bold]")
        sweep_results = self._sweep_all_configs(cached, reranked)

        # Find best configs and generate report
        best_configs = self._find_best_configs(sweep_results)
        report = self._generate_report(
            dataset, cached, sweep_results, best_configs, reranker_latency
        )

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
                        raw_results=search_results,
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
    # Reranking (optional single pass)
    # ------------------------------------------------------------------

    def _run_reranking(
        self, cached: list[CachedVectorResult]
    ) -> list[list[RankedItem]]:
        """Rerank all cached vector results using the reranker.

        Returns parallel list of RankedItem lists in reranked order.
        Original similarity scores are preserved for min_similarity filtering.
        """
        console.print(
            f"\n[bold]Reranking vector results[/bold] ({len(cached)} queries)..."
        )
        reranked_all: list[list[RankedItem]] = []
        total = len(cached)

        for idx, cr in enumerate(cached, 1):
            if cr.error or not cr.raw_results:
                reranked_all.append([])
                continue

            # Build similarity map from original results
            sim_map = {r.chunk.chunk_id: r.similarity for r in cr.raw_results}

            # Convert to reranker input format
            reranker_input = [
                {"rank": r.rank, "combined_score": r.similarity, "chunk": r.chunk}
                for r in cr.raw_results
            ]

            try:
                reranked = self.reranker.rerank(query=cr.query, results=reranker_input)

                # Convert back to RankedItem, preserving original similarity
                reranked_items = [
                    RankedItem(
                        chunk_id=r["chunk"].chunk_id,
                        article_id=r["chunk"].parent_article_id,
                        similarity=sim_map.get(r["chunk"].chunk_id, 0.0),
                    )
                    for r in reranked
                ]
                reranked_all.append(reranked_items)
            except Exception as e:
                logger.error(f"Reranking failed for query {idx}: {e}")
                # Fall back to original order
                reranked_all.append(cr.results)

            if idx % max(1, total // 10) == 0 or idx == total:
                console.print(f"  [{idx}/{total}] reranked")

        return reranked_all

    # ------------------------------------------------------------------
    # Post-hoc filtering
    # ------------------------------------------------------------------

    def _apply_config(
        self,
        cached: list[CachedVectorResult],
        top_k: int,
        min_similarity: float,
        reranked_items: list[list[RankedItem]] | None = None,
    ) -> tuple[list[RetrievalResult], int, float]:
        """
        Post-hoc filter cached results for a specific (top_k, min_similarity).

        Args:
            cached: Cached vector search results.
            top_k: Number of results to return.
            min_similarity: Minimum similarity threshold.
            reranked_items: Optional parallel list of reranked RankedItem lists.
                If provided, uses reranked ordering instead of original.

        Returns:
            (retrieval_results, zero_result_count, avg_results_per_query)
        """
        is_reranked = reranked_items is not None
        method_prefix = "vector_reranked" if is_reranked else "vector"
        results: list[RetrievalResult] = []
        total_result_count = 0
        zero_count = 0

        for i, cr in enumerate(cached):
            if cr.error is not None:
                results.append(
                    RetrievalResult(
                        query=cr.query,
                        expected_chunk_id=cr.expected_chunk_id,
                        expected_article_id=cr.expected_article_id,
                        retrieved_chunk_ids=[],
                        retrieved_article_ids=[],
                        retrieval_method=f"{method_prefix}_k{top_k}_sim{min_similarity}",
                        latency_ms=cr.latency_ms,
                        error=cr.error,
                    )
                )
                zero_count += 1
                continue

            # Use reranked items if provided, otherwise original
            items = reranked_items[i] if is_reranked else cr.results

            # Filter by similarity threshold, then truncate to top_k
            filtered = [
                item for item in items if item.similarity >= min_similarity
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
                    retrieval_method=f"{method_prefix}_k{top_k}_sim{min_similarity}",
                    latency_ms=cr.latency_ms,
                )
            )

        valid_count = len(cached)
        avg_results = total_result_count / valid_count if valid_count else 0.0

        return results, zero_count, avg_results

    def _sweep_all_configs(
        self,
        cached: list[CachedVectorResult],
        reranked: list[list[RankedItem]] | None = None,
    ) -> list[SweepResult]:
        """Iterate over all (top_k, min_similarity) combinations."""
        cfg = self.config
        sweep_results: list[SweepResult] = []
        reranker_active = reranked is not None
        multiplier = 2 if reranker_active else 1

        total = len(cfg.top_k_values) * len(cfg.min_similarity_values) * multiplier
        count = 0

        for min_sim in cfg.min_similarity_values:
            for top_k in cfg.top_k_values:
                # Non-reranked
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
                        reranked=False,
                    )
                )

                count += 1
                if count % max(1, total // 5) == 0 or count == total:
                    console.print(f"  [{count}/{total}] configurations evaluated")

                # Reranked
                if reranker_active:
                    retrieval_results, zero_count, avg_results = self._apply_config(
                        cached, top_k, min_sim, reranked_items=reranked
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
                            reranked=True,
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
                    "reranked": best_sr.reranked,
                    "value": round(best_val, 4),
                    "zero_result_queries": best_sr.zero_result_queries,
                }

        return best

    # ------------------------------------------------------------------
    # Reranker impact analysis
    # ------------------------------------------------------------------

    def _compute_reranker_impact(self, sweep_results: list[SweepResult]) -> dict:
        """
        Compare reranked vs non-reranked at each (top_k, min_similarity) pair.

        Returns dict keyed by min_similarity with per-metric deltas.
        """
        lookup: dict[tuple[int, float, bool], SweepResult] = {}
        for sr in sweep_results:
            lookup[(sr.top_k, sr.min_similarity, sr.reranked)] = sr

        impact: dict[str, dict] = {}
        min_sim_values = sorted(set(sr.min_similarity for sr in sweep_results))
        top_k_values = sorted(set(sr.top_k for sr in sweep_results))

        impact_metrics = [
            ("chunk_mrr", "chunk_level", "mrr", None),
            ("chunk_hit_rate_at_5", "chunk_level", "hit_rate", "@5"),
            ("chunk_ndcg_at_5", "chunk_level", "ndcg", "@5"),
            ("article_mrr", "article_level", "mrr", None),
        ]

        for min_sim in min_sim_values:
            sim_key = f"min_sim_{min_sim}"
            sim_impact: dict[str, dict] = {}

            for top_k in top_k_values:
                base = lookup.get((top_k, min_sim, False))
                reranked = lookup.get((top_k, min_sim, True))
                if base is None or reranked is None:
                    continue

                tk_key = f"top_k_{top_k}"
                tk_impact: dict[str, dict] = {}

                for m_key, level, metric_name, k_label in impact_metrics:
                    if k_label is not None:
                        base_val = base.metrics[level][metric_name].get(k_label, 0.0)
                        reranked_val = reranked.metrics[level][metric_name].get(
                            k_label, 0.0
                        )
                    else:
                        base_val = base.metrics[level][metric_name]
                        reranked_val = reranked.metrics[level][metric_name]

                    tk_impact[m_key] = {
                        "without_reranker": round(base_val, 4),
                        "with_reranker": round(reranked_val, 4),
                        "delta": round(reranked_val - base_val, 4),
                    }

                sim_impact[tk_key] = tk_impact

            impact[sim_key] = sim_impact

        return impact

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
        reranker_latency_ms: float = 0.0,
    ) -> dict:
        """Generate the complete sweep report."""
        cfg = self.config

        latencies = [c.latency_ms for c in cached if c.latency_ms is not None]
        total_latency = sum(latencies)
        avg_latency = total_latency / len(latencies) if latencies else 0.0

        reranker_active = self.reranker is not None and cfg.include_reranker
        num_queries = len(cached)

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
                and sr.reranked == rec_source.get("reranked", False)
            ):
                rec_metrics = sr.metrics
                break

        recommendation = {
            "top_k": rec_source.get("top_k"),
            "min_similarity": rec_source.get("min_similarity"),
            "reranked": rec_source.get("reranked", False),
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
                    "reranked": sr.reranked,
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
                "reranker_included": reranker_active,
                "reranker_latency": {
                    "total_ms": round(reranker_latency_ms, 1),
                    "avg_ms_per_query": (
                        round(reranker_latency_ms / num_queries, 1)
                        if num_queries
                        else 0.0
                    ),
                },
            },
            "best_configs": best_configs,
            "recommendation": recommendation,
            "similarity_distribution": self._compute_similarity_distribution(cached),
            "reranker_impact": (
                self._compute_reranker_impact(sweep_results)
                if reranker_active
                else {}
            ),
            "all_configurations": all_configs,
            "heatmaps": heatmaps,
        }

    def _build_heatmap_data(self, sweep_results: list[SweepResult]) -> dict:
        """Build heatmap grids for key metrics, separate for reranked/non-reranked."""
        cfg = self.config

        # Index results for fast lookup
        lookup: dict[tuple[int, float, bool], SweepResult] = {}
        for sr in sweep_results:
            lookup[(sr.top_k, sr.min_similarity, sr.reranked)] = sr

        has_reranked = any(sr.reranked for sr in sweep_results)

        heatmap_defs = [
            ("chunk_mrr", "chunk_level", "mrr", None),
            ("chunk_hit_rate_at_5", "chunk_level", "hit_rate", "@5"),
            ("chunk_ndcg_at_5", "chunk_level", "ndcg", "@5"),
            ("chunk_hit_rate_at_1", "chunk_level", "hit_rate", "@1"),
        ]

        heatmaps = {}

        for reranked_flag in [False, True] if has_reranked else [False]:
            prefix = "reranked_" if reranked_flag else ""

            for hm_key, level, metric_name, k_label in heatmap_defs:
                rows = []
                for min_sim in cfg.min_similarity_values:
                    row = []
                    for top_k in cfg.top_k_values:
                        sr = lookup.get((top_k, min_sim, reranked_flag))
                        if sr is None:
                            row.append(None)
                        elif k_label is not None:
                            row.append(
                                sr.metrics[level][metric_name].get(k_label, 0.0)
                            )
                        else:
                            row.append(sr.metrics[level][metric_name])
                    rows.append(row)

                heatmaps[f"{prefix}{hm_key}"] = {
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
        if meta.get("reranker_included"):
            console.print(
                f"Reranker time: {meta['reranker_latency']['total_ms'] / 1000:.1f}s "
                f"({meta['reranker_latency']['avg_ms_per_query']:.1f}ms avg)"
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

        # Print reranker impact
        if report.get("reranker_impact"):
            self._print_reranker_impact(report["reranker_impact"])

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
        has_reranked = any(cfg.get("reranked") for cfg in best_configs.values())

        table = Table(title="Best Configuration Per Metric")
        table.add_column("Metric", style="cyan")
        table.add_column("top_k", justify="right")
        table.add_column("min_sim", justify="right")
        if has_reranked:
            table.add_column("Reranked", justify="right")
        table.add_column("Value", justify="right")
        table.add_column("Zero-Result Queries", justify="right")

        for metric_key, cfg in best_configs.items():
            row = [
                metric_key,
                str(cfg["top_k"]),
                f"{cfg['min_similarity']:.1f}",
            ]
            if has_reranked:
                row.append(str(cfg.get("reranked", False)))
            row.extend([
                f"{cfg['value']:.4f}",
                str(cfg["zero_result_queries"]),
            ])
            table.add_row(*row)

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

    def _print_reranker_impact(self, impact: dict) -> None:
        """Print reranker impact analysis."""
        table = Table(title="Reranker Impact (chunk_mrr delta)")
        table.add_column("min_sim", style="cyan", justify="right")
        table.add_column("top_k", justify="right")
        table.add_column("Vector Only", justify="right")
        table.add_column("Reranked", justify="right")
        table.add_column("Delta", justify="right")

        for sim_key, top_k_data in sorted(impact.items()):
            for tk_key, metrics in sorted(top_k_data.items()):
                mrr_data = metrics.get("chunk_mrr", {})
                if not mrr_data:
                    continue
                delta = mrr_data["delta"]
                delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
                style = "[green]" if delta > 0 else "[red]" if delta < 0 else ""
                end_style = (
                    "[/green]" if delta > 0 else "[/red]" if delta < 0 else ""
                )

                table.add_row(
                    sim_key.replace("min_sim_", ""),
                    tk_key.replace("top_k_", ""),
                    f"{mrr_data['without_reranker']:.4f}",
                    f"{mrr_data['with_reranker']:.4f}",
                    f"{style}{delta_str}{end_style}",
                )

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
        if "reranked" in rec:
            console.print(f"  reranked       = {rec['reranked']}")
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


def cli_main():
    """Unified CLI entry point for all parameter sweeps."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parameter sweep for search configuration optimization"
    )
    parser.add_argument(
        "mode",
        choices=["vector", "hybrid", "vector-rerank", "hybrid-rerank"],
        help="Sweep mode: vector, hybrid, vector-rerank, or hybrid-rerank",
    )

    # Common arguments
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

    # Vector-specific arguments
    parser.add_argument(
        "--min-similarity-values",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help="min_similarity thresholds to sweep (vector modes only)",
    )

    # Hybrid-specific arguments
    parser.add_argument(
        "--rrf-k-values",
        type=int,
        nargs="+",
        default=[1, 10, 30, 60, 100],
        help="RRF k values to sweep (hybrid modes only)",
    )

    args = parser.parse_args()

    mode = args.mode
    use_reranker = mode.endswith("-rerank")
    is_hybrid = mode.startswith("hybrid")

    sample_size = args.sample_size if args.sample_size > 0 else None

    # Initialize reranker if needed
    reranker = None
    if use_reranker:
        try:
            reranker = Reranker()
            logger.info("Reranker initialized for sweep")
        except Exception as e:
            logger.error(f"Failed to initialize Reranker: {e}")
            console.print(
                f"[red]Failed to initialize reranker: {e}[/red]\n"
                "Running sweep without reranker."
            )
            use_reranker = False

    if is_hybrid:
        from core.storage_chunk import PostgresClient

        config = HybridSweepConfig(
            dataset_path=args.dataset,
            rrf_k_values=args.rrf_k_values,
            top_k_values=args.top_k_values,
            k_values_for_metrics=args.k_values,
            sample_size=sample_size,
            quality_filter=args.quality_filter,
            random_seed=args.seed,
            fetch_top_k=args.fetch_top_k,
            output_dir=args.output_dir,
            primary_metric=args.primary_metric,
            include_reranker=use_reranker,
        )

        postgres_client = PostgresClient()
        bm25 = BM25Retriever(postgres_client=postgres_client)
        vector = VectorRetriever()

        sweep = HybridParamSweep(
            bm25_retriever=bm25,
            vector_retriever=vector,
            reranker=reranker,
            config=config,
        )
        sweep.run()
    else:
        config = SweepConfig(
            dataset_path=args.dataset,
            top_k_values=args.top_k_values,
            min_similarity_values=args.min_similarity_values,
            k_values_for_metrics=args.k_values,
            sample_size=sample_size,
            quality_filter=args.quality_filter,
            random_seed=args.seed,
            fetch_top_k=args.fetch_top_k,
            output_dir=args.output_dir,
            primary_metric=args.primary_metric,
            include_reranker=use_reranker,
        )

        vector = VectorRetriever()

        sweep = VectorParamSweep(
            vector_retriever=vector,
            config=config,
            reranker=reranker,
        )
        sweep.run()


# ---------------------------------------------------------------------------
# Hybrid parameter sweep
# ---------------------------------------------------------------------------


class HybridParamSweep:
    """
    Orchestrates a parameter sweep over hybrid search configurations.

    Strategy:
    1. Load eval dataset
    2. Run BM25 (batch) and vector search ONCE with wide net
    3. Cache raw results per query
    4. For each rrf_k value, apply RRF fusion (CPU-only)
    5. Optionally rerank fused results per rrf_k (one API call per query)
    6. For each top_k, truncate and compute metrics
    7. Rank configurations and generate report with reranker impact analysis
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        reranker: Optional[Reranker] = None,
        config: Optional[HybridSweepConfig] = None,
    ):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.reranker = reranker
        self.config = config or HybridSweepConfig()

        if self.config.fetch_top_k < max(self.config.top_k_values):
            raise ValueError(
                f"fetch_top_k ({self.config.fetch_top_k}) must be >= "
                f"max(top_k_values) ({max(self.config.top_k_values)})"
            )

    def run(self) -> dict:
        """Execute the full hybrid parameter sweep. Returns the sweep report."""
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

        reranker_active = self.reranker is not None and cfg.include_reranker
        multiplier = 2 if reranker_active else 1
        total_configs = len(cfg.rrf_k_values) * len(cfg.top_k_values) * multiplier
        console.print(
            f"Sweeping {len(cfg.rrf_k_values)} rrf_k x "
            f"{len(cfg.top_k_values)} top_k"
            f"{' x 2 (reranked/not)' if reranker_active else ''}"
            f" = {total_configs} configurations"
        )

        # Single-pass searches
        cached = self._run_searches(dataset.questions)

        # Sweep all configurations
        console.print("\n[bold]Evaluating configurations...[/bold]")
        sweep_results, reranker_latency_ms = self._sweep_all_configs(cached)

        # Find best configs and generate report
        best_configs = self._find_best_configs(sweep_results)
        report = self._generate_report(
            dataset, cached, sweep_results, best_configs, reranker_latency_ms
        )

        # Save and display
        json_path, _ = self.save_report(report, cfg.output_dir)
        console.print(f"\nReport saved to: {json_path}")
        self.print_summary(report)

        return report

    # ------------------------------------------------------------------
    # Search (single pass)
    # ------------------------------------------------------------------

    def _run_searches(self, questions: list[EvalQuestion]) -> list[CachedHybridResult]:
        """Run BM25 batch search and vector search once for all questions."""
        console.print(
            f"\n[bold]Running BM25 + vector searches[/bold] "
            f"({len(questions)} queries, fetch_top_k={self.config.fetch_top_k})..."
        )

        # BM25 batch search
        query_strings = [q.question for q in questions]
        bm25_start = time.time()
        try:
            bm25_batch = self.bm25.batch_search(
                queries=query_strings,
                top_k=self.config.fetch_top_k,
            )
            bm25_ms = (time.time() - bm25_start) * 1000
            console.print(f"  BM25 batch complete: {bm25_ms:.0f}ms")
        except Exception as e:
            logger.error(f"BM25 batch search failed: {e}")
            bm25_batch = {}
            bm25_ms = (time.time() - bm25_start) * 1000

        # Vector search per query
        cached: list[CachedHybridResult] = []
        total = len(questions)

        for idx, q in enumerate(questions, 1):
            bm25_results = bm25_batch.get(q.question, [])

            start = time.time()
            try:
                vector_results = self.vector.search(
                    query=q.question,
                    top_k=self.config.fetch_top_k,
                    include_system_prompts=False,
                )
                latency = (time.time() - start) * 1000

                cached.append(
                    CachedHybridResult(
                        query=q.question,
                        expected_chunk_id=q.chunk_id,
                        expected_article_id=q.parent_article_id,
                        bm25_results=bm25_results,
                        vector_results=vector_results,
                        search_latency_ms=latency,
                    )
                )
            except Exception as e:
                latency = (time.time() - start) * 1000
                logger.error(f"Vector search failed for query {idx}: {e}")
                cached.append(
                    CachedHybridResult(
                        query=q.question,
                        expected_chunk_id=q.chunk_id,
                        expected_article_id=q.parent_article_id,
                        bm25_results=bm25_results,
                        vector_results=[],
                        search_latency_ms=latency,
                        error=str(e),
                    )
                )

            if idx % max(1, total // 10) == 0 or idx == total:
                elapsed = sum(c.search_latency_ms for c in cached)
                avg = elapsed / idx
                remaining = avg * (total - idx)
                console.print(
                    f"  [{idx}/{total}] "
                    f"avg={avg:.0f}ms/query, "
                    f"~{remaining / 1000:.0f}s remaining"
                )

        return cached

    # ------------------------------------------------------------------
    # Fusion and reranking
    # ------------------------------------------------------------------

    def _fuse_results(
        self, cached: list[CachedHybridResult], rrf_k: int
    ) -> list[list[dict]]:
        """Apply RRF fusion for all cached queries with a given rrf_k."""
        fused: list[list[dict]] = []
        for cr in cached:
            if cr.error is not None:
                fused.append([])
                continue
            fused.append(
                reciprocal_rank_fusion(
                    bm25_results=cr.bm25_results,
                    vector_results=cr.vector_results,
                    k=rrf_k,
                )
            )
        return fused

    def _rerank_fused(self, query: str, fused_results: list[dict]) -> list[dict]:
        """Rerank a single query's fused results. Returns reranked list."""
        if not fused_results or self.reranker is None:
            return fused_results if fused_results else []
        try:
            return self.reranker.rerank(query=query, results=fused_results)
        except Exception as e:
            logger.error(f"Reranking failed for query '{query[:50]}': {e}")
            return fused_results  # fallback to RRF order

    @staticmethod
    def _apply_top_k(
        results: list[dict],
        top_k: int,
        cached_entry: CachedHybridResult,
        method_label: str,
    ) -> RetrievalResult:
        """Truncate fused/reranked results to top_k and build RetrievalResult."""
        truncated = results[:top_k]

        chunk_ids = []
        article_ids = []
        for r in truncated:
            chunk = r["chunk"]
            chunk_ids.append(chunk.chunk_id)
            article_ids.append(chunk.parent_article_id)

        return RetrievalResult(
            query=cached_entry.query,
            expected_chunk_id=cached_entry.expected_chunk_id,
            expected_article_id=cached_entry.expected_article_id,
            retrieved_chunk_ids=chunk_ids,
            retrieved_article_ids=article_ids,
            retrieval_method=method_label,
            latency_ms=cached_entry.search_latency_ms,
            error=cached_entry.error,
        )

    # ------------------------------------------------------------------
    # Sweep all configurations
    # ------------------------------------------------------------------

    def _sweep_all_configs(
        self, cached: list[CachedHybridResult]
    ) -> tuple[list[HybridSweepResult], float]:
        """
        Iterate over all (rrf_k, top_k, reranked) combinations.

        Returns:
            (sweep_results, total_reranker_latency_ms)
        """
        cfg = self.config
        sweep_results: list[HybridSweepResult] = []
        reranker_active = self.reranker is not None and cfg.include_reranker
        multiplier = 2 if reranker_active else 1
        total = len(cfg.rrf_k_values) * len(cfg.top_k_values) * multiplier
        count = 0
        total_reranker_latency = 0.0

        for rrf_k in cfg.rrf_k_values:
            # RRF fusion for this rrf_k (CPU-only)
            fused = self._fuse_results(cached, rrf_k)

            # Non-reranked configs
            for top_k in cfg.top_k_values:
                retrieval_results = [
                    self._apply_top_k(f, top_k, c, f"hybrid_rrf{rrf_k}_k{top_k}")
                    for f, c in zip(fused, cached)
                ]
                metrics = compute_all_metrics(
                    retrieval_results, cfg.k_values_for_metrics
                )

                total_result_count = sum(
                    len(r.retrieved_chunk_ids) for r in retrieval_results
                )
                valid_count = len(cached)
                avg_results = total_result_count / valid_count if valid_count else 0.0
                zero_count = sum(
                    1 for r in retrieval_results if not r.retrieved_chunk_ids
                )

                sweep_results.append(
                    HybridSweepResult(
                        rrf_k=rrf_k,
                        top_k=top_k,
                        reranked=False,
                        metrics=metrics,
                        avg_results_per_query=round(avg_results, 2),
                        zero_result_queries=zero_count,
                    )
                )
                count += 1
                if count % max(1, total // 5) == 0 or count == total:
                    console.print(f"  [{count}/{total}] configurations evaluated")

            # Reranked configs
            if reranker_active:
                console.print(f"  Reranking for rrf_k={rrf_k}...")
                reranked_fused: list[list[dict]] = []
                for f, c in zip(fused, cached):
                    rerank_start = time.time()
                    reranked = self._rerank_fused(c.query, f)
                    total_reranker_latency += (time.time() - rerank_start) * 1000
                    reranked_fused.append(reranked)

                for top_k in cfg.top_k_values:
                    retrieval_results = [
                        self._apply_top_k(
                            r, top_k, c, f"hybrid_reranked_rrf{rrf_k}_k{top_k}"
                        )
                        for r, c in zip(reranked_fused, cached)
                    ]
                    metrics = compute_all_metrics(
                        retrieval_results, cfg.k_values_for_metrics
                    )

                    total_result_count = sum(
                        len(r.retrieved_chunk_ids) for r in retrieval_results
                    )
                    valid_count = len(cached)
                    avg_results = (
                        total_result_count / valid_count if valid_count else 0.0
                    )
                    zero_count = sum(
                        1 for r in retrieval_results if not r.retrieved_chunk_ids
                    )

                    sweep_results.append(
                        HybridSweepResult(
                            rrf_k=rrf_k,
                            top_k=top_k,
                            reranked=True,
                            metrics=metrics,
                            avg_results_per_query=round(avg_results, 2),
                            zero_result_queries=zero_count,
                        )
                    )
                    count += 1
                    if count % max(1, total // 5) == 0 or count == total:
                        console.print(f"  [{count}/{total}] configurations evaluated")

        return sweep_results, total_reranker_latency

    # ------------------------------------------------------------------
    # Best config identification
    # ------------------------------------------------------------------

    def _find_best_configs(self, sweep_results: list[HybridSweepResult]) -> dict:
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
            best_sr: Optional[HybridSweepResult] = None

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
                    "rrf_k": best_sr.rrf_k,
                    "top_k": best_sr.top_k,
                    "reranked": best_sr.reranked,
                    "value": round(best_val, 4),
                    "zero_result_queries": best_sr.zero_result_queries,
                }

        return best

    # ------------------------------------------------------------------
    # Reranker impact analysis
    # ------------------------------------------------------------------

    def _compute_reranker_impact(self, sweep_results: list[HybridSweepResult]) -> dict:
        """
        Compare reranked vs non-reranked at each (rrf_k, top_k) pair.

        Returns dict keyed by rrf_k with per-metric deltas.
        """
        # Index results for lookup
        lookup: dict[tuple[int, int, bool], HybridSweepResult] = {}
        for sr in sweep_results:
            lookup[(sr.rrf_k, sr.top_k, sr.reranked)] = sr

        impact: dict[str, dict] = {}
        rrf_k_values = sorted(set(sr.rrf_k for sr in sweep_results))
        top_k_values = sorted(set(sr.top_k for sr in sweep_results))

        impact_metrics = [
            ("chunk_mrr", "chunk_level", "mrr", None),
            ("chunk_hit_rate_at_5", "chunk_level", "hit_rate", "@5"),
            ("chunk_ndcg_at_5", "chunk_level", "ndcg", "@5"),
            ("article_mrr", "article_level", "mrr", None),
        ]

        for rrf_k in rrf_k_values:
            rrf_key = f"rrf_k_{rrf_k}"
            rrf_impact: dict[str, dict] = {}

            for top_k in top_k_values:
                base = lookup.get((rrf_k, top_k, False))
                reranked = lookup.get((rrf_k, top_k, True))
                if base is None or reranked is None:
                    continue

                tk_key = f"top_k_{top_k}"
                tk_impact: dict[str, dict] = {}

                for m_key, level, metric_name, k_label in impact_metrics:
                    if k_label is not None:
                        base_val = base.metrics[level][metric_name].get(k_label, 0.0)
                        reranked_val = reranked.metrics[level][metric_name].get(
                            k_label, 0.0
                        )
                    else:
                        base_val = base.metrics[level][metric_name]
                        reranked_val = reranked.metrics[level][metric_name]

                    tk_impact[m_key] = {
                        "without_reranker": round(base_val, 4),
                        "with_reranker": round(reranked_val, 4),
                        "delta": round(reranked_val - base_val, 4),
                    }

                rrf_impact[tk_key] = tk_impact

            impact[rrf_key] = rrf_impact

        return impact

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_report(
        self,
        dataset: EvalDataset,
        cached: list[CachedHybridResult],
        sweep_results: list[HybridSweepResult],
        best_configs: dict,
        reranker_latency_ms: float,
    ) -> dict:
        """Generate the complete hybrid sweep report."""
        cfg = self.config

        latencies = [c.search_latency_ms for c in cached if c.search_latency_ms]
        total_latency = sum(latencies)
        avg_latency = total_latency / len(latencies) if latencies else 0.0

        reranker_active = self.reranker is not None and cfg.include_reranker
        num_queries = len(cached)

        # Build recommendation from primary metric
        primary_key = f"{cfg.primary_level}_{cfg.primary_metric}"
        rec_source = best_configs.get(primary_key, best_configs.get("chunk_mrr", {}))

        # Find the full HybridSweepResult for the recommendation
        rec_metrics = {}
        for sr in sweep_results:
            if (
                sr.rrf_k == rec_source.get("rrf_k")
                and sr.top_k == rec_source.get("top_k")
                and sr.reranked == rec_source.get("reranked")
            ):
                rec_metrics = sr.metrics
                break

        recommendation = {
            "rrf_k": rec_source.get("rrf_k"),
            "top_k": rec_source.get("top_k"),
            "reranked": rec_source.get("reranked"),
            "primary_metric": primary_key,
            "primary_value": rec_source.get("value"),
            "metrics": rec_metrics,
        }

        # Serialize all configurations
        all_configs = []
        for sr in sweep_results:
            all_configs.append(
                {
                    "rrf_k": sr.rrf_k,
                    "top_k": sr.top_k,
                    "reranked": sr.reranked,
                    "avg_results_per_query": sr.avg_results_per_query,
                    "zero_result_queries": sr.zero_result_queries,
                    "metrics": sr.metrics,
                }
            )

        # Build heatmap data
        heatmaps = self._build_heatmap_data(sweep_results)

        # Reranker impact
        reranker_impact = {}
        if reranker_active:
            reranker_impact = self._compute_reranker_impact(sweep_results)

        return {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": "hybrid_param_sweep",
                "dataset_file": dataset.source_file,
                "total_chunks_in_source": dataset.total_chunks_in_source,
                "total_qa_pairs_in_source": dataset.total_qa_pairs_in_source,
                "questions_evaluated": len(dataset.questions),
                "quality_distribution": dataset.quality_distribution,
                "sample_size": cfg.sample_size,
                "random_seed": cfg.random_seed,
                "fetch_top_k": cfg.fetch_top_k,
                "rrf_k_values": cfg.rrf_k_values,
                "top_k_values": cfg.top_k_values,
                "total_configurations": len(sweep_results),
                "reranker_included": reranker_active,
                "k_values_for_metrics": cfg.k_values_for_metrics,
                "primary_metric": primary_key,
                "search_latency": {
                    "total_ms": round(total_latency, 1),
                    "avg_ms_per_query": round(avg_latency, 1),
                },
                "reranker_latency": {
                    "total_ms": round(reranker_latency_ms, 1),
                    "avg_ms_per_query": (
                        round(reranker_latency_ms / num_queries, 1)
                        if num_queries
                        else 0.0
                    ),
                },
            },
            "best_configs": best_configs,
            "recommendation": recommendation,
            "reranker_impact": reranker_impact,
            "all_configurations": all_configs,
            "heatmaps": heatmaps,
        }

    def _build_heatmap_data(self, sweep_results: list[HybridSweepResult]) -> dict:
        """Build heatmap grids for key metrics, separate for reranked/non-reranked."""
        cfg = self.config

        # Index results for fast lookup
        lookup: dict[tuple[int, int, bool], HybridSweepResult] = {}
        for sr in sweep_results:
            lookup[(sr.rrf_k, sr.top_k, sr.reranked)] = sr

        has_reranked = any(sr.reranked for sr in sweep_results)

        heatmap_defs = [
            ("chunk_mrr", "chunk_level", "mrr", None),
            ("chunk_hit_rate_at_5", "chunk_level", "hit_rate", "@5"),
            ("chunk_ndcg_at_5", "chunk_level", "ndcg", "@5"),
            ("chunk_hit_rate_at_1", "chunk_level", "hit_rate", "@1"),
        ]

        heatmaps = {}

        for reranked_flag in [False, True] if has_reranked else [False]:
            prefix = "reranked_" if reranked_flag else "rrf_"

            for hm_key, level, metric_name, k_label in heatmap_defs:
                rows = []
                for rrf_k in cfg.rrf_k_values:
                    row = []
                    for top_k in cfg.top_k_values:
                        sr = lookup.get((rrf_k, top_k, reranked_flag))
                        if sr is None:
                            row.append(None)
                        elif k_label is not None:
                            row.append(sr.metrics[level][metric_name].get(k_label, 0.0))
                        else:
                            row.append(sr.metrics[level][metric_name])
                    rows.append(row)

                heatmaps[f"{prefix}{hm_key}"] = {
                    "rows_label": "rrf_k",
                    "rows": cfg.rrf_k_values,
                    "cols_label": "top_k",
                    "cols": cfg.top_k_values,
                    "values": rows,
                }

        return heatmaps

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_report(self, report: dict, output_dir: str) -> tuple[str, str]:
        """Save hybrid sweep report to JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_path / f"hybrid_sweep_report_{timestamp}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        latest_path = output_path / "hybrid_sweep_report_latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        return str(json_path), str(latest_path)

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def print_summary(self, report: dict) -> None:
        """Print formatted summary tables to console using rich."""
        console.print("\n[bold]=" * 60)
        console.print("[bold]HYBRID PARAMETER SWEEP RESULTS[/bold]")
        console.print("[bold]=" * 60)

        meta = report["metadata"]
        console.print(f"Questions evaluated: {meta['questions_evaluated']}")
        console.print(f"Configurations tested: {meta['total_configurations']}")
        console.print(
            f"Search time: {meta['search_latency']['total_ms'] / 1000:.1f}s "
            f"({meta['search_latency']['avg_ms_per_query']:.1f}ms avg)"
        )
        if meta["reranker_included"]:
            console.print(
                f"Reranker time: {meta['reranker_latency']['total_ms'] / 1000:.1f}s "
                f"({meta['reranker_latency']['avg_ms_per_query']:.1f}ms avg)"
            )
        console.print(f"Primary metric: {meta['primary_metric']}")
        console.print()

        # Print heatmaps
        heatmaps = report.get("heatmaps", {})
        for hm_key, hm_data in heatmaps.items():
            self._print_heatmap_table(hm_key, hm_data)

        # Print best configs
        self._print_best_configs_table(report.get("best_configs", {}))

        # Print reranker impact
        if report.get("reranker_impact"):
            self._print_reranker_impact(report["reranker_impact"])

        # Print recommendation
        self._print_recommendation(report.get("recommendation", {}))

    def _print_heatmap_table(self, title: str, hm_data: dict) -> None:
        """Print a heatmap as a rich table."""
        rows_vals = hm_data["rows"]
        cols_vals = hm_data["cols"]
        values = hm_data["values"]

        all_vals = [v for row in values for v in row if v is not None]
        max_val = max(all_vals) if all_vals else 0.0

        table = Table(title=title.replace("_", " ").title())
        table.add_column(hm_data["rows_label"], style="cyan", justify="right")
        for c in cols_vals:
            table.add_column(f"k={c}", justify="right")

        for i, row_label in enumerate(rows_vals):
            row_cells = [str(row_label)]
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
        table.add_column("rrf_k", justify="right")
        table.add_column("top_k", justify="right")
        table.add_column("Reranked", justify="right")
        table.add_column("Value", justify="right")

        for metric_key, cfg in best_configs.items():
            table.add_row(
                metric_key,
                str(cfg["rrf_k"]),
                str(cfg["top_k"]),
                str(cfg["reranked"]),
                f"{cfg['value']:.4f}",
            )

        console.print(table)
        console.print()

    def _print_reranker_impact(self, impact: dict) -> None:
        """Print reranker impact analysis."""
        table = Table(title="Reranker Impact (chunk_mrr delta)")
        table.add_column("rrf_k", style="cyan", justify="right")
        table.add_column("top_k", justify="right")
        table.add_column("RRF Only", justify="right")
        table.add_column("Reranked", justify="right")
        table.add_column("Delta", justify="right")

        for rrf_key, top_k_data in sorted(impact.items()):
            for tk_key, metrics in sorted(top_k_data.items()):
                mrr_data = metrics.get("chunk_mrr", {})
                if not mrr_data:
                    continue
                delta = mrr_data["delta"]
                delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
                style = "[green]" if delta > 0 else "[red]" if delta < 0 else ""
                end_style = "[/green]" if delta > 0 else "[/red]" if delta < 0 else ""

                table.add_row(
                    rrf_key.replace("rrf_k_", ""),
                    tk_key.replace("top_k_", ""),
                    f"{mrr_data['without_reranker']:.4f}",
                    f"{mrr_data['with_reranker']:.4f}",
                    f"{style}{delta_str}{end_style}",
                )

        console.print(table)
        console.print()

    def _print_recommendation(self, rec: dict) -> None:
        """Print the recommendation."""
        if not rec or rec.get("rrf_k") is None:
            return

        console.print("[bold]RECOMMENDATION[/bold]")
        console.print("=" * 60)
        console.print(f"  rrf_k    = {rec['rrf_k']}")
        console.print(f"  top_k    = {rec['top_k']}")
        console.print(f"  reranked = {rec['reranked']}")
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


if __name__ == "__main__":
    cli_main()
