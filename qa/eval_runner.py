"""
Retrieval Evaluation Orchestrator

Runs evaluation queries through BM25, vector, and hybrid retrieval methods,
computes IR metrics, and generates comparison reports.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from core.bm25_search import BM25Retriever, BM25SearchResult
from core.hyde_generator import HyDEGenerator
from core.reranker import Reranker
from core.vector_search import VectorRetriever, VectorSearchResult
from api.utils.hybrid_search import hybrid_search, reciprocal_rank_fusion
from qa.eval_dataset import EvalDataset, EvalQuestion, load_eval_dataset
from qa.eval_metrics import RetrievalResult, compute_all_metrics
from utils.logger import get_logger

logger = get_logger(__name__)
console = Console()


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    dataset_path: str = "data/qa_pairs.jsonl"
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    methods: list[str] = field(default_factory=lambda: ["bm25", "vector", "hybrid"])
    sample_size: Optional[int] = 200
    quality_filter: Optional[str] = None
    random_seed: int = 42
    max_top_k: int = 20
    output_dir: str = "data"
    rrf_k: int = 60


class RetrievalEvaluator:
    """
    Runs retrieval evaluation against the QA dataset.

    Accepts retrievers via dependency injection.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: Optional[VectorRetriever] = None,
        hyde_generator: Optional[HyDEGenerator] = None,
        reranker: Optional[Reranker] = None,
        config: Optional[EvalConfig] = None,
    ):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.hyde = hyde_generator
        self.reranker = reranker
        self.config = config or EvalConfig()

    def run(self) -> dict:
        """
        Execute the full evaluation pipeline.

        Returns:
            Complete evaluation report as dict
        """
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

        # Ensure max_top_k covers all k_values
        cfg.max_top_k = max(cfg.max_top_k, max(cfg.k_values))

        # Run evaluations per method
        results_by_method: dict[str, list[RetrievalResult]] = {}

        for method in cfg.methods:
            if method == "bm25":
                results_by_method["bm25"] = self._evaluate_bm25(dataset.questions)
            elif method == "vector":
                if self.vector is None:
                    logger.warning(
                        "Skipping vector evaluation: no VectorRetriever provided"
                    )
                    continue
                results_by_method["vector"] = self._evaluate_vector(dataset.questions)
            elif method == "hybrid":
                if self.vector is None:
                    logger.warning(
                        "Skipping hybrid evaluation: no VectorRetriever provided"
                    )
                    continue
                results_by_method["hybrid"] = self._evaluate_hybrid(dataset.questions)
            elif method == "hybrid_reranked":
                if self.vector is None:
                    logger.warning(
                        "Skipping hybrid_reranked evaluation: no VectorRetriever provided"
                    )
                    continue
                if self.reranker is None:
                    logger.warning(
                        "Skipping hybrid_reranked evaluation: no Reranker provided"
                    )
                    continue
                results_by_method["hybrid_reranked"] = self._evaluate_hybrid_reranked(
                    dataset.questions
                )
            elif method == "hyde":
                if self.vector is None:
                    logger.warning(
                        "Skipping hyde evaluation: no VectorRetriever provided"
                    )
                    continue
                if self.hyde is None:
                    logger.warning(
                        "Skipping hyde evaluation: no HyDEGenerator provided"
                    )
                    continue
                results_by_method["hyde"] = self._evaluate_hyde(dataset.questions)
            else:
                logger.warning(f"Unknown method '{method}', skipping")

        # Compute metrics per method
        metrics_by_method: dict[str, dict] = {}
        for method, results in results_by_method.items():
            metrics_by_method[method] = compute_all_metrics(results, cfg.k_values)

        # Compute quality tier breakdown
        quality_breakdown = self._compute_quality_breakdown(
            dataset.questions, results_by_method
        )

        # Generate report
        report = self._generate_report(
            dataset, results_by_method, metrics_by_method, quality_breakdown
        )

        # Save and display
        json_path, summary_path = self.save_report(report, cfg.output_dir)
        console.print(f"\nReport saved to: {json_path}")
        self.print_summary(report)

        return report

    def _evaluate_bm25(self, questions: list[EvalQuestion]) -> list[RetrievalResult]:
        """Run all questions through BM25 using batch_search."""
        console.print(f"\n[bold]Evaluating BM25[/bold] ({len(questions)} queries)...")

        query_strings = [q.question for q in questions]

        start = time.time()
        batch_results = self.bm25.batch_search(
            queries=query_strings,
            top_k=self.config.max_top_k,
        )
        total_ms = (time.time() - start) * 1000
        avg_ms = total_ms / len(questions) if questions else 0

        console.print(
            f"  BM25 batch complete: {total_ms:.0f}ms total, "
            f"{avg_ms:.1f}ms avg per query"
        )

        results = []
        for q in questions:
            search_results = batch_results.get(q.question, [])
            results.append(
                self._build_retrieval_result(
                    question=q,
                    bm25_results=search_results,
                    method="bm25",
                    latency_ms=avg_ms,
                )
            )

        return results

    def _evaluate_vector(self, questions: list[EvalQuestion]) -> list[RetrievalResult]:
        """Run all questions through vector search sequentially."""
        console.print(f"\n[bold]Evaluating Vector[/bold] ({len(questions)} queries)...")

        results = []
        total = len(questions)

        for idx, q in enumerate(questions, 1):
            start = time.time()
            try:
                search_results = self.vector.search(
                    query=q.question,
                    top_k=self.config.max_top_k,
                    include_system_prompts=False,
                )
                latency = (time.time() - start) * 1000
                results.append(
                    self._build_retrieval_result(
                        question=q,
                        vector_results=search_results,
                        method="vector",
                        latency_ms=latency,
                    )
                )
            except Exception as e:
                latency = (time.time() - start) * 1000
                logger.error(f"Vector search failed for query {idx}: {e}")
                results.append(
                    RetrievalResult(
                        query=q.question,
                        expected_chunk_id=q.chunk_id,
                        expected_article_id=q.parent_article_id,
                        retrieved_chunk_ids=[],
                        retrieved_article_ids=[],
                        retrieval_method="vector",
                        latency_ms=latency,
                        error=str(e),
                    )
                )

            # Progress reporting
            if idx % max(1, total // 10) == 0 or idx == total:
                elapsed = sum(r.latency_ms for r in results if r.latency_ms is not None)
                avg = elapsed / idx
                remaining = avg * (total - idx)
                console.print(
                    f"  [{idx}/{total}] "
                    f"avg={avg:.0f}ms/query, "
                    f"~{remaining / 1000:.0f}s remaining"
                )

        return results

    def _evaluate_hybrid(self, questions: list[EvalQuestion]) -> list[RetrievalResult]:
        """Run all questions through hybrid search (RRF-only, no reranker)."""
        console.print(
            f"\n[bold]Evaluating Hybrid (RRF)[/bold] ({len(questions)} queries)..."
        )

        results = []
        total = len(questions)

        for idx, q in enumerate(questions, 1):
            start = time.time()
            try:
                hybrid_results, _metadata = hybrid_search(
                    query=q.question,
                    bm25_retriever=self.bm25,
                    vector_retriever=self.vector,
                    reranker=None,
                    top_k=self.config.max_top_k,
                    rrf_k=self.config.rrf_k,
                )
                latency = (time.time() - start) * 1000
                results.append(
                    self._build_retrieval_result(
                        question=q,
                        hybrid_results=hybrid_results,
                        method="hybrid",
                        latency_ms=latency,
                    )
                )
            except Exception as e:
                latency = (time.time() - start) * 1000
                logger.error(f"Hybrid search failed for query {idx}: {e}")
                results.append(
                    RetrievalResult(
                        query=q.question,
                        expected_chunk_id=q.chunk_id,
                        expected_article_id=q.parent_article_id,
                        retrieved_chunk_ids=[],
                        retrieved_article_ids=[],
                        retrieval_method="hybrid",
                        latency_ms=latency,
                        error=str(e),
                    )
                )

            if idx % max(1, total // 10) == 0 or idx == total:
                elapsed = sum(r.latency_ms for r in results if r.latency_ms is not None)
                avg = elapsed / idx
                remaining = avg * (total - idx)
                console.print(
                    f"  [{idx}/{total}] "
                    f"avg={avg:.0f}ms/query, "
                    f"~{remaining / 1000:.0f}s remaining"
                )

        return results

    def _evaluate_hybrid_reranked(
        self, questions: list[EvalQuestion]
    ) -> list[RetrievalResult]:
        """Run all questions through hybrid search with Cohere reranking."""
        console.print(
            f"\n[bold]Evaluating Hybrid (Reranked)[/bold] ({len(questions)} queries)..."
        )

        results = []
        total = len(questions)

        for idx, q in enumerate(questions, 1):
            start = time.time()
            try:
                hybrid_results, _metadata = hybrid_search(
                    query=q.question,
                    bm25_retriever=self.bm25,
                    vector_retriever=self.vector,
                    reranker=self.reranker,
                    top_k=self.config.max_top_k,
                    rrf_k=self.config.rrf_k,
                )
                latency = (time.time() - start) * 1000
                results.append(
                    self._build_retrieval_result(
                        question=q,
                        hybrid_results=hybrid_results,
                        method="hybrid_reranked",
                        latency_ms=latency,
                    )
                )
            except Exception as e:
                latency = (time.time() - start) * 1000
                logger.error(f"Hybrid reranked search failed for query {idx}: {e}")
                results.append(
                    RetrievalResult(
                        query=q.question,
                        expected_chunk_id=q.chunk_id,
                        expected_article_id=q.parent_article_id,
                        retrieved_chunk_ids=[],
                        retrieved_article_ids=[],
                        retrieval_method="hybrid_reranked",
                        latency_ms=latency,
                        error=str(e),
                    )
                )

            if idx % max(1, total // 10) == 0 or idx == total:
                elapsed = sum(r.latency_ms for r in results if r.latency_ms is not None)
                avg = elapsed / idx
                remaining = avg * (total - idx)
                console.print(
                    f"  [{idx}/{total}] "
                    f"avg={avg:.0f}ms/query, "
                    f"~{remaining / 1000:.0f}s remaining"
                )

        return results

    def _evaluate_hyde(self, questions: list[EvalQuestion]) -> list[RetrievalResult]:
        """Run all questions through HyDE search (BM25 + HyDE-vector + RRF, no reranker)."""
        console.print(
            f"\n[bold]Evaluating HyDE (RRF)[/bold] ({len(questions)} queries)..."
        )

        results = []
        total = len(questions)

        for idx, q in enumerate(questions, 1):
            start = time.time()
            try:
                # Step 1: Generate hypothetical document
                try:
                    hypothetical_doc, _token_usage = (
                        self.hyde.generate_hypothetical_document_sync(
                            query=q.question,
                        )
                    )
                except Exception as hyde_err:
                    # Graceful degradation: fall back to original query
                    logger.warning(
                        f"HyDE generation failed for query {idx}, "
                        f"falling back to original query: {hyde_err}"
                    )
                    hypothetical_doc = q.question

                # Step 2: BM25 search with original query
                bm25_results = self.bm25.search(
                    query=q.question,
                    top_k=self.config.max_top_k,
                )

                # Step 3: Vector search with hypothetical document
                vector_results = self.vector.search(
                    query=hypothetical_doc,
                    top_k=self.config.max_top_k,
                    include_system_prompts=False,
                )

                # Step 4: RRF fusion
                fused_results = reciprocal_rank_fusion(
                    bm25_results=bm25_results,
                    vector_results=vector_results,
                    k=self.config.rrf_k,
                )

                latency = (time.time() - start) * 1000
                results.append(
                    self._build_retrieval_result(
                        question=q,
                        hybrid_results=fused_results,
                        method="hyde",
                        latency_ms=latency,
                    )
                )
            except Exception as e:
                latency = (time.time() - start) * 1000
                logger.error(f"HyDE search failed for query {idx}: {e}")
                results.append(
                    RetrievalResult(
                        query=q.question,
                        expected_chunk_id=q.chunk_id,
                        expected_article_id=q.parent_article_id,
                        retrieved_chunk_ids=[],
                        retrieved_article_ids=[],
                        retrieval_method="hyde",
                        latency_ms=latency,
                        error=str(e),
                    )
                )

            if idx % max(1, total // 10) == 0 or idx == total:
                elapsed = sum(r.latency_ms for r in results if r.latency_ms is not None)
                avg = elapsed / idx
                remaining = avg * (total - idx)
                console.print(
                    f"  [{idx}/{total}] "
                    f"avg={avg:.0f}ms/query, "
                    f"~{remaining / 1000:.0f}s remaining"
                )

        return results

    def _build_retrieval_result(
        self,
        question: EvalQuestion,
        bm25_results: Optional[list[BM25SearchResult]] = None,
        vector_results: Optional[list[VectorSearchResult]] = None,
        hybrid_results: Optional[list[dict]] = None,
        method: str = "",
        latency_ms: float = 0.0,
    ) -> RetrievalResult:
        """
        Normalize retriever-specific results into a uniform RetrievalResult.

        Handles:
        - BM25SearchResult: .chunk.chunk_id, .chunk.parent_article_id
        - VectorSearchResult: .chunk.chunk_id, .chunk.parent_article_id
        - Hybrid dict: dict["chunk"].chunk_id, dict["chunk"].parent_article_id
        """
        chunk_ids = []
        article_ids = []

        if bm25_results is not None:
            for r in bm25_results:
                chunk_ids.append(r.chunk.chunk_id)
                article_ids.append(r.chunk.parent_article_id)
        elif vector_results is not None:
            for r in vector_results:
                chunk_ids.append(r.chunk.chunk_id)
                article_ids.append(r.chunk.parent_article_id)
        elif hybrid_results is not None:
            for r in hybrid_results:
                chunk = r["chunk"]
                chunk_ids.append(chunk.chunk_id)
                article_ids.append(chunk.parent_article_id)

        return RetrievalResult(
            query=question.question,
            expected_chunk_id=question.chunk_id,
            expected_article_id=question.parent_article_id,
            retrieved_chunk_ids=chunk_ids,
            retrieved_article_ids=article_ids,
            retrieval_method=method,
            latency_ms=latency_ms,
        )

    def _compute_quality_breakdown(
        self,
        questions: list[EvalQuestion],
        results_by_method: dict[str, list[RetrievalResult]],
    ) -> dict:
        """Compute metrics broken down by quality tier."""
        # Group question indices by quality
        quality_groups: dict[str, list[int]] = {}
        for idx, q in enumerate(questions):
            quality_groups.setdefault(q.overall_quality, []).append(idx)

        breakdown = {}
        for quality, indices in quality_groups.items():
            breakdown[quality] = {"question_count": len(indices)}
            for method, all_results in results_by_method.items():
                tier_results = [all_results[i] for i in indices]
                breakdown[quality][method] = compute_all_metrics(
                    tier_results, self.config.k_values
                )

        return breakdown

    def _generate_report(
        self,
        dataset: EvalDataset,
        results_by_method: dict[str, list[RetrievalResult]],
        metrics_by_method: dict[str, dict],
        quality_breakdown: dict,
    ) -> dict:
        """Generate the complete evaluation report."""
        return {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset_file": dataset.source_file,
                "total_chunks_in_source": dataset.total_chunks_in_source,
                "total_qa_pairs_in_source": dataset.total_qa_pairs_in_source,
                "questions_after_filtering": dataset.filtered_count,
                "questions_evaluated": len(dataset.questions),
                "quality_distribution": dataset.quality_distribution,
                "k_values": self.config.k_values,
                "methods_evaluated": list(metrics_by_method.keys()),
                "sample_size": self.config.sample_size,
                "random_seed": self.config.random_seed,
                "rrf_k": self.config.rrf_k,
            },
            "results": metrics_by_method,
            "quality_breakdown": quality_breakdown,
        }

    def save_report(self, report: dict, output_dir: str) -> tuple[str, str]:
        """Save evaluation report to JSON and text summary."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_path / f"eval_report_{timestamp}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        # Also write a latest symlink-style copy
        latest_path = output_path / "eval_report_latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        return str(json_path), str(latest_path)

    def print_summary(self, report: dict) -> None:
        """Print formatted summary tables to console using rich."""
        results = report.get("results", {})
        k_values = report["metadata"]["k_values"]
        methods = list(results.keys())

        if not methods:
            console.print("[yellow]No results to display[/yellow]")
            return

        console.print("\n[bold]=" * 60)
        console.print("[bold]RETRIEVAL EVALUATION RESULTS[/bold]")
        console.print("[bold]=" * 60)

        meta = report["metadata"]
        console.print(f"Questions evaluated: {meta['questions_evaluated']}")
        console.print(f"Methods: {', '.join(methods)}")
        console.print(f"K values: {k_values}")
        console.print()

        # Chunk-level Hit Rate table
        self._print_metric_table(
            "Hit Rate @ K (Chunk Level)",
            methods,
            k_values,
            results,
            level="chunk_level",
            metric="hit_rate",
        )

        # Article-level Hit Rate table
        self._print_metric_table(
            "Hit Rate @ K (Article Level)",
            methods,
            k_values,
            results,
            level="article_level",
            metric="hit_rate",
        )

        # MRR table
        mrr_table = Table(title="Mean Reciprocal Rank (MRR)")
        mrr_table.add_column("Method", style="cyan")
        mrr_table.add_column("Chunk-level", justify="right")
        mrr_table.add_column("Article-level", justify="right")
        for method in methods:
            chunk_mrr = results[method]["chunk_level"]["mrr"]
            article_mrr = results[method]["article_level"]["mrr"]
            mrr_table.add_row(
                method.upper(),
                f"{chunk_mrr:.4f}",
                f"{article_mrr:.4f}",
            )
        console.print(mrr_table)
        console.print()

        # NDCG table
        self._print_metric_table(
            "NDCG @ K (Chunk Level)",
            methods,
            k_values,
            results,
            level="chunk_level",
            metric="ndcg",
        )

        # Latency summary
        latency_table = Table(title="Latency Summary")
        latency_table.add_column("Method", style="cyan")
        latency_table.add_column("Avg Latency (ms)", justify="right")
        latency_table.add_column("Total Queries", justify="right")
        latency_table.add_column("Failed", justify="right")
        for method in methods:
            m = results[method]
            avg = m.get("avg_latency_ms")
            avg_str = f"{avg:.1f}" if avg is not None else "N/A"
            latency_table.add_row(
                method.upper(),
                avg_str,
                str(m["total_queries"]),
                str(m["failed_queries"]),
            )
        console.print(latency_table)

    def _print_metric_table(
        self,
        title: str,
        methods: list[str],
        k_values: list[int],
        results: dict,
        level: str,
        metric: str,
    ) -> None:
        """Print a metric comparison table."""
        table = Table(title=title)
        table.add_column("Method", style="cyan")
        for k in k_values:
            table.add_column(f"@{k}", justify="right")

        for method in methods:
            row = [method.upper()]
            for k in k_values:
                val = results[method][level][metric].get(f"@{k}", 0)
                row.append(f"{val:.4f}")
            table.add_row(*row)

        console.print(table)
        console.print()


def main():
    """Standalone entry point for running evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation against QA dataset"
    )
    parser.add_argument(
        "--dataset",
        default="data/qa_pairs.jsonl",
        help="Path to QA pairs JSONL file",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["bm25", "vector", "hybrid", "hybrid_reranked", "hyde"],
        default=["bm25", "vector", "hybrid"],
    )
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument(
        "--quality-filter", choices=["high", "medium", "low"], default=None
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data")

    args = parser.parse_args()

    config = EvalConfig(
        dataset_path=args.dataset,
        k_values=args.k_values,
        methods=args.methods,
        sample_size=args.sample_size,
        quality_filter=args.quality_filter,
        random_seed=args.seed,
        output_dir=args.output_dir,
    )

    # Initialize retrievers
    from core.storage_chunk import PostgresClient

    postgres_client = PostgresClient()
    bm25 = BM25Retriever(postgres_client=postgres_client)

    needs_vector = {"vector", "hybrid", "hybrid_reranked", "hyde"} & set(args.methods)
    vector = VectorRetriever() if needs_vector else None

    hyde_generator = None
    if "hyde" in args.methods:
        hyde_generator = HyDEGenerator()

    reranker = None
    if "hybrid_reranked" in args.methods:
        try:
            reranker = Reranker()
            logger.info("Reranker initialized for evaluation")
        except Exception as e:
            logger.error(f"Failed to initialize Reranker: {e}")
            console.print(
                f"[red]Failed to initialize reranker: {e}[/red]\n"
                "hybrid_reranked method will be skipped."
            )

    evaluator = RetrievalEvaluator(
        bm25_retriever=bm25,
        vector_retriever=vector,
        hyde_generator=hyde_generator,
        reranker=reranker,
        config=config,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
