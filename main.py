#!/usr/bin/env python3
"""
Main CLI entry point for the Helpdesk Chatbot RAG Backend.

This module provides a command-line interface for scheduled task execution
of various pipeline operations including article ingestion, processing,
embedding generation, and vector storage.

Usage examples:
    # Ingest articles from TDX API
    python main.py ingest

    # Process all articles into chunks
    python main.py process

    # Generate embeddings for processed chunks
    python main.py embed

    # Run full pipeline (ingest + process + embed)
    python main.py pipeline

    # Database migrations (use Alembic)
    alembic upgrade head
    alembic downgrade -1
"""

import argparse
import sys
from datetime import datetime

from core.pipeline import RAGPipeline
from core.ingestion import ArticleProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


def setup_argparse() -> argparse.ArgumentParser:
    """
    Configure and return the argument parser for the CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Helpdesk Chatbot RAG Backend - CLI for scheduled task execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --log-level DEBUG pipeline

  # Only ingest new articles (no processing/embedding)
  python main.py ingest

  # Process existing articles into chunks
  python main.py process

  # Generate embeddings for processed chunks
  python main.py --log-level INFO embed

  # Database migrations (use Alembic instead)
  alembic upgrade head
  alembic downgrade -1
        """,
    )

    # Add global options to parent parser
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # ========== INGEST COMMAND ==========
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest articles from TDX API into raw storage",
        description="Fetch articles from TeamDynamix API and store them in the database.",
    )
    ingest_parser.add_argument(
        "--stats", action="store_true", help="Print detailed statistics after ingestion"
    )

    # ========== PROCESS COMMAND ==========
    process_parser = subparsers.add_parser(
        "process",
        help="Process articles into text chunks",
        description="Convert HTML articles to clean text and create chunks for embedding.",
    )
    process_parser.add_argument(
        "--article-ids",
        type=int,
        nargs="+",
        metavar="ID",
        help="Specific article IDs to process (default: process all)",
    )

    # ========== EMBED COMMAND ==========
    embed_parser = subparsers.add_parser(
        "embed",
        help="Generate embeddings for text chunks",
        description="Generate vector embeddings for processed text chunks.",
    )
    embed_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="Number of chunks to process in each batch (default: 100)",
    )

    # ========== PIPELINE COMMAND ==========
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the complete RAG pipeline (ingest + process + embed)",
        description="Execute the full pipeline: ingestion, processing, and embedding generation.",
    )
    pipeline_parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip article ingestion (use existing data)",
    )
    pipeline_parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip text processing (use existing chunks)",
    )
    pipeline_parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding generation (dry run)",
    )
    pipeline_parser.add_argument(
        "--article-ids",
        type=int,
        nargs="+",
        metavar="ID",
        help="Specific article IDs to process (default: process all)",
    )
    pipeline_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="Number of chunks to embed per API call (default: 100)",
    )

    # ========== EVALUATE COMMAND ==========
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Run retrieval evaluation against QA dataset",
        description="Evaluate BM25, vector, and hybrid retrieval against ground-truth QA pairs.",
    )
    eval_parser.add_argument(
        "--dataset",
        default="data/qa_pairs.jsonl",
        help="Path to QA pairs JSONL file (default: data/qa_pairs.jsonl)",
    )
    eval_parser.add_argument(
        "--methods",
        nargs="+",
        choices=["bm25", "vector", "hybrid"],
        default=["bm25", "vector", "hybrid"],
        help="Retrieval methods to evaluate (default: all three)",
    )
    eval_parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20],
        help="K values for top-K metrics (default: 1 3 5 10 20)",
    )
    eval_parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of questions to sample (default: 200, use 0 for all)",
    )
    eval_parser.add_argument(
        "--quality-filter",
        choices=["high", "medium", "low"],
        default=None,
        help="Filter questions by quality tier (default: all)",
    )
    eval_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    eval_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for output files (default: data)",
    )

    # ========== SWEEP COMMAND ==========
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Run vector search parameter sweep",
        description="Sweep over (top_k, min_similarity) configurations to find optimal vector search parameters.",
    )
    sweep_parser.add_argument(
        "--dataset",
        default="data/qa_pairs.jsonl",
        help="Path to QA pairs JSONL file (default: data/qa_pairs.jsonl)",
    )
    sweep_parser.add_argument(
        "--top-k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7, 10, 15, 20],
        help="top_k values to sweep (default: 1 3 5 7 10 15 20)",
    )
    sweep_parser.add_argument(
        "--min-similarity-values",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help="min_similarity thresholds to sweep (default: 0.0 0.3 0.4 0.5 0.6 0.7 0.8)",
    )
    sweep_parser.add_argument(
        "--fetch-top-k",
        type=int,
        default=50,
        help="Results to fetch per query for caching (default: 50)",
    )
    sweep_parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of questions to sample (default: 200, use 0 for all)",
    )
    sweep_parser.add_argument(
        "--quality-filter",
        choices=["high", "medium", "low"],
        default=None,
        help="Filter questions by quality tier (default: all)",
    )
    sweep_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    sweep_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for output files (default: data)",
    )
    sweep_parser.add_argument(
        "--primary-metric",
        choices=["mrr", "hit_rate_at_5", "ndcg_at_5"],
        default="mrr",
        help="Primary metric to rank configurations by (default: mrr)",
    )
    sweep_parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20],
        help="K values for computing metrics (default: 1 3 5 10 20)",
    )

    return parser


def command_ingest(args: argparse.Namespace) -> int:
    """
    Execute the ingest command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("=" * 80)
    logger.info("COMMAND: INGEST ARTICLES")
    logger.info("=" * 80)

    try:
        processor = ArticleProcessor()
        stats = processor.ingest_and_store()

        logger.info("\n" + "=" * 80)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"New articles:       {stats['new_count']}")
        logger.info(f"Updated articles:   {stats['updated_count']}")
        logger.info(f"Unchanged articles: {stats['unchanged_count']}")
        logger.info(f"Skipped articles:   {stats['skipped_count']}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        return 1


def command_process(args: argparse.Namespace) -> int:
    """
    Execute the process command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("=" * 80)
    logger.info("COMMAND: PROCESS ARTICLES")
    logger.info("=" * 80)

    try:
        # Initialize pipeline with processing enabled
        pipeline = RAGPipeline(skip_ingestion=True, skip_embedding=True)

        # Run processing
        stats = pipeline.run_processing(article_ids=args.article_ids)

        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Articles processed: {stats['processed_count']}")
        logger.info(f"Chunks created:     {stats['chunk_count']}")
        logger.info("=" * 80)

        pipeline.cleanup()
        return 0

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        return 1


def command_embed(args: argparse.Namespace) -> int:
    """
    Execute the embed command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("=" * 80)
    logger.info("COMMAND: GENERATE EMBEDDINGS")
    logger.info("=" * 80)

    try:
        # Initialize pipeline with embedding enabled
        pipeline = RAGPipeline(
            skip_ingestion=True, skip_processing=True
        )

        # Get chunk count first
        chunk_count = pipeline.raw_store.get_chunk_count()

        if chunk_count == 0:
            logger.warning("No chunks found in database. Run 'process' command first.")
            pipeline.cleanup()
            return 1

        logger.info(f"Found {chunk_count} chunks in database")
        logger.info(f"Processing in batches of {args.batch_size}")

        # Fetch and process chunks in batches
        all_embeddings = []
        processed_count = 0
        offset = 0

        while offset < chunk_count:
            # Fetch batch of chunks
            logger.info(
                f"Fetching chunks {offset + 1} to {min(offset + args.batch_size, chunk_count)}"
            )
            chunks = pipeline.raw_store.get_all_chunks(
                limit=args.batch_size, offset=offset
            )

            if not chunks:
                logger.warning(f"No chunks retrieved at offset {offset}, stopping")
                break

            # Generate embeddings for this batch
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            try:
                embeddings = pipeline.run_embedding(chunks)
                all_embeddings.extend(embeddings)
                processed_count += len(chunks)

                # Store embeddings immediately after generating each batch
                if embeddings:
                    logger.info(f"Storing {len(embeddings)} embeddings")
                    stored = pipeline.run_storage(embeddings)
                    logger.info(f"Stored {stored} embeddings successfully")

            except Exception as e:
                logger.error(f"Failed to process batch at offset {offset}: {str(e)}")
                # Continue with next batch
                pass

            offset += args.batch_size

            # Progress update
            progress = min(offset, chunk_count)
            percent = (progress / chunk_count) * 100
            logger.info(f"Progress: {progress}/{chunk_count} chunks ({percent:.1f}%)")

        logger.info("\n" + "=" * 80)
        logger.info("EMBEDDING GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total chunks processed: {processed_count}")
        logger.info(f"Total embeddings generated: {len(all_embeddings)}")
        logger.info("=" * 80)

        pipeline.cleanup()
        return 0

    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
        return 1


def command_pipeline(args: argparse.Namespace) -> int:
    """
    Execute the full pipeline command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("=" * 80)
    logger.info("COMMAND: FULL PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Skip ingestion:  {args.skip_ingestion}")
    logger.info(f"Skip processing: {args.skip_processing}")
    logger.info(f"Skip embedding:  {args.skip_embedding}")
    logger.info("=" * 80 + "\n")

    try:
        # Initialize pipeline with specified configuration
        with RAGPipeline(
            skip_ingestion=args.skip_ingestion,
            skip_processing=args.skip_processing,
            skip_embedding=args.skip_embedding,
        ) as pipeline:
            # Run the full pipeline
            stats = pipeline.run_full_pipeline(
                article_ids=args.article_ids,
                batch_size=args.batch_size,
            )

            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE EXECUTION COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Duration: {stats['duration_seconds']:.2f} seconds")

            if stats.get("ingestion"):
                logger.info("\nIngestion:")
                logger.info(f"  New:       {stats['ingestion'].get('new_count', 0)}")
                logger.info(
                    f"  Updated:   {stats['ingestion'].get('updated_count', 0)}"
                )
                logger.info(
                    f"  Unchanged: {stats['ingestion'].get('unchanged_count', 0)}"
                )

            if stats.get("processing"):
                logger.info("\nProcessing:")
                logger.info(
                    f"  Articles:  {stats['processing'].get('processed_count', 0)}"
                )
                logger.info(f"  Chunks:    {stats['processing'].get('chunk_count', 0)}")

            if stats.get("embedding"):
                logger.info("\nEmbedding:")
                logger.info(
                    f"  Generated: {stats['embedding'].get('embedding_count', 0)}"
                )

            if stats.get("storage"):
                logger.info("\nStorage:")
                logger.info(f"  Stored:    {stats['storage'].get('stored_count', 0)}")

            logger.info("=" * 80)

            return 0

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return 1


def command_evaluate(args: argparse.Namespace) -> int:
    """
    Execute the evaluate command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("=" * 80)
    logger.info("COMMAND: RETRIEVAL EVALUATION")
    logger.info("=" * 80)

    try:
        from core.bm25_search import BM25Retriever
        from core.vector_search import VectorRetriever
        from core.storage_chunk import PostgresClient
        from qa.eval_runner import RetrievalEvaluator, EvalConfig

        sample_size = args.sample_size if args.sample_size > 0 else None

        config = EvalConfig(
            dataset_path=args.dataset,
            k_values=args.k_values,
            methods=args.methods,
            sample_size=sample_size,
            quality_filter=args.quality_filter,
            random_seed=args.seed,
            output_dir=args.output_dir,
        )

        # Initialize retrievers
        postgres_client = PostgresClient()
        bm25 = BM25Retriever(postgres_client=postgres_client)

        vector = None
        if "vector" in args.methods or "hybrid" in args.methods:
            vector = VectorRetriever()

        evaluator = RetrievalEvaluator(
            bm25_retriever=bm25,
            vector_retriever=vector,
            config=config,
        )
        evaluator.run()

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        return 1


def command_sweep(args: argparse.Namespace) -> int:
    """
    Execute the sweep command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("=" * 80)
    logger.info("COMMAND: VECTOR PARAMETER SWEEP")
    logger.info("=" * 80)

    try:
        from core.vector_search import VectorRetriever
        from qa.param_sweep import VectorParamSweep, SweepConfig

        sample_size = args.sample_size if args.sample_size > 0 else None

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
        )

        vector = VectorRetriever()

        sweep = VectorParamSweep(vector_retriever=vector, config=config)
        sweep.run()

        return 0

    except Exception as e:
        logger.error(f"Parameter sweep failed: {str(e)}", exc_info=True)
        return 1


def main() -> int:
    """
    Main entry point for the CLI application.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse command-line arguments
    parser = setup_argparse()
    args = parser.parse_args()

    # Configure logging level
    import logging

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Log startup information
    start_time = datetime.now()
    logger.info(f"Helpdesk Chatbot RAG Backend - Starting at {start_time.isoformat()}")
    logger.info(f"Command: {args.command or 'none'}")

    # Validate that a command was provided
    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate command handler
    exit_code = 1
    try:
        if args.command == "ingest":
            exit_code = command_ingest(args)
        elif args.command == "process":
            exit_code = command_process(args)
        elif args.command == "embed":
            exit_code = command_embed(args)
        elif args.command == "pipeline":
            exit_code = command_pipeline(args)
        elif args.command == "evaluate":
            exit_code = command_evaluate(args)
        elif args.command == "sweep":
            exit_code = command_sweep(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
            exit_code = 1

    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user")
        exit_code = 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        exit_code = 1

    # Log completion
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"\nCompleted in {duration:.2f} seconds")
    logger.info(f"Exit code: {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
