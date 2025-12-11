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
    python main.py embed --provider openai

    # Run full pipeline (ingest + process + embed)
    python main.py pipeline --provider cohere

    # Database bootstrap operations
    python main.py bootstrap --status
    python main.py bootstrap --dry-run
    python main.py bootstrap --full-reset
"""

import argparse
import sys
from datetime import datetime

from core.pipeline import RAGPipeline
from core.ingestion import ArticleProcessor
from utils.bootstrap_db import DatabaseBootstrap
from utils.logger import get_logger
from core.config import get_settings

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
  # Run full pipeline with OpenAI embeddings
  python main.py --log-level DEBUG pipeline --provider openai

  # Only ingest new articles (no processing/embedding)
  python main.py ingest

  # Process existing articles into chunks
  python main.py process

  # Generate embeddings for processed chunks
  python main.py --log-level INFO embed --provider cohere

  # Bootstrap database (check status first)
  python main.py bootstrap --status
  python main.py bootstrap --dry-run
  python main.py bootstrap
        """
    )

    # Add global options to parent parser
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # ========== INGEST COMMAND ==========
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest articles from TDX API into raw storage",
        description="Fetch articles from TeamDynamix API and store them in the database."
    )
    ingest_parser.add_argument(
        "--stats",
        action="store_true",
        help="Print detailed statistics after ingestion"
    )

    # ========== PROCESS COMMAND ==========
    process_parser = subparsers.add_parser(
        "process",
        help="Process articles into text chunks",
        description="Convert HTML articles to clean text and create chunks for embedding."
    )
    process_parser.add_argument(
        "--article-ids",
        type=int,
        nargs="+",
        metavar="ID",
        help="Specific article IDs to process (default: process all)"
    )

    # ========== EMBED COMMAND ==========
    embed_parser = subparsers.add_parser(
        "embed",
        help="Generate embeddings for text chunks",
        description="Generate vector embeddings for processed text chunks."
    )
    embed_parser.add_argument(
        "--provider",
        choices=["openai", "cohere"],
        default="openai",
        help="Embedding provider to use (default: openai)"
    )
    embed_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="Number of chunks to process in each batch (default: 100)"
    )

    # ========== PIPELINE COMMAND ==========
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the complete RAG pipeline (ingest + process + embed)",
        description="Execute the full pipeline: ingestion, processing, and embedding generation."
    )
    pipeline_parser.add_argument(
        "--provider",
        choices=["openai", "cohere"],
        default="openai",
        help="Embedding provider to use (default: openai)"
    )
    pipeline_parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip article ingestion (use existing data)"
    )
    pipeline_parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip text processing (use existing chunks)"
    )
    pipeline_parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding generation (dry run)"
    )
    pipeline_parser.add_argument(
        "--article-ids",
        type=int,
        nargs="+",
        metavar="ID",
        help="Specific article IDs to process (default: process all)"
    )

    # ========== BOOTSTRAP COMMAND ==========
    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="Bootstrap database tables and extensions",
        description="Set up or reset the database schema, tables, and pgvector extension."
    )
    bootstrap_group = bootstrap_parser.add_mutually_exclusive_group()
    bootstrap_group.add_argument(
        "--status",
        action="store_true",
        help="Check current database status"
    )
    bootstrap_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them"
    )
    bootstrap_group.add_argument(
        "--full-reset",
        action="store_true",
        help="Drop all tables and recreate (WARNING: deletes all data)"
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
        pipeline = RAGPipeline(
            skip_ingestion=True,
            skip_embedding=True
        )

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
    logger.info(f"COMMAND: GENERATE EMBEDDINGS ({args.provider.upper()})")
    logger.info("=" * 80)

    try:
        # Initialize pipeline with embedding enabled
        pipeline = RAGPipeline(
            embedding_provider=args.provider,
            skip_ingestion=True,
            skip_processing=True
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
            logger.info(f"Fetching chunks {offset + 1} to {min(offset + args.batch_size, chunk_count)}")
            chunks = pipeline.raw_store.get_all_chunks(limit=args.batch_size, offset=offset)

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
    logger.info(f"COMMAND: FULL PIPELINE ({args.provider.upper()})")
    logger.info("=" * 80)
    logger.info(f"Skip ingestion:  {args.skip_ingestion}")
    logger.info(f"Skip processing: {args.skip_processing}")
    logger.info(f"Skip embedding:  {args.skip_embedding}")
    logger.info("=" * 80 + "\n")

    try:
        # Initialize pipeline with specified configuration
        with RAGPipeline(
            embedding_provider=args.provider,
            skip_ingestion=args.skip_ingestion,
            skip_processing=args.skip_processing,
            skip_embedding=args.skip_embedding
        ) as pipeline:
            # Run the full pipeline
            stats = pipeline.run_full_pipeline(article_ids=args.article_ids)

            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE EXECUTION COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Duration: {stats['duration_seconds']:.2f} seconds")

            if stats.get('ingestion'):
                logger.info("\nIngestion:")
                logger.info(f"  New:       {stats['ingestion'].get('new_count', 0)}")
                logger.info(f"  Updated:   {stats['ingestion'].get('updated_count', 0)}")
                logger.info(f"  Unchanged: {stats['ingestion'].get('unchanged_count', 0)}")

            if stats.get('processing'):
                logger.info("\nProcessing:")
                logger.info(f"  Articles:  {stats['processing'].get('processed_count', 0)}")
                logger.info(f"  Chunks:    {stats['processing'].get('chunk_count', 0)}")

            if stats.get('embedding'):
                logger.info("\nEmbedding:")
                logger.info(f"  Generated: {stats['embedding'].get('embedding_count', 0)}")

            if stats.get('storage'):
                logger.info("\nStorage:")
                logger.info(f"  Stored:    {stats['storage'].get('stored_count', 0)}")

            logger.info("=" * 80)

            return 0

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return 1


def command_bootstrap(args: argparse.Namespace) -> int:
    """
    Execute the bootstrap command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("=" * 80)
    logger.info("COMMAND: DATABASE BOOTSTRAP")
    logger.info("=" * 80)

    try:
        if args.status:
            logger.info("Checking database status...")
            bootstrap = DatabaseBootstrap(dry_run=False)
            bootstrap.check_status()

        elif args.dry_run:
            logger.info("Running in DRY-RUN mode (no changes will be made)")
            bootstrap = DatabaseBootstrap(dry_run=True)
            bootstrap.setup_database()

        elif args.full_reset:
            logger.warning("=" * 80)
            logger.warning("WARNING: FULL RESET MODE")
            logger.warning("This will DELETE ALL DATA in the database!")
            logger.warning("=" * 80)
            response = input("Type 'yes' to confirm full reset: ")
            if response.lower() == "yes":
                bootstrap = DatabaseBootstrap(dry_run=False)
                bootstrap.setup_database(full_reset=True)
                logger.info("Database has been reset and recreated")
            else:
                logger.info("Reset cancelled")
                return 0

        else:
            logger.info("Bootstrapping database...")
            bootstrap = DatabaseBootstrap(dry_run=False)
            bootstrap.setup_database()
            logger.info("Database bootstrap complete")

        return 0

    except Exception as e:
        logger.error(f"Bootstrap failed: {str(e)}", exc_info=True)
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

    # Validate configuration
    try:
        get_settings()
        logger.debug("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Configuration error: {str(e)}")
        logger.error("Please ensure all required environment variables are set in .env file")
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
        elif args.command == "bootstrap":
            exit_code = command_bootstrap(args)
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
