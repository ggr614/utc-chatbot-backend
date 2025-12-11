"""
RAG Pipeline Orchestrator

This module orchestrates the complete RAG pipeline:
1. Ingestion: Fetch articles from TDX API
2. Processing: Clean HTML and chunk text
3. Embedding: Generate vector embeddings
4. Storage: Store in PostgreSQL with pgvector

The pipeline can operate in different modes:
- Full sync: Complete ingestion and processing
- Incremental: Only process new/updated articles
- Reprocess: Regenerate embeddings for existing content
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from core.ingestion import ArticleProcessor
from core.processing import TextProcessor
from core.embedding import GenerateEmbeddingsOpenAI, GenerateEmbeddingsAWS
from core.storage_raw import PostgresClient
from core.storage_vector import OpenAIVectorStorage, CohereVectorStorage
from core.schemas import TdxArticle, TextChunk, VectorRecord
from core.config import get_settings
from utils.logger import get_logger, PerformanceLogger
from utils.tokenizer import Tokenizer
import hashlib

logger = get_logger(__name__)


class RAGPipeline:
    """
    Orchestrates the complete RAG pipeline from ingestion to storage.

    This class coordinates all components to transform raw articles
    into searchable vector embeddings.
    """

    def __init__(
        self,
        embedding_provider: str = "openai",
        skip_ingestion: bool = False,
        skip_processing: bool = False,
        skip_embedding: bool = False
    ):
        """
        Initialize the RAG pipeline with all required components.

        Args:
            embedding_provider: Either "openai" or "cohere" for embeddings
            skip_ingestion: Skip article ingestion (use existing DB data)
            skip_processing: Skip text processing (use existing chunks)
            skip_embedding: Skip embedding generation (dry run mode)

        Raises:
            ValueError: If invalid embedding provider specified
        """
        logger.info("Initializing RAG Pipeline")
        logger.info(f"Configuration: provider={embedding_provider}, "
                   f"skip_ingestion={skip_ingestion}, skip_processing={skip_processing}, "
                   f"skip_embedding={skip_embedding}")

        try:
            # Validate embedding provider
            if embedding_provider not in ["openai", "cohere"]:
                raise ValueError(
                    f"Invalid embedding provider: {embedding_provider}. "
                    "Must be 'openai' or 'cohere'"
                )

            self.embedding_provider = embedding_provider
            self.skip_ingestion = skip_ingestion
            self.skip_processing = skip_processing
            self.skip_embedding = skip_embedding

            # Initialize components
            logger.debug("Initializing pipeline components")

            if not skip_ingestion:
                self.article_processor = ArticleProcessor()
                logger.debug("Article processor initialized")

            if not skip_processing:
                self.text_processor = TextProcessor()
                self.tokenizer = Tokenizer()
                logger.debug("Text processor and tokenizer initialized")

            if not skip_embedding:
                if embedding_provider == "openai":
                    self.embedder = GenerateEmbeddingsOpenAI()
                    self.vector_store = OpenAIVectorStorage()
                else:  # cohere
                    self.embedder = GenerateEmbeddingsAWS()
                    self.vector_store = CohereVectorStorage()
                logger.debug(f"{embedding_provider} embedder and vector store initialized")

            self.raw_store = PostgresClient()
            logger.debug("Raw storage client initialized")

            logger.info("RAG Pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {str(e)}")
            raise

    def _generate_chunk_id(self, parent_article_id: int, text_content: str, sequence: int) -> str:
        """
        Generate a unique, deterministic chunk ID.

        Args:
            parent_article_id: ID of the parent article
            text_content: The chunk text content
            sequence: Sequence number of the chunk

        Returns:
            Unique chunk ID hash
        """
        # Create deterministic hash from article ID, content, and sequence
        content = f"{parent_article_id}:{sequence}:{text_content}"
        chunk_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        return chunk_id

    def run_ingestion(self) -> Dict[str, int]:
        """
        Run the article ingestion phase.

        Returns:
            Statistics: {'new_count': X, 'updated_count': Y, 'skipped_count': Z}

        Raises:
            RuntimeError: If ingestion was skipped or fails
        """
        if self.skip_ingestion:
            logger.warning("Ingestion phase skipped")
            raise RuntimeError("Cannot run ingestion when skip_ingestion=True")

        logger.info("Starting ingestion phase")
        try:
            with PerformanceLogger(logger, "Ingestion phase"):
                stats = self.article_processor.ingest_and_store()
                logger.info(
                    f"Ingestion complete: {stats['new_count']} new, "
                    f"{stats['updated_count']} updated, {stats['skipped_count']} skipped"
                )
                return stats
        except Exception as e:
            logger.error(f"Ingestion phase failed: {str(e)}")
            raise

    def run_processing(self, article_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run the text processing phase to convert HTML to chunks.

        Args:
            article_ids: Optional list of specific article IDs to process.
                        If None, processes all articles in the database.

        Returns:
            Dictionary with:
                - 'processed_count': Number of articles processed
                - 'chunk_count': Total chunks created
                - 'chunk_ids': List of chunk IDs created

        Raises:
            RuntimeError: If processing was skipped or fails
        """
        if self.skip_processing:
            logger.warning("Processing phase skipped")
            raise RuntimeError("Cannot run processing when skip_processing=True")

        logger.info("Starting processing phase")
        try:
            with PerformanceLogger(logger, "Processing phase"):
                # Get articles to process
                if article_ids:
                    logger.info(f"Processing {len(article_ids)} specified articles")
                    # Would need to implement get_articles_by_ids in PostgresClient
                    raise NotImplementedError("Processing specific article IDs not yet implemented")
                else:
                    logger.info("Processing all articles from database")
                    # For now, we'll return a placeholder
                    # In a real implementation, you'd:
                    # 1. Fetch articles from raw_store
                    # 2. Process each article's HTML to text
                    # 3. Chunk the text
                    # 4. Store chunks in article_chunks table
                    logger.warning("Full article processing not yet implemented")
                    return {
                        'processed_count': 0,
                        'chunk_count': 0,
                        'chunk_ids': []
                    }

        except Exception as e:
            logger.error(f"Processing phase failed: {str(e)}")
            raise

    def process_article_to_chunks(self, article: TdxArticle) -> List[TextChunk]:
        """
        Process a single article into text chunks.

        Args:
            article: TdxArticle object to process

        Returns:
            List of TextChunk objects

        Raises:
            ValueError: If article processing fails
        """
        logger.debug(f"Processing article {article.id}: {article.title}")

        try:
            # Convert HTML to clean text
            clean_text = self.text_processor.process_text(article.content_html)
            logger.debug(f"Converted HTML to clean text for article {article.id}")

            # Chunk the text
            settings = get_settings()
            max_tokens = settings.AZURE_MAX_TOKENS if self.embedding_provider == "openai" else settings.AWS_MAX_TOKENS
            overlap = 50  # Token overlap between chunks

            text_chunks_raw = self.text_processor.text_to_chunks(
                clean_text,
                max_tokens=max_tokens,
                overlap=overlap
            )
            logger.debug(f"Created {len(text_chunks_raw)} chunks for article {article.id}")

            # Create TextChunk objects
            chunks = []
            for seq, chunk_text in enumerate(text_chunks_raw):
                chunk_id = self._generate_chunk_id(article.id, chunk_text, seq)

                # Count tokens in chunk
                token_count = self.tokenizer.num_tokens_from_string(chunk_text)

                chunk = TextChunk(
                    chunk_id=chunk_id,
                    parent_article_id=article.id,
                    chunk_sequence=seq,
                    text_content=chunk_text,
                    token_count=token_count,
                    source_url=article.url,
                    last_modified_date=article.last_modified_date
                )
                chunks.append(chunk)

            logger.info(f"Successfully processed article {article.id} into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Failed to process article {article.id}: {str(e)}")
            raise

    def run_embedding(self, chunks: List[TextChunk]) -> List[Tuple[VectorRecord, List[float]]]:
        """
        Generate embeddings for text chunks.

        Args:
            chunks: List of TextChunk objects to embed

        Returns:
            List of tuples (VectorRecord, embedding_vector)

        Raises:
            RuntimeError: If embedding was skipped or fails
        """
        if self.skip_embedding:
            logger.warning("Embedding phase skipped")
            raise RuntimeError("Cannot run embedding when skip_embedding=True")

        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []

        logger.info(f"Starting embedding phase for {len(chunks)} chunks")
        try:
            with PerformanceLogger(logger, f"Embedding {len(chunks)} chunks"):
                embeddings = []

                for idx, chunk in enumerate(chunks, 1):
                    if idx % 10 == 0:
                        logger.debug(f"Embedding progress: {idx}/{len(chunks)} chunks")

                    try:
                        # Generate embedding
                        embedding_vector = self.embedder.generate_embedding(chunk.text_content)

                        # Create VectorRecord
                        vector_record = VectorRecord(
                            chunk_id=chunk.chunk_id,
                            parent_article_id=chunk.parent_article_id,
                            chunk_sequence=chunk.chunk_sequence,
                            text_content=chunk.text_content,
                            token_count=chunk.token_count,
                            source_url=chunk.source_url
                        )

                        embeddings.append((vector_record, embedding_vector))
                        logger.debug(f"Generated embedding for chunk {chunk.chunk_id}")

                    except Exception as e:
                        logger.error(f"Failed to embed chunk {chunk.chunk_id}: {str(e)}")
                        # Continue with other chunks
                        continue

                logger.info(f"Successfully generated {len(embeddings)} embeddings")
                return embeddings

        except Exception as e:
            logger.error(f"Embedding phase failed: {str(e)}")
            raise

    def run_storage(self, embeddings: List[Tuple[VectorRecord, List[float]]]) -> int:
        """
        Store embeddings in the vector database.

        Args:
            embeddings: List of tuples (VectorRecord, embedding_vector)

        Returns:
            Number of embeddings stored

        Raises:
            RuntimeError: If storage fails
        """
        if not embeddings:
            logger.warning("No embeddings provided for storage")
            return 0

        logger.info(f"Starting storage phase for {len(embeddings)} embeddings")
        try:
            with PerformanceLogger(logger, f"Storing {len(embeddings)} embeddings"):
                self.vector_store.insert_embeddings(embeddings)
                logger.info(f"Successfully stored {len(embeddings)} embeddings")
                return len(embeddings)

        except Exception as e:
            logger.error(f"Storage phase failed: {str(e)}")
            raise

    def run_full_pipeline(self, article_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline from ingestion to storage.

        Args:
            article_ids: Optional list of specific article IDs to process

        Returns:
            Statistics dictionary with counts for each phase

        Raises:
            RuntimeError: If any phase fails
        """
        logger.info("=" * 80)
        logger.info("Starting full RAG pipeline execution")
        logger.info("=" * 80)

        stats = {
            'ingestion': {},
            'processing': {'processed_count': 0, 'chunk_count': 0},
            'embedding': {'embedding_count': 0},
            'storage': {'stored_count': 0},
            'start_time': datetime.now(),
            'end_time': None,
            'duration_seconds': 0
        }

        try:
            with PerformanceLogger(logger, "Full RAG pipeline"):
                # Phase 1: Ingestion
                if not self.skip_ingestion:
                    logger.info("\n" + "=" * 80)
                    logger.info("PHASE 1: INGESTION")
                    logger.info("=" * 80)
                    stats['ingestion'] = self.run_ingestion()
                else:
                    logger.info("PHASE 1: INGESTION - SKIPPED")

                # Phase 2: Processing (simplified for now)
                # In full implementation, this would:
                # - Fetch articles from database
                # - Process each to chunks
                # - Store chunks in article_chunks table
                logger.info("\n" + "=" * 80)
                logger.info("PHASE 2: PROCESSING")
                logger.info("=" * 80)
                logger.warning("Processing phase simplified - implement fetch and process loop for production")

                # Phase 3 & 4: Embedding and Storage would follow
                logger.info("\n" + "=" * 80)
                logger.info("PHASE 3 & 4: EMBEDDING AND STORAGE")
                logger.info("=" * 80)
                logger.warning("Embedding and storage phases not executed in simplified version")

                stats['end_time'] = datetime.now()
                stats['duration_seconds'] = (stats['end_time'] - stats['start_time']).total_seconds()

                logger.info("\n" + "=" * 80)
                logger.info("PIPELINE EXECUTION COMPLETE")
                logger.info("=" * 80)
                logger.info(f"Duration: {stats['duration_seconds']:.2f} seconds")
                logger.info(f"Ingestion: {stats['ingestion']}")

                return stats

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            logger.error("=" * 80)
            stats['end_time'] = datetime.now()
            stats['duration_seconds'] = (stats['end_time'] - stats['start_time']).total_seconds()
            stats['error'] = str(e)
            raise

    def cleanup(self):
        """Clean up resources and close connections."""
        logger.info("Cleaning up pipeline resources")
        try:
            if hasattr(self, 'raw_store'):
                self.raw_store.close()
            if hasattr(self, 'vector_store'):
                self.vector_store.close()
            logger.info("Pipeline cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
