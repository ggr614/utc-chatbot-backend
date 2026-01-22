"""
Vector storage module for managing embeddings in PostgreSQL with pgvector.

This module provides clients for storing and retrieving vector embeddings
from both OpenAI (3072 dimensions) and AWS Cohere (1536 dimensions) models.
"""

from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple
from core.config import get_settings
import psycopg
from psycopg import Connection
from core.schemas import VectorRecord
from utils.logger import get_logger, PerformanceLogger

logger = get_logger(__name__)


class VectorStorageClient:
    """
    Base class for vector storage operations.

    Handles connection management and common database operations
    for vector embeddings storage.
    """

    def __init__(self, table_name: str, embedding_dim: int):
        """
        Initialize the vector storage client.

        Args:
            table_name: Name of the embeddings table (e.g., 'embeddings_openai')
            embedding_dim: Expected dimension of embeddings (e.g., 3072 for OpenAI)
        """
        logger.info(f"Initializing VectorStorageClient for table '{table_name}'")
        try:
            settings = get_settings()

            # Validate configuration
            if not settings.DB_HOST:
                raise ValueError("DB_HOST is not configured")
            if not settings.DB_USER:
                raise ValueError("DB_USER is not configured")
            if not settings.DB_NAME:
                raise ValueError("DB_NAME is not configured")

            self.db_host = settings.DB_HOST
            self.db_user = settings.DB_USER
            self.db_password = settings.DB_PASSWORD.get_secret_value()
            self.db_name = settings.DB_NAME
            self.db_port = 5432
            self.table_name = table_name
            self.embedding_dim = embedding_dim
            self._conn: Optional[Connection] = None

            self._connection_params = {
                "host": self.db_host,
                "user": self.db_user,
                "password": self.db_password,
                "dbname": self.db_name,
                "port": self.db_port,
            }

            logger.info(
                f"VectorStorageClient configured for {self.db_host}:{self.db_port}/{self.db_name} "
                f"(table: {self.table_name}, dim: {self.embedding_dim})"
            )
            logger.debug(f"Database user: {self.db_user}")
        except Exception as e:
            logger.error(f"Failed to initialize VectorStorageClient: {str(e)}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Get a database connection with automatic cleanup.

        Yields:
            Connection: PostgreSQL connection object

        Raises:
            ConnectionError: If unable to connect to database
        """
        conn = None
        try:
            logger.debug(f"Connecting to database {self.db_name}@{self.db_host}")
            conn = psycopg.connect(**self._connection_params)
            logger.debug("Database connection established successfully")
            yield conn
            conn.commit()
            logger.debug("Transaction committed successfully")
        except psycopg.OperationalError as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {str(e)}")
            raise ConnectionError(f"Unable to connect to database: {e}") from e
        except psycopg.Error as e:
            if conn:
                conn.rollback()
                logger.warning("Transaction rolled back due to error")
            logger.error(f"Database error: {str(e)}")
            raise ConnectionError(f"Database error: {e}") from e
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Unexpected error during database operation: {str(e)}")
            raise
        finally:
            if conn and not conn.closed:
                conn.close()
                logger.debug("Database connection closed")

    def close(self):
        """Close the database connection if open."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None
            logger.debug("Closed persistent connection")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit with cleanup."""
        self.close()

    def get_existing_chunk_ids(self) -> Set[str]:
        """
        Get set of all chunk IDs currently in the embeddings table.

        Returns:
            Set of chunk IDs

        Raises:
            ConnectionError: If database operation fails
        """
        logger.debug(f"Fetching existing chunk IDs from {self.table_name}")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT chunk_id FROM {self.table_name}")  # type: ignore
                    chunk_ids = {row[0] for row in cur.fetchall()}
                    logger.info(
                        f"Retrieved {len(chunk_ids)} existing chunk IDs from {self.table_name}"
                    )
                    return chunk_ids
        except Exception as e:
            logger.error(f"Failed to fetch existing chunk IDs: {str(e)}")
            raise

    def get_chunks_by_article_id(self, article_id: int) -> List[Dict]:
        """
        Retrieve all chunks for a specific article.

        Args:
            article_id: The parent article ID

        Returns:
            List of chunk dictionaries with metadata

        Raises:
            ConnectionError: If database operation fails
        """
        logger.debug(f"Fetching chunks for article {article_id} from {self.table_name}")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT chunk_id, parent_article_id, chunk_sequence, text_content,
                               token_count, source_url, created_at
                        FROM {self.table_name}
                        WHERE parent_article_id = %s
                        ORDER BY chunk_sequence
                        """,  # type: ignore
                        (article_id,),
                    )
                    rows = cur.fetchall()
                    chunks = []
                    for row in rows:
                        chunks.append(
                            {
                                "chunk_id": row[0],
                                "parent_article_id": row[1],
                                "chunk_sequence": row[2],
                                "text_content": row[3],
                                "token_count": row[4],
                                "source_url": row[5],
                                "created_at": row[6],
                            }
                        )
                    logger.info(
                        f"Retrieved {len(chunks)} chunks for article {article_id}"
                    )
                    return chunks
        except Exception as e:
            logger.error(f"Failed to fetch chunks for article {article_id}: {str(e)}")
            raise

    def insert_embeddings(
        self, records: List[Tuple[VectorRecord, List[float]]]
    ) -> None:
        """
        Insert new embedding records into the database.

        Args:
            records: List of tuples (VectorRecord, embedding_vector)

        Raises:
            ConnectionError: If database operation fails
            ValueError: If embedding dimensions don't match expected
        """
        if not records:
            logger.warning("No embeddings to insert")
            return

        logger.info(f"Inserting {len(records)} embeddings into {self.table_name}")

        try:
            # Validate embedding dimensions
            for idx, (record, embedding) in enumerate(records):
                if len(embedding) != self.embedding_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch for record {idx}: "
                        f"expected {self.embedding_dim}, got {len(embedding)}"
                    )

            with PerformanceLogger(logger, f"Insert {len(records)} embeddings"):
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        for idx, (record, embedding) in enumerate(records, 1):
                            try:
                                cur.execute(
                                    f"""
                                    INSERT INTO {self.table_name}
                                    (chunk_id, parent_article_id, chunk_sequence, text_content,
                                     token_count, source_url, embedding)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                                    """,  # type: ignore
                                    (
                                        record.chunk_id,
                                        record.parent_article_id,
                                        record.chunk_sequence,
                                        record.text_content,
                                        record.token_count,
                                        str(record.source_url),
                                        embedding,
                                    ),
                                )
                                logger.debug(
                                    f"Inserted embedding {record.chunk_id} for article {record.parent_article_id}"
                                )
                            except psycopg.IntegrityError as e:
                                logger.error(
                                    f"Integrity error inserting embedding {record.chunk_id}: {str(e)}"
                                )
                                raise
                            except psycopg.Error as e:
                                logger.error(
                                    f"Database error inserting embedding {record.chunk_id}: {str(e)}"
                                )
                                raise

                logger.info(f"Successfully inserted {len(records)} embeddings")

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to insert embeddings: {str(e)}")
            raise

    def update_embeddings(
        self, records: List[Tuple[VectorRecord, List[float]]]
    ) -> None:
        """
        Update existing embedding records in the database.

        Args:
            records: List of tuples (VectorRecord, embedding_vector)

        Raises:
            ConnectionError: If database operation fails
            ValueError: If embedding dimensions don't match expected
        """
        if not records:
            logger.warning("No embeddings to update")
            return

        logger.info(f"Updating {len(records)} embeddings in {self.table_name}")

        try:
            # Validate embedding dimensions
            for idx, (record, embedding) in enumerate(records):
                if len(embedding) != self.embedding_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch for record {idx}: "
                        f"expected {self.embedding_dim}, got {len(embedding)}"
                    )

            with PerformanceLogger(logger, f"Update {len(records)} embeddings"):
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        for record, embedding in records:
                            try:
                                cur.execute(
                                    f"""
                                    UPDATE {self.table_name}
                                    SET parent_article_id = %s,
                                        chunk_sequence = %s,
                                        text_content = %s,
                                        token_count = %s,
                                        source_url = %s,
                                        embedding = %s
                                    WHERE chunk_id = %s
                                    """,  # type: ignore
                                    (
                                        record.parent_article_id,
                                        record.chunk_sequence,
                                        record.text_content,
                                        record.token_count,
                                        str(record.source_url),
                                        embedding,
                                        record.chunk_id,
                                    ),
                                )
                                logger.debug(
                                    f"Updated embedding {record.chunk_id} for article {record.parent_article_id}"
                                )
                            except psycopg.Error as e:
                                logger.error(
                                    f"Database error updating embedding {record.chunk_id}: {str(e)}"
                                )
                                raise

                logger.info(f"Successfully updated {len(records)} embeddings")

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to update embeddings: {str(e)}")
            raise

    def delete_embeddings_by_article_id(self, article_id: int) -> int:
        """
        Delete all embeddings for a specific article.

        Args:
            article_id: The parent article ID

        Returns:
            Number of embeddings deleted

        Raises:
            ConnectionError: If database operation fails
        """
        logger.info(
            f"Deleting embeddings for article {article_id} from {self.table_name}"
        )
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {self.table_name} WHERE parent_article_id = %s",  # type: ignore
                        (article_id,),
                    )
                    deleted_count = cur.rowcount
                    logger.info(
                        f"Deleted {deleted_count} embeddings for article {article_id}"
                    )
                    return deleted_count
        except Exception as e:
            logger.error(
                f"Failed to delete embeddings for article {article_id}: {str(e)}"
            )
            raise

    def delete_embeddings_by_chunk_ids(self, chunk_ids: List[str]) -> int:
        """
        Delete embeddings by chunk IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of embeddings deleted

        Raises:
            ConnectionError: If database operation fails
        """
        if not chunk_ids:
            logger.warning("No chunk IDs provided for deletion")
            return 0

        logger.info(f"Deleting {len(chunk_ids)} embeddings from {self.table_name}")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {self.table_name} WHERE chunk_id = ANY(%s)",  # type: ignore
                        (chunk_ids,),
                    )
                    deleted_count = cur.rowcount
                    logger.info(f"Deleted {deleted_count} embeddings")
                    return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {str(e)}")
            raise

    def get_count(self) -> int:
        """
        Get the total number of embeddings in the table.

        Returns:
            Count of embeddings

        Raises:
            ConnectionError: If database operation fails
        """
        logger.debug(f"Getting count from {self.table_name}")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")  # type: ignore
                    count = cur.fetchone()[0]
                    logger.debug(f"Count: {count} embeddings in {self.table_name}")
                    return count
        except Exception as e:
            logger.error(f"Failed to get embedding count: {str(e)}")
            raise

    def get_embedding_by_chunk_id(self, chunk_id: str) -> Optional[List[float]]:
        """
        Retrieve embedding vector for a specific chunk.

        Args:
            chunk_id: The chunk ID to retrieve

        Returns:
            Embedding vector as list of floats, or None if not found

        Raises:
            ConnectionError: If database operation fails
        """
        logger.debug(f"Retrieving embedding for chunk {chunk_id}")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT embedding::text FROM {self.table_name} WHERE chunk_id = %s",  # type: ignore
                        (chunk_id,),
                    )
                    row = cur.fetchone()
                    if row:
                        # Convert pgvector string representation to list of floats
                        # Format is like "[0.1,0.2,0.3]"
                        embedding_str = row[0]
                        if embedding_str.startswith('[') and embedding_str.endswith(']'):
                            # Remove brackets and split by comma
                            values = embedding_str[1:-1].split(',')
                            return [float(v) for v in values]
                        # If already a list, return as is
                        elif isinstance(embedding_str, list):
                            return embedding_str
                    return None
        except Exception as e:
            logger.error(f"Failed to get embedding for chunk {chunk_id}: {str(e)}")
            raise

    def search_similar_vectors(
        self, query_vector: List[float], limit: int = 10, min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar vectors using cosine similarity.

        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of dictionaries with chunk data and similarity scores

        Raises:
            ConnectionError: If database operation fails
            ValueError: If query vector dimension doesn't match expected
        """
        if len(query_vector) != self.embedding_dim:
            raise ValueError(
                f"Query vector dimension mismatch: "
                f"expected {self.embedding_dim}, got {len(query_vector)}"
            )

        logger.debug(f"Searching for {limit} similar vectors in {self.table_name}")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Use cosine similarity (1 - cosine distance)
                    cur.execute(
                        f"""
                        SELECT chunk_id, parent_article_id, chunk_sequence, text_content,
                               token_count, source_url, created_at,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM {self.table_name}
                        WHERE 1 - (embedding <=> %s::vector) >= %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,  # type: ignore
                        (
                            query_vector,
                            query_vector,
                            min_similarity,
                            query_vector,
                            limit,
                        ),
                    )
                    rows = cur.fetchall()
                    results = []
                    for row in rows:
                        results.append(
                            {
                                "chunk_id": row[0],
                                "parent_article_id": row[1],
                                "chunk_sequence": row[2],
                                "text_content": row[3],
                                "token_count": row[4],
                                "source_url": row[5],
                                "created_at": row[6],
                                "similarity": float(row[7]),
                            }
                        )
                    logger.info(f"Found {len(results)} similar vectors")
                    return results
        except Exception as e:
            logger.error(f"Failed to search similar vectors: {str(e)}")
            raise


class OpenAIVectorStorage(VectorStorageClient):
    """
    Vector storage client for OpenAI embeddings (3072 dimensions).
    Uses the text-embedding-3-large model.
    """

    def __init__(self):
        """Initialize OpenAI vector storage client."""
        super().__init__(table_name="embeddings_openai", embedding_dim=3072)
        logger.info("OpenAIVectorStorage client initialized")


class CohereVectorStorage(VectorStorageClient):
    """
    Vector storage client for AWS Cohere embeddings (1536 dimensions).
    Uses the cohere.embed-english-v3 model.
    """

    def __init__(self):
        """Initialize Cohere vector storage client."""
        super().__init__(table_name="embeddings_cohere", embedding_dim=1536)
        logger.info("CohereVectorStorage client initialized")
