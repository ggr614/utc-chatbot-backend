"""
Vector storage module for managing embeddings in PostgreSQL with pgvector.

This module provides clients for storing and retrieving vector embeddings
from OpenAI (3072 dimensions) models.
"""

from typing import Dict, List, Optional, Set, Tuple
import psycopg
from core.storage_base import BaseStorageClient
from core.schemas import VectorRecord
from utils.logger import get_logger, PerformanceLogger

logger = get_logger(__name__)


class VectorStorageClient(BaseStorageClient):
    """
    Base class for vector storage operations.

    Handles connection management and common database operations
    for vector embeddings storage.
    """

    def __init__(self, table_name: str, embedding_dim: int, connection_pool=None):
        """
        Initialize the vector storage client.

        Args:
            table_name: Name of the embeddings table (e.g., 'embeddings_openai')
            embedding_dim: Expected dimension of embeddings (e.g., 3072 for OpenAI)
            connection_pool: Optional DatabaseConnectionPool instance for API mode
        """
        super().__init__(connection_pool=connection_pool)
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        logger.info(
            f"VectorStorageClient configured for table '{self.table_name}' "
            f"with embedding dimension {self.embedding_dim}"
        )

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

    def get_chunks_by_article_id(self, article_id) -> List[Dict]:
        """
        Retrieve all chunks for a specific article.

        Args:
            article_id: The parent article ID (UUID)

        Returns:
            List of chunk dictionaries with metadata from article_chunks

        Raises:
            ConnectionError: If database operation fails
        """
        logger.debug(f"Fetching chunks for article {article_id} from {self.table_name}")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT e.chunk_id, c.parent_article_id, c.chunk_sequence,
                               c.text_content, c.token_count, c.url, e.created_at
                        FROM {self.table_name} e
                        JOIN article_chunks c ON e.chunk_id = c.id
                        WHERE c.parent_article_id = %s
                        ORDER BY c.chunk_sequence
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
                Note: Only chunk_id is used; other VectorRecord fields reference article_chunks

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
                                    (chunk_id, embedding)
                                    VALUES (%s, %s)
                                    """,  # type: ignore
                                    (
                                        record.chunk_id,
                                        embedding,
                                    ),
                                )
                                logger.debug(
                                    f"Inserted embedding for chunk {record.chunk_id}"
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
                Note: Only chunk_id and embedding are updated

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
                                    SET embedding = %s
                                    WHERE chunk_id = %s
                                    """,  # type: ignore
                                    (
                                        embedding,
                                        record.chunk_id,
                                    ),
                                )
                                logger.debug(
                                    f"Updated embedding for chunk {record.chunk_id}"
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

    def delete_embeddings_by_article_id(self, article_id) -> int:
        """
        Delete all embeddings for a specific article.

        Args:
            article_id: The parent article ID (UUID)

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
                    # Delete embeddings by joining with article_chunks
                    cur.execute(
                        f"""
                        DELETE FROM {self.table_name}
                        WHERE chunk_id IN (
                            SELECT id FROM article_chunks
                            WHERE parent_article_id = %s
                        )
                        """,  # type: ignore
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
                        if embedding_str.startswith("[") and embedding_str.endswith(
                            "]"
                        ):
                            # Remove brackets and split by comma
                            values = embedding_str[1:-1].split(",")
                            return [float(v) for v in values]
                        # If already a list, return as is
                        elif isinstance(embedding_str, list):
                            return embedding_str
                    return None
        except Exception as e:
            logger.error(f"Failed to get embedding for chunk {chunk_id}: {str(e)}")
            raise

    def search_similar_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        min_similarity: float = 0.0,
        status_names: Optional[List[str]] = None,
        category_names: Optional[List[str]] = None,
        is_public: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        include_system_prompts: bool = True,
    ) -> List[Dict]:
        """
        Search for similar vectors using cosine similarity with optional article metadata filtering.

        Filters are applied using AND logic. Within list filters (status_names, category_names, tags),
        OR logic is used (ANY match). For tags, articles must have at least one of the provided tags.

        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            status_names: Filter by article status (e.g., ['Approved', 'Published'])
            category_names: Filter by article category (e.g., ['IT Help', 'Documentation'])
            is_public: Filter by article visibility (True for public, False for private)
            tags: Filter by article tags (ANY match using array overlap operator)
            include_system_prompts: Include system prompts resolved from article tags (default: True)

        Returns:
            List of dictionaries with chunk data (from article_chunks), similarity scores,
            and optional system_prompt field (if include_system_prompts=True)

        Raises:
            ConnectionError: If database operation fails
            ValueError: If query vector dimension doesn't match expected
        """
        if len(query_vector) != self.embedding_dim:
            raise ValueError(
                f"Query vector dimension mismatch: "
                f"expected {self.embedding_dim}, got {len(query_vector)}"
            )

        logger.debug(
            f"Searching for {limit} similar vectors in {self.table_name} with filters: "
            f"status={status_names}, category={category_names}, is_public={is_public}, tags={tags}"
        )
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Build WHERE clause conditions
                    where_conditions = ["1 - (e.embedding <=> %s::vector) >= %s"]
                    params = [query_vector, query_vector, min_similarity]

                    if status_names is not None:
                        where_conditions.append("a.status_name = ANY(%s)")
                        params.append(status_names)

                    if category_names is not None:
                        where_conditions.append("a.category_name = ANY(%s)")
                        params.append(category_names)

                    if is_public is not None:
                        where_conditions.append("a.is_public = %s")
                        params.append(is_public)

                    if tags is not None:
                        # Use PostgreSQL array overlap operator (&&)
                        where_conditions.append("a.tags && %s")
                        params.append(tags)

                    # Add order by and limit parameters
                    params.extend([query_vector, limit])

                    # Build full query with article JOIN
                    where_clause = " AND ".join(where_conditions)

                    if include_system_prompts:
                        # Query with system prompt resolution CTE
                        query = f"""
                            WITH article_prompts AS (
                                SELECT
                                    a.id AS article_id,
                                    tsp.system_prompt,
                                    ROW_NUMBER() OVER (
                                        PARTITION BY a.id
                                        ORDER BY tsp.priority DESC, tsp.tag_name ASC
                                    ) AS rn
                                FROM articles a
                                LEFT JOIN LATERAL UNNEST(a.tags) AS tag ON true
                                LEFT JOIN tag_system_prompts tsp ON tsp.tag_name = tag
                                WHERE tsp.system_prompt IS NOT NULL
                            )
                            SELECT e.chunk_id, c.parent_article_id, c.chunk_sequence,
                                   c.text_content, c.token_count, c.url, e.created_at,
                                   1 - (e.embedding <=> %s::vector) AS similarity,
                                   COALESCE(
                                       (SELECT system_prompt FROM article_prompts WHERE article_id = c.parent_article_id AND rn = 1),
                                       (SELECT system_prompt FROM tag_system_prompts WHERE tag_name = '__default__')
                                   ) AS system_prompt
                            FROM {self.table_name} e
                            JOIN article_chunks c ON e.chunk_id = c.id
                            JOIN articles a ON c.parent_article_id = a.id
                            WHERE {where_clause}
                            ORDER BY e.embedding <=> %s::vector
                            LIMIT %s
                        """
                    else:
                        # Original query without system prompts
                        query = f"""
                            SELECT e.chunk_id, c.parent_article_id, c.chunk_sequence,
                                   c.text_content, c.token_count, c.url, e.created_at,
                                   1 - (e.embedding <=> %s::vector) AS similarity
                            FROM {self.table_name} e
                            JOIN article_chunks c ON e.chunk_id = c.id
                            JOIN articles a ON c.parent_article_id = a.id
                            WHERE {where_clause}
                            ORDER BY e.embedding <=> %s::vector
                            LIMIT %s
                        """

                    logger.debug(
                        f"Executing query with {len(params)} parameters (include_system_prompts={include_system_prompts})"
                    )

                    cur.execute(query, params)  # type: ignore
                    rows = cur.fetchall()
                    results = []
                    for row in rows:
                        result_dict = {
                            "chunk_id": row[0],
                            "parent_article_id": row[1],
                            "chunk_sequence": row[2],
                            "text_content": row[3],
                            "token_count": row[4],
                            "source_url": row[5],
                            "created_at": row[6],
                            "similarity": float(row[7]),
                        }
                        if include_system_prompts:
                            result_dict["system_prompt"] = row[8]
                        results.append(result_dict)
                    logger.info(
                        f"Found {len(results)} similar vectors "
                        f"(filters: status={status_names}, category={category_names}, "
                        f"is_public={is_public}, tags={tags})"
                    )
                    return results
        except Exception as e:
            logger.error(f"Failed to search similar vectors: {str(e)}")
            raise


class VectorStorage(VectorStorageClient):
    """
    Vector storage client for embeddings (3072 dimensions).
    Uses the embeddings_openai table in PostgreSQL with pgvector.

    Note: Table name remains 'embeddings_openai' for backward compatibility.
    The storage layer is provider-agnostic; the table name is a historical artifact.
    """

    def __init__(self, connection_pool=None):
        """
        Initialize vector storage client.

        Args:
            connection_pool: Optional DatabaseConnectionPool instance for API mode
        """
        super().__init__(
            table_name="embeddings_openai",
            embedding_dim=3072,
            connection_pool=connection_pool,
        )
        logger.info("VectorStorage client initialized")


# Deprecated alias for backward compatibility
OpenAIVectorStorage = VectorStorage
