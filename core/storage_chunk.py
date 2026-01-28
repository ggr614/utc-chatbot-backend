from datetime import datetime
from typing import Dict, List, Set
import psycopg
from core.storage_base import BaseStorageClient
from core.schemas import TextChunk
from utils.logger import get_logger

logger = get_logger(__name__)


class PostgresClient(BaseStorageClient):
    """
    Storage client for text chunk data.

    Handles CRUD operations for the article_chunks table, including
    storage and retrieval of processed text chunks.
    """

    def __init__(self, connection_pool=None):
        """
        Initialize the PostgresClient for chunks storage.

        Args:
            connection_pool: Optional DatabaseConnectionPool instance for API mode
        """
        super().__init__(connection_pool=connection_pool)

    def get_article_metadata(self) -> Dict[int, datetime]:
        """
        Retrieve ID and last modified date for all articles in database.

        Returns:
            Dictionary mapping article_id -> last_modified_date
        """
        logger.debug("Fetching article metadata from database")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, last_modified_date FROM articles")
                    metadata = {row[0]: row[1] for row in cur.fetchall()}
                    logger.debug(f"Retrieved metadata for {len(metadata)} articles")
                    return metadata
        except Exception as e:
            logger.error(f"Failed to fetch article metadata: {str(e)}")
            raise

    def get_existing_article_ids(self) -> Set[int]:
        """
        Get set of all article IDs currently in database.

        Returns:
            Set of article IDs
        """
        logger.debug("Fetching existing article IDs from database")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM articles")
                    article_ids = {row[0] for row in cur.fetchall()}
                    logger.debug(f"Found {len(article_ids)} article IDs")
                    return article_ids
        except Exception as e:
            logger.error(f"Failed to fetch article IDs: {str(e)}")
            raise

    def insert_chunks(self, chunks: List[TextChunk]) -> None:
        """
        Insert new chunks into database.

        Args:
            chunks: List of validated TextChunk objects

        Raises:
            ConnectionError: If database operation fails
        """
        if not chunks:
            logger.warning("No chunks to insert")
            return

        logger.info(f"Inserting {len(chunks)} chunks into database")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for chunk in chunks:
                        try:
                            cur.execute(
                                """
                                INSERT INTO article_chunks (id, parent_article_id, chunk_sequence, text_content, token_count, url, last_modified_date)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    chunk.chunk_id,
                                    chunk.parent_article_id,
                                    chunk.chunk_sequence,
                                    chunk.text_content,
                                    chunk.token_count,
                                    str(chunk.source_url),
                                    chunk.last_modified_date,
                                ),
                            )
                        except psycopg.IntegrityError as e:
                            logger.error(
                                f"Integrity error inserting chunk {chunk.chunk_id}: {str(e)}"
                            )
                            raise
                        except psycopg.Error as e:
                            logger.error(
                                f"Database error inserting chunk {chunk.chunk_id}: {str(e)}"
                            )
                            raise

            logger.info(f"Successfully inserted {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to insert chunks: {str(e)}")
            raise

    def update_chunks(self, chunks: List[TextChunk]) -> None:
        """
        Update existing chunks in database.

        Args:
            chunks: List of validated TextChunk objects

        Raises:
            ConnectionError: If database operation fails
        """
        if not chunks:
            logger.warning("No chunks to update")
            return

        logger.info(f"Updating {len(chunks)} chunks in database")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for chunk in chunks:
                        try:
                            cur.execute(
                                """
                                UPDATE article_chunks
                                SET parent_article_id = %s,
                                    chunk_sequence = %s,
                                    text_content = %s,
                                    token_count = %s,
                                    url = %s,
                                    last_modified_date = %s
                                WHERE id = %s
                                """,
                                (
                                    chunk.parent_article_id,
                                    chunk.chunk_sequence,
                                    chunk.text_content,
                                    chunk.token_count,
                                    str(chunk.source_url),
                                    chunk.last_modified_date,
                                    chunk.chunk_id,
                                ),
                            )
                            logger.debug(f"Updated chunk {chunk.chunk_id}")
                        except psycopg.Error as e:
                            logger.error(
                                f"Database error updating chunk {chunk.chunk_id}: {str(e)}"
                            )
                            raise

            logger.info(f"Successfully updated {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to update chunks: {str(e)}")
            raise

    def get_all_chunks(self) -> List[TextChunk]:
        """
        Retrieve all chunks from the database.

        Returns:
            List of TextChunk objects

        Raises:
            ConnectionError: If database operation fails
        """
        logger.debug("Fetching all chunks from database")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, parent_article_id, chunk_sequence, text_content, token_count, url, last_modified_date
                        FROM article_chunks
                        ORDER BY parent_article_id, chunk_sequence
                        """
                    )
                    rows = cur.fetchall()
                    chunks = []
                    for row in rows:
                        chunk = TextChunk(
                            chunk_id=row[0],
                            parent_article_id=row[1],
                            chunk_sequence=row[2],
                            text_content=row[3],
                            token_count=row[4],
                            source_url=row[5],
                            last_modified_date=row[6],
                        )
                        chunks.append(chunk)
                    logger.info(f"Retrieved {len(chunks)} chunks from database")
                    return chunks
        except Exception as e:
            logger.error(f"Failed to fetch chunks: {str(e)}")
            raise
