from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Set
from uuid import UUID
from core.config import get_settings
import psycopg
from psycopg import Connection
from core.schemas import TdxArticle
from utils.logger import get_logger, PerformanceLogger

logger = get_logger(__name__)


class PostgresClient:
    def __init__(self):
        logger.info("Initializing PostgresClient")
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
            self._conn: Optional[Connection] = None
            self._connection_params = {
                "host": self.db_host,
                "user": self.db_user,
                "password": self.db_password,
                "dbname": self.db_name,
                "port": self.db_port,
            }
            logger.info(
                f"PostgresClient configured for {self.db_host}:{self.db_port}/{self.db_name}"
            )
            logger.debug(f"Database user: {self.db_user}")
        except Exception as e:
            logger.error(f"Failed to initialize PostgresClient: {str(e)}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Get a database connection with automatic cleanup.

        Yields:
            Connection: PostgreSQL connection object

        Raises:
            ConnectionError: If unable to connect to database

        Example:
            >>> store = KBStore(host="localhost", user="admin", password="pass", db_name="kb")
            >>> with store.get_connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM articles")
        """
        conn = None
        try:
            logger.debug(f"Connecting to database {self.db_name}@{self.db_host}")
            conn = psycopg.connect(**self._connection_params)
            logger.debug("Database connection established successfully")
            yield conn
            conn.commit()  # Auto-commit on success
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

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit with cleanup."""
        self.close()

    def get_article_metadata(self) -> Dict[int, datetime]:
        """
        Retrieve TDX article ID and last modified date for all articles in database.

        Returns:
            Dictionary mapping tdx_article_id (int) -> last_modified_date
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT tdx_article_id, last_modified_date FROM articles")
                return {row[0]: row[1] for row in cur.fetchall()}

    def get_existing_article_ids(self) -> Set[int]:
        """
        Get set of all TDX article IDs currently in database.

        Returns:
            Set of TDX article IDs (integers)
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT tdx_article_id FROM articles")
                return {row[0] for row in cur.fetchall()}

    def insert_articles(self, articles: List[TdxArticle]) -> None:
        """
        Insert new articles into database.

        Args:
            articles: List of validated TdxArticle objects

        Raises:
            ConnectionError: If database operation fails
        """
        if not articles:
            logger.warning("No articles to insert")
            return

        logger.info(f"Inserting {len(articles)} articles into database")

        try:
            with PerformanceLogger(logger, f"Insert {len(articles)} articles"):
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        for idx, article in enumerate(articles, 1):
                            try:
                                cur.execute(
                                    """
                                    INSERT INTO articles (tdx_article_id, title, url, content_html, last_modified_date, raw_ingestion_date)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    RETURNING id
                                    """,
                                    (
                                        article.tdx_article_id,
                                        article.title,
                                        str(article.url),
                                        article.content_html,
                                        article.last_modified_date,
                                        article.raw_ingestion_date,
                                    ),
                                )
                                # Get the auto-generated UUID
                                generated_id = cur.fetchone()[0]
                                article.id = generated_id
                                logger.debug(
                                    f"Inserted article TDX ID {article.tdx_article_id} (UUID: {article.id}): {article.title}"
                                )
                            except psycopg.IntegrityError as e:
                                logger.error(
                                    f"Integrity error inserting article {article.id}: {str(e)}"
                                )
                                raise
                            except psycopg.Error as e:
                                logger.error(
                                    f"Database error inserting article {article.id}: {str(e)}"
                                )
                                raise

                logger.info(f"Successfully inserted {len(articles)} articles")

        except Exception as e:
            logger.error(f"Failed to insert articles: {str(e)}")
            raise

    def update_articles(self, articles: List[TdxArticle]) -> None:
        """
        Update existing articles in database.

        Args:
            articles: List of validated TdxArticle objects

        Raises:
            ConnectionError: If database operation fails
        """
        if not articles:
            logger.warning("No articles to update")
            return

        logger.info(f"Updating {len(articles)} articles in database")

        try:
            with PerformanceLogger(logger, f"Update {len(articles)} articles"):
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        for article in articles:
                            try:
                                cur.execute(
                                    """
                                    UPDATE articles
                                    SET title = %s,
                                        url = %s,
                                        content_html = %s,
                                        last_modified_date = %s,
                                        raw_ingestion_date = %s
                                    WHERE tdx_article_id = %s
                                    RETURNING id
                                    """,
                                    (
                                        article.title,
                                        str(article.url),
                                        article.content_html,
                                        article.last_modified_date,
                                        article.raw_ingestion_date,
                                        article.tdx_article_id,
                                    ),
                                )
                                # Get the UUID for this article
                                result = cur.fetchone()
                                if result:
                                    article.id = result[0]
                                logger.debug(
                                    f"Updated article TDX ID {article.tdx_article_id} (UUID: {article.id}): {article.title}"
                                )
                            except psycopg.Error as e:
                                logger.error(
                                    f"Database error updating article {article.id}: {str(e)}"
                                )
                                raise

                logger.info(f"Successfully updated {len(articles)} articles")

        except Exception as e:
            logger.error(f"Failed to update articles: {str(e)}")
            raise

    def get_all_articles(self) -> List[TdxArticle]:
        """
        Retrieve all articles from the database.

        Returns:
            List of TdxArticle objects

        Raises:
            ConnectionError: If database operation fails
        """
        logger.debug("Fetching all articles from database")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, tdx_article_id, title, url, content_html, last_modified_date
                        FROM articles
                        ORDER BY id
                        """
                    )
                    rows = cur.fetchall()
                    articles = []
                    for row in rows:
                        articles.append(
                            TdxArticle(
                                id=row[0],
                                tdx_article_id=row[1],
                                title=row[2],
                                url=row[3],
                                content_html=row[4],
                                last_modified_date=row[5],
                            )
                        )
                    logger.info(f"Retrieved {len(articles)} articles from database")
                    return articles
        except Exception as e:
            logger.error(f"Failed to fetch all articles: {str(e)}")
            raise

    def get_articles_by_ids(self, article_ids: List[UUID]) -> List[TdxArticle]:
        """
        Retrieve specific articles by their IDs.

        Args:
            article_ids: List of article IDs (UUIDs) to retrieve

        Returns:
            List of TdxArticle objects

        Raises:
            ConnectionError: If database operation fails
        """
        if not article_ids:
            logger.warning("No article IDs provided")
            return []

        logger.debug(f"Fetching {len(article_ids)} articles by ID")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, tdx_article_id, title, url, content_html, last_modified_date
                        FROM articles
                        WHERE id = ANY(%s)
                        ORDER BY id
                        """,
                        (article_ids,),
                    )
                    rows = cur.fetchall()
                    articles = []
                    for row in rows:
                        articles.append(
                            TdxArticle(
                                id=row[0],
                                tdx_article_id=row[1],
                                title=row[2],
                                url=row[3],
                                content_html=row[4],
                                last_modified_date=row[5],
                            )
                        )
                    logger.info(f"Retrieved {len(articles)} articles from database")
                    return articles
        except Exception as e:
            logger.error(f"Failed to fetch articles by IDs: {str(e)}")
            raise

    def store_chunks(self, chunks: List["TextChunk"]) -> int:
        """
        Store text chunks in the article_chunks table.

        Args:
            chunks: List of TextChunk objects to store

        Returns:
            Number of chunks stored

        Raises:
            ConnectionError: If database operation fails
        """
        if not chunks:
            logger.warning("No chunks provided for storage")
            return 0

        logger.info(f"Storing {len(chunks)} chunks in article_chunks table")
        try:
            with PerformanceLogger(logger, f"Store {len(chunks)} chunks"):
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        for chunk in chunks:
                            cur.execute(
                                """
                                INSERT INTO article_chunks
                                (id, parent_article_id, chunk_sequence, text_content,
                                 token_count, url, last_modified_date)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (id) DO UPDATE SET
                                    text_content = EXCLUDED.text_content,
                                    token_count = EXCLUDED.token_count,
                                    last_modified_date = EXCLUDED.last_modified_date
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
                logger.info(f"Successfully stored {len(chunks)} chunks")
                return len(chunks)
        except Exception as e:
            logger.error(f"Failed to store chunks: {str(e)}")
            raise

    def get_chunk_count(self) -> int:
        """
        Get the total number of chunks in the database.

        Returns:
            Count of chunks

        Raises:
            ConnectionError: If database operation fails
        """
        logger.debug("Getting chunk count from database")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM article_chunks")
                    count = cur.fetchone()[0]  # type: ignore
                    logger.debug(f"Found {count} chunks in database")
                    return count
        except Exception as e:
            logger.error(f"Failed to get chunk count: {str(e)}")
            raise

    def get_all_chunks(
        self, limit: Optional[int] = None, offset: int = 0
    ) -> List["TextChunk"]:
        """
        Retrieve all chunks from the database.

        Args:
            limit: Maximum number of chunks to retrieve (None for all)
            offset: Number of chunks to skip

        Returns:
            List of TextChunk objects

        Raises:
            ConnectionError: If database operation fails
        """
        logger.debug(f"Fetching chunks from database (limit={limit}, offset={offset})")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if limit:
                        cur.execute(
                            """
                            SELECT id, parent_article_id, chunk_sequence, text_content,
                                   token_count, url, last_modified_date
                            FROM article_chunks
                            ORDER BY parent_article_id, chunk_sequence
                            LIMIT %s OFFSET %s
                            """,
                            (limit, offset),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, parent_article_id, chunk_sequence, text_content,
                                   token_count, url, last_modified_date
                            FROM article_chunks
                            ORDER BY parent_article_id, chunk_sequence
                            """
                        )

                    rows = cur.fetchall()
                    chunks = []
                    for row in rows:
                        from core.schemas import TextChunk

                        chunks.append(
                            TextChunk(
                                chunk_id=row[0],
                                parent_article_id=row[1],
                                chunk_sequence=row[2],
                                text_content=row[3],
                                token_count=row[4],
                                source_url=row[5],
                                last_modified_date=row[6],
                            )
                        )
                    logger.info(f"Retrieved {len(chunks)} chunks from database")
                    return chunks
        except Exception as e:
            logger.error(f"Failed to fetch chunks: {str(e)}")
            raise
