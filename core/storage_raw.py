from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Set
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
            logger.info(f"PostgresClient configured for {self.db_host}:{self.db_port}/{self.db_name}")
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
        Retrieve ID and last modified date for all articles in database.

        Returns:
            Dictionary mapping article_id -> last_modified_date
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, last_modified_date FROM articles")
                return {row[0]: row[1] for row in cur.fetchall()}

    def get_existing_article_ids(self) -> Set[int]:
        """
        Get set of all article IDs currently in database.

        Returns:
            Set of article IDs
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM articles")
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
                                    INSERT INTO articles (id, title, url, content_html, last_modified_date, raw_ingestion_date)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    """,
                                    (
                                        article.id,
                                        article.title,
                                        str(article.url),
                                        article.content_html,
                                        article.last_modified_date,
                                        article.raw_ingestion_date,
                                    ),
                                )
                                logger.debug(f"Inserted article {article.id}: {article.title}")
                            except psycopg.IntegrityError as e:
                                logger.error(f"Integrity error inserting article {article.id}: {str(e)}")
                                raise
                            except psycopg.Error as e:
                                logger.error(f"Database error inserting article {article.id}: {str(e)}")
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
                                    WHERE id = %s
                                    """,
                                    (
                                        article.title,
                                        str(article.url),
                                        article.content_html,
                                        article.last_modified_date,
                                        article.raw_ingestion_date,
                                        article.id,
                                    ),
                                )
                                logger.debug(f"Updated article {article.id}: {article.title}")
                            except psycopg.Error as e:
                                logger.error(f"Database error updating article {article.id}: {str(e)}")
                                raise

                logger.info(f"Successfully updated {len(articles)} articles")

        except Exception as e:
            logger.error(f"Failed to update articles: {str(e)}")
            raise
