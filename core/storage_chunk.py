from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Set
from core.config import get_settings
import psycopg
from psycopg import Connection
from core.schemas import TextChunk


class PostgresClient:
    def __init__(self):
        self.db_host = get_settings().DB_HOST
        self.db_user = get_settings().DB_USER
        self.db_password = get_settings().DB_PASSWORD.get_secret_value()
        self.db_name = get_settings().DB_NAME
        self.db_port = 5432
        self._conn: Optional[Connection] = None
        self._connection_params = {
            "host": self.db_host,
            "user": self.db_user,
            "password": self.db_password,
            "dbname": self.db_name,
            "port": self.db_port,
        }

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
            conn = psycopg.connect(**self._connection_params)
            yield conn
            conn.commit()  # Auto-commit on success
        except psycopg.Error as e:
            if conn:
                conn.rollback()  # Auto-rollback on error
            raise ConnectionError(f"Database error: {e}") from e
        finally:
            if conn and not conn.closed:
                conn.close()

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

    def insert_chunks(self, chunks: List[TextChunk]) -> None:
        """
        Insert new articles into database.

        Args:
            articles: List of validated TdxArticle objects
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for chunk in chunks:
                    cur.execute(
                        """
                        INSERT INTO article_chunks (id, parent_article_id, chunk_sequence, text_content, token_count, url, last_modified_date)
                        VALUES (%s, %s, %s, %s, %s, %s)
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

    def update_articles(self, chunks: List[TextChunk]) -> None:
        """
        Update existing chunks in database.

        Args:
            chunks: List of validated TdxArticle objects
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for chunk in chunks:
                    cur.execute(
                        """
                        UPDATE article_chunks
                        SET parent_article_id = %s,
                            chunk_sequence = %s,
                            text_content = %s,
                            token_count = %s,
                            url = %s
                        WHERE id = %s
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
