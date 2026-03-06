"""
Base storage module for PostgreSQL database operations.

This module provides a base class with common database connection management,
error handling, and logging for all storage clients.
"""

from contextlib import contextmanager
from typing import Optional
from core.config import get_database_settings
import psycopg
from psycopg import Connection
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseStorageClient:
    """
    Base class for all PostgreSQL storage operations.

    Provides common functionality for database connection management,
    error handling, logging, and context manager protocol.

    Supports both connection pooling (for API mode) and individual
    connections (for CLI mode) for maximum flexibility.
    """

    def __init__(self, connection_pool=None):
        """
        Initialize the storage client with database configuration.

        Args:
            connection_pool: Optional DatabaseConnectionPool instance for API mode.
                           If None, creates individual connections (CLI mode).

        Raises:
            ValueError: If required database configuration is missing
        """
        logger.info(f"Initializing {self.__class__.__name__}")
        try:
            self._connection_pool = connection_pool

            if self._connection_pool is None:
                # CLI mode: Store connection params for individual connections
                settings = get_database_settings()

                # Validate configuration
                if not settings.HOST:
                    raise ValueError("DB_HOST is not configured")
                if not settings.USER:
                    raise ValueError("DB_USER is not configured")
                if not settings.NAME:
                    raise ValueError("DB_NAME is not configured")

                self.db_host = settings.HOST
                self.db_user = settings.USER
                self.db_password = settings.PASSWORD.get_secret_value()
                self.db_name = settings.NAME
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
                    f"{self.__class__.__name__} configured for {self.db_host}:{self.db_port}/{self.db_name} (CLI mode)"
                )
                logger.debug(f"Database user: {self.db_user}")
            else:
                # API mode: Using connection pool
                logger.info(
                    f"{self.__class__.__name__} configured with connection pool (API mode)"
                )
        except Exception as e:
            logger.error(f"Failed to initialize {self.__class__.__name__}: {str(e)}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Get a database connection with automatic cleanup.

        Uses connection pool if available (API mode), otherwise creates
        individual connection (CLI mode).

        Yields:
            Connection: PostgreSQL connection object

        Raises:
            ConnectionError: If unable to connect to database

        Example:
            >>> with client.get_connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM table")
        """
        if self._connection_pool:
            # API mode: Use pooled connection
            with self._connection_pool.get_connection() as conn:
                try:
                    logger.debug("Using pooled database connection")
                    yield conn
                    conn.commit()  # Auto-commit on success
                    logger.debug("Transaction committed successfully")
                except psycopg.Error as e:
                    conn.rollback()
                    logger.warning("Transaction rolled back due to error")
                    logger.error(f"Database error: {str(e)}")
                    raise ConnectionError(f"Database error: {e}") from e
                except Exception as e:
                    conn.rollback()
                    logger.error(
                        f"Unexpected error during database operation: {str(e)}"
                    )
                    raise
        else:
            # CLI mode: Create individual connection
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
            logger.debug(f"{self.__class__.__name__} connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit with cleanup."""
        self.close()
