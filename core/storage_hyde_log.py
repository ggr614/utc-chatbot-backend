"""
Storage module for HyDE logging and analytics.

This module provides a client for logging HyDE (Hypothetical Document Embeddings)
generation performance with hypothetical documents, token usage, and latency metrics.
The hyde_logs table uses BIGSERIAL as primary key (not UUID) and is append-only for analytics.
"""

from typing import Optional

import psycopg

from core.storage_base import BaseStorageClient
from utils.logger import get_logger

logger = get_logger(__name__)


class HyDELogClient(BaseStorageClient):
    """
    Storage client for HyDE logging and analytics.

    Handles logging of HyDE generation operations with the hyde_logs table.
    Links 1:1 with query_logs table via query_log_id.

    Key differences from other storage clients:
    - Uses BIGSERIAL primary key (not UUID)
    - Links 1:1 with query_logs (UNIQUE constraint on query_log_id)
    - Tracks token usage and latency for cost analysis
    - Stores generated hypothetical documents for quality review
    """

    def __init__(self, connection_pool=None):
        """
        Initialize the HyDELogClient for hyde_logs table.

        Args:
            connection_pool: Optional DatabaseConnectionPool instance for API mode
        """
        super().__init__(connection_pool=connection_pool)
        logger.info("HyDELogClient initialized")

    def log_hyde_generation(
        self,
        query_log_id: int,
        hypothetical_document: str,
        generation_status: str,
        model_name: str,
        generation_latency_ms: int,
        embedding_latency_ms: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> int:
        """
        Log HyDE generation operation to the hyde_logs table.

        Args:
            query_log_id: FK to query_logs.id
            hypothetical_document: Generated hypothetical document text
            generation_status: Status ('success' or 'failed_fallback')
            model_name: Model used for generation (e.g., 'gpt-4o')
            generation_latency_ms: Time to generate hypothetical document (ms)
            embedding_latency_ms: Time to embed hypothetical document (ms)
            prompt_tokens: Number of tokens in prompt
            completion_tokens: Number of tokens in completion
            total_tokens: Total tokens (prompt + completion)
            error_message: Error message if generation failed

        Returns:
            The hyde_logs.id (BIGSERIAL)

        Raises:
            ValueError: If validation fails
            ConnectionError: If database operation fails

        Example:
            >>> client = HyDELogClient()
            >>> log_id = client.log_hyde_generation(
            ...     query_log_id=12345,
            ...     hypothetical_document="To reset your password, navigate to...",
            ...     generation_status="success",
            ...     model_name="gpt-4o",
            ...     generation_latency_ms=850,
            ...     embedding_latency_ms=320,
            ...     prompt_tokens=45,
            ...     completion_tokens=67,
            ...     total_tokens=112
            ... )
        """
        # Validation
        if not hypothetical_document or not hypothetical_document.strip():
            raise ValueError("hypothetical_document cannot be empty")

        if generation_status not in ("success", "failed_fallback"):
            raise ValueError(
                f"generation_status must be 'success' or 'failed_fallback', got '{generation_status}'"
            )

        if generation_latency_ms < 0:
            raise ValueError(
                f"generation_latency_ms must be >= 0, got {generation_latency_ms}"
            )

        if embedding_latency_ms is not None and embedding_latency_ms < 0:
            raise ValueError(
                f"embedding_latency_ms must be >= 0, got {embedding_latency_ms}"
            )

        if prompt_tokens is not None and prompt_tokens < 0:
            raise ValueError(f"prompt_tokens must be >= 0, got {prompt_tokens}")

        if completion_tokens is not None and completion_tokens < 0:
            raise ValueError(f"completion_tokens must be >= 0, got {completion_tokens}")

        if total_tokens is not None and total_tokens < 0:
            raise ValueError(f"total_tokens must be >= 0, got {total_tokens}")

        logger.debug(
            f"Logging HyDE generation for query_log_id {query_log_id}: "
            f"status={generation_status}, latency={generation_latency_ms}ms, "
            f"tokens={total_tokens}"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO hyde_logs
                        (query_log_id, hypothetical_document, generation_status,
                         model_name, generation_latency_ms, embedding_latency_ms,
                         prompt_tokens, completion_tokens, total_tokens, error_message)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            query_log_id,
                            hypothetical_document,
                            generation_status,
                            model_name,
                            generation_latency_ms,
                            embedding_latency_ms,
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                            error_message,
                        ),
                    )

                    result = cur.fetchone()
                    if not result:
                        raise ConnectionError("Failed to retrieve hyde_log ID")
                    hyde_log_id = result[0]

                    logger.info(
                        f"HyDE generation logged successfully: ID {hyde_log_id}, "
                        f"query_log_id {query_log_id}, status {generation_status}"
                    )

                    return hyde_log_id

        except psycopg.IntegrityError as e:
            # Likely duplicate query_log_id (UNIQUE constraint)
            logger.error(f"Integrity error logging HyDE generation: {str(e)}")
            raise ConnectionError(
                f"HyDE generation already logged for query_log_id {query_log_id}"
            ) from e

        except Exception as e:
            logger.error(f"Failed to log HyDE generation: {str(e)}")
            raise
