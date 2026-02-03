"""
Query log endpoints for analytics and LLM response logging.

Provides endpoints for:
- Logging LLM responses
- Retrieving query logs with responses
- Analytics queries
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from datetime import datetime, timezone

from api.dependencies import verify_api_key, get_query_log_client
from api.models.requests import LogLLMResponseRequest
from api.models.responses import LogLLMResponseResponse
from core.storage_query_log import QueryLogClient
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/{query_log_id}/response",
    response_model=LogLLMResponseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Log LLM Response",
    description="Log an LLM-generated response for a query. Links response to query log for full conversation tracking.",
    tags=["Query Logs"],
)
def log_llm_response(
    query_log_id: int,
    request: LogLLMResponseRequest,
    api_key: Annotated[str, Depends(verify_api_key)],
    query_log_client: Annotated[QueryLogClient, Depends(get_query_log_client)],
) -> LogLLMResponseResponse:
    """
    Log an LLM-generated response for a specific query.

    **Use Case:**
    After retrieving RAG context and generating an LLM response, log the
    full response text along with metadata (model, tokens, latency, citations).

    **Request:**
    - `query_log_id`: Path parameter - the query log ID from search endpoint
    - `response_text`: Required - full LLM response text
    - `model_name`: Optional - LLM model identifier
    - `llm_latency_ms`: Optional - generation latency
    - `prompt_tokens`, `completion_tokens`, `total_tokens`: Optional - token counts
    - `citations`: Optional - JSONB with source URLs and chunk IDs
    - `metadata`: Optional - JSONB with model parameters

    **Response:**
    - `id`: LLM response ID
    - `query_log_id`: Associated query log ID
    - `created_at`: Timestamp
    - `message`: Success message

    **Errors:**
    - `400`: Invalid request (empty response, negative tokens, etc.)
    - `404`: Query log not found
    - `409`: Response already logged for this query
    - `500`: Database error

    **Note:** This endpoint is idempotent - attempting to log a second response
    for the same query_log_id will return 409 Conflict.
    """
    logger.info(
        f"Logging LLM response for query_log_id {query_log_id}: "
        f"model={request.model_name}, "
        f"response_length={len(request.response_text)} chars"
    )

    try:
        # Verify query log exists by fetching it
        query_log = query_log_client.get_query_by_id(query_log_id)
        if not query_log:
            logger.warning(f"Query log {query_log_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Query log {query_log_id} not found",
            )

        # Log the LLM response
        response_id = query_log_client.log_llm_response(
            query_log_id=query_log_id,
            response_text=request.response_text,
            model_name=request.model_name,
            llm_latency_ms=request.llm_latency_ms,
            prompt_tokens=request.prompt_tokens,
            completion_tokens=request.completion_tokens,
            total_tokens=request.total_tokens,
            citations=request.citations,
            metadata=request.metadata,
        )

        logger.info(
            f"LLM response logged successfully: response_id={response_id}, "
            f"query_log_id={query_log_id}"
        )

        # Get created_at timestamp
        created_at = datetime.now(timezone.utc)

        return LogLLMResponseResponse(
            id=response_id,
            query_log_id=query_log_id,
            created_at=created_at,
        )

    except ValueError as e:
        # Validation errors or integrity constraint violations
        error_msg = str(e)
        logger.warning(f"Validation error logging LLM response: {error_msg}")

        # Check if it's a duplicate (already logged)
        if "already logged" in error_msg.lower():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error_msg)

        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)

    except Exception as e:
        # Unexpected errors
        logger.error(
            f"Failed to log LLM response for query_log_id {query_log_id}: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to log LLM response",
        )


@router.get(
    "/{query_log_id}/response",
    summary="Get LLM Response",
    description="Retrieve the LLM response for a specific query log.",
    tags=["Query Logs"],
)
def get_llm_response(
    query_log_id: int,
    api_key: Annotated[str, Depends(verify_api_key)],
    query_log_client: Annotated[QueryLogClient, Depends(get_query_log_client)],
):
    """
    Retrieve the LLM response for a specific query log.

    **Use Case:**
    Fetch the full conversation: query → search results → LLM response

    **Returns:**
    - Full LLM response data if exists
    - 404 if no response logged for this query
    """
    logger.debug(f"Fetching LLM response for query_log_id {query_log_id}")

    try:
        response = query_log_client.get_llm_response_by_query_log_id(query_log_id)

        if not response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No LLM response found for query_log_id {query_log_id}",
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to fetch LLM response for query_log_id {query_log_id}: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch LLM response",
        )
