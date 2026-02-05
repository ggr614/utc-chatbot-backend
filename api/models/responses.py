"""
Pydantic response models for search API endpoints.

Provides consistent response structure for:
- Search results (BM25, vector, hybrid)
- BM25 validation results (minimal format for keyword validation)
- Health checks
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Literal, Optional
from uuid import UUID
from datetime import datetime


class SearchResultChunk(BaseModel):
    """
    Individual search result chunk with metadata.

    Contains the ranked search result with score, chunk content,
    and source article information.
    """

    rank: int = Field(
        ..., description="Result ranking position (1-indexed)", examples=[1]
    )
    score: float = Field(
        ...,
        description="Relevance score (BM25 score, similarity, or combined hybrid score)",
        examples=[5.5],
    )
    chunk_id: UUID = Field(..., description="Unique chunk identifier")
    parent_article_id: UUID = Field(..., description="Source article UUID")
    chunk_sequence: int = Field(
        ..., description="Chunk position within article (0-indexed)", examples=[0]
    )
    text_content: str = Field(
        ...,
        description="Clean text content of the chunk",
        examples=["To reset your password, navigate to the login page..."],
    )
    token_count: int = Field(
        ..., description="Number of tokens in chunk", examples=[150]
    )
    source_url: HttpUrl = Field(
        ...,
        description="Source article URL",
        examples=["https://help.example.com/article/123"],
    )
    last_modified_date: datetime = Field(
        ..., description="Article last modified timestamp"
    )


class SearchResponse(BaseModel):
    """
    Search API response with ranked results and metadata.

    Returned by all search endpoints (BM25, vector, hybrid).
    """

    query: str = Field(
        ..., description="Original query text", examples=["password reset"]
    )
    method: Literal["bm25", "vector", "hybrid", "hyde"] = Field(
        ..., description="Search method used", examples=["bm25"]
    )
    results: List[SearchResultChunk] = Field(
        ..., description="Ranked search results (empty list if no results)"
    )
    total_results: int = Field(
        ..., description="Number of results returned", examples=[10]
    )
    latency_ms: int = Field(
        ..., description="Query processing latency in milliseconds", examples=[156]
    )
    metadata: dict = Field(
        default_factory=dict,
        description=(
            "Additional method-specific metadata. "
            "For hybrid/hyde endpoints: rrf_k (int), reranked (bool), "
            "reranker_status ('success'|'failed'|'unavailable'), "
            "reranker_latency_ms (int), num_candidates_reranked (int), "
            "fallback_method (str), error (str). "
            "For hyde: hyde_latency_ms (int), hypothetical_document (str), token_usage (dict)."
        ),
        examples=[
            {
                "rrf_k": 60,
                "reranked": True,
                "reranker_status": "success",
                "reranker_latency_ms": 245,
            }
        ],
    )
    query_log_id: Optional[int] = Field(
        default=None,
        description="Query log ID for this search (for LLM response logging)",
        examples=[12345],
    )


class BM25ValidationResult(BaseModel):
    """
    Minimal validation result with IDs and scores only.

    Used by the BM25 validation endpoint for keyword query validation
    and score analysis. Excludes text content for minimal response size.
    """

    rank: int = Field(
        ..., description="Result ranking position (1-indexed)", examples=[1]
    )
    score: float = Field(..., description="BM25 relevance score", examples=[5.5])
    chunk_id: UUID = Field(..., description="Unique chunk identifier")
    parent_article_id: UUID = Field(..., description="Source article UUID")
    chunk_sequence: int = Field(
        ..., description="Chunk position within article (0-indexed)", examples=[0]
    )
    source_url: HttpUrl = Field(
        ...,
        description="Source article URL",
        examples=["https://help.example.com/article/123"],
    )


class BM25ValidationResponse(BaseModel):
    """
    Validation search response with minimal result format.

    Returns all matching chunks with scores (IDs only, no text content).
    Designed for keyword validation and score distribution analysis.
    """

    query: str = Field(
        ..., description="Original query text", examples=["password reset"]
    )
    total_results: int = Field(
        ..., description="Number of results returned", examples=[150]
    )
    results: List[BM25ValidationResult] = Field(
        ..., description="All ranked results (minimal format)"
    )
    latency_ms: int = Field(
        ..., description="Query processing latency in milliseconds", examples=[85]
    )
    metadata: dict = Field(
        default_factory=dict,
        description=(
            "Additional metadata: min_score_filter (float|None), "
            "top_k_limit (int|None), returned_all_results (bool)"
        ),
        examples=[
            {
                "min_score_filter": None,
                "top_k_limit": None,
                "returned_all_results": True,
            }
        ],
    )


class HealthResponse(BaseModel):
    """
    Health check response with component status.

    Used by monitoring systems and load balancers to verify service health.
    """

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Overall health status", examples=["healthy"]
    )
    timestamp: datetime = Field(..., description="Health check timestamp")
    checks: dict = Field(
        ...,
        description="Individual component health checks (bm25, vector, database)",
        examples=[
            {
                "bm25": {"status": "healthy", "num_chunks": 1500, "cached": True},
                "vector": {"status": "healthy", "num_embeddings": 1500},
                "database": {"status": "healthy", "pool_size": 10},
            }
        ],
    )


class LogLLMResponseResponse(BaseModel):
    """
    Response model for successful LLM response logging.
    """

    id: int = Field(..., description="LLM response ID", examples=[12345])
    query_log_id: int = Field(
        ..., description="Associated query log ID", examples=[67890]
    )
    created_at: datetime = Field(..., description="Timestamp when logged")
    message: str = Field(
        default="LLM response logged successfully", description="Success message"
    )
