from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import UUID


# Raw article
class TdxArticle(BaseModel):
    """
    Schema for the data retrieved from the TDX API
    """

    id: UUID | None = Field(
        default=None, description="Unique UUID in the database (auto-generated)."
    )
    tdx_article_id: int = Field(..., description="Original article ID from TDX API.")
    title: str
    content_html: str = Field(..., description="Raw HTML content of the article.")
    last_modified_date: datetime
    url: HttpUrl = Field(..., description="Public URL")
    raw_ingestion_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    status_name: str | None = Field(
        default=None,
        description="Article status from TDX API (e.g., 'Approved', 'Draft', 'Archived')",
    )
    category_name: str | None = Field(
        default=None,
        description="Article category from TDX API (e.g., 'IT Help', 'Documentation')",
    )
    is_public: bool | None = Field(
        default=None, description="Whether article is publicly visible in TDX"
    )
    summary: str | None = Field(
        default=None, description="Short summary/description of the article from TDX"
    )
    tags: List[str] | None = Field(
        default=None, description="List of tags associated with the article in TDX"
    )


# Chunk
class TextChunk(BaseModel):
    """
    Schema for a single text chunk derived from a TdxArticle.
    Used by: processing.py
    """

    # Unique identifier for the chunk
    chunk_id: UUID = Field(..., description="UUID for the chunk in the database.")

    # Parent/Metadata Linkage
    parent_article_id: UUID = Field(
        ..., description="The UUID of the source TdxArticle."
    )
    chunk_sequence: int = Field(
        ..., description="Sequence number of the chunk within the parent article."
    )

    # Content
    text_content: str = Field(
        ..., description="The clean, Markdown/Text content of the chunk."
    )

    # Contextual fields
    token_count: int
    source_url: HttpUrl
    last_modified_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Article metadata (optional, populated during filtered queries)
    article_tags: List[str] | None = Field(
        default=None, description="Tags from parent article (used for BM25 tag search)"
    )


# Embedding
class VectorRecord(BaseModel):
    """
    schema for the final record to be inserted into the database
    """

    # Unique identifier for the chunk
    chunk_id: UUID = Field(..., description="UUID for the chunk in the database.")

    # Parent/Metadata Linkage
    parent_article_id: UUID = Field(
        ..., description="The UUID of the source TdxArticle."
    )
    chunk_sequence: int = Field(
        ..., description="Sequence number of the chunk within the parent article."
    )

    # Content
    text_content: str = Field(
        ..., description="The clean, Markdown/Text content of the chunk."
    )

    # Contextual fields
    token_count: int
    source_url: HttpUrl
    last_modified_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# Cache Metrics
class CacheMetric(BaseModel):
    """
    Schema for cache performance metrics.

    Used for tracking cache hit rates, latency, and user analytics.
    This is an append-only table for analytics purposes.
    """

    # Primary key - auto-generated BIGSERIAL
    id: int | None = Field(
        default=None, description="Auto-generated BIGSERIAL primary key."
    )

    # Foreign key to warm_cache_entries (optional - can be NULL)
    cache_entry_id: UUID | None = Field(
        default=None,
        description="Reference to warm_cache_entries.id (NULL for cache misses).",
    )

    # Timestamp - auto-defaults to NOW() in database
    request_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the cache request occurred.",
    )

    # Cache metadata
    cache_type: str = Field(
        ..., description="Type of cache event: 'hit', 'miss', 'warm_hit', etc."
    )

    # Performance metrics
    latency_ms: int | None = Field(
        default=None, description="Response latency in milliseconds."
    )

    # User tracking
    user_id: str | None = Field(
        default=None, description="User identifier for analytics."
    )


# Cache analytics aggregation models
class CacheHitRateStats(BaseModel):
    """Aggregated cache hit rate statistics."""

    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float  # Percentage 0.0 to 100.0
    time_period_start: datetime
    time_period_end: datetime


class CacheLatencyStats(BaseModel):
    """Aggregated cache latency statistics."""

    avg_latency_ms: float
    min_latency_ms: int
    max_latency_ms: int
    p50_latency_ms: float  # Median
    p95_latency_ms: float
    p99_latency_ms: float
    total_requests: int
    time_period_start: datetime
    time_period_end: datetime


# Query Logging
class QueryLog(BaseModel):
    """
    Schema for query logging and analytics.

    Used for tracking user queries, their embeddings, cache performance,
    and latency metrics. This is an append-only table for analytics purposes.
    """

    # Primary key - auto-generated BIGSERIAL
    id: int | None = Field(
        default=None, description="Auto-generated BIGSERIAL primary key."
    )

    # Query data
    raw_query: str = Field(..., description="The original user query text.")

    # Vector embedding (optional - may not be generated for all queries)
    query_embedding: List[float] | None = Field(
        default=None,
        description="Vector embedding of the query (3072 dimensions for OpenAI).",
    )

    # Cache result tracking
    cache_result: str = Field(
        ...,
        description="Cache result type: 'hit', 'miss', 'warm_hit', 'partial_hit', etc.",
    )

    # Performance metrics
    latency_ms: int | None = Field(
        default=None, description="Query response latency in milliseconds."
    )

    # User tracking
    user_id: str | None = Field(
        default=None, description="User identifier for analytics."
    )

    # Command tracking
    command: str | None = Field(
        default=None,
        description="Command type used: 'bypass', 'q', 'qlong', 'debug', 'debuglong', or NULL.",
    )

    # Timestamp - auto-defaults to NOW() in database
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the query was logged.",
    )


# Query analytics aggregation models
class QueryLatencyStats(BaseModel):
    """Aggregated query latency statistics."""

    avg_latency_ms: float
    min_latency_ms: int
    max_latency_ms: int
    p50_latency_ms: float  # Median
    p95_latency_ms: float
    p99_latency_ms: float
    total_queries: int
    time_period_start: datetime
    time_period_end: datetime


class PopularQuery(BaseModel):
    """Popular query with frequency count."""

    raw_query: str
    query_count: int
    avg_latency_ms: float
    cache_hit_rate: float  # Percentage 0.0 to 100.0
    last_queried_at: datetime


class QueryCacheStats(BaseModel):
    """Cache performance statistics for queries."""

    total_queries: int
    cache_hits: int
    cache_misses: int
    hit_rate: float  # Percentage 0.0 to 100.0
    time_period_start: datetime
    time_period_end: datetime


# Query Results Logging
class QueryResult(BaseModel):
    """
    Schema for a single query result record.

    Represents one result (chunk) returned for a specific query.
    Used for logging retrieval results to enable effectiveness evaluation.
    """

    # Primary key - auto-generated BIGSERIAL
    id: int | None = Field(
        default=None, description="Auto-generated BIGSERIAL primary key."
    )

    # Foreign key to query_logs
    query_log_id: int = Field(..., description="Reference to query_logs.id")

    # Search metadata
    search_method: str = Field(
        ..., description="Search method used: 'bm25', 'vector', or 'hybrid'"
    )

    # Result metadata
    rank: int = Field(..., description="Result position (1-indexed)", ge=1)

    score: float = Field(
        ..., description="Relevance score (BM25 score, similarity, or hybrid score)"
    )

    # Chunk/article references
    chunk_id: UUID = Field(..., description="Reference to article_chunks.id")

    parent_article_id: UUID = Field(..., description="Reference to articles.id")

    # Timestamp
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the result was logged.",
    )


# Query results analytics aggregation models
class ArticleTopKStats(BaseModel):
    """Aggregated statistics for article appearance in top-k results."""

    article_id: UUID
    search_method: str | None  # None = all methods
    top_k: int
    total_queries: int
    top_k_appearances: int
    top_k_rate: float  # Percentage 0.0 to 100.0
    avg_rank: float  # When appearing
    avg_score: float  # When appearing
    time_period_start: datetime | None
    time_period_end: datetime | None


class RetrievalMethodStats(BaseModel):
    """Aggregated statistics for a retrieval method's effectiveness."""

    search_method: str
    total_queries: int
    avg_score: float
    min_score: float
    max_score: float
    p50_score: float
    p95_score: float
    p99_score: float
    time_period_start: datetime | None
    time_period_end: datetime | None


# LLM Response Logging
class LLMResponse(BaseModel):
    """
    Schema for logging LLM-generated responses.

    Used for tracking full conversation lifecycle: query → search → LLM response.
    Links to query_logs table via query_log_id for complete conversation analytics.
    """

    # Primary key - auto-generated BIGSERIAL
    id: int | None = Field(
        default=None, description="Auto-generated BIGSERIAL primary key."
    )

    # Foreign key to query_logs (1:1 relationship)
    query_log_id: int = Field(
        ..., description="Reference to query_logs.id (1:1 relationship)"
    )

    # LLM response data
    response_text: str = Field(
        ..., description="Full text response generated by the LLM"
    )

    model_name: str | None = Field(
        default=None,
        description="LLM model identifier (e.g., 'gpt-4', 'claude-3-sonnet')",
    )

    # Performance metrics
    llm_latency_ms: int | None = Field(
        default=None, description="LLM generation latency in milliseconds"
    )

    # Token counts (for cost tracking)
    prompt_tokens: int | None = Field(
        default=None, description="Number of tokens in prompt"
    )

    completion_tokens: int | None = Field(
        default=None, description="Number of tokens in completion"
    )

    total_tokens: int | None = Field(
        default=None, description="Total tokens (prompt + completion)"
    )

    # Citations/sources (JSONB)
    citations: Dict[str, Any] | None = Field(
        default=None,
        description="Citations data: num_documents_used, source_urls, chunk_ids",
    )

    # Additional metadata (JSONB)
    metadata: Dict[str, Any] | None = Field(
        default=None,
        description="LLM parameters: temperature, top_p, max_tokens, etc.",
    )

    # Timestamp
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the response was logged",
    )


# ============================================================================
# Reranker Logging Schemas
# ============================================================================


class RerankerLog(BaseModel):
    """
    Schema for reranker operation logging.

    Tracks aggregate metadata about a reranking operation.
    Links 1:1 with query_logs table.
    """

    id: int | None = Field(
        default=None, description="Auto-generated BIGSERIAL primary key"
    )

    query_log_id: int = Field(
        ..., description="Reference to query_logs.id (1:1 relationship)"
    )

    reranker_status: str = Field(
        ..., description="Reranking status: 'success', 'failed', 'skipped'"
    )

    model_name: str | None = Field(
        default=None, description="Reranker model ID (e.g., 'cohere.rerank-v3-5:0')"
    )

    reranker_latency_ms: int | None = Field(
        default=None, description="Reranking API latency in milliseconds"
    )

    num_candidates: int = Field(
        ..., description="Number of RRF results sent to reranker"
    )

    num_reranked: int | None = Field(
        default=None, description="Number of results returned from reranker"
    )

    fallback_method: str | None = Field(
        default=None, description="Fallback method if reranking failed (e.g., 'rrf')"
    )

    error_message: str | None = Field(
        default=None, description="Error message if reranking failed"
    )

    avg_rank_change: float | None = Field(
        default=None, description="Average rank movement across all results"
    )

    top_k_stability_score: float | None = Field(
        default=None, description="% of top-k results that stayed in top-k"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When reranking was logged",
    )


class RerankerResult(BaseModel):
    """
    Schema for individual reranked result.

    Tracks a single result's ranking before (RRF) and after (Cohere reranking).
    """

    id: int | None = Field(
        default=None, description="Auto-generated BIGSERIAL primary key"
    )

    query_log_id: int = Field(..., description="Reference to query_logs.id")

    chunk_id: UUID = Field(..., description="Reference to article_chunks.id")

    parent_article_id: UUID = Field(..., description="Reference to articles.id")

    rrf_rank: int = Field(
        ..., description="Initial rank from RRF fusion (1-indexed)", ge=1
    )

    rrf_score: float = Field(..., description="Initial RRF score", ge=0.0)

    reranked_rank: int = Field(
        ..., description="Final rank after Cohere reranking (1-indexed)", ge=1
    )

    reranked_score: float = Field(
        ..., description="Final Cohere relevance score (0-1)", ge=0.0, le=1.0
    )

    rank_change: int = Field(
        ...,
        description="Rank movement (rrf_rank - reranked_rank, positive = moved up)",
    )

    model_name: str = Field(..., description="Reranker model ID")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When result was logged",
    )
