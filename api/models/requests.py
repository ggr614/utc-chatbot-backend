"""
Pydantic request models for search API endpoints.

Provides validation for:
- BM25 keyword search requests
- BM25 validation requests (keyword query validation)
- Vector semantic search requests
- Hybrid search requests (combining BM25 and vector)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional


class SearchRequest(BaseModel):
    """
    Base search request model with common fields.

    All search endpoints accept these base parameters.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query text",
        examples=["password reset troubleshooting"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
        examples=[10],
    )
    email: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional user email address for analytics",
        examples=["user@example.com"],
    )
    command: Optional[str] = Field(
        default=None,
        max_length=10,
        description="Command type used: 'bypass', 'q', 'qlong', 'debug', 'debuglong', or NULL",
        examples=["q"],
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate that query is not empty or whitespace only."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or whitespace only")
        return stripped

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: Optional[str]) -> Optional[str]:
        """Validate that command is one of the allowed values."""
        if v is None:
            return v
        valid_commands = {"bypass", "q", "qlong", "debug", "debuglong"}
        if v not in valid_commands:
            raise ValueError(f"Command must be one of {valid_commands}, got '{v}'")
        return v


class ArticleFilterParams(BaseModel):
    """
    Common article metadata filters for all search methods.

    Filters are applied using AND logic. Within list filters (status_names, category_names, tags),
    OR logic is used (ANY match). For tags, articles must have at least one of the provided tags.

    By default, only 'Approved' articles are searched to ensure quality results.
    Override by explicitly setting status_names to null or other values.
    """

    status_names: Optional[List[str]] = Field(
        default=["Approved"],
        description="Filter by article status (defaults to ['Approved']). Set to null to search all statuses.",
        examples=[["Approved", "Published"], ["Approved"]],
    )
    category_names: Optional[List[str]] = Field(
        default=None,
        description="Filter by article category (e.g., ['IT Help', 'Documentation'])",
        examples=[["IT Help", "Network"], ["Documentation"]],
    )
    is_public: Optional[bool] = Field(
        default=None,
        description="Filter by article visibility (true for public, false for private)",
        examples=[True],
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Filter by article tags (ANY match, e.g., ['vpn', 'network'])",
        examples=[["vpn", "network", "remote-access"]],
    )


class BM25SearchRequest(SearchRequest, ArticleFilterParams):
    """
    BM25 sparse keyword retrieval request.

    Best for exact keyword matching, technical terms, and identifiers.
    """

    min_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Minimum BM25 score threshold (unbounded, typically 0-20)",
        examples=[1.0],
    )


class BM25ValidationRequest(BaseModel):
    """
    BM25 validation request for keyword query validation.

    Returns ALL BM25 scores (or up to top_k if specified) with minimal
    response format (IDs and scores only). Intended for keyword validation
    and score analysis.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query text for validation",
        examples=["password reset"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10000,
        description="Optional limit on number of results (default: return all)",
        examples=[100],
    )
    min_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Minimum BM25 score threshold",
        examples=[1.0],
    )
    email: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional user email address for analytics",
        examples=["user@example.com"],
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate that query is not empty or whitespace only."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or whitespace only")
        return stripped


class VectorSearchRequest(SearchRequest, ArticleFilterParams):
    """
    Vector dense semantic retrieval request.

    Best for natural language queries, conceptual similarity, and paraphrases.
    Note: Makes API call to Azure OpenAI for query embedding (~500ms-1s latency).
    """

    min_similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity threshold (0.0-1.0)",
        examples=[0.7],
    )


class HybridSearchRequest(SearchRequest, ArticleFilterParams):
    """
    Hybrid search request with RRF fusion and Cohere reranking.

    Combines BM25 keyword search and vector semantic search using
    Reciprocal Rank Fusion, then applies Cohere neural reranking
    for optimal relevance.

    Workflow:
    1. BM25 + Vector search (fetch_top_k results each)
    2. RRF fusion (combines and deduplicates)
    3. Cohere reranking (neural relevance scoring)
    4. Return top_k results
    """

    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results to return after reranking",
        examples=[5],
    )
    fetch_top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of candidates to fetch from each retriever (BM25 and vector) before fusion",
        examples=[20],
    )
    min_bm25_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Minimum BM25 score threshold",
        examples=[1.0],
    )
    min_vector_similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum vector similarity threshold (0.0-1.0)",
        examples=[0.7],
    )
    rrf_k: int = Field(
        default=1,
        ge=1,
        description="RRF constant (controls rank-based weighting). Lower values give more weight to top ranks.",
        examples=[1],
    )


class HyDESearchRequest(SearchRequest, ArticleFilterParams):
    """
    HyDE search request with hypothetical document generation.

    Generates a hypothetical document from the query using an LLM,
    then performs hybrid search (BM25 + vector) with reranking.

    HyDE (Hypothetical Document Embeddings) improves semantic matching
    by embedding a generated answer rather than the raw query.

    Workflow:
    1. Generate hypothetical document via LLM (~500-1000ms, uses model default temperature)
    2. BM25 search with original query (fetch_top_k results)
    3. Vector search with hypothetical document (fetch_top_k results)
    4. RRF fusion
    5. Cohere reranking with original query
    6. Return top_k results

    Note: Higher latency (~1.5-2.5s) due to LLM generation step.
    """

    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results to return after reranking",
        examples=[5],
    )
    fetch_top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of candidates to fetch from each retriever (BM25 and vector) before fusion",
        examples=[20],
    )
    min_bm25_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Minimum BM25 score threshold",
        examples=[1.0],
    )
    min_vector_similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum vector similarity threshold (0.0-1.0)",
        examples=[0.7],
    )
    rrf_k: int = Field(
        default=1,
        ge=1,
        description="RRF constant (controls rank-based weighting). Lower values give more weight to top ranks.",
        examples=[1],
    )


class LogLLMResponseRequest(BaseModel):
    """
    Request model for logging LLM-generated responses.

    Required: response_text
    Optional: All other fields for metadata and analytics
    """

    response_text: str = Field(
        ...,
        description="Full text response generated by the LLM",
        min_length=1,
        max_length=100000,  # 100k chars max
    )

    model_name: Optional[str] = Field(
        default=None,
        description="LLM model identifier (e.g., 'gpt-4', 'claude-3-sonnet')",
        max_length=100,
    )

    llm_latency_ms: Optional[int] = Field(
        default=None,
        description="LLM generation latency in milliseconds",
        ge=0,
    )

    prompt_tokens: Optional[int] = Field(
        default=None,
        description="Number of tokens in prompt",
        ge=0,
    )

    completion_tokens: Optional[int] = Field(
        default=None,
        description="Number of tokens in completion",
        ge=0,
    )

    total_tokens: Optional[int] = Field(
        default=None,
        description="Total tokens (prompt + completion)",
        ge=0,
    )

    citations: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Citations metadata (source URLs, chunk IDs, etc.)",
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="LLM parameters (temperature, top_p, etc.)",
    )

    @field_validator("response_text")
    @classmethod
    def validate_response_text(cls, v: str) -> str:
        """Validate that response text is not empty or whitespace only."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Response text cannot be empty or whitespace only")
        return stripped
