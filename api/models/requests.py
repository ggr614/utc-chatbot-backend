"""
Pydantic request models for search API endpoints.

Provides validation for:
- BM25 keyword search requests
- Vector semantic search requests
- Hybrid search requests (combining BM25 and vector)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal


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
    user_id: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional user identifier for analytics",
        examples=["user123"],
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate that query is not empty or whitespace only."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or whitespace only")
        return stripped


class BM25SearchRequest(SearchRequest):
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


class VectorSearchRequest(SearchRequest):
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


class HybridSearchRequest(SearchRequest):
    """
    Hybrid search request combining BM25 and vector retrieval.

    Provides best of both worlds: keyword precision + semantic understanding.
    Uses Reciprocal Rank Fusion (RRF) or weighted score fusion to combine results.
    """

    bm25_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 results (0-1). Vector weight = 1 - bm25_weight. Only used with 'weighted' fusion.",
        examples=[0.5],
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
    fusion_method: Literal["rrf", "weighted"] = Field(
        default="rrf",
        description="Fusion method: 'rrf' (Reciprocal Rank Fusion, rank-based) or 'weighted' (score-based)",
        examples=["rrf"],
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="RRF constant (only used with 'rrf' fusion). Typical value: 60.",
        examples=[60],
    )
