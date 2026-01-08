from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime, timezone
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
