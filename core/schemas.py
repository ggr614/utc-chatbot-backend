from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime, timezone

# Raw article
class TdxArticle(BaseModel):
    """
    Schema for the data retrieved from the TDX API
    """
    id: int = Field(...,description="Unique ID provided by the TDX system.")
    title: str
    content_html: str = Field(...,description="Raw HTML content of the article.")
    last_modified_date: datetime
    url: HttpUrl = Field(..., description="Public URL")
    raw_ingestion_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Chunk
class TextChunk(BaseModel):
    """
    Schema for a single text chunk derived from a TdxArticle.
    Used by: processing.py
    """
    # Unique identifier for the chunk
    chunk_id: str = Field(..., description="A unique hash generated from parent ID, text, and sequence.")
    
    # Parent/Metadata Linkage
    parent_article_id: int = Field(..., description="The ID of the source TdxArticle.")
    chunk_sequence: int = Field(..., description="Sequence number of the chunk within the parent article.")
    
    # Content
    text_content: str = Field(..., description="The clean, Markdown/Text content of the chunk.")
    
    # Contextual fields
    token_count: int
    source_url: HttpUrl

#Embedding
class VectorRecord(BaseModel):
    """
    schema for the final record to be inserted into the database
    """
    # Unique identifier for the chunk
    chunk_id: str = Field(..., description="A unique hash generated from parent ID, text, and sequence.")
    
    # Parent/Metadata Linkage
    parent_article_id: int = Field(..., description="The ID of the source TdxArticle.")
    chunk_sequence: int = Field(..., description="Sequence number of the chunk within the parent article.")
    
    # Content
    text_content: str = Field(..., description="The clean, Markdown/Text content of the chunk.")
    
    # Contextual fields
    token_count: int
    source_url: HttpUrl