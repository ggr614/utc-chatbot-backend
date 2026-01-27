"""initial_schema

Revision ID: f6c4e9a6ae4d
Revises:
Create Date: 2026-01-27 12:01:13.914991

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f6c4e9a6ae4d"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "vector"')

    # Create articles table
    op.execute("""
        CREATE TABLE articles (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tdx_article_id INTEGER UNIQUE NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            content_html TEXT NOT NULL,
            last_modified_date TIMESTAMP WITH TIME ZONE NOT NULL,
            raw_ingestion_date TIMESTAMP WITH TIME ZONE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indexes for articles
    op.create_index("idx_articles_tdx_id", "articles", ["tdx_article_id"])
    op.create_index("idx_articles_last_modified", "articles", ["last_modified_date"])
    op.create_index("idx_articles_ingestion_date", "articles", ["raw_ingestion_date"])

    # Create article_chunks table
    op.execute("""
        CREATE TABLE article_chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            parent_article_id UUID NOT NULL,
            chunk_sequence INTEGER NOT NULL,
            text_content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            url TEXT NOT NULL,
            last_modified_date TIMESTAMP WITH TIME ZONE NOT NULL,
            FOREIGN KEY (parent_article_id) REFERENCES articles(id) ON DELETE CASCADE
        )
    """)

    # Create indexes for article_chunks
    op.create_index(
        "idx_chunks_parent_article", "article_chunks", ["parent_article_id"]
    )
    op.create_index(
        "idx_chunks_chunk_sequence",
        "article_chunks",
        ["parent_article_id", "chunk_sequence"],
    )

    # Create embeddings_openai table
    op.execute("""
        CREATE TABLE embeddings_openai (
            chunk_id UUID PRIMARY KEY,
            embedding vector(3072),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chunk_id) REFERENCES article_chunks(id) ON DELETE CASCADE
        )
    """)

    # Create index for embeddings_openai
    op.create_index(
        "idx_embeddings_openai_created_at", "embeddings_openai", ["created_at"]
    )

    # Create warm_cache_entries table
    op.execute("""
        CREATE TABLE warm_cache_entries (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            canonical_question TEXT NOT NULL,
            verified_answer TEXT NOT NULL,
            query_embedding vector(3072),
            article_id UUID NOT NULL,
            is_active BOOL NOT NULL,
            FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
        )
    """)

    # Create cache_metrics table
    op.execute("""
        CREATE TABLE cache_metrics (
            id BIGSERIAL PRIMARY KEY,
            cache_entry_id UUID REFERENCES warm_cache_entries(id) ON DELETE SET NULL,
            request_timestamp TIMESTAMPTZ DEFAULT NOW(),
            cache_type TEXT NOT NULL,
            latency_ms INT,
            user_id TEXT
        )
    """)

    # Create query_logs table
    op.execute("""
        CREATE TABLE query_logs (
            id BIGSERIAL PRIMARY KEY,
            raw_query TEXT NOT NULL,
            query_embedding vector(3072),
            cache_result TEXT NOT NULL,
            latency_ms INT,
            user_id TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop tables in reverse order (respecting foreign key constraints)
    op.drop_table("query_logs")
    op.drop_table("cache_metrics")
    op.drop_table("warm_cache_entries")
    op.drop_table("embeddings_openai")
    op.drop_table("article_chunks")
    op.drop_table("articles")

    # Drop extensions
    op.execute('DROP EXTENSION IF EXISTS "vector"')
