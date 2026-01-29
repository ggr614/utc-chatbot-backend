"""add_query_results

Revision ID: af41a58144ab
Revises: f6c4e9a6ae4d
Create Date: 2026-01-29 10:20:41.222013

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'af41a58144ab'
down_revision: Union[str, Sequence[str], None] = 'f6c4e9a6ae4d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create query_results table for retrieval effectiveness evaluation."""

    # Create query_results table
    op.execute("""
        CREATE TABLE query_results (
            id BIGSERIAL PRIMARY KEY,
            query_log_id BIGINT NOT NULL,
            search_method TEXT NOT NULL,
            rank INTEGER NOT NULL,
            score DOUBLE PRECISION NOT NULL,
            chunk_id UUID NOT NULL,
            parent_article_id UUID NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),

            FOREIGN KEY (query_log_id) REFERENCES query_logs(id) ON DELETE CASCADE,
            FOREIGN KEY (chunk_id) REFERENCES article_chunks(id) ON DELETE CASCADE,
            FOREIGN KEY (parent_article_id) REFERENCES articles(id) ON DELETE CASCADE,

            CHECK (rank >= 1),
            CHECK (search_method IN ('bm25', 'vector', 'hybrid'))
        )
    """)

    # Create indexes for analytics queries
    op.create_index(
        'idx_query_results_query_log',
        'query_results',
        ['query_log_id', 'rank']
    )

    op.create_index(
        'idx_query_results_article',
        'query_results',
        ['parent_article_id', 'rank']
    )

    op.create_index(
        'idx_query_results_method',
        'query_results',
        ['search_method']
    )

    op.create_index(
        'idx_query_results_chunk',
        'query_results',
        ['chunk_id']
    )

    op.create_index(
        'idx_query_results_created_at',
        'query_results',
        ['created_at']
    )

    op.create_index(
        'idx_query_results_article_rank',
        'query_results',
        ['parent_article_id', 'rank', 'search_method']
    )


def downgrade() -> None:
    """Drop query_results table."""
    op.drop_table('query_results')
