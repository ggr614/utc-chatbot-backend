"""add_article_metadata_fields

Revision ID: bff14a8411a7
Revises: 6edea7230754
Create Date: 2026-02-05 10:45:25.942553

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bff14a8411a7'
down_revision: Union[str, Sequence[str], None] = '6edea7230754'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add metadata fields to articles table: status_name, category_name, is_public, summary, tags."""

    # Add new columns (all nullable for backward compatibility)
    op.execute("""
        ALTER TABLE articles
        ADD COLUMN status_name TEXT,
        ADD COLUMN category_name TEXT,
        ADD COLUMN is_public BOOLEAN,
        ADD COLUMN summary TEXT,
        ADD COLUMN tags TEXT[]
    """)

    # Create indexes for query optimization
    # B-tree indexes for scalar fields (fast filtering)
    op.create_index('idx_articles_status_name', 'articles', ['status_name'])
    op.create_index('idx_articles_category_name', 'articles', ['category_name'])
    op.create_index('idx_articles_is_public', 'articles', ['is_public'])

    # GIN index for array field (supports array containment queries)
    op.execute("""
        CREATE INDEX idx_articles_tags ON articles USING GIN(tags)
    """)


def downgrade() -> None:
    """Remove metadata fields from articles table."""

    # Drop indexes first (prevent foreign key constraint errors)
    op.execute("DROP INDEX IF EXISTS idx_articles_tags")
    op.drop_index('idx_articles_is_public', table_name='articles')
    op.drop_index('idx_articles_category_name', table_name='articles')
    op.drop_index('idx_articles_status_name', table_name='articles')

    # Drop columns (data will be lost!)
    op.execute("""
        ALTER TABLE articles
        DROP COLUMN IF EXISTS tags,
        DROP COLUMN IF EXISTS summary,
        DROP COLUMN IF EXISTS is_public,
        DROP COLUMN IF EXISTS category_name,
        DROP COLUMN IF EXISTS status_name
    """)
