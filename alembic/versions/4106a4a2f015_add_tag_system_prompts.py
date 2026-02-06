"""add_tag_system_prompts

Revision ID: 4106a4a2f015
Revises: 68c23f1aaa54
Create Date: 2026-02-06 10:30:39.355412

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4106a4a2f015'
down_revision: Union[str, Sequence[str], None] = '68c23f1aaa54'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add tag_system_prompts table for storing tag-based system prompts."""

    # Create tag_system_prompts table
    op.execute("""
        CREATE TABLE tag_system_prompts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tag_name TEXT UNIQUE NOT NULL,
            system_prompt TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            description TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Create indexes for query optimization
    op.create_index("idx_tag_system_prompts_tag", "tag_system_prompts", ["tag_name"])
    op.execute("CREATE INDEX idx_tag_system_prompts_priority ON tag_system_prompts(priority DESC)")

    # Add table comment for documentation
    op.execute("""
        COMMENT ON TABLE tag_system_prompts IS
        'Maps article tags to system prompts for LLM responses'
    """)

    op.execute("""
        COMMENT ON COLUMN tag_system_prompts.tag_name IS
        'Tag name from articles.tags array. Use "__default__" for fallback prompt.'
    """)

    op.execute("""
        COMMENT ON COLUMN tag_system_prompts.priority IS
        'Higher priority wins when article has multiple tags. Default: 0.'
    """)

    # Seed default prompt (high priority fallback)
    op.execute("""
        INSERT INTO tag_system_prompts (tag_name, system_prompt, priority, description)
        VALUES (
            '__default__',
            'You are a helpful IT helpdesk assistant. Provide clear, accurate answers based on the knowledge base.',
            1000,
            'Default system prompt used when no tag-specific prompt matches'
        )
    """)


def downgrade() -> None:
    """Remove tag_system_prompts table."""

    # Drop indexes first
    op.execute("DROP INDEX IF EXISTS idx_tag_system_prompts_priority")
    op.drop_index("idx_tag_system_prompts_tag", table_name="tag_system_prompts")

    # Drop table (data will be lost!)
    op.drop_table("tag_system_prompts")
