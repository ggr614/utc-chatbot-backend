"""add_llm_responses

Revision ID: 46eb84522090
Revises: af41a58144ab
Create Date: 2026-01-29 14:12:43.561334

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "46eb84522090"
down_revision: Union[str, Sequence[str], None] = "af41a58144ab"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create llm_responses table for logging LLM-generated answers."""

    # Create llm_responses table
    op.execute("""
        CREATE TABLE llm_responses (
            id BIGSERIAL PRIMARY KEY,
            query_log_id BIGINT NOT NULL UNIQUE,
            response_text TEXT NOT NULL,
            model_name TEXT,
            llm_latency_ms INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            citations JSONB,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),

            FOREIGN KEY (query_log_id) REFERENCES query_logs(id) ON DELETE CASCADE,

            CHECK (llm_latency_ms >= 0),
            CHECK (prompt_tokens >= 0),
            CHECK (completion_tokens >= 0),
            CHECK (total_tokens >= 0)
        )
    """)

    # Create indexes
    op.create_index(
        "idx_llm_responses_query_log", "llm_responses", ["query_log_id"], unique=True
    )

    op.create_index("idx_llm_responses_created_at", "llm_responses", ["created_at"])

    op.create_index("idx_llm_responses_model", "llm_responses", ["model_name"])

    # GIN indexes for JSONB columns (for JSON queries)
    op.execute("""
        CREATE INDEX idx_llm_responses_citations_gin
        ON llm_responses USING GIN (citations)
    """)

    op.execute("""
        CREATE INDEX idx_llm_responses_metadata_gin
        ON llm_responses USING GIN (metadata)
    """)


def downgrade() -> None:
    """Drop llm_responses table."""
    op.drop_table("llm_responses")
