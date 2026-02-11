"""add_hyde_logs

Revision ID: 41344e1f8aa5
Revises: e93e7b757904
Create Date: 2026-02-03 11:46:19.864754

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "41344e1f8aa5"
down_revision: Union[str, Sequence[str], None] = "e93e7b757904"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create hyde_logs table for tracking HyDE generation performance."""

    # Create hyde_logs table
    op.execute("""
        CREATE TABLE hyde_logs (
            id BIGSERIAL PRIMARY KEY,
            query_log_id BIGINT NOT NULL UNIQUE,
            hypothetical_document TEXT NOT NULL,
            generation_status TEXT NOT NULL,
            model_name TEXT,
            generation_latency_ms INTEGER,
            embedding_latency_ms INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            error_message TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),

            FOREIGN KEY (query_log_id) REFERENCES query_logs(id) ON DELETE CASCADE,

            CHECK (generation_status IN ('success', 'failed_fallback')),
            CHECK (generation_latency_ms >= 0),
            CHECK (embedding_latency_ms >= 0),
            CHECK (prompt_tokens >= 0),
            CHECK (completion_tokens >= 0),
            CHECK (total_tokens >= 0)
        )
    """)

    # Create indexes for hyde_logs
    op.create_index(
        "idx_hyde_logs_query_log", "hyde_logs", ["query_log_id"], unique=True
    )

    op.create_index("idx_hyde_logs_status", "hyde_logs", ["generation_status"])

    op.create_index("idx_hyde_logs_created_at", "hyde_logs", ["created_at"])

    op.create_index("idx_hyde_logs_model", "hyde_logs", ["model_name"])


def downgrade() -> None:
    """Drop hyde_logs table."""
    # Indexes are dropped automatically with the table
    op.drop_table("hyde_logs")
