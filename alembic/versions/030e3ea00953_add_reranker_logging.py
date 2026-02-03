"""add_reranker_logging

Revision ID: 030e3ea00953
Revises: 46eb84522090
Create Date: 2026-02-02 16:29:50.926943

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "030e3ea00953"
down_revision: Union[str, Sequence[str], None] = "46eb84522090"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create reranker_logs and reranker_results tables for tracking reranking performance."""

    # Create reranker_logs table
    op.execute("""
        CREATE TABLE reranker_logs (
            id BIGSERIAL PRIMARY KEY,
            query_log_id BIGINT NOT NULL UNIQUE,
            reranker_status TEXT NOT NULL,
            model_name TEXT,
            reranker_latency_ms INTEGER,
            num_candidates INTEGER NOT NULL,
            num_reranked INTEGER,
            fallback_method TEXT,
            error_message TEXT,
            avg_rank_change DOUBLE PRECISION,
            top_k_stability_score DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW(),

            FOREIGN KEY (query_log_id) REFERENCES query_logs(id) ON DELETE CASCADE,

            CHECK (reranker_status IN ('success', 'failed', 'skipped')),
            CHECK (num_candidates >= 0),
            CHECK (num_reranked >= 0),
            CHECK (reranker_latency_ms >= 0)
        )
    """)

    # Create reranker_results table
    op.execute("""
        CREATE TABLE reranker_results (
            id BIGSERIAL PRIMARY KEY,
            query_log_id BIGINT NOT NULL,
            chunk_id UUID NOT NULL,
            parent_article_id UUID NOT NULL,
            rrf_rank INTEGER NOT NULL,
            rrf_score DOUBLE PRECISION NOT NULL,
            reranked_rank INTEGER NOT NULL,
            reranked_score DOUBLE PRECISION NOT NULL,
            rank_change INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),

            FOREIGN KEY (query_log_id) REFERENCES query_logs(id) ON DELETE CASCADE,
            FOREIGN KEY (chunk_id) REFERENCES article_chunks(id) ON DELETE CASCADE,
            FOREIGN KEY (parent_article_id) REFERENCES articles(id) ON DELETE CASCADE,

            CHECK (rrf_rank >= 1),
            CHECK (reranked_rank >= 1),
            CHECK (rrf_score >= 0),
            CHECK (reranked_score >= 0 AND reranked_score <= 1)
        )
    """)

    # Create indexes for reranker_logs
    op.create_index(
        "idx_reranker_logs_query_log", "reranker_logs", ["query_log_id"], unique=True
    )

    op.create_index("idx_reranker_logs_status", "reranker_logs", ["reranker_status"])

    op.create_index("idx_reranker_logs_created_at", "reranker_logs", ["created_at"])

    op.create_index("idx_reranker_logs_model", "reranker_logs", ["model_name"])

    # Create indexes for reranker_results
    op.create_index(
        "idx_reranker_results_query_log",
        "reranker_results",
        ["query_log_id", "reranked_rank"],
    )

    op.create_index(
        "idx_reranker_results_rank_change", "reranker_results", ["rank_change"]
    )

    op.create_index(
        "idx_reranker_results_article",
        "reranker_results",
        ["parent_article_id", "rank_change"],
    )

    op.create_index(
        "idx_reranker_results_created_at", "reranker_results", ["created_at"]
    )

    op.create_index("idx_reranker_results_model", "reranker_results", ["model_name"])

    op.create_index("idx_reranker_results_rrf_rank", "reranker_results", ["rrf_rank"])


def downgrade() -> None:
    """Drop reranker tables."""
    # Drop tables in reverse order (results first, then logs)
    # Indexes are dropped automatically with tables
    op.drop_table("reranker_results")
    op.drop_table("reranker_logs")
