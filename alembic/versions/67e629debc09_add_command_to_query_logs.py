"""add_command_to_query_logs

Revision ID: 67e629debc09
Revises: 41344e1f8aa5
Create Date: 2026-02-03 13:47:43.836348

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "67e629debc09"
down_revision: Union[str, Sequence[str], None] = "41344e1f8aa5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add command column to query_logs table."""
    # Add command column (nullable TEXT)
    op.execute("""
        ALTER TABLE query_logs
        ADD COLUMN command TEXT
    """)

    # Add CHECK constraint for valid command values
    op.execute("""
        ALTER TABLE query_logs
        ADD CONSTRAINT check_command_values
        CHECK (command IS NULL OR command IN ('bypass', 'q', 'qlong', 'debug'))
    """)

    # Create index on command column for analytics queries
    op.create_index("idx_query_logs_command", "query_logs", ["command"])


def downgrade() -> None:
    """Remove command column from query_logs table."""
    # Drop index
    op.drop_index("idx_query_logs_command", table_name="query_logs")

    # Drop CHECK constraint
    op.execute("""
        ALTER TABLE query_logs
        DROP CONSTRAINT IF EXISTS check_command_values
    """)

    # Drop command column
    op.execute("""
        ALTER TABLE query_logs
        DROP COLUMN command
    """)
