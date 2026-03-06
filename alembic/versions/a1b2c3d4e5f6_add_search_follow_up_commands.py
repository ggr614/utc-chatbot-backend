"""add_search_follow_up_commands

Revision ID: a1b2c3d4e5f6
Revises: 4106a4a2f015
Create Date: 2026-02-25 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "4106a4a2f015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add 'search' and 'follow_up' to command CHECK constraint."""
    op.execute("""
        ALTER TABLE query_logs
        DROP CONSTRAINT IF EXISTS check_command_values
    """)

    # Keep old values for backward compatibility with historical data
    op.execute("""
        ALTER TABLE query_logs
        ADD CONSTRAINT check_command_values
        CHECK (command IS NULL OR command IN ('bypass', 'q', 'qlong', 'debug', 'debuglong', 'search', 'follow_up'))
    """)


def downgrade() -> None:
    """Remove 'search' and 'follow_up' from command CHECK constraint."""
    op.execute("""
        ALTER TABLE query_logs
        DROP CONSTRAINT IF EXISTS check_command_values
    """)

    op.execute("""
        ALTER TABLE query_logs
        ADD CONSTRAINT check_command_values
        CHECK (command IS NULL OR command IN ('bypass', 'q', 'qlong', 'debug', 'debuglong'))
    """)
