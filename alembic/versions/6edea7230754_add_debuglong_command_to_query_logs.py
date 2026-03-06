"""add_debuglong_command_to_query_logs

Revision ID: 6edea7230754
Revises: 67e629debc09
Create Date: 2026-02-03 14:15:34.703229

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "6edea7230754"
down_revision: Union[str, Sequence[str], None] = "67e629debc09"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add 'debuglong' to command CHECK constraint."""
    # Drop the old CHECK constraint
    op.execute("""
        ALTER TABLE query_logs
        DROP CONSTRAINT IF EXISTS check_command_values
    """)

    # Add new CHECK constraint with 'debuglong' included
    op.execute("""
        ALTER TABLE query_logs
        ADD CONSTRAINT check_command_values
        CHECK (command IS NULL OR command IN ('bypass', 'q', 'qlong', 'debug', 'debuglong'))
    """)


def downgrade() -> None:
    """Remove 'debuglong' from command CHECK constraint."""
    # Drop the CHECK constraint with 'debuglong'
    op.execute("""
        ALTER TABLE query_logs
        DROP CONSTRAINT IF EXISTS check_command_values
    """)

    # Restore original CHECK constraint without 'debuglong'
    op.execute("""
        ALTER TABLE query_logs
        ADD CONSTRAINT check_command_values
        CHECK (command IS NULL OR command IN ('bypass', 'q', 'qlong', 'debug'))
    """)
