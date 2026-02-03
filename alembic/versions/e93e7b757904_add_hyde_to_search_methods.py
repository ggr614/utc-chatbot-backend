"""add_hyde_to_search_methods

Revision ID: e93e7b757904
Revises: 030e3ea00953
Create Date: 2026-02-03 11:15:29.596766

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e93e7b757904'
down_revision: Union[str, Sequence[str], None] = '030e3ea00953'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add 'hyde' to query_results search_method constraint."""
    # Drop old constraint
    op.execute("""
        ALTER TABLE query_results
        DROP CONSTRAINT query_results_search_method_check
    """)

    # Add new constraint with 'hyde' included
    op.execute("""
        ALTER TABLE query_results
        ADD CONSTRAINT query_results_search_method_check
        CHECK (search_method IN ('bm25', 'vector', 'hybrid', 'hyde'))
    """)


def downgrade() -> None:
    """Remove 'hyde' from query_results search_method constraint."""
    # Drop new constraint
    op.execute("""
        ALTER TABLE query_results
        DROP CONSTRAINT query_results_search_method_check
    """)

    # Restore old constraint without 'hyde'
    op.execute("""
        ALTER TABLE query_results
        ADD CONSTRAINT query_results_search_method_check
        CHECK (search_method IN ('bm25', 'vector', 'hybrid'))
    """)
