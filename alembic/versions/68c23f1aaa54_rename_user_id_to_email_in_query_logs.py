"""rename_user_id_to_email_in_query_logs

Revision ID: 68c23f1aaa54
Revises: bff14a8411a7
Create Date: 2026-02-05 14:19:12.598594

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "68c23f1aaa54"
down_revision: Union[str, Sequence[str], None] = "bff14a8411a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Rename user_id column to email in query_logs table."""
    op.execute(
        """
        ALTER TABLE query_logs
        RENAME COLUMN user_id TO email
        """
    )


def downgrade() -> None:
    """Downgrade schema: Rename email column back to user_id in query_logs table."""
    op.execute(
        """
        ALTER TABLE query_logs
        RENAME COLUMN email TO user_id
        """
    )
