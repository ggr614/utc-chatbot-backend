"""add_admin_users

Revision ID: 440ecd494453
Revises: a1b2c3d4e5f6
Create Date: 2026-03-21 00:26:26.752043

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '440ecd494453'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add admin_users table for admin authentication."""
    op.execute("""
        CREATE TABLE admin_users (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            username        TEXT NOT NULL UNIQUE,
            password_hash   TEXT NOT NULL,
            display_name    TEXT,
            is_active       BOOLEAN NOT NULL DEFAULT true,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_login_at   TIMESTAMPTZ
        )
    """)

    op.create_index("idx_admin_users_username", "admin_users", ["username"])

    op.execute("""
        COMMENT ON TABLE admin_users IS
        'Admin users for authenticated access to admin endpoints'
    """)


def downgrade() -> None:
    """Remove admin_users table."""
    op.drop_index("idx_admin_users_username", table_name="admin_users")
    op.drop_table("admin_users")
