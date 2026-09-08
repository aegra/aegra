"""add explicit marker for stateless-run threads

Revision ID: c4e8d2f1a6b7
Revises: a3f7c1d9e2b4
Create Date: 2026-09-08 00:00:00.000000
"""

import sqlalchemy as sa

from alembic import op

revision = "c4e8d2f1a6b7"
down_revision = "a3f7c1d9e2b4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("thread", sa.Column("is_ephemeral", sa.Boolean(), server_default=sa.text("false"), nullable=False))
    op.create_index("idx_thread_ephemeral_updated", "thread", ["is_ephemeral", "updated_at"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_thread_ephemeral_updated", table_name="thread")
    op.drop_column("thread", "is_ephemeral")
