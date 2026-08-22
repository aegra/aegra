"""add_thread_ttl_table

Revision ID: a3f7c1d9e2b4
Revises: b88bb61be638
Create Date: 2026-08-22 00:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "a3f7c1d9e2b4"
down_revision = "b88bb61be638"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "thread_ttl",
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("strategy", sa.Text(), server_default=sa.text("'delete'"), nullable=False),
        sa.Column("ttl_minutes", sa.Float(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["thread_id"], ["thread.thread_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("thread_id"),
    )
    op.create_index("idx_thread_ttl_expires_at", "thread_ttl", ["expires_at"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_thread_ttl_expires_at", table_name="thread_ttl")
    op.drop_table("thread_ttl")
