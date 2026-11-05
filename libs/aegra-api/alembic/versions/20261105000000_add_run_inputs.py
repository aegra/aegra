"""add_run_inputs

Revision ID: 70b7216bfef2
Revises: a3f7c1d9e2b4
Create Date: 2026-11-05 00:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "70b7216bfef2"
down_revision = "a3f7c1d9e2b4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("accepts_input_inject", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    op.create_table(
        "run_inputs",
        sa.Column("seq", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("message_id", sa.Text(), nullable=False),
        sa.Column("message", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("taken_by_run", sa.Text(), nullable=True),
        sa.Column("taken_by_attempt", sa.Text(), nullable=True),
        sa.Column("delivered", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["thread_id"], ["thread.thread_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("seq"),
    )
    op.create_index("idx_run_inputs_thread_message", "run_inputs", ["thread_id", "message_id"], unique=True)


def downgrade() -> None:
    op.drop_index("idx_run_inputs_thread_message", table_name="run_inputs")
    op.drop_table("run_inputs")
    op.drop_column("runs", "accepts_input_inject")
