"""add_thread_values_columns

Caches the latest checkpoint values and per-task interrupts on the thread row
so Thread responses (get/search) carry them like LangGraph Platform does.

Revision ID: c5e9a2f1d7b3
Revises: a3f7c1d9e2b4
Create Date: 2026-09-27 00:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision = "c5e9a2f1d7b3"
down_revision = "a3f7c1d9e2b4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("thread", sa.Column("values_json", JSONB(), nullable=True))
    op.add_column(
        "thread",
        sa.Column("interrupts_json", JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
    )
    op.add_column("thread", sa.Column("values_checkpoint_id", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("thread", "values_checkpoint_id")
    op.drop_column("thread", "interrupts_json")
    op.drop_column("thread", "values_json")
