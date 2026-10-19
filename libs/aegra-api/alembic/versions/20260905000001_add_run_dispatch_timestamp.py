"""add delayed run dispatch timestamp

Revision ID: 7c3e0a2b5d8f
Revises: 6b2d9f1a4c7e
Create Date: 2026-09-05 00:00:01.000000
"""

import sqlalchemy as sa

from alembic import op

revision = "7c3e0a2b5d8f"
down_revision = "6b2d9f1a4c7e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("dispatched_at", sa.TIMESTAMP(timezone=True), nullable=True))
    op.create_index(
        "idx_runs_dispatch",
        "runs",
        ["status", "not_before", "dispatched_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_runs_dispatch", table_name="runs")
    op.drop_column("runs", "dispatched_at")
