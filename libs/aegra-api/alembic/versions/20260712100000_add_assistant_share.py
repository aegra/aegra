"""add assistant_share table for cross-tenant / public assistant sharing

Revision ID: d1e2f3a4b5c6
Revises: a7b8c9d0e1f2
Create Date: 2026-07-12

Adds the assistant_share table: an owner can share an assistant with a specific user, a specific
tenant, or fully public (share_type). Grantees can read and execute (execution reuses the owner config, including the real api_key).
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d1e2f3a4b5c6"
down_revision: str | Sequence[str] | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "assistant_share",
        sa.Column("share_id", sa.Text(), server_default=sa.text("gen_random_uuid()::text"), nullable=False),
        sa.Column("assistant_id", sa.Text(), nullable=False),
        sa.Column("owner_user_id", sa.Text(), nullable=False),
        sa.Column("owner_tenant_id", sa.Text(), nullable=True),
        sa.Column("share_type", sa.Text(), nullable=False),
        sa.Column("target_user_id", sa.Text(), nullable=True),
        sa.Column("target_tenant_id", sa.Text(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["assistant_id"], ["assistant.assistant_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("share_id"),
    )
    op.create_index("idx_assistant_share_assistant", "assistant_share", ["assistant_id"])
    op.create_index("idx_assistant_share_target_user", "assistant_share", ["target_user_id"])
    op.create_index("idx_assistant_share_target_tenant", "assistant_share", ["target_tenant_id"])
    op.create_index("idx_assistant_share_owner", "assistant_share", ["owner_user_id"])


def downgrade() -> None:
    op.drop_index("idx_assistant_share_owner", table_name="assistant_share")
    op.drop_index("idx_assistant_share_target_tenant", table_name="assistant_share")
    op.drop_index("idx_assistant_share_target_user", table_name="assistant_share")
    op.drop_index("idx_assistant_share_assistant", table_name="assistant_share")
    op.drop_table("assistant_share")
