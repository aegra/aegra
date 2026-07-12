"""Multi-tenant isolation, assistant sharing, and encrypted assistant secrets.

tenant_id is an optional isolation dimension supplied by the auth handler
(server-authoritative); NULL falls back to pure user_id isolation. The query
layer matches with `tenant_id IS NOT DISTINCT FROM :t`, so the column is
nullable with no default. The assistant unique index is rebuilt to include
tenant_id via coalesce(tenant_id, '') (NULL-safe uniqueness without PG15+
NULLS NOT DISTINCT), keeping the md5(config) expression for large configs.

assistant_share holds one row per grant: grantee is "user:<id>",
"tenant:<id>", or "public"; the composite primary key deduplicates grants.

assistant.secrets stores per-assistant secrets as {name: fernet_token},
encrypted at rest and never returned in responses.

Revision ID: e2f3a4b5c6d7
Revises: b88bb61be638
Create Date: 2026-07-12 00:00:00.000000
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "e2f3a4b5c6d7"
down_revision = "b88bb61be638"
branch_labels = None
depends_on = None

# Index names follow each table's existing prefix (runs plural, cron singular), matching orm.py.
_TENANT_INDEXES = {
    "assistant": "idx_assistant_tenant_user",
    "thread": "idx_thread_tenant_user",
    "runs": "idx_runs_tenant_user",
    "crons": "idx_cron_tenant_user",
}


def upgrade() -> None:
    for table, index_name in _TENANT_INDEXES.items():
        op.add_column(table, sa.Column("tenant_id", sa.Text(), nullable=True))
        op.create_index(index_name, table, ["tenant_id", "user_id"])

    # Rebuild the assistant unique index to include tenant_id (coalesce normalizes NULL), keeping md5(config).
    # CREATE INDEX cannot use CONCURRENTLY in a transaction; the assistant table is small, so a one-shot ShareLock is acceptable.
    op.drop_index("idx_assistant_user_graph_config", table_name="assistant")
    op.execute(
        "CREATE UNIQUE INDEX idx_assistant_user_graph_config "
        "ON assistant (coalesce(tenant_id, ''), user_id, graph_id, md5(config::text))"
    )

    op.create_table(
        "assistant_share",
        sa.Column("assistant_id", sa.Text(), nullable=False),
        sa.Column("grantee", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["assistant_id"], ["assistant.assistant_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("assistant_id", "grantee"),
    )
    op.create_index("idx_assistant_share_grantee", "assistant_share", ["grantee"])

    op.add_column(
        "assistant",
        sa.Column("secrets", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("assistant", "secrets")

    op.drop_index("idx_assistant_share_grantee", table_name="assistant_share")
    op.drop_table("assistant_share")

    op.drop_index("idx_assistant_user_graph_config", table_name="assistant")
    op.execute(
        "CREATE UNIQUE INDEX idx_assistant_user_graph_config "
        "ON assistant (user_id, graph_id, md5(config::text))"
    )
    for table, index_name in _TENANT_INDEXES.items():
        op.drop_index(index_name, table_name=table)
        op.drop_column(table, "tenant_id")
