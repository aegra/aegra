"""Add optional tenant_id to assistant/thread/runs/crons for multi-tenant isolation.

tenant_id is an optional multi-tenant isolation dimension supplied by the auth handler
(server-authoritative); when NULL it falls back to pure user_id isolation. The query layer uses
`tenant_id IS NOT DISTINCT FROM :t` for NULL-safe matching, so the column is nullable with no default.

The assistant unique index idx_assistant_user_graph_config is rebuilt to include tenant_id: it uses
coalesce(tenant_id, '') to normalize NULL into one group, avoiding the Postgres default NULL!=NULL
weakening uniqueness in the tenant-less case (without relying on PG15+ NULLS NOT DISTINCT). The md5(config)
expression is kept to support large configs.

Revision ID: a7b8c9d0e1f2
Revises: b88bb61be638
Create Date: 2026-07-12 00:00:00.000000
"""

import sqlalchemy as sa

from alembic import op

revision = "a7b8c9d0e1f2"
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


def downgrade() -> None:
    op.drop_index("idx_assistant_user_graph_config", table_name="assistant")
    op.execute(
        "CREATE UNIQUE INDEX idx_assistant_user_graph_config "
        "ON assistant (user_id, graph_id, md5(config::text))"
    )
    for table, index_name in _TENANT_INDEXES.items():
        op.drop_index(index_name, table_name=table)
        op.drop_column(table, "tenant_id")
