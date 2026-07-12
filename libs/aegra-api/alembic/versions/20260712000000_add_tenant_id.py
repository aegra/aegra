"""Add optional tenant_id to assistant/thread/runs/crons for multi-tenant isolation.

tenant_id 是可选的多租户隔离维度,由 auth handler 提供(服务端权威),为 NULL 时
退回纯 user_id 隔离。查询层用 `tenant_id IS NOT DISTINCT FROM :t` 做 NULL-safe
匹配,故列可空、无默认值。

Assistant 唯一索引 idx_assistant_user_graph_config 重建为纳入 tenant_id:用
coalesce(tenant_id, '') 把 NULL 归一为同一分组,避免 Postgres 默认 NULL!=NULL
在无租户场景削弱唯一性(不依赖 PG15+ 的 NULLS NOT DISTINCT)。保留 md5(config)
表达式以支持大 config。

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

# 索引名沿用各表既有命名前缀(runs 复数、cron 单数),与 orm.py 保持一致。
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

    # 重建 Assistant 唯一索引,纳入 tenant_id(coalesce 归一 NULL),保留 md5(config)。
    # CREATE INDEX 不能在事务内用 CONCURRENTLY;assistant 表小,一次性 ShareLock 可接受。
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
