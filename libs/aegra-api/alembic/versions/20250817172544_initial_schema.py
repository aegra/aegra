"""Initial database schema for Aegra Agent Protocol server

This migration creates the core tables for the Agent Protocol implementation:
- assistant: Stores assistant configurations and metadata
- thread: Manages conversation threads with status tracking
- runs: Tracks execution runs with input/output and status
- run_events: Stores streaming events for real-time communication

Revision ID: 7b79bfd12626
Revises:
Create Date: 2025-08-17 17:25:44.338823

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "7b79bfd12626"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial database schema for Aegra Agent Protocol server."""

    # Create PostgreSQL extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp" SCHEMA public;')

    # Create assistant table
    op.create_table(
        "assistant",
        sa.Column(
            "assistant_id",
            sa.Text(),
            server_default=sa.text("public.uuid_generate_v4()::text"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("graph_id", sa.Text(), nullable=False),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("assistant_id"),
    )

    # Create run_events table for streaming functionality
    op.create_table(
        "run_events",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("event", sa.Text(), nullable=False),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create thread table
    op.create_table(
        "thread",
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default=sa.text("'idle'"), nullable=False),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("thread_id"),
    )

    # Create runs table with foreign key constraints
    op.create_table(
        "runs",
        sa.Column(
            "run_id",
            sa.Text(),
            server_default=sa.text("public.uuid_generate_v4()::text"),
            nullable=False,
        ),
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("assistant_id", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), server_default=sa.text("'pending'"), nullable=False),
        sa.Column(
            "input",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=True,
        ),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("output", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["assistant_id"], ["assistant.assistant_id"]),
        sa.ForeignKeyConstraint(["thread_id"], ["thread.thread_id"]),
        sa.PrimaryKeyConstraint("run_id"),
    )

    # Create indexes for performance optimization
    # Assistant indexes
    op.create_index("idx_assistant_user", "assistant", ["user_id"])
    op.create_index("idx_assistant_user_graph", "assistant", ["user_id", "graph_id"], unique=True)

    # Run events indexes
    op.create_index("idx_run_events_run_id", "run_events", ["run_id"])
    op.create_index("idx_run_events_seq", "run_events", ["run_id", "seq"])

    # Thread indexes
    op.create_index("idx_thread_user", "thread", ["user_id"])

    # Runs indexes
    op.create_index("idx_runs_assistant_id", "runs", ["assistant_id"])
    op.create_index("idx_runs_created_at", "runs", ["created_at"])
    op.create_index("idx_runs_status", "runs", ["status"])
    op.create_index("idx_runs_thread_id", "runs", ["thread_id"])
    op.create_index("idx_runs_user", "runs", ["user_id"])


def downgrade() -> None:
    """Drop all tables and indexes created in this migration."""

    # Drop indexes in reverse order (respecting dependencies)
    # Runs indexes
    op.drop_index("idx_runs_user", table_name="runs")
    op.drop_index("idx_runs_thread_id", table_name="runs")
    op.drop_index("idx_runs_status", table_name="runs")
    op.drop_index("idx_runs_created_at", table_name="runs")
    op.drop_index("idx_runs_assistant_id", table_name="runs")

    # Thread indexes
    op.drop_index("idx_thread_user", table_name="thread")

    # Run events indexes
    op.drop_index("idx_run_events_seq", table_name="run_events")
    op.drop_index("idx_run_events_run_id", table_name="run_events")

    # Assistant indexes
    op.drop_index("idx_assistant_user_graph", table_name="assistant")
    op.drop_index("idx_assistant_user", table_name="assistant")

    # Drop tables in reverse order (respecting foreign key constraints)
    op.drop_table("runs")
    op.drop_table("thread")
    op.drop_table("run_events")
    op.drop_table("assistant")
