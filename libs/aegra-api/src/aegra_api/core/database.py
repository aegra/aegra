"""Database manager with LangGraph integration"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from aegra_api.config import load_store_config
from aegra_api.settings import settings

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = structlog.get_logger(__name__)


class _NoOpStore:
    """Stub store that raises on any operation when running in memory mode.

    Returned by ``get_store()`` in memory mode so callers always receive an
    object (avoiding ``AttributeError`` on ``None``), but get a descriptive
    error if they attempt store operations that require PostgreSQL.
    """

    async def aget(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Store is disabled in memory mode (DATABASE_ENABLED=false)")

    async def aput(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Store is disabled in memory mode (DATABASE_ENABLED=false)")

    async def adelete(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Store is disabled in memory mode (DATABASE_ENABLED=false)")

    async def alist_namespaces(self, *args: Any, **kwargs: Any) -> list:
        raise RuntimeError("Store is disabled in memory mode (DATABASE_ENABLED=false)")


class DatabaseManager:
    """Manages database connections and LangGraph persistence components.

    Supports two modes controlled by ``settings.db.DATABASE_ENABLED``:
      - **postgres** (default): full PostgreSQL-backed checkpointing and store.
      - **memory**: lightweight in-process MemorySaver — no external DB required.
    """

    def __init__(self) -> None:
        self.engine: AsyncEngine | None = None
        self._memory_mode: bool = False

        # Shared pool for LangGraph components (Checkpointer + Store)
        self.lg_pool: AsyncConnectionPool | None = None
        self._checkpointer: BaseCheckpointSaver | None = None
        self._store: AsyncPostgresStore | Any | None = None
        self._database_url = settings.db.database_url

    async def initialize_memory_mode(self) -> None:
        """Initialize with in-memory checkpointing (no database required).

        Use when DATABASE_ENABLED=false. State is ephemeral and lost on restart,
        but the application starts without requiring any external database.
        """
        if self._memory_mode and self._checkpointer is not None:
            return
        self._memory_mode = True
        self._checkpointer = MemorySaver()
        self._store = _NoOpStore()
        logger.info("✅ In-memory checkpointing initialized (no PostgreSQL)")

    async def initialize(self) -> None:
        """Initialize database connections and LangGraph components"""
        # Idempotency check: if already initialized, do nothing
        if self.engine:
            return

        # 1. SQLAlchemy Engine (app metadata, uses asyncpg)
        # We strictly limit this pool because the main load
        # is handled by LangGraph components.
        self.engine = create_async_engine(
            self._database_url,
            pool_size=settings.pool.SQLALCHEMY_POOL_SIZE,
            max_overflow=settings.pool.SQLALCHEMY_MAX_OVERFLOW,
            pool_pre_ping=True,
            echo=settings.db.DB_ECHO_LOG,
            connect_args={"prepared_statement_cache_size": 0},  # PgBouncer compatibility
        )

        lg_max = settings.pool.LANGGRAPH_MAX_POOL_SIZE
        lg_kwargs = {
            "autocommit": True,
            "prepare_threshold": None,  # Disable prepared statements for PgBouncer compatibility
            "row_factory": dict_row,  # LangGraph requires dictionary rows, not tuples
        }

        # Create a single shared pool.
        # 'open=False' is important to avoid RuntimeWarning; we open it explicitly below.
        self.lg_pool = AsyncConnectionPool(
            conninfo=settings.db.database_url_sync,
            min_size=settings.pool.LANGGRAPH_MIN_POOL_SIZE,
            max_size=lg_max,
            open=False,
            kwargs=lg_kwargs,
            check=AsyncConnectionPool.check_connection,
        )

        # Explicitly open the pool
        await self.lg_pool.open()

        # 2. Initialize LangGraph components using the shared pool
        # Passing 'conn=self.lg_pool' prevents components from creating their own pools.

        logger.info(f"Initializing LangGraph components with shared pool (max {lg_max} conns)...")

        self._checkpointer = AsyncPostgresSaver(conn=self.lg_pool)
        await self._checkpointer.setup()  # Ensure tables exist

        # Load store configuration for semantic search (if configured)
        store_config = load_store_config()
        index_config = store_config.get("index") if store_config else None

        self._store = AsyncPostgresStore(conn=self.lg_pool, index=index_config)
        await self._store.setup()  # Ensure tables exist

        if index_config:
            embed_model = index_config.get("embed", "unknown")
            logger.info(f"Semantic store enabled with embeddings: {embed_model}")

        logger.info("✅ Database and LangGraph components initialized")

    async def close(self) -> None:
        """Close database connections (no-op in memory mode)."""
        if self._memory_mode:
            self._checkpointer = None
            logger.info("✅ In-memory checkpointer released")
            return

        # Close SQLAlchemy engine
        if self.engine:
            await self.engine.dispose()
            self.engine = None

        # Close shared LangGraph pool
        if self.lg_pool:
            await self.lg_pool.close()
            self.lg_pool = None
            self._checkpointer = None
            self._store = None

        logger.info("✅ Database connections closed")

    def get_checkpointer(self) -> BaseCheckpointSaver:
        """Return the checkpointer (Postgres or MemorySaver depending on mode)."""
        if self._checkpointer is None:
            raise RuntimeError("Database not initialized — call initialize() or initialize_memory_mode() first")
        return self._checkpointer

    def get_store(self) -> "AsyncPostgresStore | _NoOpStore":
        """Return the store backend.

        In memory mode, returns a ``_NoOpStore`` stub that raises a descriptive
        ``RuntimeError`` on any operation, preventing silent ``AttributeError``
        crashes in downstream callers.
        """
        if self._memory_mode:
            return self._store  # type: ignore[return-value]  # _NoOpStore
        if self._store is None:
            raise RuntimeError("Database not initialized")
        return self._store

    def get_engine(self) -> AsyncEngine:
        """Get the SQLAlchemy engine for metadata tables"""
        if not self.engine:
            raise RuntimeError("Database not initialized")
        return self.engine

    @property
    def is_memory_mode(self) -> bool:
        """Whether the manager is running in memory-only mode (no PostgreSQL)."""
        return self._memory_mode


# Global database manager instance
db_manager = DatabaseManager()
