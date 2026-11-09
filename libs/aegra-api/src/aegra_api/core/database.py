"""Database manager with LangGraph integration"""

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from aegra_api.config import get_config_dir, load_store_config, setup_dependencies
from aegra_api.core.store_index import resolve_store_index
from aegra_api.settings import settings

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """Manages database connections and LangGraph persistence components"""

    def __init__(self) -> None:
        self.engine: AsyncEngine | None = None

        # Shared pool for LangGraph components (Checkpointer + Store)
        self.lg_pool: AsyncConnectionPool | None = None
        self._checkpointer: AsyncPostgresSaver | None = None
        self._store: AsyncPostgresStore | None = None
        self._database_url = settings.db.database_url

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
        store_index = None
        if index_config:
            # An embedding file may import from `dependencies` like a graph file, and graphs load after this.
            setup_dependencies()
            store_index = resolve_store_index(index_config, base_dir=get_config_dir())

        self._store = AsyncPostgresStore(conn=self.lg_pool, index=store_index)
        await self._store.setup()  # Ensure tables exist

        if index_config:
            embed_model = index_config.get("embed", "unknown")
            logger.info(f"Semantic store enabled with embeddings: {embed_model}")

        logger.info("✅ Database and LangGraph components initialized")

    async def close(self) -> None:
        """Close database connections"""
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

    def get_checkpointer(self) -> AsyncPostgresSaver:
        """Return the live AsyncPostgresSaver instance."""
        if self._checkpointer is None:
            raise RuntimeError("Database not initialized")
        return self._checkpointer

    def get_store(self) -> AsyncPostgresStore:
        """Return the live AsyncPostgresStore instance."""
        if self._store is None:
            raise RuntimeError("Database not initialized")
        return self._store

    def get_engine(self) -> AsyncEngine:
        """Get the SQLAlchemy engine for metadata tables"""
        if not self.engine:
            raise RuntimeError("Database not initialized")
        return self.engine


# Global database manager instance
db_manager = DatabaseManager()
