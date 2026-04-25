import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.core.database import DatabaseManager
from aegra_api.settings import settings


class TestDatabaseManager:
    @pytest.fixture
    def db_manager(self):
        """Creates a fresh DatabaseManager instance for each test."""
        return DatabaseManager()

    @pytest.fixture
    def mock_env(self):
        """
        Sets up the environment variables required for the database configuration.
        Using patch.dict ensures these are reset after the test.
        """
        env_vars = {
            "DATABASE_URL": settings.db.database_url,
            "SQLALCHEMY_POOL_SIZE": str(settings.pool.SQLALCHEMY_POOL_SIZE),
            "LANGGRAPH_MAX_POOL_SIZE": str(settings.pool.LANGGRAPH_MAX_POOL_SIZE),
            "LANGGRAPH_MIN_POOL_SIZE": str(settings.pool.LANGGRAPH_MIN_POOL_SIZE),
        }
        with patch.dict(os.environ, env_vars):
            yield

    @pytest.fixture
    def mock_db_deps(self, mock_env):
        """
        Mocks all external database dependencies (SQLAlchemy, LangGraph pools, etc.).
        Returns a dictionary of mocks to be used for assertions in tests.
        """
        with (
            patch("aegra_api.core.database.create_async_engine") as mock_create_engine,
            patch("aegra_api.core.database.AsyncConnectionPool") as mock_pool_cls,
            patch("aegra_api.core.database.AsyncPostgresSaver") as mock_saver_cls,
            patch("aegra_api.core.database.AsyncPostgresStore") as mock_store_cls,
            patch("aegra_api.core.database.load_store_config") as mock_load_store_config,
        ):
            # 1. Setup SQLAlchemy Engine Mock
            mock_engine = AsyncMock()
            mock_create_engine.return_value = mock_engine

            # 2. Setup LangGraph Connection Pool Mock
            mock_pool_instance = AsyncMock()
            mock_pool_cls.return_value = mock_pool_instance
            # Mock the static/class method check_connection
            mock_pool_cls.check_connection = MagicMock()

            mock_conn = AsyncMock()

            @asynccontextmanager
            async def _fake_connection():
                yield mock_conn

            mock_pool_instance.connection = _fake_connection

            # 3. Setup Saver and Store Mocks
            mock_saver_instance = AsyncMock()
            mock_saver_cls.return_value = mock_saver_instance

            mock_store_instance = AsyncMock()
            mock_store_cls.return_value = mock_store_instance

            # 4. Default: no store config
            mock_load_store_config.return_value = None

            # Yield a dictionary so tests can access specific mocks for assertions
            yield {
                "create_engine": mock_create_engine,
                "engine_instance": mock_engine,
                "pool_cls": mock_pool_cls,
                "pool_instance": mock_pool_instance,
                "saver_cls": mock_saver_cls,
                "saver_instance": mock_saver_instance,
                "store_cls": mock_store_cls,
                "store_instance": mock_store_instance,
                "load_store_config": mock_load_store_config,
            }

    async def test_initialize_success(self, db_manager, mock_db_deps):
        """Test successful initialization checking against ACTUAL settings."""

        # --- EXECUTE ---
        await db_manager.initialize()

        # --- ASSERTIONS ---

        # 1. Verify SQLAlchemy engine creation
        mock_db_deps["create_engine"].assert_called_once()
        _, kwargs = mock_db_deps["create_engine"].call_args

        assert kwargs["pool_size"] == settings.pool.SQLALCHEMY_POOL_SIZE
        assert kwargs["pool_pre_ping"] is True

        # 2. Verify LangGraph Pool creation
        mock_db_deps["pool_cls"].assert_called_once()
        _, lg_kwargs = mock_db_deps["pool_cls"].call_args

        assert lg_kwargs["min_size"] == settings.pool.LANGGRAPH_MIN_POOL_SIZE
        assert lg_kwargs["max_size"] == settings.pool.LANGGRAPH_MAX_POOL_SIZE
        assert lg_kwargs["open"] is False
        assert "check" in lg_kwargs

        # Verify pool options inside kwargs
        inner_kwargs = lg_kwargs["kwargs"]
        assert inner_kwargs["prepare_threshold"] is None
        assert inner_kwargs["autocommit"] is True

        # Verify explicit open was awaited
        mock_db_deps["pool_instance"].open.assert_awaited_once()

        # 3. Verify Components initialization
        mock_db_deps["saver_cls"].assert_called_with(conn=mock_db_deps["pool_instance"])
        mock_db_deps["saver_instance"].setup.assert_awaited_once()

        # Store is initialized with index=None when no store config is provided
        mock_db_deps["store_cls"].assert_called_with(conn=mock_db_deps["pool_instance"], index=None)
        mock_db_deps["store_instance"].setup.assert_awaited_once()

        # 4. Verify internal state
        assert db_manager.engine == mock_db_deps["engine_instance"]
        assert db_manager.lg_pool == mock_db_deps["pool_instance"]

    @pytest.mark.asyncio
    async def test_initialize_idempotency(self, db_manager, mock_db_deps):
        """Test that initialize returns early if the database is already initialized."""

        # Simulate an existing engine to trigger the guard clause
        db_manager.engine = AsyncMock()

        # --- EXECUTE ---
        await db_manager.initialize()

        # --- ASSERTIONS ---
        # Ensure create_engine was NOT called because we already had an engine
        mock_db_deps["create_engine"].assert_not_called()

    @pytest.mark.asyncio
    async def test_close_resources(self, db_manager):
        """Test proper resource cleanup during close()."""

        # Setup fake resources manually since we are testing teardown
        mock_engine = AsyncMock()
        mock_pool = AsyncMock()

        db_manager.engine = mock_engine
        db_manager.lg_pool = mock_pool
        db_manager._checkpointer = AsyncMock()
        db_manager._store = AsyncMock()

        # --- EXECUTE ---
        await db_manager.close()

        # --- ASSERTIONS ---
        mock_engine.dispose.assert_awaited_once()
        mock_pool.close.assert_awaited_once()

        # Verify state is reset to None
        assert db_manager.engine is None
        assert db_manager.lg_pool is None
        assert db_manager._checkpointer is None
        assert db_manager._store is None

    @pytest.mark.asyncio
    async def test_getters_success(self, db_manager):
        """Test successful retrieval of components via getters."""

        # Manually inject mocks into the manager
        db_manager.engine = MagicMock()
        db_manager._checkpointer = MagicMock()
        db_manager._store = MagicMock()

        # Synchronous getter
        assert db_manager.get_engine() is not None

        # Asynchronous getters (ensure they are awaited)
        assert db_manager.get_checkpointer() is not None
        assert db_manager.get_store() is not None

    @pytest.mark.asyncio
    async def test_getters_failure(self, db_manager):
        """Test that getters raise RuntimeError when the manager is not initialized."""

        # Ensure state is empty
        db_manager.engine = None
        db_manager._checkpointer = None
        db_manager._store = None

        # Synchronous getter check
        with pytest.raises(RuntimeError, match="Database not initialized"):
            db_manager.get_engine()

        # Asynchronous getter checks
        with pytest.raises(RuntimeError, match="Database not initialized"):
            db_manager.get_checkpointer()

        with pytest.raises(RuntimeError, match="Database not initialized"):
            db_manager.get_store()

    @pytest.mark.asyncio
    async def test_initialize_with_store_index_config(self, db_manager, mock_db_deps):
        """Test that store is initialized with index config when provided."""
        # Configure mock to return store config with index
        index_config = {
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }
        mock_db_deps["load_store_config"].return_value = {"index": index_config}

        # --- EXECUTE ---
        await db_manager.initialize()

        # --- ASSERTIONS ---
        # Store should be initialized with the index config
        mock_db_deps["store_cls"].assert_called_with(conn=mock_db_deps["pool_instance"], index=index_config)
        mock_db_deps["store_instance"].setup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_with_store_config_no_index(self, db_manager, mock_db_deps):
        """Test that store is initialized with index=None when store config has no index."""
        # Configure mock to return store config without index
        mock_db_deps["load_store_config"].return_value = {}

        # --- EXECUTE ---
        await db_manager.initialize()

        # --- ASSERTIONS ---
        # Store should be initialized with index=None
        mock_db_deps["store_cls"].assert_called_with(conn=mock_db_deps["pool_instance"], index=None)
        mock_db_deps["store_instance"].setup.assert_awaited_once()


class TestDatabaseManagerSchemaIsolation:
    """Tests for POSTGRES_SCHEMA-based schema isolation in DatabaseManager."""

    @pytest.fixture
    def db_manager(self) -> DatabaseManager:
        """Creates a fresh DatabaseManager instance for each test."""
        return DatabaseManager()

    @pytest.fixture
    def mock_env_with_schema(self):
        """Sets environment with POSTGRES_SCHEMA configured."""
        env_vars = {
            "DATABASE_URL": settings.db.database_url,
            "SQLALCHEMY_POOL_SIZE": str(settings.pool.SQLALCHEMY_POOL_SIZE),
            "LANGGRAPH_MAX_POOL_SIZE": str(settings.pool.LANGGRAPH_MAX_POOL_SIZE),
            "LANGGRAPH_MIN_POOL_SIZE": str(settings.pool.LANGGRAPH_MIN_POOL_SIZE),
            "POSTGRES_SCHEMA": "tenant_x",
        }
        with patch.dict(os.environ, env_vars), patch.object(settings.db, "POSTGRES_SCHEMA", "tenant_x"):
            yield

    @pytest.fixture
    def mock_db_deps_with_schema(self, mock_env_with_schema):
        """Mocks all external database dependencies with schema isolation active."""
        with (
            patch("aegra_api.core.database.create_async_engine") as mock_create_engine,
            patch("aegra_api.core.database.AsyncConnectionPool") as mock_pool_cls,
            patch("aegra_api.core.database.AsyncPostgresSaver") as mock_saver_cls,
            patch("aegra_api.core.database.AsyncPostgresStore") as mock_store_cls,
            patch("aegra_api.core.database.load_store_config") as mock_load_store_config,
        ):
            mock_engine = AsyncMock()
            mock_create_engine.return_value = mock_engine

            mock_pool_instance = AsyncMock()
            mock_pool_cls.return_value = mock_pool_instance
            mock_pool_cls.check_connection = MagicMock()

            mock_conn = AsyncMock()

            @asynccontextmanager
            async def _fake_connection():
                yield mock_conn

            mock_pool_instance.connection = _fake_connection

            mock_saver_instance = AsyncMock()
            mock_saver_cls.return_value = mock_saver_instance

            mock_store_instance = AsyncMock()
            mock_store_cls.return_value = mock_store_instance

            mock_load_store_config.return_value = None

            yield {
                "create_engine": mock_create_engine,
                "engine_instance": mock_engine,
                "pool_cls": mock_pool_cls,
                "pool_instance": mock_pool_instance,
                "pool_conn": mock_conn,
                "saver_cls": mock_saver_cls,
                "saver_instance": mock_saver_instance,
                "store_cls": mock_store_cls,
                "store_instance": mock_store_instance,
                "load_store_config": mock_load_store_config,
            }

    @pytest.mark.asyncio
    async def test_sqlalchemy_connect_args_include_search_path(
        self, db_manager: DatabaseManager, mock_db_deps_with_schema: dict
    ) -> None:
        """When POSTGRES_SCHEMA is set, SQLAlchemy connect_args must include server_settings with search_path."""
        await db_manager.initialize()

        _, kwargs = mock_db_deps_with_schema["create_engine"].call_args
        connect_args = kwargs["connect_args"]
        assert "server_settings" in connect_args
        assert connect_args["server_settings"]["search_path"] == "tenant_x, public"

    @pytest.mark.asyncio
    async def test_sqlalchemy_connect_args_no_search_path_without_schema(self, db_manager: DatabaseManager) -> None:
        """When POSTGRES_SCHEMA is None, connect_args should NOT include server_settings."""
        env_vars = {
            "DATABASE_URL": settings.db.database_url,
            "SQLALCHEMY_POOL_SIZE": str(settings.pool.SQLALCHEMY_POOL_SIZE),
            "LANGGRAPH_MAX_POOL_SIZE": str(settings.pool.LANGGRAPH_MAX_POOL_SIZE),
            "LANGGRAPH_MIN_POOL_SIZE": str(settings.pool.LANGGRAPH_MIN_POOL_SIZE),
        }
        with (
            patch.dict(os.environ, env_vars),
            patch.object(settings.db, "POSTGRES_SCHEMA", None),
            patch("aegra_api.core.database.create_async_engine") as mock_create_engine,
            patch("aegra_api.core.database.AsyncConnectionPool") as mock_pool_cls,
            patch("aegra_api.core.database.AsyncPostgresSaver") as mock_saver_cls,
            patch("aegra_api.core.database.AsyncPostgresStore") as mock_store_cls,
            patch("aegra_api.core.database.load_store_config") as mock_load_store_config,
        ):
            mock_create_engine.return_value = AsyncMock()
            mock_pool_instance = AsyncMock()
            mock_pool_cls.return_value = mock_pool_instance
            mock_pool_cls.check_connection = MagicMock()

            mock_conn = AsyncMock()

            @asynccontextmanager
            async def _fake_connection():
                yield mock_conn

            mock_pool_instance.connection = _fake_connection

            mock_saver_cls.return_value = AsyncMock()
            mock_store_cls.return_value = AsyncMock()
            mock_load_store_config.return_value = None

            await db_manager.initialize()

            _, kwargs = mock_create_engine.call_args
            connect_args = kwargs["connect_args"]
            assert "server_settings" not in connect_args

    @pytest.mark.asyncio
    async def test_langgraph_pool_receives_configure_callback(
        self, db_manager: DatabaseManager, mock_db_deps_with_schema: dict
    ) -> None:
        """When POSTGRES_SCHEMA is set, the LangGraph pool must receive a configure callback."""
        await db_manager.initialize()

        _, kwargs = mock_db_deps_with_schema["pool_cls"].call_args
        assert kwargs["configure"] is not None
        assert callable(kwargs["configure"])

    @pytest.mark.asyncio
    async def test_langgraph_pool_configure_is_none_without_schema(self, db_manager: DatabaseManager) -> None:
        """When POSTGRES_SCHEMA is None, the pool configure callback must be None."""
        env_vars = {
            "DATABASE_URL": settings.db.database_url,
            "SQLALCHEMY_POOL_SIZE": str(settings.pool.SQLALCHEMY_POOL_SIZE),
            "LANGGRAPH_MAX_POOL_SIZE": str(settings.pool.LANGGRAPH_MAX_POOL_SIZE),
            "LANGGRAPH_MIN_POOL_SIZE": str(settings.pool.LANGGRAPH_MIN_POOL_SIZE),
        }
        with (
            patch.dict(os.environ, env_vars),
            patch.object(settings.db, "POSTGRES_SCHEMA", None),
            patch("aegra_api.core.database.create_async_engine") as mock_create_engine,
            patch("aegra_api.core.database.AsyncConnectionPool") as mock_pool_cls,
            patch("aegra_api.core.database.AsyncPostgresSaver") as mock_saver_cls,
            patch("aegra_api.core.database.AsyncPostgresStore") as mock_store_cls,
            patch("aegra_api.core.database.load_store_config") as mock_load_store_config,
        ):
            mock_create_engine.return_value = AsyncMock()
            mock_pool_instance = AsyncMock()
            mock_pool_cls.return_value = mock_pool_instance
            mock_pool_cls.check_connection = MagicMock()

            mock_conn = AsyncMock()

            @asynccontextmanager
            async def _fake_connection():
                yield mock_conn

            mock_pool_instance.connection = _fake_connection

            mock_saver_cls.return_value = AsyncMock()
            mock_store_cls.return_value = AsyncMock()
            mock_load_store_config.return_value = None

            await db_manager.initialize()

            _, kwargs = mock_pool_cls.call_args
            assert kwargs["configure"] is None

    @pytest.mark.asyncio
    async def test_schema_created_on_initialize(
        self, db_manager: DatabaseManager, mock_db_deps_with_schema: dict
    ) -> None:
        """When POSTGRES_SCHEMA is set, CREATE SCHEMA IF NOT EXISTS is executed via the pool."""
        await db_manager.initialize()

        mock_conn = mock_db_deps_with_schema["pool_conn"]
        mock_conn.execute.assert_awaited()

        sql_obj = mock_conn.execute.call_args_list[0][0][0]
        executed_sql = sql_obj.as_string(None)
        assert "CREATE SCHEMA IF NOT EXISTS" in executed_sql
        assert '"tenant_x"' in executed_sql

    @pytest.mark.asyncio
    async def test_no_schema_created_without_postgres_schema(self, db_manager: DatabaseManager) -> None:
        """When POSTGRES_SCHEMA is None, no CREATE SCHEMA is executed."""
        env_vars = {
            "DATABASE_URL": settings.db.database_url,
            "SQLALCHEMY_POOL_SIZE": str(settings.pool.SQLALCHEMY_POOL_SIZE),
            "LANGGRAPH_MAX_POOL_SIZE": str(settings.pool.LANGGRAPH_MAX_POOL_SIZE),
            "LANGGRAPH_MIN_POOL_SIZE": str(settings.pool.LANGGRAPH_MIN_POOL_SIZE),
        }
        with (
            patch.dict(os.environ, env_vars),
            patch.object(settings.db, "POSTGRES_SCHEMA", None),
            patch("aegra_api.core.database.create_async_engine") as mock_create_engine,
            patch("aegra_api.core.database.AsyncConnectionPool") as mock_pool_cls,
            patch("aegra_api.core.database.AsyncPostgresSaver") as mock_saver_cls,
            patch("aegra_api.core.database.AsyncPostgresStore") as mock_store_cls,
            patch("aegra_api.core.database.load_store_config") as mock_load_store_config,
        ):
            mock_create_engine.return_value = AsyncMock()
            mock_pool_instance = AsyncMock()
            mock_pool_cls.return_value = mock_pool_instance
            mock_pool_cls.check_connection = MagicMock()

            mock_conn = AsyncMock()

            @asynccontextmanager
            async def _fake_connection():
                yield mock_conn

            mock_pool_instance.connection = _fake_connection

            mock_saver_cls.return_value = AsyncMock()
            mock_store_cls.return_value = AsyncMock()
            mock_load_store_config.return_value = None

            await db_manager.initialize()

            # The pool's connection context manager should never be called
            # since POSTGRES_SCHEMA is None — the _fake_connection should not
            # have been entered, meaning mock_conn.execute was not called.
            mock_conn.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_configure_callback_sets_search_path(
        self, db_manager: DatabaseManager, mock_db_deps_with_schema: dict
    ) -> None:
        """The pool configure callback should execute SET search_path when called."""
        await db_manager.initialize()

        _, kwargs = mock_db_deps_with_schema["pool_cls"].call_args
        configure_fn = kwargs["configure"]

        fake_conn = AsyncMock()
        await configure_fn(fake_conn)

        fake_conn.execute.assert_awaited_once()
        sql_obj = fake_conn.execute.call_args[0][0]
        executed_sql = sql_obj.as_string(None)
        assert "SET search_path TO" in executed_sql
        assert '"tenant_x"' in executed_sql

    @pytest.mark.asyncio
    async def test_pgbouncer_cache_size_preserved_with_schema(
        self, db_manager: DatabaseManager, mock_db_deps_with_schema: dict
    ) -> None:
        """PgBouncer compatibility setting must be preserved when schema is set."""
        await db_manager.initialize()

        _, kwargs = mock_db_deps_with_schema["create_engine"].call_args
        connect_args = kwargs["connect_args"]
        assert connect_args["prepared_statement_cache_size"] == 0
