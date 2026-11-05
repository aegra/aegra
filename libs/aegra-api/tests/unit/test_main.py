"""Tests for application lifespan and startup logic"""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aegra_api.main as main_module
from aegra_api.observability.base import get_observability_manager
from aegra_api.services import run_preparation


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_registers_otel_provider(monkeypatch):
    """Test that the lifespan function registers the OpenTelemetry provider during startup."""
    # Configure OTEL_TARGETS so the provider is enabled
    monkeypatch.setenv("OTEL_TARGETS", "LANGFUSE")

    # 1. Reload settings
    import aegra_api.settings as settings_module

    importlib.reload(settings_module)

    # 2. Reload otel module (creates new Provider class/instance)
    import aegra_api.observability.otel as otel_module

    importlib.reload(otel_module)
    # 3. Reload setup module (crucial! so it imports the NEW otel_provider)
    import aegra_api.observability.setup as setup_module
    from aegra_api.observability.otel import OpenTelemetryProvider

    importlib.reload(setup_module)

    # 4. Reload main module
    import aegra_api.main as main_module

    importlib.reload(main_module)

    # Mock all the dependencies
    with (
        patch("aegra_api.main.run_migrations_async", new_callable=AsyncMock),
        patch("aegra_api.main.db_manager") as mock_db_manager,
        patch("aegra_api.main.get_langgraph_service") as mock_get_langgraph_service,
    ):
        mock_db_manager.initialize = AsyncMock()
        mock_db_manager.close = AsyncMock()

        mock_langgraph_service = MagicMock()
        mock_langgraph_service.initialize = AsyncMock()
        mock_get_langgraph_service.return_value = mock_langgraph_service

        # Clear the manager
        manager = get_observability_manager()
        manager._providers.clear()

        mock_app = MagicMock()

        async with main_module.lifespan(mock_app):
            # Verify OpenTelemetryProvider is registered
            otel_providers = [p for p in manager._providers if isinstance(p, OpenTelemetryProvider)]
            assert len(otel_providers) == 1, "OpenTelemetry provider should be registered during lifespan startup"

        mock_db_manager.close.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_calls_required_initialization():
    """Test that lifespan calls all required initialization functions."""
    # Reload main to ensure clean state
    import aegra_api.main as main_module

    importlib.reload(main_module)

    with (
        patch("aegra_api.main.run_migrations_async", new_callable=AsyncMock) as mock_migrations,
        patch("aegra_api.main.db_manager") as mock_db_manager,
        patch("aegra_api.main.get_langgraph_service") as mock_get_langgraph_service,
        patch("aegra_api.main.setup_observability") as mock_setup_observability,
    ):
        # Setup mocks
        mock_db_manager.initialize = AsyncMock()
        mock_db_manager.close = AsyncMock()

        mock_langgraph_service = MagicMock()
        mock_langgraph_service.initialize = AsyncMock()
        mock_get_langgraph_service.return_value = mock_langgraph_service

        mock_app = MagicMock()

        # Run the lifespan function
        async with main_module.lifespan(mock_app):
            pass

        # Verify migrations run first, then initialization
        mock_migrations.assert_called_once()
        mock_db_manager.initialize.assert_called_once()
        mock_langgraph_service.initialize.assert_called_once()

        # Verify observability setup was called
        mock_setup_observability.assert_called_once()

        # Verify cleanup
        mock_db_manager.close.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_skips_migrations_when_disabled(monkeypatch):
    """When RUN_MIGRATIONS_ON_STARTUP=false, lifespan must not call alembic."""
    import aegra_api.main as main_module
    from aegra_api.settings import settings

    importlib.reload(main_module)

    monkeypatch.setattr(settings.app, "RUN_MIGRATIONS_ON_STARTUP", False)

    with (
        patch("aegra_api.main.run_migrations_async", new_callable=AsyncMock) as mock_migrations,
        patch("aegra_api.main.db_manager") as mock_db_manager,
        patch("aegra_api.main.get_langgraph_service") as mock_get_langgraph_service,
        patch("aegra_api.main.setup_observability"),
    ):
        mock_db_manager.initialize = AsyncMock()
        mock_db_manager.close = AsyncMock()

        mock_langgraph_service = MagicMock()
        mock_langgraph_service.initialize = AsyncMock()
        mock_get_langgraph_service.return_value = mock_langgraph_service

        async with main_module.lifespan(MagicMock()):
            pass

        mock_migrations.assert_not_called()
        # The rest of startup must still happen.
        mock_db_manager.initialize.assert_called_once()
        mock_langgraph_service.initialize.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_runs_migrations_when_enabled(monkeypatch):
    """Default (RUN_MIGRATIONS_ON_STARTUP=true) keeps the auto-migrate behavior."""
    import aegra_api.main as main_module
    from aegra_api.settings import settings

    importlib.reload(main_module)

    monkeypatch.setattr(settings.app, "RUN_MIGRATIONS_ON_STARTUP", True)

    with (
        patch("aegra_api.main.run_migrations_async", new_callable=AsyncMock) as mock_migrations,
        patch("aegra_api.main.db_manager") as mock_db_manager,
        patch("aegra_api.main.get_langgraph_service") as mock_get_langgraph_service,
        patch("aegra_api.main.setup_observability"),
    ):
        mock_db_manager.initialize = AsyncMock()
        mock_db_manager.close = AsyncMock()

        mock_langgraph_service = MagicMock()
        mock_langgraph_service.initialize = AsyncMock()
        mock_get_langgraph_service.return_value = mock_langgraph_service

        async with main_module.lifespan(MagicMock()):
            pass

        mock_migrations.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_starts_ttl_sweeper_when_configured() -> None:
    """The TTL sweeper starts when a TTL config resolves and stops before DB close."""
    import aegra_api.main as main_module

    importlib.reload(main_module)

    with (
        patch("aegra_api.main.run_migrations_async", new_callable=AsyncMock),
        patch("aegra_api.main.db_manager") as mock_db_manager,
        patch("aegra_api.main.get_langgraph_service") as mock_get_langgraph_service,
        patch("aegra_api.main.setup_observability"),
        patch("aegra_api.main.get_thread_ttl_config", return_value=MagicMock()),
        patch("aegra_api.main.thread_ttl_sweeper") as mock_sweeper,
    ):
        mock_db_manager.initialize = AsyncMock()
        mock_db_manager.close = AsyncMock()
        mock_langgraph_service = MagicMock()
        mock_langgraph_service.initialize = AsyncMock()
        mock_get_langgraph_service.return_value = mock_langgraph_service
        mock_sweeper.start = AsyncMock()
        mock_sweeper.stop = AsyncMock()

        async with main_module.lifespan(MagicMock()):
            mock_sweeper.start.assert_awaited_once()
            mock_sweeper.stop.assert_not_awaited()

        mock_sweeper.stop.assert_awaited_once()
        mock_db_manager.close.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_skips_ttl_sweeper_without_config() -> None:
    """No TTL config resolved → the sweeper never starts."""
    import aegra_api.main as main_module

    importlib.reload(main_module)

    with (
        patch("aegra_api.main.run_migrations_async", new_callable=AsyncMock),
        patch("aegra_api.main.db_manager") as mock_db_manager,
        patch("aegra_api.main.get_langgraph_service") as mock_get_langgraph_service,
        patch("aegra_api.main.setup_observability"),
        patch("aegra_api.main.get_thread_ttl_config", return_value=None),
        patch("aegra_api.main.thread_ttl_sweeper") as mock_sweeper,
    ):
        mock_db_manager.initialize = AsyncMock()
        mock_db_manager.close = AsyncMock()
        mock_langgraph_service = MagicMock()
        mock_langgraph_service.initialize = AsyncMock()
        mock_get_langgraph_service.return_value = mock_langgraph_service
        mock_sweeper.start = AsyncMock()
        mock_sweeper.stop = AsyncMock()

        async with main_module.lifespan(MagicMock()):
            pass

        mock_sweeper.start.assert_not_awaited()
        mock_sweeper.stop.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_fails_before_any_db_work_on_invalid_durability_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo in the durability default stops the boot instead of failing every run."""
    import aegra_api.main as main_module

    importlib.reload(main_module)
    # Patch the module's own settings reference: an earlier test in this file reloads aegra_api.settings.
    monkeypatch.setattr(run_preparation.settings.checkpointer, "AEGRA_CHECKPOINT_DURABILITY", "eventually")
    run_preparation.get_default_durability.cache_clear()
    try:
        with (
            patch("aegra_api.main.run_migrations_async", new_callable=AsyncMock) as mock_migrations,
            patch("aegra_api.main.db_manager") as mock_db_manager,
        ):
            mock_db_manager.initialize = AsyncMock()

            with pytest.raises(ValueError, match="AEGRA_CHECKPOINT_DURABILITY"):
                async with main_module.lifespan(MagicMock()):
                    pass

            mock_migrations.assert_not_called()
            mock_db_manager.initialize.assert_not_called()
    finally:
        run_preparation.get_default_durability.cache_clear()


def _recording(events: list[str], label: str, error: Exception | None = None) -> AsyncMock:
    """An awaitable that appends ``label`` to ``events`` when awaited, then raises ``error`` if given."""

    async def record() -> None:
        events.append(label)
        if error is not None:
            raise error

    return AsyncMock(side_effect=record)


@pytest.fixture
def lifespan_events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace every lifespan component with a stub that logs its start/stop calls, in call order."""
    events: list[str] = []
    components = {
        "db_manager": ("db", "initialize", "close"),
        "redis_manager": ("redis", "initialize", "close"),
        "broker_manager": ("broker", "start", "stop"),
        "executor": ("executor", "start", "stop"),
        "lease_reaper": ("reaper", "start", "stop"),
        "cron_scheduler": ("cron", "start", "stop"),
        "thread_ttl_sweeper": ("ttl", "start", "stop"),
    }
    for attr, (label, start_name, stop_name) in components.items():
        stub = MagicMock()
        setattr(stub, start_name, _recording(events, f"{label}.{start_name}"))
        setattr(stub, stop_name, _recording(events, f"{label}.{stop_name}"))
        monkeypatch.setattr(main_module, attr, stub)

    graph_service = MagicMock()
    graph_service.initialize = _recording(events, "graphs.initialize")
    monkeypatch.setattr(main_module, "get_langgraph_service", lambda: graph_service)
    monkeypatch.setattr(main_module, "setup_observability", lambda: None)
    monkeypatch.setattr(main_module, "get_thread_ttl_config", MagicMock)
    # Patch the module's own settings reference: an earlier test in this file reloads aegra_api.settings.
    monkeypatch.setattr(main_module.settings.app, "RUN_MIGRATIONS_ON_STARTUP", False)
    monkeypatch.setattr(main_module.settings.redis, "REDIS_BROKER_ENABLED", True)
    monkeypatch.setattr(main_module.settings.cron, "CRON_ENABLED", True)
    return events


@pytest.mark.unit
@pytest.mark.asyncio
async def test_closes_db_manager_when_graph_load_fails(
    monkeypatch: pytest.MonkeyPatch, lifespan_events: list[str]
) -> None:
    """A graph that fails to import must still close the DB pools, or the process hangs on exit (#683)."""
    graph_service = MagicMock()
    graph_service.initialize = _recording(lifespan_events, "graphs.initialize", error=ImportError("broken graph"))
    monkeypatch.setattr(main_module, "get_langgraph_service", lambda: graph_service)

    with pytest.raises(ImportError, match="broken graph"):
        async with main_module.lifespan(MagicMock()):
            pass

    assert lifespan_events == ["db.initialize", "graphs.initialize", "db.close"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stops_components_in_reverse_start_order_on_shutdown(lifespan_events: list[str]) -> None:
    """Shutdown must drain the executor before closing the broker, Redis and DB it depends on."""
    async with main_module.lifespan(MagicMock()):
        pass

    assert lifespan_events == [
        "db.initialize",
        "graphs.initialize",
        "redis.initialize",
        "broker.start",
        "executor.start",
        "reaper.start",
        "cron.start",
        "ttl.start",
        "ttl.stop",
        "cron.stop",
        "reaper.stop",
        "executor.stop",
        "broker.stop",
        "redis.close",
        "db.close",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unwinds_started_components_when_executor_start_fails(
    monkeypatch: pytest.MonkeyPatch, lifespan_events: list[str]
) -> None:
    """A failed executor start stops what already started, but not the executor or the steps after it."""
    failing_executor = MagicMock()
    failing_executor.start = _recording(lifespan_events, "executor.start", error=RuntimeError("no workers"))
    failing_executor.stop = _recording(lifespan_events, "executor.stop")
    monkeypatch.setattr(main_module, "executor", failing_executor)

    with pytest.raises(RuntimeError, match="no workers"):
        async with main_module.lifespan(MagicMock()):
            pass

    assert lifespan_events == [
        "db.initialize",
        "graphs.initialize",
        "redis.initialize",
        "broker.start",
        "executor.start",
        "broker.stop",
        "redis.close",
        "db.close",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_keeps_stopping_remaining_components_when_one_stop_fails(
    monkeypatch: pytest.MonkeyPatch, lifespan_events: list[str]
) -> None:
    """A failing cleanup must not skip later cleanups, so the DB pools still close."""
    failing_executor = MagicMock()
    failing_executor.start = _recording(lifespan_events, "executor.start")
    failing_executor.stop = _recording(lifespan_events, "executor.stop", error=RuntimeError("stop failed"))
    monkeypatch.setattr(main_module, "executor", failing_executor)

    with pytest.raises(RuntimeError, match="stop failed"):
        async with main_module.lifespan(MagicMock()):
            pass

    assert lifespan_events == [
        "db.initialize",
        "graphs.initialize",
        "redis.initialize",
        "broker.start",
        "executor.start",
        "reaper.start",
        "cron.start",
        "ttl.start",
        "ttl.stop",
        "cron.stop",
        "reaper.stop",
        "executor.stop",
        "broker.stop",
        "redis.close",
        "db.close",
    ]
