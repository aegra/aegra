"""Tests for application lifespan and startup logic"""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aegra_api.main as aegra_main
from aegra_api.observability.base import get_observability_manager
from aegra_api.services import run_preparation, webhook_policy


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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_fails_before_any_db_work_on_invalid_webhooks_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A webhook header that references an unset variable stops the boot instead of every delivery."""
    importlib.reload(aegra_main)
    monkeypatch.delenv("LG_WEBHOOK_TOKEN", raising=False)
    monkeypatch.setattr(
        webhook_policy,
        "load_webhooks_config",
        lambda: {"headers": {"Authorization": "Bearer ${{ env.LG_WEBHOOK_TOKEN }}"}},
    )
    webhook_policy.get_webhook_settings.cache_clear()

    with (
        patch("aegra_api.main.run_migrations_async", new_callable=AsyncMock) as mock_migrations,
        patch("aegra_api.main.db_manager") as mock_db_manager,
    ):
        mock_db_manager.initialize = AsyncMock()

        with pytest.raises(ValueError, match="LG_WEBHOOK_TOKEN"):
            async with aegra_main.lifespan(MagicMock()):
                pass

        mock_migrations.assert_not_called()
        mock_db_manager.initialize.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_drains_webhooks_after_the_executor() -> None:
    """Runs finalized while the executor drains queue their own webhooks, so the dispatcher stops after it."""
    importlib.reload(aegra_main)
    stopped: list[str] = []

    with (
        patch("aegra_api.main.run_migrations_async", new_callable=AsyncMock),
        patch("aegra_api.main.db_manager") as mock_db_manager,
        patch("aegra_api.main.get_langgraph_service") as mock_get_langgraph_service,
        patch("aegra_api.main.setup_observability"),
        patch("aegra_api.main.cron_scheduler") as mock_cron_scheduler,
        patch("aegra_api.main.executor") as mock_executor,
        patch("aegra_api.main.webhook_dispatcher") as mock_dispatcher,
        patch("aegra_api.main.broker_manager") as mock_broker,
    ):
        mock_db_manager.initialize = AsyncMock()
        mock_db_manager.close = AsyncMock(side_effect=lambda: stopped.append("db"))
        mock_get_langgraph_service.return_value.initialize = AsyncMock()
        mock_cron_scheduler.start = AsyncMock()
        mock_cron_scheduler.stop = AsyncMock()
        mock_executor.start = AsyncMock()
        mock_executor.stop = AsyncMock(side_effect=lambda: stopped.append("executor"))
        mock_dispatcher.stop = AsyncMock(side_effect=lambda: stopped.append("webhooks"))
        mock_broker.start = AsyncMock()
        mock_broker.stop = AsyncMock(side_effect=lambda: stopped.append("broker"))

        async with aegra_main.lifespan(MagicMock()):
            mock_dispatcher.stop.assert_not_awaited()

    assert stopped == ["executor", "webhooks", "broker", "db"]
