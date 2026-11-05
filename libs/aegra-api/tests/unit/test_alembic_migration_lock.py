"""Exercise online Alembic execution and cleanup through its actual environment."""

import asyncio
import runpy
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from alembic.config import Config

from aegra_api.core import migrations
from alembic import context


@pytest.mark.parametrize("upgrade_fails", [False, True])
@pytest.mark.parametrize("cleanup_error", [None, RuntimeError("dispose failed"), asyncio.CancelledError()])
def test_online_upgrade_disposes_engine_before_releasing_lock(
    upgrade_fails: bool, cleanup_error: BaseException | None
) -> None:
    order: list[str] = []
    upgrade_error = RuntimeError("migration failed")
    engine = MagicMock()
    connection = MagicMock()

    @contextmanager
    def lock() -> Iterator[None]:
        order.append("lock")
        try:
            yield
        finally:
            order.append("unlock")

    async def enter() -> MagicMock:
        order.append("connect")
        return connection

    async def exit_connection(*_args: object) -> None:
        order.append("disconnect")

    async def apply_migrations(_callback: object) -> None:
        order.append("upgrade")
        if upgrade_fails:
            raise upgrade_error

    async def dispose() -> None:
        order.append("dispose")
        if cleanup_error is not None:
            raise cleanup_error

    engine.connect.return_value.__aenter__ = AsyncMock(side_effect=enter)
    engine.connect.return_value.__aexit__ = AsyncMock(side_effect=exit_connection)
    engine.dispose = AsyncMock(side_effect=dispose)
    connection.run_sync = AsyncMock(side_effect=apply_migrations)
    env_path = Path(__file__).resolve().parents[2] / "alembic" / "env.py"

    with (
        patch.object(context, "config", Config(), create=True),
        patch.object(context, "is_offline_mode", return_value=False),
        patch.object(migrations, "migration_advisory_lock", lock),
        patch("sqlalchemy.ext.asyncio.async_engine_from_config", return_value=engine),
    ):
        expected_error = upgrade_error if upgrade_fails else cleanup_error
        if expected_error is not None:
            with pytest.raises(type(expected_error)) as exc_info:
                runpy.run_path(str(env_path))
            assert exc_info.value is expected_error
        else:
            runpy.run_path(str(env_path))

    assert order == ["lock", "connect", "upgrade", "disconnect", "dispose", "unlock"]
    engine.dispose.assert_awaited_once_with()
