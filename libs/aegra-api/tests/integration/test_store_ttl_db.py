"""Store TTL tests that need a real PostgreSQL.

Whether an expired item is actually deleted is a property of the store's
sweeper running against real rows, which no mock can show. These drive the
real ``DatabaseManager`` with a ``store.ttl`` block and skip when no database
is reachable.
"""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import patch
from uuid import uuid4

import psycopg
import pytest

from aegra_api.core.database import DatabaseManager
from aegra_api.settings import settings

# Sweep every 0.6s: the shortest interval the store accepts is expressed in minutes.
_FAST_SWEEP = {"ttl": {"default_ttl": 60, "sweep_interval_minutes": 0.01}}
_WAIT_TIMEOUT = 15.0


@pytest.fixture(autouse=True)
async def _require_database() -> None:
    """Skip fast: DatabaseManager would otherwise wait out the pool timeout without a database."""
    try:
        async with await psycopg.AsyncConnection.connect(settings.db.database_url_sync, connect_timeout=3):
            pass
    except psycopg.Error as exc:
        pytest.skip(f"PostgreSQL test database is unavailable: {exc}")


@pytest.fixture
async def manager() -> AsyncIterator[DatabaseManager]:
    """A DatabaseManager that sweeps quickly, or skip without a database."""
    mgr = DatabaseManager()
    with patch("aegra_api.core.database.load_store_config", return_value=_FAST_SWEEP):
        try:
            await mgr.initialize()
        except Exception as exc:  # noqa: BLE001 - any connect failure means "no database here"
            pytest.skip(f"PostgreSQL test database is unavailable: {exc}")
    try:
        yield mgr
    finally:
        await mgr.close()


async def _wait_until_gone(manager: DatabaseManager, namespace: tuple[str, ...], key: str) -> bool:
    store = manager.get_store()
    async with asyncio.timeout(_WAIT_TIMEOUT):
        while await store.aget(namespace, key, refresh_ttl=False) is not None:
            await asyncio.sleep(0.2)
    return True


@pytest.mark.asyncio
async def test_sweeper_deletes_an_item_whose_ttl_elapsed(manager: DatabaseManager) -> None:
    store = manager.get_store()
    namespace = ("store-ttl-test", uuid4().hex)
    await store.aput(namespace, "short", {"n": 1}, ttl=0.01)  # 0.6 seconds
    await store.aput(namespace, "default", {"n": 2})  # default_ttl: an hour

    assert await _wait_until_gone(manager, namespace, "short")

    kept = await store.aget(namespace, "default")
    assert kept is not None
    assert kept.value == {"n": 2}
    await store.adelete(namespace, "default")


@pytest.mark.asyncio
async def test_close_stops_the_sweeper() -> None:
    mgr = DatabaseManager()
    with patch("aegra_api.core.database.load_store_config", return_value=_FAST_SWEEP):
        try:
            await mgr.initialize()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"PostgreSQL test database is unavailable: {exc}")
    task = mgr.get_store()._ttl_sweeper_task
    assert task is not None
    assert not task.done()

    await mgr.close()

    assert task.done()
