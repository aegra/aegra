"""keep_latest pruning against a real PostgreSQL checkpointer.

A DeltaChannel's value is rebuilt from ancestor checkpoints' writes, which only a
real checkpointer shows. Lives under e2e because only those CI jobs have Postgres.
"""

import operator
from collections.abc import AsyncIterator, Sequence
from typing import Annotated, Any, TypedDict
from unittest.mock import patch
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from aegra_api.services.thread_ttl import _prune_checkpoint_history
from aegra_api.settings import settings

_TURNS = 3


def _append(state: list[str] | None, writes: Sequence[Any]) -> list[str]:
    merged = list(state or [])
    for write in writes:
        merged.extend(write if isinstance(write, list) else [write])
    return merged


class DeltaState(TypedDict):
    messages: Annotated[list, DeltaChannel(_append, snapshot_frequency=50)]


class PlainState(TypedDict):
    messages: Annotated[list, operator.add]


def _graph(builder: StateGraph[Any], saver: AsyncPostgresSaver) -> CompiledStateGraph:
    builder.add_node("say", lambda state: {"messages": ["hi"]})
    builder.add_edge(START, "say")
    return builder.compile(checkpointer=saver)


@pytest.fixture
async def pool() -> AsyncIterator[AsyncConnectionPool[AsyncConnection[DictRow]]]:
    pool = AsyncConnectionPool(
        conninfo=settings.db.database_url_sync,
        connection_class=AsyncConnection[DictRow],
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row},
    )
    try:
        await pool.open(wait=True, timeout=10)
    except PoolTimeout as exc:
        pytest.skip(f"PostgreSQL test database is unavailable: {exc}")
    try:
        await AsyncPostgresSaver(pool).setup()
        yield pool
    finally:
        await pool.close()


async def _checkpoint_count(pool: AsyncConnectionPool[AsyncConnection[DictRow]], thread_id: str) -> int:
    async with pool.connection() as conn:
        cursor = await conn.execute("SELECT count(*) AS n FROM checkpoints WHERE thread_id = %s", (thread_id,))
        row = await cursor.fetchone()
    assert row is not None
    return row["n"]


async def _run_turns(graph: CompiledStateGraph, config: RunnableConfig) -> None:
    for _ in range(_TURNS):
        await graph.ainvoke({"messages": []}, config)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_keep_latest_keeps_delta_channel_state(pool: AsyncConnectionPool[AsyncConnection[DictRow]]) -> None:
    saver = AsyncPostgresSaver(pool)
    graph = _graph(StateGraph(DeltaState), saver)
    thread_id = str(uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        await _run_turns(graph, config)
        before = (await graph.aget_state(config)).values

        with patch("aegra_api.services.thread_ttl.db_manager.get_checkpointer", return_value=saver):
            await _prune_checkpoint_history(thread_id)

        assert before == {"messages": ["hi"] * _TURNS}
        assert (await graph.aget_state(config)).values == before
    finally:
        await saver.adelete_thread(thread_id)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_keep_latest_compacts_plain_channel_thread_to_one_checkpoint(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
) -> None:
    saver = AsyncPostgresSaver(pool)
    graph = _graph(StateGraph(PlainState), saver)
    thread_id = str(uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        await _run_turns(graph, config)
        before = (await graph.aget_state(config)).values

        with patch("aegra_api.services.thread_ttl.db_manager.get_checkpointer", return_value=saver):
            await _prune_checkpoint_history(thread_id)

        assert await _checkpoint_count(pool, thread_id) == 1
        assert (await graph.aget_state(config)).values == before
    finally:
        await saver.adelete_thread(thread_id)
