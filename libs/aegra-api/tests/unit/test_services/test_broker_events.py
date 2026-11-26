"""Unit tests for the default BaseRunBroker.events: the replay-then-aiter join brokers inherit."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from aegra_api.services.base_broker import BaseRunBroker
from aegra_api.services.broker import RunBroker


class _V012Broker(BaseRunBroker):
    """A third-party broker written against 0.12.0: it implements only the abstract methods.

    Its ``aiter`` re-yields part of the stored events before the live ones, as an aiter that
    includes its backlog does.
    """

    def __init__(self, stored: list[tuple[str, Any]], live: list[tuple[str, Any]]) -> None:
        self._stored = stored
        self._live = live
        self.aiter_calls = 0

    async def put(self, event_id: str, payload: Any, *, resumable: bool = True) -> None:
        self._live.append((event_id, payload))

    async def aiter(self) -> AsyncIterator[tuple[str, Any]]:
        self.aiter_calls += 1
        for event in self._live:
            yield event

    async def replay(self, last_event_id: str | None) -> list[tuple[str, Any]]:
        return list(self._stored)

    def mark_finished(self) -> None:
        pass

    def is_finished(self) -> bool:
        return False


def _event(event_id: str) -> tuple[str, tuple[str, dict[str, Any]]]:
    return event_id, ("values", {"id": event_id})


END = ("e-end", ("end", {"status": "success"}))


async def _ids(events: AsyncIterator[tuple[str, Any]]) -> list[str]:
    return [event_id async for event_id, _ in events]


class TestDefaultEvents:
    def test_a_subclass_without_events_still_instantiates(self) -> None:
        broker = _V012Broker([], [])

        assert isinstance(broker, BaseRunBroker)

    async def test_default_events_replay_then_follow_aiter_without_repeating_the_replay(self) -> None:
        broker = _V012Broker([_event("e1"), _event("e2")], [_event("e2"), _event("e3"), END])

        sent = await _ids(broker.events(follow=True))

        assert sent == ["e1", "e2", "e3", "e-end"]

    async def test_default_events_without_follow_never_iterate_aiter(self) -> None:
        broker = _V012Broker([_event("e1")], [_event("e2"), END])

        sent = await _ids(broker.events(follow=False))

        assert sent == ["e1"]
        assert broker.aiter_calls == 0

    @pytest.mark.parametrize(
        ("stored", "live", "expected", "aiter_calls"),
        [
            ([_event("e1"), END, _event("e2")], [_event("e3")], ["e1", "e-end"], 0),
            ([_event("e1")], [("e-err", ("error", {"error": "Boom"})), END], ["e1", "e-err"], 1),
        ],
        ids=["end-in-replay", "error-in-aiter"],
    )
    async def test_default_events_stop_after_the_terminal_event(
        self, stored: list[tuple[str, Any]], live: list[tuple[str, Any]], expected: list[str], aiter_calls: int
    ) -> None:
        broker = _V012Broker(stored, live)

        sent = await _ids(broker.events(follow=True))

        assert sent == expected
        assert broker.aiter_calls == aiter_calls


class TestInMemoryEvents:
    async def test_follow_delivers_events_put_while_the_replay_is_being_read(self) -> None:
        broker = RunBroker("run-1")
        await broker.put("e1", ("values", {"n": 1}))
        events = broker.events(follow=True)
        first, _ = await anext(events)

        await broker.put("e2", ("values", {"n": 2}))
        await broker.put("e3", ("values", {"n": 3}))
        await broker.put(*END)

        assert [first, *await _ids(events)] == ["e1", "e2", "e3", "e-end"]

    async def test_follow_on_a_finished_broker_ends_after_its_stored_end(self) -> None:
        broker = RunBroker("run-1")
        await broker.put("e1", ("values", {"n": 1}))
        await broker.put(*END)

        sent = await asyncio.wait_for(_ids(broker.events(follow=True)), 1)

        assert sent == ["e1", "e-end"]
