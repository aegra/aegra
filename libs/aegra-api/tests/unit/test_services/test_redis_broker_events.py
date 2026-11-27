"""Unit tests for RedisRunBroker.events: paged replay and the gap-free live join.

fakeredis, not MagicMock: the join is only correct under real list and pub/sub semantics
(RPUSH before PUBLISH, LTRIM shifting positions, LPOS, a subscriber missing what was published
before it subscribed). Orderings Redis does not expose are forced by wrapping ``client.lrange``.
"""

import asyncio
import json
import random
import tracemalloc
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from typing import Any
from unittest.mock import MagicMock

import fakeredis
import pytest
from redis import RedisError
from redis.asyncio.client import PubSub
from redis.exceptions import ResponseError

from aegra_api.core.redis_manager import redis_manager
from aegra_api.services import redis_broker
from aegra_api.services.base_broker import ReplayGapError
from aegra_api.services.redis_broker import RedisRunBroker

RUN_ID = "run-123"
CACHE_KEY = f"aegra:run:cache:{RUN_ID}"
END_ID = f"{RUN_ID}_event_end"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> fakeredis.FakeAsyncRedis:
    """One fake server shared by writers and readers; retries and resubscribes without sleeping."""
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    monkeypatch.setattr(redis_manager, "get_client", lambda: redis)
    monkeypatch.setattr(redis_broker, "_backoff_delay", lambda attempt: 0.0)
    return redis


def _broker(run_id: str = RUN_ID) -> RedisRunBroker:
    """A broker object of its own, as another process (or a fresh cache entry) would hold."""
    return RedisRunBroker(run_id, f"aegra:run:{run_id}", f"aegra:run:cache:{run_id}", f"aegra:run:counter:{run_id}")


def _event_id(i: int) -> str:
    return f"{RUN_ID}_event_{2 * i + 1}"


def _event(i: int) -> tuple[str, dict[str, Any]]:
    return ("messages", {"params": {"namespace": [], "data": {"event": f"m{i}"}}})


def _ids(n: int, *, end: bool = False) -> list[str]:
    return [_event_id(i) for i in range(n)] + ([END_ID] if end else [])


async def _write(writer: RedisRunBroker, indexes: range, *, end: bool = False) -> None:
    for i in indexes:
        await writer.put(_event_id(i), _event(i))
    if end:
        await writer.put(END_ID, ("end", {"status": "success"}))


async def _collect(events: AsyncGenerator[tuple[str, Any], None]) -> list[str]:
    async with aclosing(events) as stream:
        return [event_id async for event_id, _ in stream]


async def _until(condition: Callable[[], Awaitable[bool]], timeout: float = 2.0) -> None:
    async def poll() -> None:
        while not await condition():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


async def _subscribed(client: fakeredis.FakeAsyncRedis, run_id: str = RUN_ID) -> bool:
    [(_, count)] = await client.pubsub_numsub(f"aegra:run:{run_id}")
    return count > 0


async def _failing_publish(message: str) -> None:
    raise RedisError("publish dropped")


async def _rejected_lpos(key: str, element: str) -> int | None:
    raise ResponseError("ERR unknown command 'LPOS'")


def _counting_lpos(client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count LPOS calls: each is a full scan of the buffer on the server."""
    calls: list[int] = []
    lpos = client.lpos

    async def counted(key: str, element: str) -> int | None:
        calls.append(1)
        return await lpos(key, element)

    monkeypatch.setattr(client, "lpos", counted)
    return calls


def _wrap_lrange(
    client: fakeredis.FakeAsyncRedis,
    monkeypatch: pytest.MonkeyPatch,
    before: Callable[[int, int, int], Awaitable[None]] | None = None,
    after: Callable[[list[str], int, int], Awaitable[None]] | None = None,
) -> None:
    """Run ``before(call, start, end)`` / ``after(page, start, end)`` around every LRANGE."""
    lrange = client.lrange
    calls = 0

    async def wrapped(key: str, start: int, end: int) -> list[str]:
        nonlocal calls
        calls += 1
        if before is not None:
            await before(calls, start, end)
        page = await lrange(key, start, end)
        if after is not None:
            await after(page, start, end)
        return page

    monkeypatch.setattr(client, "lrange", wrapped)


async def _peak_bytes(work: Awaitable[None]) -> int:
    tracemalloc.start()
    try:
        await work
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


class TestStoredEvents:
    @pytest.mark.parametrize("seed", range(30))
    async def test_stored_events_match_replay(self, client: fakeredis.FakeAsyncRedis, seed: int) -> None:
        """Random buffers of up to four pages, with adjacent retry copies: the same events as replay(None)."""
        rng = random.Random(seed)
        await _write(_broker(), range(rng.randint(0, 4 * redis_broker._REPLAY_PAGE_EVENTS)), end=rng.random() < 0.7)
        for _ in range(rng.randint(0, 3)):  # what a timed-out-but-applied cache pipeline leaves behind
            length = await client.llen(CACHE_KEY)
            if length:
                copy = await client.lindex(CACHE_KEY, rng.randrange(length))
                await client.linsert(CACHE_KEY, "AFTER", copy, copy)

        paged = [event async for event in _broker().events(follow=False)]

        whole = await _broker().replay(None)
        cut = next((i + 1 for i, (_, payload) in enumerate(whole) if payload[0] == "end"), len(whole))
        assert paged == whole[:cut]

    async def test_stored_events_skip_a_retry_copy_across_a_page_boundary(
        self, client: fakeredis.FakeAsyncRedis
    ) -> None:
        page = redis_broker._REPLAY_PAGE_EVENTS
        await _write(_broker(), range(2 * page), end=True)
        last_of_first_page = await client.lindex(CACHE_KEY, page - 1)
        await client.linsert(CACHE_KEY, "AFTER", last_of_first_page, last_of_first_page)

        sent = [event_id async for event_id, _ in _broker().events(follow=False)]

        assert sent == _ids(2 * page, end=True)

    async def test_stored_events_skip_a_retry_copy_that_is_not_adjacent(self, client: fakeredis.FakeAsyncRedis) -> None:
        """Two writers can interleave a retry copy with another event; replay(None) would send it twice."""
        await _write(_broker(), range(5))
        await client.rpush(CACHE_KEY, await client.lindex(CACHE_KEY, 2))
        await _write(_broker(), range(5, 7), end=True)

        sent = [event_id async for event_id, _ in _broker().events(follow=False)]

        assert sent == _ids(7, end=True)

    @pytest.mark.parametrize("trims", [1, 3])
    async def test_stored_events_do_not_resume_after_a_later_copy_of_their_last_event(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch, trims: int
    ) -> None:
        """The first page ends at X; X's retry copy sits two after it. Head trims before the next page
        shift the real X out of it, and resuming after the copy would skip Y. (Exactly two trims
        put the copy at X's own index, where no read can tell them apart: a known gap.)"""
        page = redis_broker._REPLAY_PAGE_EVENTS
        await _write(_broker(), range(page))
        last_of_first_page = await client.lindex(CACHE_KEY, page - 1)
        await _write(_broker(), range(page, page + 1))
        await client.rpush(CACHE_KEY, last_of_first_page)
        await _write(_broker(), range(page + 1, page + 3), end=True)

        async def trim_before_the_second_page(call: int, start: int, end: int) -> None:
            for _ in range(trims if call == 2 else 0):
                await client.lpop(CACHE_KEY)
                await client.rpush(CACHE_KEY, json.dumps({"event_id": f"pad-{call}", "payload": ["messages", {}]}))

        _wrap_lrange(client, monkeypatch, before=trim_before_the_second_page)

        sent = [event_id async for event_id, _ in _broker().events(follow=False)]

        assert sent == _ids(page + 3, end=True)

    async def test_stored_events_resume_after_their_last_event_when_the_cap_trims_between_pages(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each later page sees 0-6 head trims first, which is what LTRIM at the cap does on every append."""
        await _write(_broker(), range(3 * redis_broker._REPLAY_PAGE_EVENTS), end=True)

        async def trim(call: int, start: int, end: int) -> None:
            for _ in range(call % 7 if start > 0 else 0):
                await client.lpop(CACHE_KEY)
                await client.rpush(CACHE_KEY, json.dumps({"event_id": f"pad-{call}", "payload": ["messages", {}]}))

        _wrap_lrange(client, monkeypatch, before=trim)

        sent = [event_id async for event_id, _ in _broker().events(follow=False)]

        assert sent == _ids(3 * redis_broker._REPLAY_PAGE_EVENTS, end=True)

    async def test_stored_events_raise_replay_gap_when_redis_fails_midway(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = redis_broker._REPLAY_PAGE_EVENTS
        await _write(_broker(), range(2 * page), end=True)

        async def fail_after_the_first_page(call: int, start: int, end: int) -> None:
            if start > 0:
                raise RedisError("connection reset")

        _wrap_lrange(client, monkeypatch, before=fail_after_the_first_page)
        sent: list[str] = []

        with pytest.raises(ReplayGapError) as gap:
            async for event_id, _ in _broker().events(follow=False):
                sent.append(event_id)

        assert sent == _ids(page)
        assert isinstance(gap.value.__cause__, RedisError)

    async def test_stored_events_raise_replay_gap_when_the_buffer_expires_midway(
        self, client: fakeredis.FakeAsyncRedis
    ) -> None:
        page = redis_broker._REPLAY_PAGE_EVENTS
        await _write(_broker(), range(2 * page), end=True)
        events = _broker().events(follow=False)
        first, _ = await anext(events)
        await client.delete(CACHE_KEY)
        rest: list[str] = []

        with pytest.raises(ReplayGapError, match="expired"):
            async for event_id, _ in events:
                rest.append(event_id)

        assert [first, *rest] == _ids(page)

    async def test_stored_events_raise_replay_gap_when_the_cap_trims_past_the_reader(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(redis_broker, "_REPLAY_MAX_EVENTS", 8)
        monkeypatch.setattr(redis_broker, "_REPLAY_PAGE_EVENTS", 3)
        writer = _broker()
        await _write(writer, range(8))
        events = _broker().events(follow=False)
        await anext(events)
        await _write(writer, range(8, 20), end=True)

        with pytest.raises(ReplayGapError, match="trimmed"):
            async for _ in events:
                pass

    async def test_stored_events_raise_replay_gap_when_the_server_rejects_lpos(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(redis_broker, "_REPLAY_MAX_EVENTS", 8)
        monkeypatch.setattr(redis_broker, "_REPLAY_PAGE_EVENTS", 3)
        monkeypatch.setattr(client, "lpos", _rejected_lpos)
        writer = _broker()
        await _write(writer, range(8))
        events = _broker().events(follow=False)
        await anext(events)
        await _write(writer, range(8, 12))

        with pytest.raises(ReplayGapError, match="LPOS"):
            async for _ in events:
                pass

    async def test_stored_events_of_an_empty_buffer_end_without_error(self, client: fakeredis.FakeAsyncRedis) -> None:
        assert [event async for event in _broker().events(follow=False)] == []

    async def test_stored_events_stop_after_an_error_event(self, client: fakeredis.FakeAsyncRedis) -> None:
        writer = _broker()
        await _write(writer, range(2))
        await writer.put(f"{RUN_ID}_event_error", ("error", {"error": "Boom"}))
        await writer.put(END_ID, ("end", {"status": "error"}))

        sent = [event_id async for event_id, _ in _broker().events(follow=False)]

        assert sent == [*_ids(2), f"{RUN_ID}_event_error"]

    async def test_events_mark_the_broker_finished_after_end(self, client: fakeredis.FakeAsyncRedis) -> None:
        await _write(_broker(), range(2), end=True)
        reader = _broker()

        await _collect(reader.events(follow=False))

        assert reader.is_finished()

    @pytest.mark.parametrize("follow", [False, True], ids=["finished", "running"])
    async def test_reading_a_run_holds_a_page_not_the_buffer(
        self, client: fakeredis.FakeAsyncRedis, follow: bool
    ) -> None:
        """The OOM shape: a 50 kB state resent each step. A running run's `end` arrives once the reader is live."""
        big = {"params": {"namespace": ["worker:1"], "data": {"messages": [{"content": "x" * 50_000}]}}}

        async def peak(steps: int) -> int:
            run_id = f"run-{steps}"
            writer = _broker(run_id)
            for i in range(steps):
                await writer.put(f"{run_id}_event_{2 * i + 1}", ("values", big))
            if not follow:
                await writer.put(f"{run_id}_event_end", ("end", {"status": "success"}))

            async def read() -> None:
                task = asyncio.create_task(_collect(_broker(run_id).events(follow=follow)))
                if follow:
                    await _until(lambda: _subscribed(client, run_id))
                    await writer.put(f"{run_id}_event_end", ("end", {"status": "success"}))
                await asyncio.wait_for(task, 5)

            return await _peak_bytes(read())

        short, long = await peak(40), await peak(160)
        whole = await _peak_bytes(_broker("run-160").replay(None))

        assert long < short * 1.5, (short, long)
        assert long * 5 < whole, (long, whole)


class TestFollowEvents:
    async def test_follow_delivers_an_end_written_between_the_stored_read_and_the_subscription(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: replay() then aiter() lost an event stored and published between its LRANGE and
        its SUBSCRIBE; for `end`, aiter found it in the buffer and exited without yielding it."""
        writer = _broker()
        await _write(writer, range(5))
        wrote = False

        async def end_after_the_tail_read(page: list[str], start: int, end: int) -> None:
            nonlocal wrote
            if not wrote and len(page) < end - start + 1:
                wrote = True
                await writer.put(END_ID, ("end", {"status": "success"}))

        _wrap_lrange(client, monkeypatch, after=end_after_the_tail_read)
        monkeypatch.setattr(redis_broker, "_LIVE_RECHECK_SECONDS", 60.0)

        # Far under the recheck: only the read right after subscribing can find it.
        assert await asyncio.wait_for(_collect(_broker().events(follow=True)), 2) == _ids(5, end=True)

    async def test_follow_sends_an_event_both_stored_and_published_during_the_join_once(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        writer = _broker()
        await _write(writer, range(5))

        async def write_before_the_second_read(call: int, start: int, end: int) -> None:
            if call == 2:  # the read right after subscribing: stored and published
                await _write(writer, range(5, 6), end=True)

        _wrap_lrange(client, monkeypatch, before=write_before_the_second_read)

        assert await asyncio.wait_for(_collect(_broker().events(follow=True)), 2) == _ids(6, end=True)

    @pytest.mark.parametrize("seed", range(25))
    async def test_follow_delivers_every_event_once_under_random_interleavings(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch, seed: int
    ) -> None:
        """Random pauses on both sides, republished events, one dropped pub/sub connection, page
        sizes 2, 3 or 16. The recheck is disabled so only pub/sub wake-ups can deliver."""
        rng = random.Random(seed)
        monkeypatch.setattr(redis_broker, "_REPLAY_PAGE_EVENTS", rng.choice([2, 3, 16]))
        monkeypatch.setattr(redis_broker, "_LIVE_RECHECK_SECONDS", 60.0)
        total = rng.randint(1, 60)
        backlog = rng.randint(0, total)
        writer = _broker()
        await _write(writer, range(backlog))
        get_message = PubSub.get_message
        drop = rng.random() < 0.3

        async def flaky_get_message(self: PubSub, *args: Any, **kwargs: Any) -> Any:
            nonlocal drop
            if drop and rng.random() < 0.05:
                drop = False
                raise RedisError("pubsub dropped")
            return await get_message(self, *args, **kwargs)

        monkeypatch.setattr(PubSub, "get_message", flaky_get_message)

        async def produce() -> None:
            for i in range(backlog, total):
                await asyncio.sleep(rng.choice([0, 0, 0.001, 0.003]))
                await writer.put(_event_id(i), _event(i))
                if rng.random() < 0.1:  # a publish retried after it already reached Redis
                    await client.publish(writer._channel, json.dumps({"event_id": _event_id(i), "payload": _event(i)}))
            await writer.put(END_ID, ("end", {"status": "success"}))

        async def consume() -> list[str]:
            received: list[str] = []
            async with aclosing(_broker().events(follow=True)) as events:
                async for event_id, _ in events:
                    received.append(event_id)
                    await asyncio.sleep(rng.choice([0, 0, 0.002, 0.01]))
            return received

        consumer = asyncio.create_task(consume())
        await asyncio.sleep(rng.choice([0, 0.001, 0.01]))
        await produce()

        assert await asyncio.wait_for(consumer, 10) == _ids(total, end=True)

    async def test_follow_resumes_from_the_buffer_after_a_pubsub_error(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """aiter resubscribes without backfill; events written while disconnected must still arrive."""
        monkeypatch.setattr(redis_broker, "_LIVE_RECHECK_SECONDS", 60.0)
        writer = _broker()
        await _write(writer, range(2))
        get_message = PubSub.get_message
        broken = asyncio.Event()

        async def disconnect_once(self: PubSub, *args: Any, **kwargs: Any) -> Any:
            if broken.is_set():
                return await get_message(self, *args, **kwargs)
            await _write(writer, range(2, 4))  # published while the connection is going away
            broken.set()
            raise RedisError("connection reset")

        monkeypatch.setattr(PubSub, "get_message", disconnect_once)
        consumer = asyncio.create_task(_collect(_broker().events(follow=True)))
        await asyncio.wait_for(broken.wait(), 2)
        await _write(writer, range(4, 5), end=True)

        assert await asyncio.wait_for(consumer, 2) == _ids(5, end=True)

    async def test_follow_drops_republished_and_recopied_live_events(self, client: fakeredis.FakeAsyncRedis) -> None:
        writer = _broker()
        consumer = asyncio.create_task(_collect(_broker().events(follow=True)))
        await _until(lambda: _subscribed(client))
        await _write(writer, range(1))
        message = await client.lindex(CACHE_KEY, 0)
        await client.publish(writer._channel, message)  # a publish retry
        await client.rpush(CACHE_KEY, message)  # a cache retry
        await client.publish(writer._channel, message)
        await _write(writer, range(1, 2), end=True)

        assert await asyncio.wait_for(consumer, 2) == _ids(2, end=True)

    async def test_follow_delivers_a_stored_event_whose_publish_failed_within_one_recheck(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """put() drops a PUBLISH that fails three times; the follower still reads the event from the list."""
        monkeypatch.setattr(redis_broker, "_LIVE_RECHECK_SECONDS", 0.1)
        writer = _broker()
        await _write(writer, range(3))
        received: list[str] = []

        async def consume() -> None:
            async with aclosing(_broker().events(follow=True)) as events:
                async for event_id, _ in events:
                    received.append(event_id)

        consumer = asyncio.create_task(consume())
        await _until(lambda: _subscribed(client))
        publish = writer._publish_event
        monkeypatch.setattr(writer, "_publish_event", _failing_publish)
        await writer.put(_event_id(3), _event(3))
        monkeypatch.setattr(writer, "_publish_event", publish)

        async def delivered() -> bool:
            return len(received) == 4

        await _until(delivered, timeout=1)  # nothing else is written meanwhile, so no pub/sub wake-up
        await writer.put(END_ID, ("end", {"status": "success"}))
        await asyncio.wait_for(consumer, 2)

        assert received == _ids(4, end=True)

    async def test_follow_ends_when_another_process_failed_to_publish_end(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reader's broker never sees the writer's put(), so only the stored `end` can end it."""
        monkeypatch.setattr(redis_broker, "_LIVE_RECHECK_SECONDS", 0.1)
        writer = _broker()
        await _write(writer, range(3))
        consumer = asyncio.create_task(_collect(_broker().events(follow=True)))
        await _until(lambda: _subscribed(client))
        monkeypatch.setattr(writer, "_publish_event", _failing_publish)

        await writer.put(END_ID, ("end", {"status": "success"}))

        assert await asyncio.wait_for(consumer, 1) == _ids(3, end=True)

    async def test_follow_ends_after_one_more_read_when_this_process_finished_the_broker(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parity with aiter: a locally finished broker with no stored `end` still ends its followers."""
        monkeypatch.setattr(redis_broker, "_LIVE_RECHECK_SECONDS", 0.1)
        broker = _broker()
        await _write(broker, range(2))
        consumer = asyncio.create_task(_collect(broker.events(follow=True)))
        await _until(lambda: _subscribed(client))
        await client.rpush(CACHE_KEY, json.dumps({"event_id": _event_id(2), "payload": list(_event(2))}))

        broker.mark_finished()

        assert await asyncio.wait_for(consumer, 1) == _ids(3)

    async def test_follow_keeps_its_place_when_the_cap_trims_a_running_buffer(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(redis_broker, "_REPLAY_MAX_EVENTS", 12)
        monkeypatch.setattr(redis_broker, "_REPLAY_PAGE_EVENTS", 3)
        writer = _broker()
        await _write(writer, range(12))  # at the cap before the reader opens
        received: list[str] = []

        async def consume() -> None:
            async with aclosing(_broker().events(follow=True)) as events:
                async for event_id, _ in events:
                    received.append(event_id)
                    await asyncio.sleep(0.001)

        async def produce() -> None:
            for i in range(12, 30):  # every append at the cap moves the reader's place one to the left
                while len(received) < i - 5:  # never a whole cap behind, however slow the host
                    await asyncio.sleep(0.001)
                await writer.put(_event_id(i), _event(i))
            await writer.put(END_ID, ("end", {"status": "success"}))

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(produce(), 5)
        await asyncio.wait_for(consumer, 5)

        assert received == _ids(30, end=True)

    async def test_follow_resumes_at_the_head_with_a_warning_when_the_cap_trims_past_the_reader(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The skipped events no longer exist anywhere; the stream stays live for the rest of the run."""
        monkeypatch.setattr(redis_broker, "_REPLAY_MAX_EVENTS", 8)
        monkeypatch.setattr(redis_broker, "_REPLAY_PAGE_EVENTS", 3)
        warning = MagicMock()
        monkeypatch.setattr(redis_broker.logger, "warning", warning)
        writer = _broker()
        await _write(writer, range(8))
        events = _broker().events(follow=True)
        first, _ = await anext(events)
        await _write(writer, range(8, 20))  # 12 appends at an 8-event cap while the reader is suspended
        rest = asyncio.create_task(_collect(events))
        await _until(lambda: _subscribed(client))

        await writer.put(END_ID, ("end", {"status": "success"}))

        # The first page (e0-e2) was already read when the cap dropped it; e12-e19 is what survived.
        assert [first, *await asyncio.wait_for(rest, 2)] == [*_ids(3), *_ids(20)[12:], END_ID]
        assert any("lost its place" in call.args[0] for call in warning.call_args_list)

    async def test_follow_closes_its_subscription_when_the_consumer_stops(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        closed: list[PubSub] = []
        aclose = PubSub.aclose

        async def recording_aclose(self: PubSub) -> None:
            closed.append(self)
            await aclose(self)

        monkeypatch.setattr(PubSub, "aclose", recording_aclose)
        writer = _broker()
        events = _broker().events(follow=True)
        reading = asyncio.ensure_future(anext(events))
        await _until(lambda: _subscribed(client))
        await _write(writer, range(1))
        await asyncio.wait_for(reading, 2)

        await events.aclose()

        assert len(closed) == 1
        assert closed[0].connection is None

    async def test_follow_raises_instead_of_resending_from_the_head_when_the_server_rejects_lpos(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without LPOS no read can find the place again; restarting at the head would resend the run."""
        monkeypatch.setattr(redis_broker, "_REPLAY_MAX_EVENTS", 8)
        monkeypatch.setattr(redis_broker, "_REPLAY_PAGE_EVENTS", 3)
        monkeypatch.setattr(client, "lpos", _rejected_lpos)
        writer = _broker()
        await _write(writer, range(8))
        events = _broker().events(follow=True)
        await anext(events)
        await _write(writer, range(8, 12))

        with pytest.raises(ReplayGapError, match="LPOS"):
            await asyncio.wait_for(_collect(events), 2)

    async def test_follow_at_the_cap_finds_its_place_without_an_lpos_per_live_event(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At the cap every append moves the place left; LPOS scans the whole buffer on the server."""
        monkeypatch.setattr(redis_broker, "_REPLAY_MAX_EVENTS", 12)
        monkeypatch.setattr(redis_broker, "_REPLAY_PAGE_EVENTS", 3)
        writer = _broker()
        await _write(writer, range(12))
        lpos_calls = _counting_lpos(client, monkeypatch)
        received: list[str] = []

        async def consume() -> None:
            async with aclosing(_broker().events(follow=True)) as events:
                async for event_id, _ in events:
                    received.append(event_id)

        async def caught_up(count: int) -> bool:
            return len(received) == count

        consumer = asyncio.create_task(consume())
        await _until(lambda: _subscribed(client))
        for i in range(12, 30):
            await writer.put(_event_id(i), _event(i))
            await _until(lambda i=i: caught_up(i + 1))
        await writer.put(END_ID, ("end", {"status": "success"}))
        await asyncio.wait_for(consumer, 2)

        assert received == _ids(30, end=True)
        assert len(lpos_calls) <= 1, lpos_calls

    async def test_follow_survives_a_subscription_that_fails_to_close(
        self, client: fakeredis.FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closing is cleanup: a failure there must not end the thread stream that is reading."""

        async def failing_aclose(self: PubSub) -> None:
            raise RedisError("close timed out")

        monkeypatch.setattr(PubSub, "aclose", failing_aclose)
        writer = _broker()
        events = _broker().events(follow=True)
        rest = asyncio.create_task(_collect(events))
        await _until(lambda: _subscribed(client))

        await _write(writer, range(3), end=True)

        assert await asyncio.wait_for(rest, 2) == _ids(3, end=True)
