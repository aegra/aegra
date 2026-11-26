"""Redis-backed broker for multi-instance SSE streaming.

Uses Redis pub/sub for live event broadcast and Redis Lists for replay storage.
The replay buffer (Redis List) stores resumable events with a TTL so they
auto-expire without a cleanup loop. On reconnect, LRANGE fetches missed events.

Event sequence counters are stored as Redis INCR keys for O(1) cross-instance
access (instead of deriving from the replay buffer with O(N) LRANGE).
"""

import asyncio
import contextlib
import json
import random
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import structlog
from redis import RedisError
from redis.asyncio.client import PubSub
from redis.exceptions import ResponseError

from aegra_api.core.active_runs import active_runs, explicit_run_cancellations
from aegra_api.core.redis_manager import redis_manager
from aegra_api.core.serializers import GeneralSerializer
from aegra_api.models.enums import RunCancellationAction
from aegra_api.services.base_broker import (
    REPLAY_RETENTION_SECONDS,
    BaseBrokerManager,
    BaseRunBroker,
    ReplayGapError,
    _is_terminal_payload,
)
from aegra_api.settings import settings
from aegra_api.utils import generate_event_id

logger = structlog.getLogger(__name__)

_serializer = GeneralSerializer()

# Every write refreshes the TTL, so the buffer expires this long after the last
# event, matching the in-memory post-completion window.
_REPLAY_TTL_SECONDS = REPLAY_RETENTION_SECONDS
# Max events in the replay buffer (prevents unbounded growth)
_REPLAY_MAX_EVENTS = 10_000

# Reconnect backoff for Redis pub/sub listeners
_BACKOFF_BASE = 0.5
_BACKOFF_MAX = 30.0
_BACKOFF_FACTOR = 2.0

# Bounded retry for the write path (put). The read path (aiter) retries
# RedisError indefinitely; the write path must be bounded so a producer can't
# block forever, but it should still retry a transient blip rather than drop the
# event — a dropped event is a permanently lost SSE token.
_PUT_MAX_ATTEMPTS = 3

# Stored events read per LRANGE: what a reader holds follows this, not the run's length.
_REPLAY_PAGE_EVENTS = 16
# Recently delivered ids a reader remembers, so a retry copy of an event is not sent twice.
_DELIVERED_EVENT_IDS = 128
# Elements re-read before the place LPOS found, in case the cap trims again before the read.
_REALIGN_SLACK = 4
# A follower re-reads the buffer this often without a pub/sub wake-up (a PUBLISH can fail).
_LIVE_RECHECK_SECONDS = 5.0


def _serialize_payload(payload: Any) -> str:
    """Serialize an event payload to a JSON string for Redis transport."""
    return json.dumps(payload, default=_serializer.serialize)


def _deserialize_payload(raw: Any) -> Any:
    """Convert JSON-deserialized data back to expected Python types.

    Event payloads are tuples like ("values", {...}). JSON has no tuple type,
    so they arrive as lists. We convert top-level lists with a string first
    element back to tuples.
    """
    if isinstance(raw, list) and len(raw) >= 1 and isinstance(raw[0], str):
        return tuple(raw)
    return raw


@dataclass
class _ReplayPlace:
    """One reader's place in a run's replay buffer; each ``events`` call owns its own.

    ``anchor`` is the raw element consumed last and ``index`` the list index just past it as
    of the last read. LTRIM at the cap shifts both left, so a read finds ``anchor`` before
    trusting ``index``. Event ids are never positions: they step by two and writers interleave.
    """

    index: int = 0
    anchor: str | None = None
    delivered: deque[str] = field(default_factory=lambda: deque(maxlen=_DELIVERED_EVENT_IDS))
    ended: bool = False


def _backoff_delay(attempt: int) -> float:
    """Calculate exponential backoff delay with jitter."""
    delay = min(_BACKOFF_BASE * (_BACKOFF_FACTOR**attempt), _BACKOFF_MAX)
    jitter = random.uniform(0, delay * 0.25)  # noqa: S311  # nosec B311
    return delay + jitter


class RedisRunBroker(BaseRunBroker):
    """Broker for a single run backed by Redis pub/sub + Redis Lists.

    Producer: RPUSH to list (replay buffer) + PUBLISH to channel (live).
    Consumer: LRANGE for replay, then SUBSCRIBE for live events.
    """

    def __init__(self, run_id: str, channel: str, cache_key: str, counter_key: str) -> None:
        self.run_id = run_id
        self._channel = channel
        self._cache_key = cache_key
        self._counter_key = counter_key
        self._finished = False
        # Serializes writes for this run so concurrent put() calls (e.g. a
        # cancel/error signal racing a stream write) keep cache-then-publish
        # ordering — without it, a backoff sleep on one put could let a later
        # event's publish overtake an earlier one.
        self._write_lock = asyncio.Lock()

    async def put(self, event_id: str, payload: Any, *, resumable: bool = True) -> None:
        """Append an event to the replay buffer and publish it to live subscribers.

        Writes are retried with bounded backoff (see ``_write_with_retry``), so
        delivery is **at-least-once**: in the rare case a write times out *after*
        Redis already applied it, a retry may re-deliver the event. ``event_id``
        is stable (allocated once upstream), so consumers de-duplicate on
        ``Last-Event-ID`` during replay.
        """
        if self._finished:
            logger.warning(f"Attempted to put event {event_id} into finished broker for run {self.run_id}")
            return

        message = json.dumps(
            {
                "event_id": event_id,
                "payload": json.loads(_serialize_payload(payload)),
            }
        )

        is_end = isinstance(payload, tuple) and len(payload) >= 1 and payload[0] == "end"

        async with self._write_lock:
            # Cache and publish are retried independently so a publish failure
            # never re-runs the cache pipeline (which would double-write the
            # replay buffer and double-increment the sequence counter).
            operation = "cache"
            try:
                if resumable:
                    await self._write_with_retry(self._cache_event, message)

                operation = "publish"
                await self._write_with_retry(self._publish_event, message)

                if is_end:
                    self._finished = True
            except RedisError as e:
                logger.error(
                    f"Redis {operation} write failed for run {self.run_id} after {_PUT_MAX_ATTEMPTS} attempts: {e}"
                )
                # Even if Redis fails, mark finished for end events so aiter() can exit
                # rather than looping forever waiting for an end event that won't arrive.
                if is_end:
                    self._finished = True

    async def _cache_event(self, message: str) -> None:
        """Append the event to the replay buffer and bump the sequence counter."""
        client = redis_manager.get_client()
        pipe = client.pipeline()
        pipe.rpush(self._cache_key, message)
        pipe.ltrim(self._cache_key, -_REPLAY_MAX_EVENTS, -1)
        pipe.expire(self._cache_key, _REPLAY_TTL_SECONDS)
        pipe.incr(self._counter_key)
        pipe.expire(self._counter_key, _REPLAY_TTL_SECONDS)
        await pipe.execute()

    async def _publish_event(self, message: str) -> None:
        """Broadcast the event to live subscribers."""
        client = redis_manager.get_client()
        await client.publish(self._channel, message)

    async def _write_with_retry(self, op: Callable[[str], Awaitable[None]], message: str) -> None:
        """Run a Redis write with bounded exponential backoff.

        Mirrors the retry/backoff the read path (aiter) already applies, so a
        transient client-side blip (e.g. socket timeout) is retried instead of
        dropping the event. Bounded by ``_PUT_MAX_ATTEMPTS`` so a producer never
        blocks indefinitely; the final ``RedisError`` propagates to ``put()``.
        """
        attempt = 0
        while True:
            try:
                await op(message)
                return
            except RedisError as e:
                attempt += 1
                if attempt >= _PUT_MAX_ATTEMPTS:
                    raise
                delay = _backoff_delay(attempt)
                logger.warning(
                    f"Redis write failed for run {self.run_id}, retrying in {delay:.1f}s: {e}",
                    attempt=attempt,
                )
                await asyncio.sleep(delay)

    async def aiter(self) -> AsyncIterator[tuple[str, Any]]:
        attempt = 0
        while not self._finished:
            try:
                async for event_id, payload in self._subscribe_and_listen():
                    attempt = 0
                    yield event_id, payload
                # Clean exit from _subscribe_and_listen (end event or finished)
                break
            except RedisError as e:
                attempt += 1
                delay = _backoff_delay(attempt)
                logger.warning(
                    f"Redis pub/sub error for run {self.run_id}, retrying in {delay:.1f}s: {e}",
                    attempt=attempt,
                )
                await asyncio.sleep(delay)

    async def _subscribe_and_listen(self) -> AsyncIterator[tuple[str, Any]]:
        """Subscribe to the channel and yield events until end or disconnect."""
        client = redis_manager.get_client()
        pubsub = client.pubsub()
        await pubsub.subscribe(self._channel)

        # After subscribing, check if the run already ended (closes the race where
        # the end event was published before we subscribed on this instance).
        end_already_in_buffer = await self._check_end_in_buffer()

        last_yielded_event_id: str | None = None
        try:
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=0.5,
                )
                if message is None:
                    if self._finished or end_already_in_buffer:
                        break
                    continue

                if message["type"] != "message":
                    continue

                data = json.loads(message["data"])
                event_id: str = data["event_id"]

                # Drop an at-least-once republish (put() retries a transient
                # publish failure). event_ids are unique per event and the write
                # lock keeps a retry adjacent, so skipping a repeat of the last
                # id de-duplicates without dropping a distinct event.
                if event_id == last_yielded_event_id:
                    continue
                last_yielded_event_id = event_id

                payload = _deserialize_payload(data["payload"])

                yield event_id, payload

                if isinstance(payload, tuple) and len(payload) >= 1 and payload[0] == "end":
                    self._finished = True
                    break
        finally:
            await pubsub.unsubscribe(self._channel)
            await pubsub.aclose()

    async def _check_end_in_buffer(self) -> bool:
        """Check if an 'end' event is already in the replay buffer.

        Used after subscribing to detect runs that finished before we subscribed,
        preventing infinite loops in cross-instance consumer scenarios.
        """
        try:
            client = redis_manager.get_client()
            raw_messages = await client.lrange(self._cache_key, -1, -1)  # type: ignore[invalid-await]
            if raw_messages:
                data = json.loads(raw_messages[0])
                payload = _deserialize_payload(data["payload"])
                if isinstance(payload, tuple) and len(payload) >= 1 and payload[0] == "end":
                    self._finished = True
                    return True
        except RedisError as e:
            logger.warning(f"Failed checking replay buffer for end event for run {self.run_id}: {e}")
        return False

    async def replay(self, last_event_id: str | None) -> list[tuple[str, Any]]:
        try:
            client = redis_manager.get_client()
            raw_messages = await client.lrange(self._cache_key, 0, _REPLAY_MAX_EVENTS - 1)  # type: ignore[invalid-await]
        except RedisError as e:
            logger.error(f"Redis replay failed for run {self.run_id}: {e}")
            return []

        if not raw_messages:
            return []

        all_events: list[tuple[str, Any]] = []
        events_after: list[tuple[str, Any]] = []
        found_last = last_event_id is None
        prev_event_id: str | None = None
        for raw in raw_messages:
            data = json.loads(raw)
            event_id: str = data["event_id"]

            # Skip an adjacent duplicate: put() retries are at-least-once, so a
            # write that timed out *after* Redis applied it can RPUSH the same
            # event twice. The per-run write lock keeps the copies adjacent, and
            # event_ids are unique per event, so dropping a run of equal ids is
            # safe and removes the duplicate from replay.
            if event_id == prev_event_id:
                continue
            prev_event_id = event_id

            payload = _deserialize_payload(data["payload"])
            all_events.append((event_id, payload))

            if not found_last:
                if event_id == last_event_id:
                    found_last = True
                continue

            events_after.append((event_id, payload))

        # If last_event_id was not found in the buffer, return all events
        if not found_last:
            return all_events

        return events_after

    async def events(self, *, follow: bool) -> AsyncGenerator[tuple[str, Any], None]:
        """Read the replay buffer a page at a time and, with ``follow``, as the run appends to it.

        The list is the only source of events, so a ``resumable=False`` event is not delivered.
        Pub/sub only wakes a follower, which also re-reads every ``_LIVE_RECHECK_SECONDS``, so
        nothing written between the stored and the live events is lost, not even an event whose
        PUBLISH failed. A RedisError ends a ``follow=False`` read as ``ReplayGapError``; a
        follower backs off, resubscribes and goes on from its place.
        """
        place = _ReplayPlace()
        if not follow:
            try:
                async for event in self._stored(place, follow=False):
                    yield event
            except RedisError as e:
                raise ReplayGapError(f"Redis replay of run {self.run_id} failed: {e}") from e
            return

        pubsub: PubSub | None = None
        attempt = 0
        try:
            while True:
                # Taken before the read, so a finish that lands during it still gets one more read.
                finished = self._finished
                try:
                    async for event in self._stored(place, follow=True):
                        attempt = 0
                        yield event
                    if place.ended or finished:
                        return
                    if pubsub is None:
                        # Subscribing after the backlog keeps a long one out of Redis's pub/sub buffer.
                        pubsub = redis_manager.get_client().pubsub()
                        await pubsub.subscribe(self._channel)
                        continue
                    await pubsub.get_message(timeout=_LIVE_RECHECK_SECONDS)
                    while await pubsub.get_message(timeout=0) is not None:
                        pass
                except RedisError as e:
                    attempt += 1
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        f"Redis follow failed for run {self.run_id}, re-reading the buffer in {delay:.1f}s: {e}",
                        attempt=attempt,
                    )
                    if pubsub is not None:
                        await pubsub.aclose()
                        pubsub = None
                    await asyncio.sleep(delay)
        finally:
            if pubsub is not None:
                await pubsub.aclose()

    async def _stored(self, place: _ReplayPlace, *, follow: bool) -> AsyncIterator[tuple[str, Any]]:
        """Yield the stored events after ``place`` up to the buffer's tail, one page in memory.

        A lost place (anchor trimmed away, or the buffer expired) raises ``ReplayGapError``, or
        with ``follow`` restarts at the head: everything still stored is newer than the anchor.
        """
        client = redis_manager.get_client()
        slack = 0
        while True:
            first = max(place.index - 1 - slack, 0) if place.anchor is not None else place.index
            last = place.index + _REPLAY_PAGE_EVENTS - 1
            raws = await client.lrange(self._cache_key, first, last)  # type: ignore[invalid-await]
            skip = 0
            if place.anchor is not None:
                # LTRIM only moves the anchor left, so a match past its last index is a later retry copy.
                at = next((i for i, raw in enumerate(raws[: place.index - first]) if raw == place.anchor), None)
                if at is None:
                    try:
                        found = await client.lpos(self._cache_key, place.anchor)  # type: ignore[invalid-await]
                    except ResponseError as e:
                        logger.warning(f"Redis LPOS failed for run {self.run_id} (needs Redis >= 6.0.6): {e}")
                        found = None
                    if found is not None:
                        place.index, slack = found + 1, _REALIGN_SLACK
                        continue
                    how = "trimmed past the reader" if raws else "expired"
                    if not follow:
                        raise ReplayGapError(f"Replay of run {self.run_id} lost its place: buffer {how}")
                    logger.warning(f"Replay of run {self.run_id} lost its place (buffer {how}), resuming at its head")
                    place.index, place.anchor = 0, None
                    continue
                skip = at + 1
            slack = 0
            for index, raw in enumerate(raws[skip:], start=first + skip + 1):
                place.anchor, place.index = raw, index
                data = json.loads(raw)
                event_id: str = data["event_id"]
                if event_id in place.delivered:
                    continue
                place.delivered.append(event_id)
                payload = _deserialize_payload(data["payload"])
                if _is_terminal_payload(payload):
                    place.ended = True
                    # Parity with aiter, which marks the broker finished at its end event.
                    self._finished = self._finished or payload[0] == "end"
                yield event_id, payload
                if place.ended:
                    return
            if len(raws) < last - first + 1:
                return

    def mark_finished(self) -> None:
        self._finished = True
        logger.debug(f"Redis broker for run {self.run_id} marked as finished")

    def is_finished(self) -> bool:
        return self._finished


class RedisBrokerManager(BaseBrokerManager):
    """Manages RedisRunBroker instances with cross-instance cancel support.

    The local _brokers dict is a cache for status checks. Redis is the source
    of truth for event data. Cancel commands are broadcast via a dedicated
    pub/sub channel so any instance can cancel a run.
    """

    def __init__(self) -> None:
        self._brokers: dict[str, RedisRunBroker] = {}
        self._channel_prefix = settings.redis.REDIS_CHANNEL_PREFIX
        self._cache_prefix = f"{self._channel_prefix}cache:"
        self._counter_prefix = f"{self._channel_prefix}counter:"
        self._cancel_channel = f"{self._channel_prefix}cancel"
        self._listener_task: asyncio.Task[None] | None = None
        self._running: bool = False

    def _make_broker(self, run_id: str) -> RedisRunBroker:
        """Create a RedisRunBroker for a run_id."""
        channel = f"{self._channel_prefix}{run_id}"
        cache_key = f"{self._cache_prefix}{run_id}"
        counter_key = f"{self._counter_prefix}{run_id}"
        return RedisRunBroker(run_id, channel, cache_key, counter_key)

    def get_or_create_broker(self, run_id: str) -> RedisRunBroker:
        if run_id not in self._brokers:
            self._brokers[run_id] = self._make_broker(run_id)
            logger.debug(f"Created Redis broker for run {run_id}")
        return self._brokers[run_id]

    def get_broker(self, run_id: str) -> RedisRunBroker | None:
        """Get an existing broker from the local cache, or None."""
        return self._brokers.get(run_id)

    def cleanup_broker(self, run_id: str) -> None:
        broker = self._brokers.pop(run_id, None)
        if broker:
            broker.mark_finished()
            logger.debug(f"Cleaned up Redis broker for run {run_id}")

    def remove_broker(self, run_id: str) -> None:
        broker = self._brokers.pop(run_id, None)
        if broker:
            broker.mark_finished()
            logger.debug(f"Removed Redis broker for run {run_id}")

    async def start(self) -> None:
        """Start the cancel command listener."""
        self._running = True
        self._listener_task = asyncio.create_task(self._listen_for_cancel_commands())
        logger.info("Redis broker manager started", cancel_channel=self._cancel_channel)

    async def stop(self) -> None:
        """Stop the cancel command listener."""
        self._running = False
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task
            self._listener_task = None
        logger.debug("Redis broker manager stopped")

    async def request_cancel(
        self,
        run_id: str,
        action: RunCancellationAction = "cancel",
        *,
        emit_end_event: bool = True,
    ) -> None:
        """Broadcast a cancel command via Redis pub/sub."""
        message = json.dumps(
            {
                "run_id": run_id,
                "action": action,
                "emit_end_event": emit_end_event,
            }
        )
        try:
            client = redis_manager.get_client()
            await client.publish(self._cancel_channel, message)
            logger.info(f"Published {action} command for run {run_id}")
        except RedisError as e:
            logger.error(f"Failed to publish {action} for run {run_id}: {e}")
            # Fall back to local execution - if the task is on this instance,
            # we can still cancel it even if Redis publish fails.
            await self._execute_cancel(run_id, emit_end_event=emit_end_event)

    async def get_event_sequence(self, run_id: str) -> int:
        """Read the current event sequence counter from Redis (O(1) GET)."""
        try:
            client = redis_manager.get_client()
            value = await client.get(f"{self._counter_prefix}{run_id}")
            if value is not None:
                return int(value)
        except (RedisError, ValueError) as e:
            logger.warning(f"Failed to read event counter for run {run_id}: {e}")
        return 0

    async def allocate_event_id(self, run_id: str) -> str:
        """Atomically allocate the next event sequence number and return the event_id.

        Uses Redis INCR which is atomic — two concurrent callers will always
        get different sequence numbers. This prevents the race condition where
        get_event_sequence + 1 could return the same value to two callers.
        """
        counter_key = f"{self._counter_prefix}{run_id}"
        try:
            client = redis_manager.get_client()
            seq = await client.incr(counter_key)
            await client.expire(counter_key, _REPLAY_TTL_SECONDS)
            return generate_event_id(run_id, int(seq))
        except RedisError as e:
            logger.warning(f"Failed to allocate event_id for run {run_id}: {e}")
            # Fallback: use timestamp-based ID (unique but not sequential)
            return generate_event_id(run_id, int(time.time() * 1000))

    async def _listen_for_cancel_commands(self) -> None:
        """Background task: subscribe to cancel channel with reconnect backoff."""
        attempt = 0
        while self._running:
            try:
                await self._subscribe_and_handle_cancels()
                attempt = 0
            except asyncio.CancelledError:
                break
            except RedisError as e:
                attempt += 1
                delay = _backoff_delay(attempt)
                logger.error(
                    f"Cancel listener disconnected, retrying in {delay:.1f}s: {e}",
                    attempt=attempt,
                )
                await asyncio.sleep(delay)

    async def _subscribe_and_handle_cancels(self) -> None:
        """Subscribe to cancel channel and process commands until stopped."""
        client = redis_manager.get_client()
        pubsub = client.pubsub()
        await pubsub.subscribe(self._cancel_channel)

        try:
            while self._running:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=0.5,
                )
                if message is None:
                    continue
                if message["type"] != "message":
                    continue

                try:
                    data = json.loads(message["data"])
                    emit_end_event = data.get("emit_end_event", True)
                    if not isinstance(emit_end_event, bool):
                        raise ValueError("emit_end_event must be a boolean")
                    await self._execute_cancel(data["run_id"], emit_end_event=emit_end_event)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Invalid cancel message: {e}")
        finally:
            await pubsub.unsubscribe(self._cancel_channel)
            await pubsub.aclose()

    async def _execute_cancel(self, run_id: str, *, emit_end_event: bool = True) -> None:
        """Cancel a locally-owned task and signal the broker."""
        task = active_runs.get(run_id)
        if task is None or task.done():
            return

        logger.info(f"Cancelling run {run_id} (local task found)")
        explicit_run_cancellations.add(run_id)
        task.cancel()

        broker = self.get_or_create_broker(run_id)
        if emit_end_event and not broker.is_finished():
            event_id = await self.allocate_event_id(run_id)
            await broker.put(event_id, ("end", {"status": "interrupted"}))
            # Do NOT call cleanup_broker here - execute_run's finally block
            # owns cleanup.  Leaving the finished broker in _brokers lets
            # signal_run_cancelled see is_finished() → True and skip the
            # duplicate end event.

    async def start_cleanup_task(self) -> None:
        """No-op. Redis TTL handles replay buffer expiry."""

    async def stop_cleanup_task(self) -> None:
        """No-op."""
