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
from collections.abc import AsyncIterator
from typing import Any

import structlog
from redis import RedisError

from aegra_api.core.active_runs import active_runs
from aegra_api.core.cancellation_state import cancellations
from aegra_api.core.redis_manager import redis_manager
from aegra_api.core.serializers import GeneralSerializer
from aegra_api.services.base_broker import BaseBrokerManager, BaseRunBroker
from aegra_api.settings import settings
from aegra_api.utils import generate_event_id

logger = structlog.getLogger(__name__)

_serializer = GeneralSerializer()

# TTL for the replay buffer — safety net for runs that crash without cleanup.
# cleanup_run() deletes the broker on normal completion; this TTL only matters
# if cleanup never fires (e.g. process crash, OOM kill).
_REPLAY_TTL_SECONDS = 600  # 10 minutes
# Max events in the replay buffer (prevents unbounded growth)
_REPLAY_MAX_EVENTS = 10_000

# Reconnect backoff for Redis pub/sub listeners
_BACKOFF_BASE = 0.5
_BACKOFF_MAX = 30.0
_BACKOFF_FACTOR = 2.0
# Maximum total time an SSE consumer will keep retrying when Redis is
# unreachable. After this deadline the consumer gives up so the ASGI
# connection can close instead of holding memory and a worker slot
# indefinitely. Bounded by ``BG_JOB_TIMEOUT_SECS`` so a stuck consumer
# can never outlive the underlying run.
_RECONNECT_DEADLINE_SECONDS = 600.0


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

    async def put(self, event_id: str, payload: Any, *, resumable: bool = True) -> None:
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

        try:
            client = redis_manager.get_client()

            if resumable:
                pipe = client.pipeline()
                pipe.rpush(self._cache_key, message)
                pipe.ltrim(self._cache_key, -_REPLAY_MAX_EVENTS, -1)
                pipe.expire(self._cache_key, _REPLAY_TTL_SECONDS)
                pipe.incr(self._counter_key)
                pipe.expire(self._counter_key, _REPLAY_TTL_SECONDS)
                await pipe.execute()  # type: ignore[invalid-await]

            await client.publish(self._channel, message)

            if is_end:
                self._finished = True
        except RedisError as e:
            logger.error(f"Redis publish failed for run {self.run_id}: {e}")
            # Even if Redis fails, mark finished for end events so aiter() can exit
            # rather than looping forever waiting for an end event that won't arrive.
            if is_end:
                self._finished = True

    async def aiter(self) -> AsyncIterator[tuple[str, Any]]:
        attempt = 0
        deadline = time.monotonic() + _RECONNECT_DEADLINE_SECONDS
        while not self._finished:
            try:
                async for event_id, payload in self._subscribe_and_listen():
                    attempt = 0
                    deadline = time.monotonic() + _RECONNECT_DEADLINE_SECONDS
                    yield event_id, payload
                # Clean exit from _subscribe_and_listen (end event or finished)
                break
            except RedisError as e:
                attempt += 1
                if time.monotonic() >= deadline:
                    logger.error(
                        f"Redis pub/sub unavailable for run {self.run_id} after "
                        f"{_RECONNECT_DEADLINE_SECONDS:.0f}s, giving up: {e}",
                        attempt=attempt,
                    )
                    self._finished = True
                    break
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
        for raw in raw_messages:
            data = json.loads(raw)
            event_id: str = data["event_id"]
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

    async def request_cancel(self, run_id: str, action: str = "cancel") -> None:
        """Broadcast a cancel command via Redis pub/sub."""
        message = json.dumps({"run_id": run_id, "action": action})
        try:
            client = redis_manager.get_client()
            await client.publish(self._cancel_channel, message)
            logger.info(f"Published {action} command for run {run_id}")
        except RedisError as e:
            logger.error(f"Failed to publish {action} for run {run_id}: {e}")
            # Fall back to local execution — if the task is on this instance,
            # we can still cancel it even if Redis publish fails.
            await self._execute_cancel(run_id)

    async def get_event_sequence(self, run_id: str) -> int:
        """Read the current event sequence counter from Redis (O(1) GET)."""
        try:
            client = redis_manager.get_client()
            value = await client.get(f"{self._counter_prefix}{run_id}")  # type: ignore[invalid-await]
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
            seq = await client.incr(counter_key)  # type: ignore[invalid-await]
            await client.expire(counter_key, _REPLAY_TTL_SECONDS)  # type: ignore[invalid-await]
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
                    await self._execute_cancel(data["run_id"])
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Invalid cancel message: {e}")
        finally:
            await pubsub.unsubscribe(self._cancel_channel)
            await pubsub.aclose()

    async def _execute_cancel(self, run_id: str) -> None:
        """Cancel a locally-owned task and signal the broker."""
        task = active_runs.get(run_id)
        if task is None or task.done():
            return

        # Mark before cancel so execute_run's CancelledError handler classifies
        # this as a user cancel (not a timeout). The HTTP cancel endpoint
        # already marks on the originating instance; this covers the
        # cross-instance case where the cancel arrives via Redis pub/sub.
        cancellations.mark(run_id, "user")

        # Recheck task.done() after marking: if the task finished between the
        # initial check and now, execute_run's finally has already cleared
        # the cancellation entry — leaving our mark behind would leak.
        # Unconditional ``clear`` also drops any leaked lease_loss tag.
        if task.done():
            cancellations.clear(run_id)
            return

        logger.info(f"Cancelling run {run_id} (local task found)")
        task.cancel()

        broker = self.get_or_create_broker(run_id)
        if not broker.is_finished():
            event_id = await self.allocate_event_id(run_id)
            await broker.put(event_id, ("end", {"status": "interrupted"}))
            # Do NOT call cleanup_broker here — execute_run's finally block
            # owns cleanup.  Leaving the finished broker in _brokers lets
            # signal_run_cancelled see is_finished() → True and skip the
            # duplicate end event.

    async def start_cleanup_task(self) -> None:
        """No-op. Redis TTL handles replay buffer expiry."""

    async def stop_cleanup_task(self) -> None:
        """No-op."""
