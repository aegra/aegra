"""Event broker for managing run-specific event queues.

The broker handles both live event broadcast (via queue) and replay storage
(via replay buffer) for SSE reconnection support.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import structlog

from aegra_api.core.active_runs import active_runs
from aegra_api.services.base_broker import BaseBrokerManager, BaseRunBroker
from aegra_api.settings import settings
from aegra_api.utils import generate_event_id

logger = structlog.getLogger(__name__)


class RunBroker(BaseRunBroker):
    """In-memory broker with per-subscriber fan-out + replay buffer.

    Each ``aiter()`` caller gets its own queue fed by ``put`` — so multiple
    concurrent SSE consumers on one run (e.g. the v2 SDK's main event stream and
    its separate lifecycle-watcher) each receive every event, matching the Redis
    pub/sub backend. The replay buffer keeps resumable events for reconnect.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.finished = asyncio.Event()
        self._replay_buffer: list[tuple[str, Any]] = []
        self._subscribers: set[asyncio.Queue[tuple[str, Any]]] = set()
        self._created_at = asyncio.get_running_loop().time()

    async def put(self, event_id: str, payload: Any, *, resumable: bool = True) -> None:
        if self.finished.is_set():
            logger.warning(f"Attempted to put event {event_id} into finished broker for run {self.run_id}")
            return

        if resumable:
            self._replay_buffer.append((event_id, payload))

        for queue in list(self._subscribers):
            queue.put_nowait((event_id, payload))

        # Check if this is an end event
        if isinstance(payload, tuple) and len(payload) >= 1 and payload[0] == "end":
            self.mark_finished()

    async def aiter(self) -> AsyncIterator[tuple[str, Any]]:
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        # Register, then replay the buffer into this iterator: any event put
        # between a caller's separate replay() and this registration is still in
        # the buffer, so nothing is dropped. Callers dedup the overlap by event_id.
        self._subscribers.add(queue)
        backlog = list(self._replay_buffer)
        try:
            for event_id, payload in backlog:
                yield event_id, payload
                if isinstance(payload, tuple) and len(payload) >= 1 and payload[0] == "end":
                    return
            while True:
                try:
                    event_id, payload = await asyncio.wait_for(queue.get(), timeout=0.1)
                except TimeoutError:
                    if self.finished.is_set() and queue.empty():
                        break
                    continue
                yield event_id, payload
                if isinstance(payload, tuple) and len(payload) >= 1 and payload[0] == "end":
                    break
        finally:
            self._subscribers.discard(queue)

    async def replay(self, last_event_id: str | None) -> list[tuple[str, Any]]:
        if not self._replay_buffer:
            return []

        if last_event_id is None:
            return list(self._replay_buffer)

        # Find the index after last_event_id
        for i, (eid, _) in enumerate(self._replay_buffer):
            if eid == last_event_id:
                return list(self._replay_buffer[i + 1 :])

        # last_event_id not found — return all
        return list(self._replay_buffer)

    def mark_finished(self) -> None:
        self.finished.set()
        logger.debug(f"Broker for run {self.run_id} marked as finished")

    def is_finished(self) -> bool:
        return self.finished.is_set()

    def is_empty(self) -> bool:
        return all(queue.empty() for queue in self._subscribers)

    def get_age(self) -> float:
        return asyncio.get_running_loop().time() - self._created_at


class BrokerManager(BaseBrokerManager):
    """Manages multiple RunBroker instances with periodic cleanup."""

    def __init__(self) -> None:
        self._brokers: dict[str, RunBroker] = {}
        self._event_counters: dict[str, int] = {}
        self._cleanup_task: asyncio.Task[None] | None = None

    def get_or_create_broker(self, run_id: str) -> RunBroker:
        if run_id not in self._brokers:
            self._brokers[run_id] = RunBroker(run_id)
            logger.debug(f"Created new broker for run {run_id}")
        return self._brokers[run_id]

    def get_broker(self, run_id: str) -> RunBroker | None:
        return self._brokers.get(run_id)

    def cleanup_broker(self, run_id: str) -> None:
        if run_id in self._brokers:
            self._brokers[run_id].mark_finished()
            logger.debug(f"Marked broker for run {run_id} for cleanup")

    def remove_broker(self, run_id: str) -> None:
        if run_id in self._brokers:
            self._brokers[run_id].mark_finished()
            del self._brokers[run_id]
            self._event_counters.pop(run_id, None)
            logger.debug(f"Removed broker for run {run_id}")

    async def start(self) -> None:
        """Start the periodic cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_old_brokers())

    async def stop(self) -> None:
        """Stop the periodic cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

    async def request_cancel(self, run_id: str, action: str = "cancel") -> None:
        """Cancel a run locally by cancelling its asyncio task."""
        task = active_runs.get(run_id)
        if task is None or task.done():
            return

        logger.info(f"Executing {action} for run {run_id} (local task)")
        task.cancel()

        broker = self.get_or_create_broker(run_id)
        if not broker.is_finished():
            event_id = await self.allocate_event_id(run_id)
            await broker.put(event_id, ("end", {"status": "interrupted"}))
            self.cleanup_broker(run_id)

    async def allocate_event_id(self, run_id: str) -> str:
        """Allocate the next event ID using an in-memory counter."""
        seq = self._event_counters.get(run_id, 0) + 1
        self._event_counters[run_id] = seq
        return generate_event_id(run_id, seq)

    async def get_event_sequence(self, run_id: str) -> int:
        """Return the current event sequence from the in-memory counter."""
        return self._event_counters.get(run_id, 0)

    async def _cleanup_old_brokers(self) -> None:
        """Remove finished brokers older than 1 hour every 5 minutes."""
        while True:
            try:
                await asyncio.sleep(300)
                to_remove = [
                    run_id
                    for run_id, broker in self._brokers.items()
                    if broker.is_finished() and broker.is_empty() and broker.get_age() > 3600
                ]
                for run_id in to_remove:
                    self.remove_broker(run_id)
                    logger.info(f"Cleaned up old broker for run {run_id}")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in broker cleanup task")


def _create_broker_manager() -> BaseBrokerManager:
    """Select broker backend based on settings.

    Returns RedisBrokerManager when REDIS_BROKER_ENABLED=true,
    otherwise the default in-memory BrokerManager.
    """
    if settings.redis.REDIS_BROKER_ENABLED:
        # Conditional import: redis is only required when the Redis broker is enabled
        from aegra_api.services.redis_broker import RedisBrokerManager

        logger.info("Using Redis broker for SSE streaming")
        return RedisBrokerManager()
    logger.info("Using in-memory broker for SSE streaming")
    return BrokerManager()


# Global broker manager instance
broker_manager: BaseBrokerManager = _create_broker_manager()
