"""Abstract base classes for the broker system"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from aegra_api.models.enums import RunCancellationAction

# How long a finished run's events stay replayable, both backends. Reconnects
# happen in seconds; this is the documented window in docs/guides/streaming.mdx.
REPLAY_RETENTION_SECONDS = 600


class ReplayGapError(Exception):
    """``BaseRunBroker.events`` cannot deliver the run's next stored event.

    The buffer read failed, or the buffer no longer holds the event after the last one
    delivered (trimmed past the reader, or expired). Events already yielded are valid.
    """


def _is_terminal_payload(payload: Any) -> bool:
    return isinstance(payload, tuple) and len(payload) >= 1 and payload[0] in ("end", "error")


class BaseRunBroker(ABC):
    """Abstract base class for a run-specific event broker.

    Handles both live event broadcast and replay storage for SSE reconnection.
    """

    @abstractmethod
    async def put(self, event_id: str, payload: Any, *, resumable: bool = True) -> None:
        """Publish an event to live subscribers and optionally store for replay.

        Args:
            event_id: Unique event identifier (format: {run_id}_event_{seq}).
            payload: The event data (typically a tuple like ("values", {...})).
            resumable: If True, store the event for replay on reconnect.
        """

    @abstractmethod
    def aiter(self) -> AsyncIterator[tuple[str, Any]]:
        """Async iterator yielding (event_id, payload) pairs.

        ``events`` joins this to the stored events; prefer it.
        """
        ...

    @abstractmethod
    async def replay(self, last_event_id: str | None) -> list[tuple[str, Any]]:
        """Return stored events for replay on reconnect.

        Args:
            last_event_id: If provided, return events after this ID.
                           If None, return all stored events.

        Returns:
            List of (event_id, payload) tuples in order.
        """

    async def events(self, *, follow: bool) -> AsyncGenerator[tuple[str, Any], None]:
        """Yield the run's events in order, each once, ending after its ``end`` or ``error``.

        ``follow=False`` yields the stored events; ``follow=True`` goes on with the live ones.
        An override raises ``ReplayGapError`` when it cannot deliver the next stored event; this
        default never does.

        This default is the replay-then-aiter join: it holds the whole replay and keeps any gap
        between ``replay`` and ``aiter``. The Redis broker overrides it to read a page at a
        time and join the live events without a gap.
        """
        replayed = await self.replay(None)
        seen = {event_id for event_id, _ in replayed}
        for event_id, payload in replayed:
            yield event_id, payload
            if _is_terminal_payload(payload):
                return
        if not follow:
            return
        async for event_id, payload in self.aiter():
            if event_id in seen:
                continue
            yield event_id, payload
            if _is_terminal_payload(payload):
                return

    @abstractmethod
    def mark_finished(self) -> None:
        """Mark this broker as finished."""

    @abstractmethod
    def is_finished(self) -> bool:
        """Check if this broker is finished."""


class BaseBrokerManager(ABC):
    """Abstract base class for managing multiple RunBroker instances"""

    @abstractmethod
    def get_or_create_broker(self, run_id: str) -> BaseRunBroker:
        """Get or create a broker for a run"""

    @abstractmethod
    def get_broker(self, run_id: str) -> BaseRunBroker | None:
        """Get an existing broker or None"""

    @abstractmethod
    def cleanup_broker(self, run_id: str) -> None:
        """Clean up a broker for a run"""

    @abstractmethod
    def remove_broker(self, run_id: str) -> None:
        """Remove a broker completely"""

    @abstractmethod
    async def start(self) -> None:
        """Start background tasks (cleanup loop, cancel listener, etc.)"""

    @abstractmethod
    async def stop(self) -> None:
        """Stop background tasks and release resources"""

    @abstractmethod
    async def request_cancel(
        self,
        run_id: str,
        action: RunCancellationAction = "cancel",
        *,
        emit_end_event: bool = True,
    ) -> None:
        """Request cancellation of a run.

        In single-instance mode, cancels the local asyncio task directly.
        In multi-instance mode, broadcasts via Redis pub/sub so the owning
        instance can act on it. ``emit_end_event`` is disabled when the API
        has already reconciled the database and owns terminal signaling.
        """

    @abstractmethod
    async def get_event_sequence(self, run_id: str) -> int:
        """Return the current (highest) event sequence number for a run.

        Used by signal methods (cancel, error) to generate the next event ID
        when they need to inject events into a run's stream.
        """

    async def allocate_event_id(self, run_id: str) -> str:
        """Atomically allocate the next event sequence number and return event_id.

        Default implementation uses get_event_sequence + 1 (not atomic).
        Redis backend overrides with atomic INCR.
        """
        from aegra_api.utils import generate_event_id

        counter = await self.get_event_sequence(run_id) + 1
        return generate_event_id(run_id, counter)
