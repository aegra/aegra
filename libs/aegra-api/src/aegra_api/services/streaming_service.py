"""Streaming service for orchestrating SSE streaming."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import structlog

from aegra_api.core.sse import create_error_event
from aegra_api.models import Run
from aegra_api.services.broker import broker_manager
from aegra_api.services.event_converter import EventConverter
from aegra_api.utils import extract_event_sequence

logger = structlog.getLogger(__name__)


class StreamingService:
    """Service to handle SSE streaming orchestration.

    Events flow through the broker which handles both live broadcast
    and replay storage. No separate event store needed.
    """

    def __init__(self) -> None:
        self.event_counters: dict[str, int] = {}
        self.event_converter = EventConverter()

    def _next_event_counter(self, run_id: str, event_id: str) -> int:
        """Update and return the next event counter for a run."""
        try:
            idx = extract_event_sequence(event_id)
            current = self.event_counters.get(run_id, 0)
            if idx > current:
                self.event_counters[run_id] = idx
                return idx
        except ValueError:
            logger.warning(f"Event counter update failed for event_id: {event_id}")
        return self.event_counters.get(run_id, 0)

    async def put_to_broker(
        self,
        run_id: str,
        event_id: str,
        raw_event: Any,
    ) -> None:
        """Put an event into the run's broker for live consumers and replay storage."""
        broker = broker_manager.get_or_create_broker(run_id)
        self._next_event_counter(run_id, event_id)
        await broker.put(event_id, raw_event)

    async def signal_run_cancelled(self, run_id: str) -> None:
        """Signal that a run was cancelled.

        Uses get_or_create_broker so this works cross-instance: any instance
        can signal cancellation since the broker is backed by Redis.
        """
        broker = broker_manager.get_or_create_broker(run_id)
        if broker.is_finished():
            return

        event_id = await broker_manager.allocate_event_id(run_id)
        await broker.put(event_id, ("end", {"status": "interrupted"}))
        broker_manager.cleanup_broker(run_id)

    async def signal_run_error(self, run_id: str, error_message: str, error_type: str = "Error") -> None:
        """Signal that a run encountered an error.

        Sends a proper 'error' event to the broker followed by an 'end' event.
        Uses get_or_create_broker so this works cross-instance.
        """
        broker = broker_manager.get_or_create_broker(run_id)
        if broker.is_finished():
            return

        error_event_id = await broker_manager.allocate_event_id(run_id)
        error_payload = {"error": error_type, "message": error_message}
        await broker.put(error_event_id, ("error", error_payload))

        end_event_id = await broker_manager.allocate_event_id(run_id)
        await broker.put(end_event_id, ("end", {"status": "error"}))
        broker_manager.cleanup_broker(run_id)

    async def stream_run_execution(
        self,
        run: Run,
        last_event_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream run execution with unified producer-consumer pattern.

        Cancellation on client disconnect is handled at the transport layer
        via ``EventSourceResponse(client_close_handler_callable=...)`` — the
        generator only propagates ``CancelledError`` cleanup without side
        effects on the broker.
        """
        run_id = run.run_id
        try:
            # Replay stored events first
            last_sent_sequence = 0
            if last_event_id:
                last_sent_sequence = extract_event_sequence(last_event_id)

            async for event_id, sse_event in self._replay_stored_events(run_id, last_event_id):
                # Track the highest replayed sequence for live dedup
                replayed_seq = extract_event_sequence(event_id)
                if replayed_seq > last_sent_sequence:
                    last_sent_sequence = replayed_seq
                yield sse_event

            # Stream live events if run is still active
            async for sse_event in self._stream_live_events(run, last_sent_sequence):
                yield sse_event

        except asyncio.CancelledError:
            logger.debug(f"Stream cancelled for run {run_id}")
            raise
        except Exception as e:
            logger.error(f"Error in stream_run_execution for run {run_id}: {e}")
            yield create_error_event(str(e))

    async def _replay_stored_events(self, run_id: str, last_event_id: str | None) -> AsyncIterator[tuple[str, str]]:
        """Replay stored events from the broker's replay buffer.

        Yields (event_id, sse_event) tuples so the caller can track
        the highest replayed sequence for live deduplication.
        """
        broker = broker_manager.get_or_create_broker(run_id)
        stored_events = await broker.replay(last_event_id)

        for event_id, raw_event in stored_events:
            sse_event = await self._convert_raw_to_sse(event_id, raw_event)
            if sse_event:
                yield event_id, sse_event

    async def _stream_live_events(self, run: Run, last_sent_sequence: int) -> AsyncIterator[str]:
        """Stream live events from broker."""
        run_id = run.run_id
        broker = broker_manager.get_broker(run_id)

        # If run is in a terminal state and broker is either missing or finished,
        # there are no live events to stream. Using get_broker (not get_or_create)
        # avoids creating a blank broker that would hang forever in aiter().
        if run.status in ["success", "error", "interrupted"] and (broker is None or broker.is_finished()):
            return

        if broker is None:
            broker = broker_manager.get_or_create_broker(run_id)

        async for event_id, raw_event in broker.aiter():
            # Skip duplicates that were already replayed
            current_sequence = extract_event_sequence(event_id)
            if current_sequence <= last_sent_sequence:
                continue

            sse_event = await self._convert_raw_to_sse(event_id, raw_event)
            if sse_event:
                yield sse_event
                last_sent_sequence = current_sequence

    async def interrupt_run(self, run_id: str) -> bool:
        """Interrupt a running execution.

        Delegates to broker_manager for cross-instance support via Redis pub/sub.
        """
        try:
            await broker_manager.request_cancel(run_id, "interrupt")
            return True
        except Exception as e:
            logger.error(f"Error interrupting run {run_id}: {e}")
            return False

    async def cancel_run(self, run_id: str) -> bool:
        """Cancel a pending or running execution.

        Delegates to broker_manager for cross-instance support via Redis pub/sub.
        """
        try:
            await broker_manager.request_cancel(run_id, "cancel")
            return True
        except Exception as e:
            logger.error(f"Error cancelling run {run_id}: {e}")
            return False

    def is_run_streaming(self, run_id: str) -> bool:
        """Check if run is currently active (has a broker)."""
        broker = broker_manager.get_broker(run_id)
        return broker is not None and not broker.is_finished()

    async def cleanup_run(self, run_id: str) -> None:
        """Clean up streaming resources for a run."""
        broker_manager.cleanup_broker(run_id)

    async def _convert_raw_to_sse(self, event_id: str, raw_event: Any) -> str | None:
        """Convert a raw event from broker to SSE format."""
        return self.event_converter.convert_raw_to_sse(event_id, raw_event)


# Global streaming service instance
streaming_service = StreamingService()
