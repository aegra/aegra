"""Unit tests for streaming_service module"""

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.core.active_runs import active_runs
from aegra_api.models import Run
from aegra_api.services.broker import RunBroker, broker_manager
from aegra_api.services.redis_broker import RedisRunBroker
from aegra_api.services.streaming_service import StreamingService, parse_broker_event


def _run(run_id: str, status: str) -> Run:
    now = datetime.now(UTC)
    return Run(
        run_id=run_id,
        status=status,
        user_id="user-1",
        thread_id="thread-1",
        assistant_id="agent",
        input={"message": "hello"},
        created_at=now,
        updated_at=now,
    )


def _cached(event_id: str, payload: list[Any]) -> str:
    return json.dumps({"event_id": event_id, "payload": payload})


class _RemoteRedisBroker(RedisRunBroker):
    """Redis broker whose live tail is scripted. Replay still reads the Redis list."""

    def __init__(self, run_id: str, live: list[tuple[str, Any]] | None = None) -> None:
        super().__init__(run_id, f"chan:{run_id}", f"cache:{run_id}", f"counter:{run_id}")
        self._live = live
        self.live_entered = False

    async def aiter(self) -> AsyncIterator[tuple[str, Any]]:
        self.live_entered = True
        if self.is_finished():
            return
        if self._live is None:
            await asyncio.Event().wait()
            return
        for event in self._live:
            yield event


class _SignalingBroker(RunBroker):
    """In-memory broker that records when a consumer enters the live tail."""

    def __init__(self, run_id: str) -> None:
        super().__init__(run_id)
        self.live_started = asyncio.Event()

    async def aiter(self) -> AsyncIterator[tuple[str, Any]]:
        self.live_started.set()
        async for event in super().aiter():
            yield event


class _EmptyLocalCache:
    """Starts with no local broker, then keeps the one replay creates."""

    def __init__(self, broker: RedisRunBroker) -> None:
        self._broker = broker
        self._visible = False

    def get_broker(self, _run_id: str) -> RedisRunBroker | None:
        if not self._visible:
            return None
        return self._broker

    def get_or_create_broker(self, _run_id: str) -> RedisRunBroker:
        self._visible = True
        return self._broker


@pytest.mark.asyncio
class TestStreamingService:
    """Test StreamingService class"""

    async def test_next_event_counter(self) -> None:
        """Test event counter update logic"""
        service = StreamingService()
        run_id = "run-123"

        # Initial counter
        count = service._next_event_counter(run_id, "run-123_event_5")
        assert count == 5
        assert service.event_counters[run_id] == 5

        # Lower counter should be ignored
        count = service._next_event_counter(run_id, "run-123_event_3")
        assert count == 5
        assert service.event_counters[run_id] == 5

        # Higher counter should update
        count = service._next_event_counter(run_id, "run-123_event_10")
        assert count == 10
        assert service.event_counters[run_id] == 10

        # Malformed event id should handle gracefully
        count = service._next_event_counter(run_id, "invalid")
        assert count == 10  # Should remain unchanged

    async def test_put_to_broker(self) -> None:
        """Test putting event to broker"""
        service = StreamingService()
        run_id = "run-123"
        event_id = "run-123_event_1"
        raw_event = {"data": "test"}

        mock_broker = AsyncMock()

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_or_create_broker.return_value = mock_broker

            await service.put_to_broker(run_id, event_id, raw_event)

            mock_manager.get_or_create_broker.assert_called_with(run_id)
            mock_broker.put.assert_awaited_with(event_id, raw_event)
            assert service.event_counters[run_id] == 1

    async def test_signal_run_cancelled(self) -> None:
        """Test signalling run cancellation"""
        service = StreamingService()
        run_id = "run-123"

        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = False
        mock_broker.put = AsyncMock()

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_or_create_broker.return_value = mock_broker
            mock_manager.allocate_event_id = AsyncMock(return_value="run-123_event_1")

            await service.signal_run_cancelled(run_id)

            # Should put end event
            mock_broker.put.assert_awaited()
            args = mock_broker.put.call_args
            assert args[0][1] == ("end", {"status": "interrupted"})

            # Should cleanup broker
            mock_manager.cleanup_broker.assert_called_with(run_id)

    async def test_signal_run_cancelled_skips_when_finished(self) -> None:
        """Test that signal_run_cancelled is a no-op when broker is finished"""
        service = StreamingService()

        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = True

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_or_create_broker.return_value = mock_broker

            await service.signal_run_cancelled("run-123")

            mock_manager.cleanup_broker.assert_not_called()

    async def test_signal_run_error(self) -> None:
        """Test signalling run error sends error event then end event"""
        service = StreamingService()
        run_id = "run-123"
        error_msg = "Something went wrong"
        error_type = "ValueError"

        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = False
        mock_broker.put = AsyncMock()

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_or_create_broker.return_value = mock_broker
            mock_manager.allocate_event_id = AsyncMock(side_effect=["run-123_event_1", "run-123_event_2"])

            await service.signal_run_error(run_id, error_msg, error_type)

            # Should put error event first, then end event
            assert mock_broker.put.await_count == 2
            call_args_list = mock_broker.put.call_args_list

            # First call should be error event
            first_call = call_args_list[0]
            event_type_val, event_data = first_call[0][1]
            assert event_type_val == "error"
            assert event_data["error"] == error_type
            assert event_data["message"] == error_msg

            # Second call should be end event
            second_call = call_args_list[1]
            end_type, end_data = second_call[0][1]
            assert end_type == "end"
            assert end_data["status"] == "error"

            # Should cleanup broker
            mock_manager.cleanup_broker.assert_called_with(run_id)

    async def test_signal_run_error_skips_when_finished(self) -> None:
        """Test that signal_run_error is a no-op when broker is finished"""
        service = StreamingService()

        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = True

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_or_create_broker.return_value = mock_broker

            await service.signal_run_error("run-123", "error msg")

            mock_manager.cleanup_broker.assert_not_called()

    async def test_signal_run_error_default_type(self) -> None:
        """Test signal_run_error with default error type"""
        service = StreamingService()
        run_id = "run-123"
        error_msg = "Generic error"

        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = False
        mock_broker.put = AsyncMock()

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_or_create_broker.return_value = mock_broker
            mock_manager.allocate_event_id = AsyncMock(side_effect=["run-123_event_1", "run-123_event_2"])

            await service.signal_run_error(run_id, error_msg)

            call_args = mock_broker.put.call_args_list[0]
            event_type_val, event_data = call_args[0][1]
            assert event_type_val == "error"
            assert event_data["error"] == "Error"
            assert event_data["message"] == error_msg

    async def test_stream_run_execution(self) -> None:
        """Test overall streaming execution with replay + live"""
        service = StreamingService()
        run = Run(
            run_id="run-123",
            status="running",
            user_id="user-1",
            thread_id="thread-1",
            assistant_id="agent",
            input={"message": "hello"},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = False
        # Replay returns 2 stored events
        mock_broker.replay = AsyncMock(
            return_value=[
                ("run-123_event_1", ("values", {"a": 1})),
                ("run-123_event_2", ("values", {"a": 2})),
            ]
        )

        async def mock_aiter() -> AsyncGenerator[tuple[str, Any], None]:
            yield "run-123_event_3", ("values", {"a": 3})
            yield "run-123_event_4", ("end", {"status": "success"})

        mock_broker.aiter = mock_aiter

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_or_create_broker.return_value = mock_broker
            mock_manager.get_broker.return_value = mock_broker
            service._convert_raw_to_sse = AsyncMock(side_effect=["sse1", "sse2", "sse3", "sse4"])  # type: ignore[assignment]

            events: list[str] = []
            async for event in service.stream_run_execution(run):
                events.append(event)

            assert len(events) == 4
            assert events == ["sse1", "sse2", "sse3", "sse4"]
            mock_broker.replay.assert_awaited_with(None)

    async def test_stream_run_execution_with_last_id(self) -> None:
        """Test streaming with last_event_id resume"""
        service = StreamingService()
        run = Run(
            run_id="run-123",
            status="running",
            user_id="user-1",
            thread_id="thread-1",
            assistant_id="agent",
            input={"message": "hello"},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        last_id = "run-123_event_5"

        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = False
        mock_broker.replay = AsyncMock(return_value=[])

        async def mock_aiter() -> AsyncGenerator[tuple[str, Any], None]:
            if False:
                yield

        mock_broker.aiter = mock_aiter

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_or_create_broker.return_value = mock_broker
            mock_manager.get_broker.return_value = mock_broker

            async for _ in service.stream_run_execution(run, last_event_id=last_id):
                pass

            # Should replay from last_id
            mock_broker.replay.assert_awaited_with(last_id)

    async def test_stream_skips_already_replayed_events(self) -> None:
        """Test that live events with sequence <= last replayed are skipped"""
        service = StreamingService()
        run = Run(
            run_id="run-123",
            status="running",
            user_id="user-1",
            thread_id="thread-1",
            assistant_id="agent",
            input={"message": "hello"},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = False
        mock_broker.replay = AsyncMock(return_value=[])

        async def mock_aiter() -> AsyncGenerator[tuple[str, Any], None]:
            # Event with sequence 3 should be skipped (last_event_id was event_5)
            yield "run-123_event_3", ("values", {"should": "skip"})
            yield "run-123_event_6", ("values", {"should": "include"})
            yield "run-123_event_7", ("end", {"status": "success"})

        mock_broker.aiter = mock_aiter

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_or_create_broker.return_value = mock_broker
            mock_manager.get_broker.return_value = mock_broker
            service._convert_raw_to_sse = AsyncMock(side_effect=["sse6", "sse7"])  # type: ignore[assignment]

            events: list[str] = []
            async for event in service.stream_run_execution(run, last_event_id="run-123_event_5"):
                events.append(event)

            # Only events after sequence 5 should be included
            assert len(events) == 2
            assert events == ["sse6", "sse7"]

    async def test_stream_deduplicates_replay_and_live_overlap(self) -> None:
        """Test that events replayed on reconnect and then re-emitted live are deduplicated.

        Simulates: client saw events 1-4, reconnects with last_event_id=event_4,
        replay returns events 5-6, live stream re-emits event_6 (overlap) plus new events.
        Event_6 should NOT be duplicated in the output.
        """
        service = StreamingService()
        run = Run(
            run_id="run-123",
            status="running",
            user_id="user-1",
            thread_id="thread-1",
            assistant_id="agent",
            input={"message": "hello"},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = False
        # Replay returns events after event_4 (the last_event_id)
        mock_broker.replay = AsyncMock(
            return_value=[
                ("run-123_event_5", ("values", {"a": 5})),
                ("run-123_event_6", ("values", {"a": 6})),
            ]
        )

        async def mock_aiter() -> AsyncGenerator[tuple[str, Any], None]:
            # Live stream re-emits event_6 (overlap) plus new events
            yield "run-123_event_6", ("values", {"a": 6})
            yield "run-123_event_7", ("values", {"a": 7})
            yield "run-123_event_8", ("end", {"status": "success"})

        mock_broker.aiter = mock_aiter

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_or_create_broker.return_value = mock_broker
            mock_manager.get_broker.return_value = mock_broker
            # Only 4 convert calls expected (event_6 skipped in live)
            service._convert_raw_to_sse = AsyncMock(  # type: ignore[assignment]
                side_effect=["sse5", "sse6", "sse7", "sse8"]
            )

            events: list[str] = []
            async for event in service.stream_run_execution(run, last_event_id="run-123_event_4"):
                events.append(event)

            # Should get replay (5, 6) + live (7, 8) — event_6 NOT duplicated
            assert len(events) == 4
            assert events == ["sse5", "sse6", "sse7", "sse8"]

    async def test_interrupt_run_delegates_to_broker_manager(self) -> None:
        """Test run interruption delegates to broker_manager"""
        service = StreamingService()
        run_id = "run-123"

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_bm:
            mock_bm.request_cancel = AsyncMock()
            success = await service.interrupt_run(run_id)

            assert success is True
            mock_bm.request_cancel.assert_awaited_once_with(run_id, "interrupt", emit_end_event=True)

    async def test_cancel_run_delegates_to_broker_manager(self) -> None:
        """Test run cancellation delegates to broker_manager"""
        service = StreamingService()
        run_id = "run-123"

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_bm:
            mock_bm.request_cancel = AsyncMock()
            success = await service.cancel_run(run_id)

            assert success is True
            mock_bm.request_cancel.assert_awaited_once_with(run_id, "cancel", emit_end_event=True)

    async def test_is_run_streaming(self) -> None:
        """Test fetching if run is streaming"""
        service = StreamingService()

        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = False

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_broker.return_value = mock_broker
            assert service.is_run_streaming("run-1") is True

            mock_broker.is_finished.return_value = True
            assert service.is_run_streaming("run-1") is False

            mock_manager.get_broker.return_value = None
            assert service.is_run_streaming("run-1") is False

    @pytest.mark.asyncio
    async def test_terminal_run_with_an_expired_redis_buffer_does_not_wait(self) -> None:
        """A finished run whose Redis list is gone returns immediately instead of tailing live."""
        service = StreamingService()
        run_id = "run-redis-expired"
        broker = _RemoteRedisBroker(run_id)
        client = AsyncMock()
        client.lrange.return_value = []

        with (
            patch("aegra_api.services.streaming_service.broker_manager", _EmptyLocalCache(broker)),
            patch("aegra_api.services.redis_broker.redis_manager") as redis,
        ):
            redis.get_client.return_value = client
            async with asyncio.timeout(1):
                events = [event async for event in service.iter_broker_events(_run(run_id, "success"), None)]

        assert events == []
        assert broker.is_finished() is True
        assert broker.live_entered is False
        assert client.lrange.await_args.args[0] == f"cache:{run_id}"

    @pytest.mark.asyncio
    async def test_terminal_run_replays_redis_history_without_a_local_broker(self) -> None:
        """A finished run streamed from another instance replays the Redis list and then stops."""
        service = StreamingService()
        run_id = "run-redis-done"
        broker = _RemoteRedisBroker(run_id)
        client = AsyncMock()
        client.lrange.return_value = [
            _cached(f"{run_id}_event_1", ["values", {"a": 1}]),
            _cached(f"{run_id}_event_2", ["end", {"status": "success"}]),
        ]

        with (
            patch("aegra_api.services.streaming_service.broker_manager", _EmptyLocalCache(broker)),
            patch("aegra_api.services.redis_broker.redis_manager") as redis,
        ):
            redis.get_client.return_value = client
            async with asyncio.timeout(1):
                events = [event async for event in service.iter_broker_events(_run(run_id, "success"), None)]

        assert events == [
            (f"{run_id}_event_1", "values", {"a": 1}),
            (f"{run_id}_event_2", "end", {"status": "success"}),
        ]
        assert broker.is_finished() is True
        assert broker.live_entered is False

    @pytest.mark.asyncio
    async def test_active_run_replays_redis_history_without_a_local_broker(self) -> None:
        """An active run on another instance replays Redis history and then follows live events."""
        service = StreamingService()
        run_id = "run-redis-live"
        broker = _RemoteRedisBroker(
            run_id,
            live=[
                (f"{run_id}_event_2", ("messages", [{"id": "m"}])),
                (f"{run_id}_event_3", ("end", {"status": "success"})),
            ],
        )
        client = AsyncMock()
        client.lrange.return_value = [_cached(f"{run_id}_event_1", ["values", {"a": 1}])]

        with (
            patch("aegra_api.services.streaming_service.broker_manager", _EmptyLocalCache(broker)),
            patch("aegra_api.services.redis_broker.redis_manager") as redis,
        ):
            redis.get_client.return_value = client
            async with asyncio.timeout(1):
                events = [event async for event in service.iter_broker_events(_run(run_id, "running"), None)]

        assert [(event_id, event_type) for event_id, event_type, _payload in events] == [
            (f"{run_id}_event_1", "values"),
            (f"{run_id}_event_2", "messages"),
            (f"{run_id}_event_3", "end"),
        ]
        assert broker.is_finished() is False
        assert broker.live_entered is True

    @pytest.mark.asyncio
    async def test_terminal_replay_waits_while_the_local_producer_is_still_running(self) -> None:
        """A finished row can commit before the end event; the live tail waits for that publish."""
        service = StreamingService()
        run_id = "run-producer-open"
        broker = _SignalingBroker(run_id)
        producer = asyncio.create_task(asyncio.sleep(3600))
        active_runs[run_id] = producer

        async def collect() -> list[tuple[str, str, Any]]:
            async with asyncio.timeout(2):
                return [event async for event in service.iter_broker_events(_run(run_id, "success"), None)]

        stream: asyncio.Task[list[tuple[str, str, Any]]] | None = None
        try:
            await broker.put(f"{run_id}_event_1", ("values", {"a": 1}))
            with (
                patch("aegra_api.services.streaming_service.broker_manager.get_broker", return_value=broker),
                patch(
                    "aegra_api.services.streaming_service.broker_manager.get_or_create_broker",
                    return_value=broker,
                ),
            ):
                stream = asyncio.create_task(collect())
                async with asyncio.timeout(1):
                    await broker.live_started.wait()
                await broker.put(f"{run_id}_event_2", ("end", {"status": "success"}))
                events = await stream
        finally:
            if stream is not None and not stream.done():
                stream.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer
            active_runs.pop(run_id, None)
            broker_manager.remove_broker(run_id)

        assert [(event_id, event_type) for event_id, event_type, _payload in events] == [
            (f"{run_id}_event_1", "values"),
            (f"{run_id}_event_2", "end"),
        ]

    @pytest.mark.asyncio
    async def test_iter_broker_events_replays_then_follows_new_events(self) -> None:
        """Replay yields stored events, and the live tail waits only while the run is active."""
        service = StreamingService()
        run_id = "run-broker-coverage"
        broker = broker_manager.get_or_create_broker(run_id)
        try:
            await broker.put(f"{run_id}_event_1", ("values", {"a": 1}))
            await broker.put(f"{run_id}_event_2", ("node", "updates", {"b": 2}))
            await broker.put(f"{run_id}_event_3", (1, 2))
            await broker.put(f"{run_id}_event_4", {"raw": True})

            run = Run(
                run_id=run_id,
                status="running",
                user_id="user-1",
                thread_id="thread-1",
                assistant_id="agent",
                input={"message": "hello"},
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )

            async def publish_live() -> None:
                await asyncio.sleep(0.01)
                await broker.put(f"{run_id}_event_0", ("values", {"skipped": True}))
                await broker.put(f"{run_id}_event_5", ("messages", [{"id": "m"}]))
                await broker.put(f"{run_id}_event_7", (1, 2, 3))
                await broker.put(f"{run_id}_event_6", ("end", {"status": "success"}))

            task = asyncio.create_task(publish_live())
            async with asyncio.timeout(2):
                events = [event async for event in service.iter_broker_events(run, f"{run_id}_event_1")]
            assert await task is None
        finally:
            broker_manager.remove_broker(run_id)

        assert [(event_id, event_type) for event_id, event_type, _payload in events] == [
            (f"{run_id}_event_2", "updates"),
            (f"{run_id}_event_4", "values"),
            (f"{run_id}_event_5", "messages"),
            (f"{run_id}_event_6", "end"),
        ]
        assert parse_broker_event(("values", {"a": 1})) == ("values", {"a": 1})
        assert parse_broker_event(("node", "updates", {"b": 2})) == ("updates", {"b": 2})
        assert parse_broker_event((1, 2, 3)) is None
        assert parse_broker_event({"raw": True}) == ("values", {"raw": True})

    @pytest.mark.asyncio
    async def test_live_tail_creates_a_broker_for_an_active_run(self) -> None:
        """A run that has not published yet still gets a broker for events that arrive later."""
        service = StreamingService()
        run_id = "run-broker-created-live"
        run = Run(
            run_id=run_id,
            status="running",
            user_id="user-1",
            thread_id="thread-1",
            assistant_id="agent",
            input={"message": "hello"},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        async def publish_live() -> None:
            broker = None
            for _ in range(50):
                broker = broker_manager.get_broker(run_id)
                if broker is not None:
                    break
                await asyncio.sleep(0.01)
            assert broker is not None
            await broker.put(f"{run_id}_event_1", ("values", {"a": 1}))
            await broker.put(f"{run_id}_event_2", ("end", {"status": "success"}))

        task = asyncio.create_task(publish_live())
        try:
            async with asyncio.timeout(2):
                events = [event async for event in service.iter_broker_events(run, None)]
            assert await task is None
        finally:
            broker_manager.remove_broker(run_id)

        assert events[-1][1] == "end"

    @pytest.mark.asyncio
    async def test_cleanup_run(self) -> None:
        """Test run cleanup"""
        service = StreamingService()
        run_id = "run-123"

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            await service.cleanup_run(run_id)
            mock_manager.cleanup_broker.assert_called_with(run_id)

    async def test_cleanup_run_drops_the_event_counter(self) -> None:
        """The counter dict is global and unbounded; a finished run must leave no entry."""
        service = StreamingService()
        service.event_counters["run-123"] = 7

        with patch("aegra_api.services.streaming_service.broker_manager"):
            await service.cleanup_run("run-123")

        assert "run-123" not in service.event_counters
