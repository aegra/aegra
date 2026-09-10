"""Unit tests for RunBroker and BrokerManager"""

import asyncio
from typing import Self

import pytest

from aegra_api.services.broker import BrokerManager, RunBroker


class TestRunBroker:
    """Test RunBroker class"""

    @pytest.mark.asyncio
    async def test_run_broker_initialization(self):
        """Test RunBroker initialization"""
        broker = RunBroker("run-123")

        assert broker.run_id == "run-123"
        assert broker._subscribers == set()
        assert not broker.finished.is_set()

    @pytest.mark.asyncio
    async def test_put_event(self):
        """Test putting an event into broker"""
        broker = RunBroker("run-123")

        await broker.put("evt-1", {"data": "test"})

        # Event is buffered for replay and delivered to a subscriber's aiter.
        replayed = await broker.replay(None)
        assert replayed == [("evt-1", {"data": "test"})]

    @pytest.mark.asyncio
    async def test_put_end_event_marks_finished(self):
        """Test that end event marks broker as finished"""
        broker = RunBroker("run-123")

        # Put end event (format: tuple with 'end' as first element)
        await broker.put("evt-end", ("end", {}))

        # Broker should be marked as finished
        assert broker.finished.is_set()

    @pytest.mark.asyncio
    async def test_put_after_finished_warns(self):
        """Test that putting after finished logs warning"""
        broker = RunBroker("run-123")
        broker.mark_finished()

        # Should not raise, just log warning
        await broker.put("evt-1", {"data": "test"})

        # Event is dropped (broker finished) — nothing buffered.
        assert await broker.replay(None) == []

    @pytest.mark.asyncio
    async def test_mark_finished(self):
        """Test marking broker as finished"""
        broker = RunBroker("run-123")

        broker.mark_finished()

        assert broker.finished.is_set()

    @pytest.mark.asyncio
    async def test_aiter_yields_events(self):
        """Test async iteration over broker events"""
        broker = RunBroker("run-123")

        # Put some events
        await broker.put("evt-1", {"data": "first"})
        await broker.put("evt-2", {"data": "second"})
        await broker.put("evt-end", ("end", {}))

        # Collect events
        events = []
        async for event_id, payload in broker.aiter():
            events.append((event_id, payload))
            if event_id == "evt-end":
                break

        assert len(events) == 3
        assert events[0] == ("evt-1", {"data": "first"})
        assert events[1] == ("evt-2", {"data": "second"})
        assert events[2] == ("evt-end", ("end", {}))

    @pytest.mark.asyncio
    async def test_aiter_stops_on_end_event(self):
        """Test that iteration stops on end event"""
        broker = RunBroker("run-123")

        await broker.put("evt-1", {"data": "test"})
        await broker.put("evt-end", ("end", {}))

        events = []
        async for event_id, payload in broker.aiter():
            events.append((event_id, payload))

        # Should get both events including end
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_two_concurrent_aiters_each_receive_every_live_event(self):
        """Regression: the v2 SDK opens two SSE on one run (main + lifecycle watcher).

        Both must receive every event. A single shared queue would split events
        between the two consumers, so the watcher would miss the interrupt.
        """
        broker = RunBroker("run-123")

        async def drain() -> list[tuple[str, object]]:
            out: list[tuple[str, object]] = []
            async for event_id, payload in broker.aiter():
                out.append((event_id, payload))
                if event_id == "evt-end":
                    break
            return out

        a = asyncio.create_task(drain())
        b = asyncio.create_task(drain())
        await asyncio.sleep(0.05)  # let both register their subscriber queues

        await broker.put("evt-1", {"data": "first"})
        await broker.put("evt-2", {"data": "second"})
        await broker.put("evt-end", ("end", {}))

        got_a, got_b = await asyncio.gather(a, b)
        assert got_a == got_b
        assert [eid for eid, _ in got_a] == ["evt-1", "evt-2", "evt-end"]

    @pytest.mark.asyncio
    async def test_get_finished_age(self: Self) -> None:
        """Test finished age is None until marked finished and tracks completion time."""
        broker = RunBroker("run-123")
        assert broker.get_finished_age() is None

        broker.mark_finished()
        finished_age = broker.get_finished_age()
        assert finished_age is not None
        assert finished_age >= 0.0


class TestBrokerManager:
    """Test BrokerManager class"""

    @pytest.mark.asyncio
    async def test_broker_manager_initialization(self):
        """Test BrokerManager initialization"""
        manager = BrokerManager()

        assert manager._brokers == {}

    @pytest.mark.asyncio
    async def test_get_or_create_broker(self):
        """Test getting or creating a broker"""
        manager = BrokerManager()

        broker1 = manager.get_or_create_broker("run-123")
        broker2 = manager.get_or_create_broker("run-123")

        # Should return the same broker instance
        assert broker1 is broker2
        assert broker1.run_id == "run-123"

    @pytest.mark.asyncio
    async def test_get_or_create_different_runs(self):
        """Test creating brokers for different runs"""
        manager = BrokerManager()

        broker1 = manager.get_or_create_broker("run-123")
        broker2 = manager.get_or_create_broker("run-456")

        # Should be different brokers
        assert broker1 is not broker2
        assert broker1.run_id == "run-123"
        assert broker2.run_id == "run-456"

    @pytest.mark.asyncio
    async def test_get_existing_broker(self):
        """Test getting an existing broker"""
        manager = BrokerManager()

        # Create a broker
        created = manager.get_or_create_broker("run-123")

        # Get it
        retrieved = manager.get_broker("run-123")

        assert retrieved is created

    @pytest.mark.asyncio
    async def test_get_nonexistent_broker(self):
        """Test getting a nonexistent broker returns None"""
        manager = BrokerManager()

        broker = manager.get_broker("nonexistent")

        assert broker is None

    @pytest.mark.asyncio
    async def test_cleanup_broker(self):
        """Test cleanup_broker marks broker as finished"""
        manager = BrokerManager()

        # Create a broker
        broker = manager.get_or_create_broker("run-123")

        # Cleanup it (marks finished but doesn't remove)
        manager.cleanup_broker("run-123")

        # Should still exist but be marked finished
        assert manager.get_broker("run-123") is broker
        assert broker.is_finished()

    @pytest.mark.asyncio
    async def test_remove_broker(self):
        """Test removing a broker"""
        manager = BrokerManager()

        # Create a broker
        manager.get_or_create_broker("run-123")

        # Remove it
        manager.remove_broker("run-123")

        # Should no longer exist
        assert manager.get_broker("run-123") is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent_broker(self):
        """Test removing a nonexistent broker doesn't error"""
        manager = BrokerManager()

        # Should not raise
        manager.remove_broker("nonexistent")

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """Test starting and stopping broker manager"""
        manager = BrokerManager()

        # Start (creates cleanup task)
        await manager.start()

        assert manager._cleanup_task is not None
        assert not manager._cleanup_task.done()

        # Stop (cancels cleanup task)
        await manager.stop()

        assert manager._cleanup_task.cancelled() or manager._cleanup_task.done()

    def test_cleanup_constants(self: Self) -> None:
        """Test configured cleanup interval and finished TTL constants."""
        assert BrokerManager.CLEANUP_INTERVAL_SECONDS == 60.0
        assert BrokerManager.FINISHED_BROKER_TTL_SECONDS == 300.0

    @pytest.mark.asyncio
    async def test_cleanup_finished_brokers_removes_old_empty_broker(self: Self) -> None:
        """Test that brokers whose finish time exceeds TTL are purged."""
        manager = BrokerManager()
        broker = manager.get_or_create_broker("run-old")
        broker.mark_finished()
        broker._finished_at = asyncio.get_running_loop().time() - 301.0
        manager._event_counters["run-old"] = 42

        removed = manager.cleanup_finished_brokers()
        assert removed == ["run-old"]
        assert manager.get_broker("run-old") is None
        assert "run-old" not in manager._event_counters

    @pytest.mark.asyncio
    async def test_cleanup_finished_brokers_retains_young_broker(self: Self) -> None:
        """Test that finished brokers within the TTL window are kept for reconnect."""
        manager = BrokerManager()
        broker = manager.get_or_create_broker("run-recent")
        broker.mark_finished()
        broker._finished_at = asyncio.get_running_loop().time() - 100.0

        removed = manager.cleanup_finished_brokers()
        assert removed == []
        assert manager.get_broker("run-recent") is not None

    @pytest.mark.asyncio
    async def test_cleanup_finished_brokers_retains_long_running_run_finished_recently(self: Self) -> None:
        """Test that long-running runs finished recently retain their full replay TTL."""
        manager = BrokerManager()
        broker = manager.get_or_create_broker("run-long")
        # Run was created 600s ago but completed only 10s ago
        broker._created_at = asyncio.get_running_loop().time() - 600.0
        broker.mark_finished()
        broker._finished_at = asyncio.get_running_loop().time() - 10.0

        removed = manager.cleanup_finished_brokers()
        assert removed == []
        assert manager.get_broker("run-long") is not None

    @pytest.mark.asyncio
    async def test_cleanup_finished_brokers_retains_unfinished_broker(self: Self) -> None:
        """Test that active runs older than TTL are not purged."""
        manager = BrokerManager()
        broker = manager.get_or_create_broker("run-active")
        broker._created_at = asyncio.get_running_loop().time() - 400.0

        removed = manager.cleanup_finished_brokers()
        assert removed == []
        assert manager.get_broker("run-active") is not None

    @pytest.mark.asyncio
    async def test_cleanup_finished_brokers_retains_broker_with_subscribers(self: Self) -> None:
        """Test that finished brokers with queued subscriber events are not purged."""
        manager = BrokerManager()
        broker = manager.get_or_create_broker("run-busy")
        broker.mark_finished()
        broker._finished_at = asyncio.get_running_loop().time() - 400.0
        sub_queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        sub_queue.put_nowait(("evt-1", {"status": "ok"}))
        broker._subscribers.add(sub_queue)

        removed = manager.cleanup_finished_brokers()
        assert removed == []
        assert manager.get_broker("run-busy") is not None
