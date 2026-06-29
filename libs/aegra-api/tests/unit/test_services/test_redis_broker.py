"""Unit tests for RedisRunBroker and RedisBrokerManager"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from aegra_api.services.redis_broker import (
    _PUT_MAX_ATTEMPTS,
    RedisBrokerManager,
    RedisRunBroker,
    _deserialize_payload,
    _serialize_payload,
)


class TestSerializationHelpers:
    """Test serialization/deserialization helpers"""

    def test_serialize_payload_simple_tuple(self) -> None:
        result = _serialize_payload(("values", {"key": "value"}))
        parsed = json.loads(result)
        assert parsed == ["values", {"key": "value"}]

    def test_serialize_payload_end_event(self) -> None:
        result = _serialize_payload(("end", {"status": "success"}))
        parsed = json.loads(result)
        assert parsed == ["end", {"status": "success"}]

    def test_serialize_payload_dict(self) -> None:
        result = _serialize_payload({"data": "test"})
        parsed = json.loads(result)
        assert parsed == {"data": "test"}

    def test_deserialize_payload_converts_list_to_tuple(self) -> None:
        result = _deserialize_payload(["values", {"key": "value"}])
        assert result == ("values", {"key": "value"})
        assert isinstance(result, tuple)

    def test_deserialize_payload_preserves_dict(self) -> None:
        result = _deserialize_payload({"data": "test"})
        assert result == {"data": "test"}

    def test_deserialize_payload_preserves_non_event_list(self) -> None:
        result = _deserialize_payload([1, 2, 3])
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_deserialize_payload_preserves_empty_list(self) -> None:
        result = _deserialize_payload([])
        assert result == []

    def test_deserialize_end_event(self) -> None:
        result = _deserialize_payload(["end", {"status": "success"}])
        assert result == ("end", {"status": "success"})
        assert isinstance(result, tuple)


class TestRedisRunBroker:
    """Test RedisRunBroker class"""

    def _make_broker(self, run_id: str = "run-123") -> RedisRunBroker:
        return RedisRunBroker(
            run_id,
            f"aegra:run:{run_id}",
            f"aegra:run:cache:{run_id}",
            f"aegra:run:counter:{run_id}",
        )

    @pytest.mark.asyncio
    async def test_put_publishes_and_caches(self) -> None:
        """Test that put() publishes to channel and stores in cache list via pipeline"""
        broker = self._make_broker()
        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock()
        mock_client = MagicMock()
        mock_client.publish = AsyncMock()
        mock_client.pipeline.return_value = mock_pipe

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            await broker.put("evt-1", ("values", {"msg": "hello"}))

            # Should publish to channel
            mock_client.publish.assert_called_once()
            channel, message = mock_client.publish.call_args[0]
            assert channel == "aegra:run:run-123"

            data = json.loads(message)
            assert data["event_id"] == "evt-1"
            assert data["payload"] == ["values", {"msg": "hello"}]

            # Should use pipeline for cache operations
            mock_client.pipeline.assert_called_once()
            mock_pipe.rpush.assert_called_once()
            cache_key, cached_msg = mock_pipe.rpush.call_args[0]
            assert cache_key == "aegra:run:cache:run-123"
            assert json.loads(cached_msg) == data

            # Should trim, set TTL, and increment counter in pipeline
            mock_pipe.ltrim.assert_called_once()
            mock_pipe.expire.assert_called()
            mock_pipe.incr.assert_called_once_with("aegra:run:counter:run-123")
            mock_pipe.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_put_non_resumable_skips_cache(self) -> None:
        """Test that put() with resumable=False skips the cache pipeline"""
        broker = self._make_broker()
        mock_client = MagicMock()
        mock_client.publish = AsyncMock()

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            await broker.put("evt-1", ("values", {"msg": "hello"}), resumable=False)

            mock_client.publish.assert_called_once()
            mock_client.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_put_end_event_marks_finished(self) -> None:
        broker = self._make_broker()
        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock()
        mock_client = MagicMock()
        mock_client.publish = AsyncMock()
        mock_client.pipeline.return_value = mock_pipe

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            await broker.put("evt-end", ("end", {"status": "success"}))

            assert broker.is_finished()

    @pytest.mark.asyncio
    async def test_put_after_finished_skips(self) -> None:
        broker = self._make_broker()
        broker.mark_finished()
        mock_client = MagicMock()
        mock_client.publish = AsyncMock()

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            await broker.put("evt-1", ("values", {"data": "test"}))

            mock_client.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_put_handles_redis_failure_gracefully(self) -> None:
        broker = self._make_broker()
        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock(side_effect=RedisConnectionError("Redis down"))
        mock_client = MagicMock()
        mock_client.publish = AsyncMock()
        mock_client.pipeline.return_value = mock_pipe

        with (
            patch("aegra_api.services.redis_broker.redis_manager") as mock_rm,
            patch("aegra_api.services.redis_broker.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_rm.get_client.return_value = mock_client

            # Should not raise even after retries are exhausted
            await broker.put("evt-1", ("values", {"data": "test"}))

        # Retried up to the bounded limit before giving up
        assert mock_pipe.execute.await_count == _PUT_MAX_ATTEMPTS

    @pytest.mark.asyncio
    async def test_put_retries_cache_write_then_succeeds(self) -> None:
        """A transient RedisError on the cache pipeline is retried, not dropped.

        Mirrors the read path (aiter), which already retries on RedisError.
        """
        broker = self._make_broker()
        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock(side_effect=[RedisConnectionError("blip"), None])
        mock_client = MagicMock()
        mock_client.publish = AsyncMock()
        mock_client.pipeline.return_value = mock_pipe

        with (
            patch("aegra_api.services.redis_broker.redis_manager") as mock_rm,
            patch("aegra_api.services.redis_broker.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            mock_rm.get_client.return_value = mock_client

            await broker.put("evt-1", ("values", {"msg": "hi"}))

        assert mock_pipe.execute.await_count == 2  # one retry
        mock_client.publish.assert_awaited_once()  # publish runs after cache succeeds
        mock_sleep.assert_awaited()  # backed off between attempts

    @pytest.mark.asyncio
    async def test_put_retries_publish_without_rerunning_cache(self) -> None:
        """A publish failure retries publish only — the cache pipeline is not
        re-run, so the replay buffer is never double-written (no duplicate RPUSH)."""
        broker = self._make_broker()
        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock()
        mock_client = MagicMock()
        mock_client.publish = AsyncMock(side_effect=[RedisConnectionError("blip"), None])
        mock_client.pipeline.return_value = mock_pipe

        with (
            patch("aegra_api.services.redis_broker.redis_manager") as mock_rm,
            patch("aegra_api.services.redis_broker.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_rm.get_client.return_value = mock_client

            await broker.put("evt-1", ("values", {"msg": "hi"}))

        assert mock_client.publish.await_count == 2  # publish retried
        mock_pipe.execute.assert_awaited_once()  # cache written exactly once

    @pytest.mark.asyncio
    async def test_put_end_event_marks_finished_even_when_dropped(self) -> None:
        """If an 'end' event can't be delivered after retries, the broker is still
        marked finished so aiter() can exit instead of looping forever."""
        broker = self._make_broker()
        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock(side_effect=RedisConnectionError("down"))
        mock_client = MagicMock()
        mock_client.publish = AsyncMock(side_effect=RedisConnectionError("down"))
        mock_client.pipeline.return_value = mock_pipe

        with (
            patch("aegra_api.services.redis_broker.redis_manager") as mock_rm,
            patch("aegra_api.services.redis_broker.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_rm.get_client.return_value = mock_client

            await broker.put("evt-end", ("end", {"status": "success"}))

        assert broker.is_finished()

    @pytest.mark.asyncio
    async def test_put_serializes_concurrent_calls(self) -> None:
        """The per-broker write lock keeps concurrent put() calls from
        interleaving: a second put cannot start writing while the first is
        mid-flight, preserving cache-then-publish event ordering."""
        broker = self._make_broker()
        order: list[str] = []
        gate = asyncio.Event()

        async def slow_first_cache(message: str) -> None:
            order.append("A-cache-start")
            await gate.wait()  # hold the write lock until released
            order.append("A-cache-end")

        async def publish_noop(message: str) -> None:
            order.append("publish")

        async def second_cache(message: str) -> None:
            order.append("B-cache")

        broker._cache_event = slow_first_cache  # type: ignore[method-assign]
        broker._publish_event = publish_noop  # type: ignore[method-assign]
        task_a = asyncio.create_task(broker.put("evt-A", ("values", {})))
        await asyncio.sleep(0)  # let A acquire the lock and enter its cache write
        assert order == ["A-cache-start"]

        # B is queued while A holds the lock — it must not write yet.
        broker._cache_event = second_cache  # type: ignore[method-assign]
        task_b = asyncio.create_task(broker.put("evt-B", ("values", {})))
        await asyncio.sleep(0)
        assert "B-cache" not in order

        gate.set()  # release A
        await asyncio.gather(task_a, task_b)

        # A fully completes (cache then publish) before B begins.
        assert order == ["A-cache-start", "A-cache-end", "publish", "B-cache", "publish"]

    @pytest.mark.asyncio
    async def test_replay_dedups_adjacent_duplicate_event(self) -> None:
        """An at-least-once retry can RPUSH the same event twice; replay() drops
        the adjacent duplicate so a resuming client doesn't receive it twice."""
        broker = self._make_broker()
        mock_client = AsyncMock()
        mock_client.lrange.return_value = [
            json.dumps({"event_id": "evt-1", "payload": ["values", {"a": 1}]}),
            json.dumps({"event_id": "evt-2", "payload": ["values", {"a": 2}]}),
            json.dumps({"event_id": "evt-2", "payload": ["values", {"a": 2}]}),  # duplicate
            json.dumps({"event_id": "evt-3", "payload": ["end", {}]}),
        ]

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            events = await broker.replay(None)

        assert [event_id for event_id, _ in events] == ["evt-1", "evt-2", "evt-3"]

    @pytest.mark.asyncio
    async def test_aiter_dedups_adjacent_duplicate_live_event(self) -> None:
        """A republished (retried) live event with the same event_id is yielded once."""
        broker = self._make_broker()

        messages = [
            {"type": "message", "data": json.dumps({"event_id": "evt-1", "payload": ["values", {"msg": "hi"}]})},
            {"type": "message", "data": json.dumps({"event_id": "evt-1", "payload": ["values", {"msg": "hi"}]})},  # dup
            {"type": "message", "data": json.dumps({"event_id": "evt-2", "payload": ["end", {"status": "success"}]})},
        ]
        call_count = 0

        async def mock_get_message(ignore_subscribe_messages: bool = True, timeout: float = 0.5) -> dict | None:
            nonlocal call_count
            if call_count < len(messages):
                msg = messages[call_count]
                call_count += 1
                return msg
            return None

        mock_pubsub = AsyncMock()
        mock_pubsub.get_message = mock_get_message
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.aclose = AsyncMock()

        mock_client = MagicMock()
        mock_client.pubsub.return_value = mock_pubsub
        mock_client.lrange = AsyncMock(return_value=[])

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            events: list[tuple[str, object]] = []
            async for event_id, payload in broker.aiter():
                events.append((event_id, payload))

        assert [event_id for event_id, _ in events] == ["evt-1", "evt-2"]

    @pytest.mark.asyncio
    async def test_aiter_yields_events(self) -> None:
        broker = self._make_broker()

        messages = [
            {"type": "message", "data": json.dumps({"event_id": "evt-1", "payload": ["values", {"msg": "hi"}]})},
            {"type": "message", "data": json.dumps({"event_id": "evt-2", "payload": ["end", {"status": "success"}]})},
        ]
        call_count = 0

        async def mock_get_message(ignore_subscribe_messages: bool = True, timeout: float = 0.5) -> dict | None:
            nonlocal call_count
            if call_count < len(messages):
                msg = messages[call_count]
                call_count += 1
                return msg
            return None

        mock_pubsub = AsyncMock()
        mock_pubsub.get_message = mock_get_message
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.aclose = AsyncMock()

        mock_client = MagicMock()
        mock_client.pubsub.return_value = mock_pubsub
        # _check_end_in_buffer calls lrange — return empty (no end in buffer yet)
        mock_client.lrange = AsyncMock(return_value=[])

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            events: list[tuple[str, object]] = []
            async for event_id, payload in broker.aiter():
                events.append((event_id, payload))

        assert len(events) == 2
        assert events[0] == ("evt-1", ("values", {"msg": "hi"}))
        assert events[1] == ("evt-2", ("end", {"status": "success"}))

    @pytest.mark.asyncio
    async def test_aiter_skips_non_message_types(self) -> None:
        broker = self._make_broker()

        messages = [
            {"type": "subscribe", "data": None},
            {"type": "message", "data": json.dumps({"event_id": "evt-1", "payload": ["end", {}]})},
        ]
        call_count = 0

        async def mock_get_message(ignore_subscribe_messages: bool = True, timeout: float = 0.5) -> dict | None:
            nonlocal call_count
            if call_count < len(messages):
                msg = messages[call_count]
                call_count += 1
                return msg
            return None

        mock_pubsub = AsyncMock()
        mock_pubsub.get_message = mock_get_message
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.aclose = AsyncMock()

        mock_client = MagicMock()
        mock_client.pubsub.return_value = mock_pubsub
        mock_client.lrange = AsyncMock(return_value=[])

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            events: list[tuple[str, object]] = []
            async for event_id, payload in broker.aiter():
                events.append((event_id, payload))

        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_replay_returns_all_when_no_last_event_id(self) -> None:
        """Test replay returns all cached events when last_event_id is None"""
        broker = self._make_broker()
        mock_client = AsyncMock()
        mock_client.lrange.return_value = [
            json.dumps({"event_id": "evt-1", "payload": ["values", {"a": 1}]}),
            json.dumps({"event_id": "evt-2", "payload": ["values", {"a": 2}]}),
        ]

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            events = await broker.replay(None)

        assert len(events) == 2
        assert events[0] == ("evt-1", ("values", {"a": 1}))
        assert events[1] == ("evt-2", ("values", {"a": 2}))

    @pytest.mark.asyncio
    async def test_replay_returns_events_after_last_event_id(self) -> None:
        """Test replay returns only events after last_event_id"""
        broker = self._make_broker()
        mock_client = AsyncMock()
        mock_client.lrange.return_value = [
            json.dumps({"event_id": "evt-1", "payload": ["values", {"a": 1}]}),
            json.dumps({"event_id": "evt-2", "payload": ["values", {"a": 2}]}),
            json.dumps({"event_id": "evt-3", "payload": ["end", {}]}),
        ]

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            events = await broker.replay("evt-1")

        assert len(events) == 2
        assert events[0] == ("evt-2", ("values", {"a": 2}))
        assert events[1] == ("evt-3", ("end", {}))

    @pytest.mark.asyncio
    async def test_replay_returns_all_when_last_event_id_not_found(self) -> None:
        """Test replay returns all events when last_event_id is not in cache"""
        broker = self._make_broker()
        mock_client = AsyncMock()
        mock_client.lrange.return_value = [
            json.dumps({"event_id": "evt-1", "payload": ["values", {"a": 1}]}),
        ]

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            events = await broker.replay("evt-999")

        assert len(events) == 1
        assert events[0] == ("evt-1", ("values", {"a": 1}))

    @pytest.mark.asyncio
    async def test_replay_returns_empty_when_no_cache(self) -> None:
        broker = self._make_broker()
        mock_client = AsyncMock()
        mock_client.lrange.return_value = []

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            events = await broker.replay(None)

        assert events == []

    @pytest.mark.asyncio
    async def test_replay_handles_redis_failure(self) -> None:
        broker = self._make_broker()
        mock_client = AsyncMock()
        mock_client.lrange.side_effect = RedisConnectionError("Redis down")

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            events = await broker.replay(None)

        assert events == []

    @pytest.mark.asyncio
    async def test_check_end_in_buffer_returns_true_when_end_present(self) -> None:
        """Test that _check_end_in_buffer detects end events in replay buffer"""
        broker = self._make_broker()
        mock_client = AsyncMock()
        mock_client.lrange.return_value = [
            json.dumps({"event_id": "evt-end", "payload": ["end", {"status": "success"}]})
        ]

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client
            result = await broker._check_end_in_buffer()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_end_in_buffer_returns_false_when_no_end(self) -> None:
        """Test that _check_end_in_buffer returns False when no end event"""
        broker = self._make_broker()
        mock_client = AsyncMock()
        mock_client.lrange.return_value = [json.dumps({"event_id": "evt-1", "payload": ["values", {"a": 1}]})]

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client
            result = await broker._check_end_in_buffer()

        assert result is False

    @pytest.mark.asyncio
    async def test_aiter_exits_when_end_already_in_buffer(self) -> None:
        """Test that aiter() exits immediately when end event is already in replay buffer"""
        broker = self._make_broker()

        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.aclose = AsyncMock()
        # Return None (no live messages) — should exit because end is in buffer
        mock_pubsub.get_message = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.pubsub.return_value = mock_pubsub
        # _check_end_in_buffer sees end event in last position
        mock_client.lrange = AsyncMock(
            return_value=[json.dumps({"event_id": "evt-end", "payload": ["end", {"status": "success"}]})]
        )

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            events: list[tuple[str, object]] = []
            async for event_id, payload in broker.aiter():
                events.append((event_id, payload))

        # Should exit with no live events since end was already in buffer
        assert len(events) == 0

    def test_mark_finished(self) -> None:
        broker = self._make_broker()
        broker.mark_finished()
        assert broker.is_finished()

    def test_is_finished_default_false(self) -> None:
        broker = self._make_broker()
        assert not broker.is_finished()


class TestRedisBrokerManager:
    """Test RedisBrokerManager class"""

    def _make_manager(self) -> RedisBrokerManager:
        return RedisBrokerManager()

    def test_get_or_create_broker_creates_new(self) -> None:
        manager = self._make_manager()
        broker = manager.get_or_create_broker("run-123")

        assert broker is not None
        assert broker.run_id == "run-123"

    def test_get_or_create_broker_returns_existing(self) -> None:
        manager = self._make_manager()
        broker1 = manager.get_or_create_broker("run-123")
        broker2 = manager.get_or_create_broker("run-123")

        assert broker1 is broker2

    def test_get_or_create_different_runs(self) -> None:
        manager = self._make_manager()
        broker1 = manager.get_or_create_broker("run-123")
        broker2 = manager.get_or_create_broker("run-456")

        assert broker1 is not broker2

    def test_get_existing_broker(self) -> None:
        manager = self._make_manager()
        created = manager.get_or_create_broker("run-123")
        retrieved = manager.get_broker("run-123")

        assert retrieved is created

    def test_get_nonexistent_broker_returns_none(self) -> None:
        manager = self._make_manager()
        broker = manager.get_broker("nonexistent")

        assert broker is None

    def test_cleanup_broker(self) -> None:
        manager = self._make_manager()
        broker = manager.get_or_create_broker("run-123")
        manager.cleanup_broker("run-123")

        assert broker.is_finished()
        assert "run-123" not in manager._brokers

    def test_remove_broker(self) -> None:
        manager = self._make_manager()
        manager.get_or_create_broker("run-123")
        manager.remove_broker("run-123")

        assert "run-123" not in manager._brokers

    def test_remove_nonexistent_broker(self) -> None:
        manager = self._make_manager()
        manager.remove_broker("nonexistent")

    def test_broker_has_correct_keys(self) -> None:
        """Test that created brokers have correct cache and counter keys"""
        manager = self._make_manager()
        broker = manager.get_or_create_broker("run-123")

        assert broker._cache_key == "aegra:run:cache:run-123"
        assert broker._channel == "aegra:run:run-123"
        assert broker._counter_key == "aegra:run:counter:run-123"

    @pytest.mark.asyncio
    async def test_get_event_sequence_reads_from_redis(self) -> None:
        """Test that get_event_sequence reads the INCR counter from Redis"""
        manager = self._make_manager()
        mock_client = AsyncMock()
        mock_client.get.return_value = "42"

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            result = await manager.get_event_sequence("run-123")

        assert result == 42
        mock_client.get.assert_awaited_once_with("aegra:run:counter:run-123")

    @pytest.mark.asyncio
    async def test_get_event_sequence_returns_zero_when_no_counter(self) -> None:
        """Test that get_event_sequence returns 0 when no counter exists"""
        manager = self._make_manager()
        mock_client = AsyncMock()
        mock_client.get.return_value = None

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            result = await manager.get_event_sequence("run-123")

        assert result == 0

    @pytest.mark.asyncio
    async def test_get_event_sequence_handles_redis_error(self) -> None:
        """Test that get_event_sequence returns 0 on Redis error"""
        manager = self._make_manager()
        mock_client = AsyncMock()
        mock_client.get.side_effect = RedisConnectionError("Redis down")

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            result = await manager.get_event_sequence("run-123")

        assert result == 0

    @pytest.mark.asyncio
    async def test_request_cancel_publishes_to_redis(self) -> None:
        """Test that request_cancel publishes cancel command to Redis"""
        manager = self._make_manager()
        mock_client = MagicMock()
        mock_client.publish = AsyncMock()

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            await manager.request_cancel("run-123", "cancel")

            mock_client.publish.assert_awaited_once()
            channel, message = mock_client.publish.call_args[0]
            assert channel == "aegra:run:cancel"
            payload = json.loads(message)
            assert payload == {"run_id": "run-123", "action": "cancel"}

    @pytest.mark.asyncio
    async def test_request_cancel_falls_back_on_redis_error(self) -> None:
        """Test that request_cancel falls back to local execution on Redis error"""
        manager = self._make_manager()
        mock_client = MagicMock()
        mock_client.publish = AsyncMock(side_effect=RedisConnectionError("Redis down"))

        with (
            patch("aegra_api.services.redis_broker.redis_manager") as mock_rm,
            patch.object(manager, "_execute_cancel", new_callable=AsyncMock) as mock_exec,
        ):
            mock_rm.get_client.return_value = mock_client

            await manager.request_cancel("run-123", "cancel")

            mock_exec.assert_awaited_once_with("run-123")

    @pytest.mark.asyncio
    async def test_execute_cancel_cancels_local_task(self) -> None:
        """Test that _execute_cancel cancels a locally-owned task"""
        manager = self._make_manager()
        mock_task = MagicMock()
        mock_task.done.return_value = False

        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = False
        mock_broker.put = AsyncMock()

        with (
            patch.dict("aegra_api.core.active_runs.active_runs", {"run-123": mock_task}, clear=True),
            patch.object(manager, "get_or_create_broker", return_value=mock_broker),
            patch.object(manager, "allocate_event_id", new_callable=AsyncMock, return_value="run-123_event_6"),
        ):
            await manager._execute_cancel("run-123")

            mock_task.cancel.assert_called_once()
            mock_broker.put.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_cancel_ignores_missing_task(self) -> None:
        """Test that _execute_cancel does nothing when task not found locally"""
        manager = self._make_manager()

        with patch.dict("aegra_api.core.active_runs.active_runs", {}, clear=True):
            await manager._execute_cancel("run-123")
            # Should not raise

    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        """Test start creates listener task and stop cancels it"""
        manager = self._make_manager()

        with patch.object(manager, "_listen_for_cancel_commands", new_callable=AsyncMock):
            await manager.start()
            assert manager._running is True
            assert manager._listener_task is not None

            await manager.stop()
            assert manager._running is False
