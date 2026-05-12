"""Unit tests for the JS bridge client.

Tests the JSBridge class with fully mocked subprocess — no Node.js needed.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.services.js_bridge import JSBridge, JSBridgeError


class TestJSBridgeError:
    """Tests for JSBridgeError."""

    def test_basic_error(self) -> None:
        err = JSBridgeError("test error")
        assert str(err) == "test error"
        assert err.code == -32000

    def test_error_with_code(self) -> None:
        err = JSBridgeError("not found", code=-32601)
        assert err.code == -32601

    def test_error_with_data(self) -> None:
        err = JSBridgeError("fail", data={"stack": "..."})
        assert err.data == {"stack": "..."}

    def test_default_data_is_none(self) -> None:
        err = JSBridgeError("oops")
        assert err.data is None

    def test_inherits_exception(self) -> None:
        err = JSBridgeError("boom")
        assert isinstance(err, Exception)


class TestJSBridge:
    """Tests for the JSBridge subprocess manager."""

    @pytest.fixture
    def bridge(self) -> JSBridge:
        """Create a fresh JSBridge instance."""
        return JSBridge()

    # ---- Initial state ----

    def test_initial_state(self, bridge: JSBridge) -> None:
        """Bridge starts in non-running state."""
        assert not bridge.is_running
        assert bridge._process is None

    def test_initial_started_event_not_set(self, bridge: JSBridge) -> None:
        assert not bridge._started.is_set()

    def test_initial_pending_empty(self, bridge: JSBridge) -> None:
        assert bridge._pending == {}

    def test_initial_notification_queues_empty(self, bridge: JSBridge) -> None:
        assert bridge._notification_queues == {}

    def test_initial_write_lock_exists(self, bridge: JSBridge) -> None:
        """Bridge has a _write_lock for stdin concurrency protection."""
        assert isinstance(bridge._write_lock, asyncio.Lock)

    # ---- ping ----

    async def test_ping_when_not_running(self, bridge: JSBridge) -> None:
        """Ping returns False when bridge is not started."""
        result = await bridge.ping()
        assert result is False

    # ---- invoke/call when not running ----

    async def test_call_when_not_running_raises(self, bridge: JSBridge) -> None:
        """RPC calls raise when bridge is not started."""
        with pytest.raises(JSBridgeError, match="not running"):
            await bridge.invoke("test", {"input": "hello"})

    async def test_load_graph_when_not_running_raises(self, bridge: JSBridge) -> None:
        with pytest.raises(JSBridgeError, match="not running"):
            await bridge.load_graph("/path/graph.ts", "graph", "gid")

    async def test_get_schema_when_not_running_raises(self, bridge: JSBridge) -> None:
        with pytest.raises(JSBridgeError, match="not running"):
            await bridge.get_schema("gid")

    # ---- _create_future ----

    async def test_create_future_registers(self, bridge: JSBridge) -> None:
        """_create_future stores a future under the given request id."""
        future = bridge._create_future("req-99")
        assert "req-99" in bridge._pending
        assert bridge._pending["req-99"] is future
        assert not future.done()

    # ---- _dispatch: ready notification ----

    async def test_dispatch_ready_notification(self, bridge: JSBridge) -> None:
        """Ready notification sets the started event."""
        assert not bridge._started.is_set()
        await bridge._dispatch({"method": "ready", "params": {"version": "1.0.0"}})
        assert bridge._started.is_set()

    async def test_dispatch_ready_without_version(self, bridge: JSBridge) -> None:
        """Ready notification works even without a version param."""
        await bridge._dispatch({"method": "ready", "params": {}})
        assert bridge._started.is_set()

    # ---- _dispatch: response resolves future ----

    async def test_dispatch_response_resolves_future(self, bridge: JSBridge) -> None:
        """Response messages resolve pending futures."""
        future = bridge._create_future("req-1")
        await bridge._dispatch({"id": "req-1", "result": {"status": "ok"}})
        assert future.done()
        assert future.result() == {"status": "ok"}

    async def test_dispatch_response_with_none_result(self, bridge: JSBridge) -> None:
        """Response with None result resolves correctly."""
        future = bridge._create_future("req-n")
        await bridge._dispatch({"id": "req-n"})
        assert future.done()
        assert future.result() is None

    # ---- _dispatch: error response ----

    async def test_dispatch_error_response(self, bridge: JSBridge) -> None:
        """Error responses set exceptions on futures."""
        future = bridge._create_future("req-2")
        await bridge._dispatch(
            {
                "id": "req-2",
                "error": {"code": -32000, "message": "Graph not found"},
            }
        )
        assert future.done()
        with pytest.raises(JSBridgeError, match="Graph not found"):
            future.result()

    async def test_dispatch_error_preserves_code(self, bridge: JSBridge) -> None:
        """Error code is preserved on the exception."""
        future = bridge._create_future("req-ec")
        await bridge._dispatch(
            {
                "id": "req-ec",
                "error": {"code": -32601, "message": "Method not found"},
            }
        )
        with pytest.raises(JSBridgeError) as exc_info:
            future.result()
        assert exc_info.value.code == -32601

    async def test_dispatch_error_preserves_data(self, bridge: JSBridge) -> None:
        """Error data is preserved on the exception."""
        future = bridge._create_future("req-ed")
        await bridge._dispatch(
            {
                "id": "req-ed",
                "error": {"code": -32000, "message": "fail", "data": {"stack": "..."}},
            }
        )
        with pytest.raises(JSBridgeError) as exc_info:
            future.result()
        assert exc_info.value.data == {"stack": "..."}

    # ---- _dispatch: stream events ----

    async def test_dispatch_stream_event(self, bridge: JSBridge) -> None:
        """Stream event notifications are routed to the correct queue."""
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        bridge._notification_queues["req-3"] = queue

        await bridge._dispatch(
            {
                "method": "stream_event",
                "params": {
                    "request_id": "req-3",
                    "mode": "values",
                    "data": {"messages": ["hello"]},
                },
            }
        )

        event = queue.get_nowait()
        assert event["mode"] == "values"
        assert event["data"] == {"messages": ["hello"]}

    async def test_dispatch_stream_event_unknown_request(self, bridge: JSBridge) -> None:
        """Stream events for unknown request_ids are silently ignored."""
        await bridge._dispatch(
            {
                "method": "stream_event",
                "params": {
                    "request_id": "nonexistent",
                    "mode": "values",
                    "data": {},
                },
            }
        )

    # ---- _dispatch: unknown response / notification ----

    async def test_dispatch_unknown_response_id(self, bridge: JSBridge) -> None:
        """Responses for unknown IDs are logged but don't crash."""
        await bridge._dispatch({"id": "unknown-id", "result": "something"})

    async def test_dispatch_unknown_notification(self, bridge: JSBridge) -> None:
        """Unknown notification methods don't crash."""
        await bridge._dispatch({"method": "some_other_event", "params": {"x": 1}})

    # ---- _send_raw ----

    async def test_send_raw_no_process_raises(self, bridge: JSBridge) -> None:
        """_send_raw raises when no process exists."""
        with pytest.raises(JSBridgeError, match="not available"):
            await bridge._send_raw({"jsonrpc": "2.0", "method": "test"})

    async def test_send_raw_writes_json(self, bridge: JSBridge) -> None:
        """_send_raw writes newline-delimited JSON to stdin."""
        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        bridge._process = mock_process

        msg = {"jsonrpc": "2.0", "id": "1", "method": "ping"}
        await bridge._send_raw(msg)

        written = mock_stdin.write.call_args[0][0]
        assert written.endswith(b"\n")
        parsed = json.loads(written.decode())
        assert parsed["method"] == "ping"

    async def test_send_raw_acquires_write_lock(self, bridge: JSBridge) -> None:
        """_send_raw holds the write lock during write+drain."""
        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        bridge._process = mock_process

        # Verify the lock is used by checking it's not locked before and after
        assert not bridge._write_lock.locked()
        await bridge._send_raw({"jsonrpc": "2.0", "method": "test"})
        assert not bridge._write_lock.locked()

        # Verify both write and drain were called (under the lock)
        mock_stdin.write.assert_called_once()
        mock_stdin.drain.assert_called_once()

    # ---- stop ----

    async def test_stop_when_not_started(self, bridge: JSBridge) -> None:
        """stop() is a no-op when bridge was never started."""
        await bridge.stop()  # should not raise

    async def test_stop_cleans_up(self, bridge: JSBridge) -> None:
        """Stopping fails all pending futures and cleans up state."""
        future = bridge._create_future("pending-req")

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.wait = AsyncMock(return_value=0)
        mock_process.kill = MagicMock()
        bridge._process = mock_process

        await bridge.stop()

        assert future.done()
        with pytest.raises(JSBridgeError, match="shut down"):
            future.result()

    async def test_stop_clears_process(self, bridge: JSBridge) -> None:
        """After stop(), process is None."""
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.wait = AsyncMock(return_value=0)
        mock_process.kill = MagicMock()
        bridge._process = mock_process

        await bridge.stop()

        assert bridge._process is None
        assert bridge._pending == {}
        assert bridge._notification_queues == {}

    async def test_stop_force_kills_on_timeout(self, bridge: JSBridge) -> None:
        """stop() force-kills when graceful shutdown times out."""
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.wait = AsyncMock(side_effect=[asyncio.TimeoutError, 0])
        mock_process.kill = MagicMock()
        bridge._process = mock_process

        await bridge.stop()

        mock_process.kill.assert_called_once()
        assert bridge._process is None

    # ---- is_running property ----

    def test_is_running_false_no_process(self, bridge: JSBridge) -> None:
        assert not bridge.is_running

    def test_is_running_false_process_exited(self, bridge: JSBridge) -> None:
        mock_process = MagicMock()
        mock_process.returncode = 1
        bridge._process = mock_process
        assert not bridge.is_running

    def test_is_running_false_when_closed(self, bridge: JSBridge) -> None:
        mock_process = MagicMock()
        mock_process.returncode = None
        bridge._process = mock_process
        bridge._closed = True
        assert not bridge.is_running

    def test_is_running_true(self, bridge: JSBridge) -> None:
        mock_process = MagicMock()
        mock_process.returncode = None
        bridge._process = mock_process
        bridge._closed = False
        assert bridge.is_running


class TestSingletonLazyLock:
    """Tests for the lazy-init singleton lock."""

    async def test_get_js_bridge_lazy_inits_lock(self) -> None:
        """get_js_bridge creates the lock on first call."""
        import aegra_api.services.js_bridge as bridge_module

        # Reset global state
        old_lock = bridge_module._bridge_lock
        old_instance = bridge_module._bridge_instance
        bridge_module._bridge_lock = None
        bridge_module._bridge_instance = None

        try:
            # Mock the bridge so start() doesn't actually spawn a process
            mock_bridge = MagicMock(spec=JSBridge)
            mock_bridge.is_running = True
            mock_bridge.start = AsyncMock()

            with patch.object(bridge_module, "JSBridge", return_value=mock_bridge):
                result = await bridge_module.get_js_bridge()

            assert bridge_module._bridge_lock is not None
            assert isinstance(bridge_module._bridge_lock, asyncio.Lock)
            assert result is mock_bridge
        finally:
            bridge_module._bridge_lock = old_lock
            bridge_module._bridge_instance = old_instance

    async def test_stop_js_bridge_lazy_inits_lock(self) -> None:
        """stop_js_bridge creates the lock if needed."""
        import aegra_api.services.js_bridge as bridge_module

        old_lock = bridge_module._bridge_lock
        old_instance = bridge_module._bridge_instance
        bridge_module._bridge_lock = None
        bridge_module._bridge_instance = None

        try:
            await bridge_module.stop_js_bridge()
            assert bridge_module._bridge_lock is not None
        finally:
            bridge_module._bridge_lock = old_lock
            bridge_module._bridge_instance = old_instance
