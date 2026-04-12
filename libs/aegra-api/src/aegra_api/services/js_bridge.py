"""Node.js subprocess bridge for LangGraph.js graph execution.

Manages a long-lived Node.js child process that loads and runs
LangGraph.js graphs.  Communication uses JSON-RPC 2.0 over
stdin/stdout (newline-delimited JSON).

Architecture
------------
- The bridge is a singleton managed by :class:`JSBridge`.
- On first use it spawns ``npx tsx <bridge-entry>``, waits for
  a ``ready`` notification, then accepts requests.
- Each request gets a unique ``id``; the response is matched by
  that id.  Streaming uses *notifications* (no ``id``) with a
  ``request_id`` field so we can correlate events to the original
  request.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Path to the JS bridge entry point.
# Resolution order:
# 1. AEGRA_JS_BRIDGE_DIR env var (Docker / custom deployments)
# 2. Bundled package data (wheel installs)
# 3. Source-tree relative path (editable / monorepo dev installs)
_PACKAGE_BRIDGE_DIR = Path(__file__).resolve().parent.parent / "_js_bridge"
_SOURCE_BRIDGE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "aegra-js-bridge"

_BRIDGE_DIR = Path(
    os.environ.get("AEGRA_JS_BRIDGE_DIR")
    or (str(_PACKAGE_BRIDGE_DIR) if _PACKAGE_BRIDGE_DIR.exists() else str(_SOURCE_BRIDGE_DIR))
)
_BRIDGE_ENTRY = _BRIDGE_DIR / "src" / "index.ts"

# Timeout for individual RPC calls (seconds)
DEFAULT_RPC_TIMEOUT = 120
# Timeout for bridge startup (seconds)
STARTUP_TIMEOUT = 30


class JSBridgeError(Exception):
    """Raised when the JS bridge reports an error."""

    def __init__(self, message: str, code: int = -32000, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class JSBridge:
    """Manages a Node.js subprocess for LangGraph.js graph execution.

    Usage::

        bridge = JSBridge()
        await bridge.start()
        info = await bridge.load_graph("/abs/path/graph.ts", "graph", "my_id")
        result = await bridge.invoke("my_id", {"messages": [...]})
        await bridge.stop()
    """

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._pending: dict[str | int, asyncio.Future[Any]] = {}
        self._notification_queues: dict[str | int, asyncio.Queue[dict[str, Any]]] = {}
        self._started = asyncio.Event()
        self._lock = asyncio.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Spawn the Node.js bridge subprocess and wait for ready."""
        async with self._lock:
            if self._process is not None and self._process.returncode is None:
                return  # already running

            await _find_node()  # Verify Node.js is installed
            npx_bin = _find_npx()

            if not _BRIDGE_ENTRY.exists():
                raise JSBridgeError(
                    f"JS bridge entry point not found: {_BRIDGE_ENTRY}. Ensure libs/aegra-js-bridge/ is present."
                )

            # Check if node_modules exists; if not, install
            node_modules = _BRIDGE_DIR / "node_modules"
            if not node_modules.exists():
                await logger.ainfo("Installing JS bridge dependencies…")
                install = await asyncio.create_subprocess_exec(
                    "npm",
                    "install",
                    cwd=str(_BRIDGE_DIR),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(install.communicate(), timeout=120)
                if install.returncode != 0:
                    raise JSBridgeError(f"npm install failed in {_BRIDGE_DIR}:\n{stderr.decode()}")

            # Spawn the bridge subprocess
            cmd = [npx_bin, "tsx", str(_BRIDGE_ENTRY)]
            env = {**os.environ}

            await logger.ainfo("Starting JS bridge", cmd=cmd)

            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_BRIDGE_DIR),
                env=env,
            )

            self._closed = False
            self._started.clear()

            # Start background reader
            self._reader_task = asyncio.create_task(self._read_loop())

            # Start stderr reader
            self._stderr_task = asyncio.create_task(self._stderr_loop())

            # Wait for ready notification
            try:
                await asyncio.wait_for(self._started.wait(), timeout=STARTUP_TIMEOUT)
            except TimeoutError:
                # Clean up inline — calling self.stop() would deadlock
                # because we already hold self._lock.
                self._closed = True
                if self._process:
                    try:
                        self._process.kill()
                        await self._process.wait()
                    except ProcessLookupError:
                        pass
                if self._reader_task and not self._reader_task.done():
                    self._reader_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._reader_task
                if self._stderr_task and not self._stderr_task.done():
                    self._stderr_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._stderr_task
                self._process = None
                self._reader_task = None
                self._stderr_task = None
                raise JSBridgeError(
                    f"JS bridge did not send ready signal within {STARTUP_TIMEOUT}s. Check Node.js installation."
                )

            await logger.ainfo("JS bridge started successfully")

    async def stop(self) -> None:
        """Gracefully shut down the bridge subprocess."""
        async with self._lock:
            if self._process is None:
                return

            self._closed = True

            try:
                # Try graceful shutdown first
                await self._send_raw(
                    {
                        "jsonrpc": "2.0",
                        "id": "_shutdown",
                        "method": "shutdown",
                    }
                )
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except (TimeoutError, OSError, JSBridgeError):
                # Force kill
                try:
                    self._process.kill()
                    await self._process.wait()
                except ProcessLookupError:
                    pass

            # Cancel reader task
            if self._reader_task and not self._reader_task.done():
                self._reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._reader_task

            # Cancel stderr task
            if self._stderr_task and not self._stderr_task.done():
                self._stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._stderr_task

            # Fail all pending requests
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(JSBridgeError("Bridge shut down"))

            self._pending.clear()
            self._notification_queues.clear()
            self._process = None
            self._reader_task = None
            self._stderr_task = None

            await logger.ainfo("JS bridge stopped")

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None and not self._closed

    # ------------------------------------------------------------------
    # RPC Methods
    # ------------------------------------------------------------------

    async def load_graph(self, file_path: str, export_name: str, graph_id: str) -> dict[str, Any]:
        """Load a LangGraph.js graph from a TypeScript/JavaScript file.

        Returns:
            GraphInfo dict with graphId, inputSchema, outputSchema.
        """
        return await self._call(
            "load_graph",
            {"path": file_path, "export_name": export_name, "graph_id": graph_id},
        )

    async def get_schema(self, graph_id: str) -> dict[str, Any]:
        """Get input/output schemas for a loaded graph."""
        return await self._call("get_schema", {"graph_id": graph_id})

    async def invoke(
        self,
        graph_id: str,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke a graph and return the final state."""
        return await self._call(
            "invoke",
            {"graph_id": graph_id, "input": input_data, "config": config or {}},
        )

    async def stream(
        self,
        graph_id: str,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
        stream_mode: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream graph execution events.

        Yields dicts with ``mode`` and ``data`` keys for each event.
        """
        request_id = str(uuid.uuid4())

        # Set up notification queue before sending
        queue: asyncio.Queue = asyncio.Queue()
        self._notification_queues[request_id] = queue

        try:
            # Send the stream request (response comes after all events)
            future = self._create_future(request_id)

            await self._send_raw(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "stream",
                    "params": {
                        "graph_id": graph_id,
                        "input": input_data,
                        "config": config or {},
                        "stream_mode": stream_mode or ["values"],
                    },
                }
            )

            # Yield events from the notification queue until we get the final response
            while True:
                # Check if the final response arrived
                if future.done():
                    # Drain any remaining events
                    while not queue.empty():
                        yield queue.get_nowait()
                    # Raises if the bridge reported an error
                    future.result()
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield event
                except TimeoutError:
                    # Check if process is still alive
                    if not self.is_running:
                        raise JSBridgeError("JS bridge process terminated unexpectedly")
                    continue

        finally:
            self._notification_queues.pop(request_id, None)
            self._pending.pop(request_id, None)

    async def ping(self) -> bool:
        """Check if the bridge is responsive."""
        try:
            result = await asyncio.wait_for(
                self._call("ping", {}),
                timeout=5,
            )
            return result.get("status") == "ok"
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal transport
    # ------------------------------------------------------------------

    def _create_future(self, request_id: str | int) -> asyncio.Future:
        """Create and register a future for a request id."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request_id] = future
        return future

    async def _call(self, method: str, params: dict[str, Any], timeout: float = DEFAULT_RPC_TIMEOUT) -> Any:
        """Send an RPC request and wait for the response."""
        if not self.is_running:
            raise JSBridgeError("JS bridge is not running. Call start() first.")

        request_id = str(uuid.uuid4())
        future = self._create_future(request_id)

        await self._send_raw(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending.pop(request_id, None)
            raise JSBridgeError(f"RPC call '{method}' timed out after {timeout}s")

        return result

    async def _send_raw(self, message: dict[str, Any]) -> None:
        """Write a JSON-RPC message to the subprocess stdin."""
        if self._process is None or self._process.stdin is None:
            raise JSBridgeError("JS bridge process not available")

        line = json.dumps(message, separators=(",", ":")) + "\n"
        self._process.stdin.write(line.encode())
        await self._process.stdin.drain()

    async def _read_loop(self) -> None:
        """Background task: read stdout lines and dispatch responses."""
        assert self._process is not None and self._process.stdout is not None

        try:
            while not self._closed:
                line_bytes = await self._process.stdout.readline()
                if not line_bytes:
                    break  # EOF — process exited

                line = line_bytes.decode().strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    await logger.awarning("JS bridge sent invalid JSON", line=line[:200])
                    continue

                await self._dispatch(msg)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            await logger.aerror("JS bridge reader crashed", exc_info=exc)

        # Process ended — fail all pending futures
        if not self._closed:
            await logger.awarning("JS bridge process exited unexpectedly")
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(JSBridgeError("JS bridge process exited unexpectedly"))

    async def _stderr_loop(self) -> None:
        """Background task: log stderr output from the bridge."""
        if self._process is None or self._process.stderr is None:
            return

        try:
            while not self._closed:
                line = await self._process.stderr.readline()
                if not line:
                    break
                await logger.adebug("JS bridge stderr", output=line.decode().rstrip())
        except asyncio.CancelledError:
            return
        except (OSError, UnicodeDecodeError) as exc:
            await logger.awarning("JS bridge stderr reader failed", exc_info=exc)

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        """Route an incoming JSON-RPC message to the right handler."""
        # Notification (no "id" field)
        if "id" not in msg:
            method = msg.get("method", "")
            params = msg.get("params", {})

            if method == "ready":
                await logger.ainfo("JS bridge ready", version=params.get("version"))
                self._started.set()
                return

            if method == "stream_event":
                # Route to the notification queue for the matching request
                request_id = params.get("request_id")
                queue = self._notification_queues.get(request_id)
                if queue is not None:
                    await queue.put(
                        {
                            "mode": params.get("mode", "values"),
                            "data": params.get("data"),
                        }
                    )
                return

            await logger.adebug("JS bridge notification", method=method, params=params)
            return

        # Response (has "id" field)
        msg_id = msg["id"]
        future = self._pending.pop(msg_id, None)

        if future is None:
            await logger.awarning("JS bridge response for unknown id", id=msg_id)
            return

        if "error" in msg:
            err = msg["error"]
            future.set_exception(
                JSBridgeError(
                    err.get("message", "Unknown error"),
                    code=err.get("code", -32000),
                    data=err.get("data"),
                )
            )
        else:
            future.set_result(msg.get("result"))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _find_node() -> str:
    """Find the Node.js binary and verify version >= 18."""
    node = shutil.which("node")
    if not node:
        raise JSBridgeError(
            "Node.js not found. Install Node.js 18+ to use LangGraph.js graphs.\nDownload from: https://nodejs.org/"
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            node,
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        version_str = stdout.decode().strip().lstrip("v")
        major = int(version_str.split(".")[0])
        if major < 18:
            raise JSBridgeError(
                f"Node.js {version_str} detected but LangGraph.js requires 18+.\nDownload from: https://nodejs.org/"
            )
    except JSBridgeError:
        raise
    except (TimeoutError, OSError, ValueError) as exc:
        logger.warning("Could not determine Node.js version", exc_info=exc)

    return node


def _find_npx() -> str:
    """Find the npx binary."""
    npx = shutil.which("npx")
    if not npx:
        raise JSBridgeError(
            "npx not found. Ensure Node.js 18+ is installed with npm.\nDownload from: https://nodejs.org/"
        )
    return npx


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_bridge_lock = asyncio.Lock()
_bridge_instance: JSBridge | None = None


async def get_js_bridge() -> JSBridge:
    """Get or create the singleton JS bridge instance."""
    global _bridge_instance
    async with _bridge_lock:
        if _bridge_instance is None:
            _bridge_instance = JSBridge()
        if not _bridge_instance.is_running:
            await _bridge_instance.start()
        return _bridge_instance


async def stop_js_bridge() -> None:
    """Stop the singleton JS bridge if running."""
    global _bridge_instance
    async with _bridge_lock:
        if _bridge_instance is not None:
            await _bridge_instance.stop()
            _bridge_instance = None
