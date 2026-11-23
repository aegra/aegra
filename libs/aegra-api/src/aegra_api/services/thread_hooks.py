"""Thread-deleted hook: deployment code that runs after Aegra deletes a thread.

``hooks.thread_deleted`` in aegra.json names an async function. Aegra calls it
once per deleted thread, after the delete commits, in a background task that
never delays, fails or undoes the delete. Delivery is at most once: a crash
between the commit and the call, a call still running past the shutdown
grace, a delete that commits after the grace ends, and a thread whose
stateless run was still running when shutdown checked it are lost.
"""

import asyncio
import importlib
import importlib.util
import inspect
import sys
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TypedDict

import structlog

from aegra_api.config import get_config_dir, load_hooks_config
from aegra_api.services.webhook_delivery import SHUTDOWN_DRAIN_SECONDS

logger = structlog.getLogger(__name__)

_KNOWN_HOOKS = ("thread_deleted",)


class ThreadDeletedEvent(TypedDict):
    """What a thread-deleted hook receives: a plain dict, safe to forward as JSON."""

    thread_id: str
    user_id: str
    """Identity of the thread's owner."""
    metadata: dict[str, Any]
    """Copy of the thread's metadata as it stood when the thread was deleted."""


ThreadDeletedHook = Callable[[ThreadDeletedEvent], Awaitable[None]]


class HookLoadError(RuntimeError):
    """A configured hook cannot be loaded. Booting without it would skip every cleanup it exists for."""


def thread_deleted_event(thread_id: str, user_id: str, metadata: Mapping[str, Any] | None) -> ThreadDeletedEvent:
    """The hook payload, captured before the delete commits: the row is gone after it."""
    return {"thread_id": thread_id, "user_id": user_id, "metadata": dict(metadata or {})}


def load_thread_deleted_hook() -> ThreadDeletedHook | None:
    """The ``hooks.thread_deleted`` function, or None when none is configured.

    Raises HookLoadError for any configured value that does not load, including
    an unknown key under ``hooks``: a mistyped event name would otherwise never fire.
    """
    hooks = load_hooks_config()
    if hooks is None:
        return None
    if not isinstance(hooks, dict):
        raise HookLoadError(f"'hooks' in aegra.json must be an object, got {type(hooks).__name__}")
    unknown = sorted(set(hooks) - set(_KNOWN_HOOKS))
    if unknown:
        raise HookLoadError(f"Unknown hooks {unknown} in aegra.json; supported: {list(_KNOWN_HOOKS)}")
    path = hooks.get("thread_deleted")
    if path is None:
        return None
    if not isinstance(path, str) or ":" not in path:
        raise HookLoadError(f"Invalid hooks.thread_deleted path (expected './file.py:name' or 'module:name'): {path!r}")

    hook = _load_attribute(path)
    if not inspect.iscoroutinefunction(hook):
        raise HookLoadError(f"hooks.thread_deleted '{path}' must be an async function")
    logger.info("Loaded thread-deleted hook", path=path)
    return hook


def _load_attribute(path: str) -> object:
    """Load './file.py:name' or 'module:name'. Relative paths resolve from the config directory."""
    module_path, attr = path.rsplit(":", 1)
    is_file_path = module_path.endswith(".py") or module_path.startswith(("./", "../"))
    if not is_file_path:
        # User code can raise anything at import time, not only ImportError.
        try:
            module = importlib.import_module(module_path)
        except Exception as e:
            raise HookLoadError(f"Error loading hook module {module_path}: {e}") from e
        return _get_attribute(module, attr, path)

    file_path = Path(module_path)
    if not file_path.is_absolute():
        file_path = ((get_config_dir() or Path.cwd()) / file_path).resolve()
    if not file_path.exists():
        raise HookLoadError(f"Hook file not found: {file_path}")
    if not file_path.is_file():
        raise HookLoadError(f"Hook path is not a file: {file_path}")

    module_name = f"aegra_hooks_{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise HookLoadError(f"Could not load hook module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise HookLoadError(f"Error loading hook from {file_path}: {e}") from e
    return _get_attribute(module, attr, path)


def _get_attribute(module: object, attr: str, path: str) -> object:
    if not hasattr(module, attr):
        raise HookLoadError(f"'{attr}' not found for hooks.thread_deleted '{path}'")
    return getattr(module, attr)


class ThreadHookDispatcher:
    """Runs the thread-deleted hook in background tasks and drains them at shutdown."""

    def __init__(self) -> None:
        self._hook: ThreadDeletedHook | None = None
        self._tasks: dict[asyncio.Task[None], str] = {}
        self._pending_deletes: Callable[[], Awaitable[Iterable[asyncio.Task[Any]]]] | None = None
        self._stopped = False

    def configure(
        self,
        hook: ThreadDeletedHook | None,
        *,
        pending_deletes: Callable[[], Awaitable[Iterable[asyncio.Task[Any]]]] | None = None,
    ) -> None:
        """Set the hook, and the coroutine function returning the deletes in flight that shutdown waits for."""
        self._hook = hook
        self._pending_deletes = pending_deletes
        self._stopped = False

    def submit(self, event: ThreadDeletedEvent) -> None:
        """Start the hook for one deleted thread; the caller never waits on it."""
        hook = self._hook
        if hook is None:
            return
        if self._stopped:
            logger.warning("Thread deleted after shutdown, hook not called", thread_id=event["thread_id"])
            return
        task = asyncio.create_task(self._call(hook, event), name=f"thread-deleted-{event['thread_id']}")
        self._tasks[task] = event["thread_id"]
        task.add_done_callback(self._discard)

    async def stop(self) -> None:
        """Wait out the drain grace for running calls and deletes still in flight, then cancel the calls left.

        While a hook is configured, the deletes ``pending_deletes`` returns are waited for under the same
        deadline and never cancelled: stateless cleanup whose run has ended goes on deleting, and the call
        its delete submits is accepted and waited for too. Cleanup still waiting on a running run is not
        waited for. A crash, a call outliving the grace, a delete that commits after the grace ends, and a
        thread whose run was still running when shutdown checked it lose their call.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SHUTDOWN_DRAIN_SECONDS
        while (remaining := deadline - loop.time()) > 0:
            deletes: Iterable[asyncio.Task[Any]] = ()
            if self._hook is not None and self._pending_deletes is not None:
                try:
                    deletes = await asyncio.wait_for(self._pending_deletes(), timeout=remaining)
                except TimeoutError:
                    break
            # Read after the check, which can suspend: a delete that commits meanwhile has submitted its call.
            waiting = [task for task in self._tasks if not task.done()]
            waiting.extend(task for task in deletes if not task.done())
            if not waiting:
                break
            await asyncio.wait(waiting, timeout=deadline - loop.time(), return_when=asyncio.FIRST_COMPLETED)
        self._stopped = True
        pending = [task for task in self._tasks if not task.done()]
        if not pending:
            return
        logger.warning(
            "Cancelling unfinished thread-deleted hooks at shutdown",
            count=len(pending),
            thread_ids=[self._tasks[task] for task in pending],
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    def _discard(self, task: asyncio.Task[None]) -> None:
        self._tasks.pop(task, None)

    async def _call(self, hook: ThreadDeletedHook, event: ThreadDeletedEvent) -> None:
        try:
            await hook(event)
        except Exception:
            # Deployment code can raise anything; the delete stands and other threads' calls go on.
            logger.exception("Thread-deleted hook failed", thread_id=event["thread_id"], user_id=event["user_id"])


thread_hook_dispatcher = ThreadHookDispatcher()


def notify_threads_deleted(events: Iterable[ThreadDeletedEvent]) -> None:
    """Start the hook for each committed delete. Synchronous and never raises, so it is safe in ``finally``."""
    for event in events:
        try:
            thread_hook_dispatcher.submit(event)
        except RuntimeError:
            logger.exception("Could not schedule thread-deleted hook", thread_id=event["thread_id"])
