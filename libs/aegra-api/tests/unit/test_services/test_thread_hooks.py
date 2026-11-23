"""Thread-deleted hook: loading from aegra.json and running in the background."""

import asyncio
import contextlib
import importlib
import json
import sys
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any

import pytest
import structlog
from redis import RedisError
from structlog.testing import capture_logs

from aegra_api.services import run_cleanup, thread_hooks
from aegra_api.services.thread_hooks import (
    HookLoadError,
    ThreadDeletedEvent,
    ThreadHookDispatcher,
    load_thread_deleted_hook,
    notify_threads_deleted,
    thread_deleted_event,
    thread_hook_dispatcher,
)

_HOOK_SOURCE = "async def on_thread_deleted(event):\n    pass\n\ndef sync_hook(event):\n    pass\n\nvalue = 1\n"


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hooks: object) -> None:
    config = tmp_path / "aegra.json"
    config.write_text(json.dumps({"graphs": {}, "hooks": hooks}))
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def logs(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[MutableMapping[str, Any]]]:
    with capture_logs() as captured:
        # An earlier test's logging setup may have bound the module logger; a fresh one binds under the capture.
        monkeypatch.setattr(thread_hooks, "logger", structlog.get_logger(thread_hooks.__name__))
        yield captured


def _text(logs: list[MutableMapping[str, Any]]) -> str:
    return " ".join(str(entry) for entry in logs)


def _event(thread_id: str = "t-1") -> ThreadDeletedEvent:
    return thread_deleted_event(thread_id, "owner-1", {"k": "v"})


class TestLoadThreadDeletedHook:
    def test_returns_none_without_hooks_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = tmp_path / "aegra.json"
        config.write_text(json.dumps({"graphs": {}}))
        monkeypatch.chdir(tmp_path)

        assert load_thread_deleted_hook() is None

    def test_loads_file_path_relative_to_config_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "hooks.py").write_text(_HOOK_SOURCE)
        _write_config(tmp_path, monkeypatch, {"thread_deleted": "./hooks.py:on_thread_deleted"})

        hook = load_thread_deleted_hook()

        assert hook is sys.modules["aegra_hooks_hooks"].on_thread_deleted

    def test_loads_module_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        package = tmp_path / "pkg_thread_hooks"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "cleanup.py").write_text(_HOOK_SOURCE)
        monkeypatch.syspath_prepend(str(tmp_path))
        _write_config(tmp_path, monkeypatch, {"thread_deleted": "pkg_thread_hooks.cleanup:on_thread_deleted"})

        hook = load_thread_deleted_hook()

        assert hook is importlib.import_module("pkg_thread_hooks.cleanup").on_thread_deleted

    @pytest.mark.parametrize(
        ("hooks", "message"),
        [
            (["./hooks.py:on_thread_deleted"], "must be an object"),
            ({"thread_delete": "./hooks.py:on_thread_deleted"}, "Unknown hooks"),
            ({"thread_deleted": 5}, "Invalid hooks.thread_deleted path"),
            ({"thread_deleted": "./hooks.py"}, "Invalid hooks.thread_deleted path"),
            ({"thread_deleted": "./missing.py:on_thread_deleted"}, "Hook file not found"),
            ({"thread_deleted": "./hooks.py:absent"}, "not found"),
            ({"thread_deleted": "./hooks.py:sync_hook"}, "must be an async function"),
            ({"thread_deleted": "./hooks.py:value"}, "must be an async function"),
            ({"thread_deleted": "no_such_module_for_hooks:fn"}, "Error loading hook module"),
        ],
    )
    def test_raises_for_a_value_that_does_not_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hooks: object, message: str
    ) -> None:
        (tmp_path / "hooks.py").write_text(_HOOK_SOURCE)
        _write_config(tmp_path, monkeypatch, hooks)

        with pytest.raises(HookLoadError, match=message):
            load_thread_deleted_hook()

    def test_raises_when_path_is_a_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "hooks.py").mkdir()
        _write_config(tmp_path, monkeypatch, {"thread_deleted": "./hooks.py:on_thread_deleted"})

        with pytest.raises(HookLoadError, match="not a file"):
            load_thread_deleted_hook()

    def test_import_error_in_file_raises_and_leaves_no_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "broken_hook.py").write_text("raise RuntimeError('boom')\n")
        _write_config(tmp_path, monkeypatch, {"thread_deleted": "./broken_hook.py:on_thread_deleted"})

        with pytest.raises(HookLoadError, match="boom") as excinfo:
            load_thread_deleted_hook()

        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert "aegra_hooks_broken_hook" not in sys.modules


class TestThreadDeletedEvent:
    def test_copies_metadata(self) -> None:
        metadata = {"tenant": "a"}

        event = thread_deleted_event("t-1", "owner-1", metadata)
        metadata["tenant"] = "b"

        assert event == {"thread_id": "t-1", "user_id": "owner-1", "metadata": {"tenant": "a"}}

    def test_missing_metadata_becomes_empty_dict(self) -> None:
        assert thread_deleted_event("t-1", "owner-1", None)["metadata"] == {}


class TestThreadHookDispatcher:
    @pytest.mark.asyncio
    async def test_calls_hook_once_per_event_with_the_payload(self) -> None:
        received: list[ThreadDeletedEvent] = []

        async def hook(event: ThreadDeletedEvent) -> None:
            received.append(event)

        dispatcher = ThreadHookDispatcher()
        dispatcher.configure(hook)
        dispatcher.submit(_event("t-1"))
        dispatcher.submit(_event("t-2"))
        await dispatcher.stop()

        assert received == [_event("t-1"), _event("t-2")]

    @pytest.mark.asyncio
    async def test_submit_does_not_wait_for_the_hook(self) -> None:
        release = asyncio.Event()

        async def hook(event: ThreadDeletedEvent) -> None:
            await release.wait()

        dispatcher = ThreadHookDispatcher()
        dispatcher.configure(hook)
        dispatcher.submit(_event())

        release.set()
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_failing_hook_is_logged_and_other_calls_still_run(self, logs: list[MutableMapping[str, Any]]) -> None:
        received: list[str] = []

        async def hook(event: ThreadDeletedEvent) -> None:
            if event["thread_id"] == "t-bad":
                raise ValueError("cleanup failed")
            received.append(event["thread_id"])

        dispatcher = ThreadHookDispatcher()
        dispatcher.configure(hook)
        dispatcher.submit(_event("t-bad"))
        dispatcher.submit(_event("t-good"))
        await dispatcher.stop()

        assert received == ["t-good"]
        failure = next(entry for entry in logs if entry["event"] == "Thread-deleted hook failed")
        assert failure["thread_id"] == "t-bad"
        assert failure["user_id"] == "owner-1"
        assert "'k'" not in _text(logs)

    @pytest.mark.asyncio
    async def test_no_hook_configured_starts_nothing(self) -> None:
        dispatcher = ThreadHookDispatcher()
        dispatcher.submit(_event())

        assert dispatcher._tasks == {}

    @pytest.mark.asyncio
    async def test_stop_waits_for_a_running_call(self) -> None:
        finished: list[str] = []

        async def hook(event: ThreadDeletedEvent) -> None:
            await asyncio.sleep(0.05)
            finished.append(event["thread_id"])

        dispatcher = ThreadHookDispatcher()
        dispatcher.configure(hook)
        dispatcher.submit(_event())
        await dispatcher.stop()

        assert finished == ["t-1"]

    @pytest.mark.asyncio
    async def test_stop_accepts_and_waits_for_a_submission_during_the_wait(self) -> None:
        finished: list[str] = []
        dispatcher = ThreadHookDispatcher()

        async def hook(event: ThreadDeletedEvent) -> None:
            if event["thread_id"] == "t-1":
                await asyncio.sleep(0.02)
                dispatcher.submit(_event("t-late"))
            finished.append(event["thread_id"])

        dispatcher.configure(hook)
        dispatcher.submit(_event("t-1"))
        await dispatcher.stop()

        assert finished == ["t-1", "t-late"]

    @pytest.mark.asyncio
    async def test_stop_cancels_after_the_grace_and_logs_the_thread_ids(
        self, monkeypatch: pytest.MonkeyPatch, logs: list[MutableMapping[str, Any]]
    ) -> None:
        monkeypatch.setattr(thread_hooks, "SHUTDOWN_DRAIN_SECONDS", 0.01)
        cancelled: list[str] = []

        async def hook(event: ThreadDeletedEvent) -> None:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(event["thread_id"])
                raise

        dispatcher = ThreadHookDispatcher()
        dispatcher.configure(hook)
        dispatcher.submit(_event("t-slow"))
        await dispatcher.stop()

        assert cancelled == ["t-slow"]
        assert "Cancelling unfinished thread-deleted hooks" in _text(logs)
        assert "t-slow" in _text(logs)

    @pytest.mark.asyncio
    async def test_submission_after_stop_starts_nothing(self, logs: list[MutableMapping[str, Any]]) -> None:
        calls: list[str] = []

        async def hook(event: ThreadDeletedEvent) -> None:
            calls.append(event["thread_id"])

        dispatcher = ThreadHookDispatcher()
        dispatcher.configure(hook)
        await dispatcher.stop()
        dispatcher.submit(_event("t-after"))
        await asyncio.sleep(0)

        assert calls == []
        assert dispatcher._tasks == {}
        assert "t-after" in _text(logs)


class _FakeRuns:
    """The executor the cleanup waits on: its runs end when ``ended`` is set, and shutdown's finished checks.

    A run in ``live`` never ends. ``check_gate`` and ``check_delay`` make the finished check suspend, as the
    Redis and database reads of a worker deployment do.
    """

    def __init__(self) -> None:
        self.ended = asyncio.Event()
        self.live: set[str] = set()
        self.checks: list[str] = []
        self.check_error: Exception | None = None
        self.check_gate: asyncio.Event | None = None
        self.check_delay = 0.0
        self.delete_released = asyncio.Event()
        self.delete_released.set()
        self.deleting = asyncio.Event()
        self.notified = asyncio.Event()

    async def wait_for_completion(self, run_id: str, timeout: float) -> None:
        if run_id in self.live:
            await asyncio.get_running_loop().create_future()
        await self.ended.wait()

    async def is_finished(self, run_id: str) -> bool:
        self.checks.append(run_id)
        if self.check_gate is not None:
            await self.check_gate.wait()
        if self.check_delay:
            await asyncio.sleep(self.check_delay)
        if self.check_error is not None:
            raise self.check_error
        return run_id not in self.live and self.ended.is_set()


async def _cancel(task: asyncio.Task[Any]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


class TestShutdownWithDeletesInFlight:
    """Shutdown waits for stateless cleanup past its run, or whose run has ended, and not for one on a live run."""

    @pytest.fixture
    def runs(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeRuns]:
        fake = _FakeRuns()

        async def delete_thread_by_id(thread_id: str, user_id: str) -> None:
            fake.deleting.set()
            await fake.delete_released.wait()
            await asyncio.sleep(0.01)
            notify_threads_deleted([thread_deleted_event(thread_id, user_id, {"k": "v"})])
            fake.notified.set()

        monkeypatch.setattr(run_cleanup, "executor", fake)
        monkeypatch.setattr(run_cleanup, "delete_thread_by_id", delete_thread_by_id)
        yield fake
        thread_hook_dispatcher.configure(None)

    @pytest.mark.asyncio
    async def test_thread_deleted_by_cleanup_during_shutdown_reaches_the_hook(self, runs: _FakeRuns) -> None:
        received: list[ThreadDeletedEvent] = []

        async def hook(event: ThreadDeletedEvent) -> None:
            received.append(event)

        thread_hook_dispatcher.configure(hook, pending_deletes=run_cleanup.cleanup_tasks_for_shutdown)
        cleanup = run_cleanup.schedule_background_cleanup("r-1", "t-1", "owner-1")
        await asyncio.sleep(0)
        # The executor stops before the dispatcher: the run has ended while its cleanup still waits on it.
        runs.ended.set()
        await thread_hook_dispatcher.stop()

        assert received == [{"thread_id": "t-1", "user_id": "owner-1", "metadata": {"k": "v"}}]
        assert cleanup.done()

    @pytest.mark.asyncio
    async def test_cleanup_on_a_live_run_does_not_hold_shutdown(self, runs: _FakeRuns) -> None:
        calls: list[str] = []

        async def hook(event: ThreadDeletedEvent) -> None:
            await asyncio.sleep(0.05)
            calls.append(event["thread_id"])

        thread_hook_dispatcher.configure(hook, pending_deletes=run_cleanup.cleanup_tasks_for_shutdown)
        cleanup = run_cleanup.schedule_background_cleanup("r-live", "t-live", "owner-1")
        await asyncio.sleep(0)
        # A call still running makes stop look at the cleanup more than once.
        thread_hook_dispatcher.submit(_event("t-call"))

        await asyncio.wait_for(thread_hook_dispatcher.stop(), timeout=1)

        assert calls == ["t-call"]
        assert runs.checks == ["r-live"]
        assert not cleanup.done()
        await _cancel(cleanup)

    @pytest.mark.asyncio
    async def test_call_submitted_while_the_runs_are_checked_is_waited_for(
        self, runs: _FakeRuns, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        received: list[str] = []

        async def hook(event: ThreadDeletedEvent) -> None:
            await asyncio.sleep(0.05)
            received.append(event["thread_id"])

        runs.ended.set()
        runs.live.add("r-live")
        # The live run's check returns only once the other cleanup has deleted its thread and submitted the call.
        runs.check_gate = runs.notified
        thread_hook_dispatcher.configure(hook, pending_deletes=run_cleanup.cleanup_tasks_for_shutdown)
        deleting = run_cleanup.schedule_background_cleanup("r-ended", "t-ended", "owner-1")
        live = run_cleanup.schedule_background_cleanup("r-live", "t-live", "owner-1")
        await runs.deleting.wait()

        with capture_logs() as captured:
            monkeypatch.setattr(thread_hooks, "logger", structlog.get_logger(thread_hooks.__name__))
            await asyncio.wait_for(thread_hook_dispatcher.stop(), timeout=2)

        assert received == ["t-ended"]
        assert deleting.done()
        assert not live.done()
        assert not any(entry["log_level"] == "warning" for entry in captured)
        await _cancel(live)

    @pytest.mark.asyncio
    async def test_a_slow_run_check_does_not_stretch_the_period(
        self, runs: _FakeRuns, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(thread_hooks, "SHUTDOWN_DRAIN_SECONDS", 0.3)

        async def hook(event: ThreadDeletedEvent) -> None:
            await asyncio.get_running_loop().create_future()

        runs.live.add("r-live")
        runs.check_delay = 0.25
        thread_hook_dispatcher.configure(hook, pending_deletes=run_cleanup.cleanup_tasks_for_shutdown)
        live = run_cleanup.schedule_background_cleanup("r-live", "t-live", "owner-1")
        await asyncio.sleep(0)
        thread_hook_dispatcher.submit(_event("t-stuck"))

        loop = asyncio.get_running_loop()
        started = loop.time()
        await asyncio.wait_for(thread_hook_dispatcher.stop(), timeout=2)

        # One period plus a margin for coarse timers; a full period restarted after the check takes about 0.55 s.
        assert loop.time() - started < 0.45
        assert not live.done()
        await _cancel(live)

    @pytest.mark.asyncio
    async def test_cleanup_past_its_run_wait_is_waited_for(
        self, runs: _FakeRuns, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(thread_hooks, "SHUTDOWN_DRAIN_SECONDS", 1.0)
        received: list[str] = []

        async def hook(event: ThreadDeletedEvent) -> None:
            received.append(event["thread_id"])

        runs.ended.set()
        runs.delete_released.clear()
        thread_hook_dispatcher.configure(hook, pending_deletes=run_cleanup.cleanup_tasks_for_shutdown)
        cleanup = run_cleanup.schedule_background_cleanup("r-1", "t-1", "owner-1")
        await runs.deleting.wait()
        # The delete finishes after shutdown has started; a cleanup this far needs no run check.
        asyncio.get_running_loop().call_later(0.05, runs.delete_released.set)

        await asyncio.wait_for(thread_hook_dispatcher.stop(), timeout=2)

        assert received == ["t-1"]
        assert cleanup.done()
        assert runs.checks == []

    @pytest.mark.asyncio
    async def test_cleanup_deleting_when_the_period_ends_is_not_cancelled(
        self, runs: _FakeRuns, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(thread_hooks, "SHUTDOWN_DRAIN_SECONDS", 0.05)
        calls: list[str] = []

        async def hook(event: ThreadDeletedEvent) -> None:
            calls.append(event["thread_id"])

        runs.ended.set()
        runs.delete_released.clear()
        thread_hook_dispatcher.configure(hook, pending_deletes=run_cleanup.cleanup_tasks_for_shutdown)
        cleanup = run_cleanup.schedule_background_cleanup("r-1", "t-1", "owner-1")
        await asyncio.sleep(0.01)

        await asyncio.wait_for(thread_hook_dispatcher.stop(), timeout=1)
        await asyncio.sleep(0)

        assert not cleanup.done()
        assert calls == []
        assert runs.checks == []
        await _cancel(cleanup)

    @pytest.mark.asyncio
    async def test_failed_run_check_waits_for_the_cleanup(
        self, runs: _FakeRuns, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(thread_hooks, "SHUTDOWN_DRAIN_SECONDS", 1.0)
        received: list[str] = []

        async def hook(event: ThreadDeletedEvent) -> None:
            received.append(event["thread_id"])

        runs.check_error = RedisError("redis down")
        thread_hook_dispatcher.configure(hook, pending_deletes=run_cleanup.cleanup_tasks_for_shutdown)
        cleanup = run_cleanup.schedule_background_cleanup("r-1", "t-1", "owner-1")
        await asyncio.sleep(0)
        asyncio.get_running_loop().call_later(0.05, runs.ended.set)

        with capture_logs() as captured:
            monkeypatch.setattr(run_cleanup, "logger", structlog.get_logger(run_cleanup.__name__))
            await asyncio.wait_for(thread_hook_dispatcher.stop(), timeout=2)

        assert received == ["t-1"]
        assert cleanup.done()
        assert any(entry.get("run_id") == "r-1" and entry["log_level"] == "error" for entry in captured)

    @pytest.mark.asyncio
    async def test_thread_deleted_after_shutdown_gets_no_call(
        self, runs: _FakeRuns, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        async def hook(event: ThreadDeletedEvent) -> None:
            calls.append(event["thread_id"])

        thread_hook_dispatcher.configure(hook, pending_deletes=run_cleanup.cleanup_tasks_for_shutdown)
        cleanup = run_cleanup.schedule_background_cleanup("r-late", "t-late", "owner-1")
        await asyncio.sleep(0)

        with capture_logs() as captured:
            monkeypatch.setattr(thread_hooks, "logger", structlog.get_logger(thread_hooks.__name__))
            await asyncio.wait_for(thread_hook_dispatcher.stop(), timeout=1)
            runs.ended.set()
            await asyncio.wait_for(cleanup, timeout=1)

        assert calls == []
        assert any(entry.get("thread_id") == "t-late" and entry["log_level"] == "warning" for entry in captured)

    @pytest.mark.asyncio
    async def test_without_a_hook_no_run_is_checked(self, runs: _FakeRuns) -> None:
        thread_hook_dispatcher.configure(None, pending_deletes=run_cleanup.cleanup_tasks_for_shutdown)
        cleanup = run_cleanup.schedule_background_cleanup("r-1", "t-1", "owner-1")
        await asyncio.sleep(0)

        await asyncio.wait_for(thread_hook_dispatcher.stop(), timeout=1)

        assert runs.checks == []
        assert not cleanup.done()
        await _cancel(cleanup)


class TestNotifyThreadsDeleted:
    def test_never_raises_without_a_running_loop(
        self, monkeypatch: pytest.MonkeyPatch, logs: list[MutableMapping[str, Any]]
    ) -> None:
        async def hook(event: ThreadDeletedEvent) -> None:
            pass

        dispatcher = ThreadHookDispatcher()
        dispatcher.configure(hook)
        monkeypatch.setattr(thread_hooks, "thread_hook_dispatcher", dispatcher)

        notify_threads_deleted([_event("t-no-loop")])

        assert "Could not schedule thread-deleted hook" in _text(logs)
        assert "t-no-loop" in _text(logs)

    @pytest.mark.asyncio
    async def test_submits_every_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received: list[str] = []

        async def hook(event: ThreadDeletedEvent) -> None:
            received.append(event["thread_id"])

        dispatcher = ThreadHookDispatcher()
        dispatcher.configure(hook)
        monkeypatch.setattr(thread_hooks, "thread_hook_dispatcher", dispatcher)

        notify_threads_deleted([_event("t-1"), _event("t-2")])
        await dispatcher.stop()

        assert received == ["t-1", "t-2"]
