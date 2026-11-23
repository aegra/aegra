"""Thread-deleted hook: loading from aegra.json and running in the background."""

import asyncio
import importlib
import json
import sys
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from aegra_api.services import thread_hooks
from aegra_api.services.thread_hooks import (
    HookLoadError,
    ThreadDeletedEvent,
    ThreadHookDispatcher,
    load_thread_deleted_hook,
    notify_threads_deleted,
    thread_deleted_event,
)

_HOOK_SOURCE = "async def on_thread_deleted(event):\n    pass\n\ndef sync_hook(event):\n    pass\n\nvalue = 1\n"


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hooks: object) -> None:
    config = tmp_path / "aegra.json"
    config.write_text(json.dumps({"graphs": {}, "hooks": hooks}))
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def logs() -> Iterator[list[MutableMapping[str, Any]]]:
    with capture_logs() as captured:
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
        assert "Thread-deleted hook failed" in _text(logs)
        assert "t-bad" in _text(logs)
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
