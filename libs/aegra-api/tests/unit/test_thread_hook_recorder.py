"""The E2E thread_deleted recorder keeps every event when the server calls it for several threads at once."""

import asyncio
import json
import uuid
from pathlib import Path

import pytest

from tests.e2e.thread_hook_recorder import LOG_ENV, record


@pytest.mark.asyncio
async def test_concurrent_calls_record_every_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "thread-hooks.jsonl"
    monkeypatch.setenv(LOG_ENV, str(log))
    marker = uuid.uuid4().hex
    thread_ids = [str(uuid.uuid4()) for _ in range(100)]
    events = [
        {"thread_id": thread_id, "user_id": "anonymous", "metadata": {"hook_case": marker}} for thread_id in thread_ids
    ]

    for batch in range(0, len(events), 20):
        await asyncio.gather(*(record(event) for event in events[batch : batch + 20]))

    recorded = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert sorted(event["thread_id"] for event in recorded) == sorted(thread_ids)
