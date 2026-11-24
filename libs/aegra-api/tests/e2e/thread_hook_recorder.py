"""thread_deleted hook for the E2E server: records each event for the tests to read.

The E2E workflow names this file under ``hooks.thread_deleted``. Each event is appended
as one JSON line to the file the ``AEGRA_E2E_THREAD_HOOK_LOG`` variable names; without
the variable the hook records nothing. Appends are serialized, because the server calls
the hook for several threads at once and, on Windows, two appends through separate
handles can land at one offset, the later overwriting the earlier.
"""

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any

LOG_ENV = "AEGRA_E2E_THREAD_HOOK_LOG"

_APPEND_LOCK = threading.Lock()


def _append(path: Path, line: str) -> None:
    with _APPEND_LOCK, path.open("a", encoding="utf-8") as log:
        log.write(line + "\n")


async def record(event: dict[str, Any]) -> None:
    """Append the event as one JSON line."""
    path = os.environ.get(LOG_ENV)
    if not path:
        return
    await asyncio.to_thread(_append, Path(path), json.dumps(event, sort_keys=True))
