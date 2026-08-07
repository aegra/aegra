import asyncio
import json
from typing import Any

import httpx
import pytest

from aegra_api.settings import settings

TERMINAL_RUN_STATES = ("success", "error", "interrupted", "timeout")

try:
    from langgraph_sdk import get_client
except Exception as e:
    raise RuntimeError(
        "langgraph-sdk is required for E2E tests. Install via extras 'e2e' or add to your environment."
    ) from e


async def await_terminal_run(
    client: Any,
    thread_id: str,
    run_id: str,
    *,
    timeout: float = 15.0,
) -> dict:
    """Poll a run until it settles.

    Cancelling a run a live worker owns is asynchronous: the worker finalizes it,
    so the state right after the cancel call is not guaranteed to be terminal.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    run = await client.runs.get(thread_id, run_id)
    while run["status"] not in TERMINAL_RUN_STATES:
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"Run {run_id} did not settle within {timeout}s, last status={run['status']!r}")
        await asyncio.sleep(0.25)
        run = await client.runs.get(thread_id, run_id)
    return run


def elog(title: str, payload):
    """Emit pretty JSON logs for E2E visibility."""
    try:
        formatted = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception:
        formatted = str(payload)
    print(f"\n=== {title} ===\n{formatted}\n")


def get_e2e_client():
    """Construct a LangGraph SDK client from env and log the target URL."""
    server_url = settings.app.SERVER_URL
    print(f"[E2E] Using SERVER_URL={server_url}")
    return get_client(url=server_url)


def check_server_has_auth(url: str | None = None) -> bool | None:
    """Check if server has auth enabled by making unauthenticated request.

    Args:
        url: Server URL (defaults to settings.app.SERVER_URL)

    Returns:
        True if auth is enabled (401 response)
        False if no auth (200 response)
        None if can't determine (server not running or error)
    """
    if url is None:
        url = settings.app.SERVER_URL

    try:
        response = httpx.get(f"{url}/assistants", timeout=2.0)
        if response.status_code == 401:
            return True
        elif response.status_code in (200, 404):
            return False
    except Exception:
        pass
    return None


def check_and_skip_if_geo_blocked(run_data: dict) -> None:
    """
    Checks if a run failed due to OpenAI geo-blocking/unsupported region.
    If so, skips the test instead of failing.

    This targets the specific error code 'unsupported_country_region_territory'
    to avoid masking other permission errors (403).
    """
    if run_data.get("status") == "error":
        msg = str(run_data.get("error_message", "")).lower()
        # Strict check as requested by maintainer
        if "unsupported_country_region_territory" in msg or "generator didn't stop" in msg:
            pytest.skip(f"⛔️ Skipped: OpenAI Geo-block detected. ({msg[:60]}...)")
