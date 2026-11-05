"""E2E test: a scheduled cron firing runs as the user ``auth.cron_user`` returns.

aegra.auth.json names ``examples/jwt_mock_auth_example.py:cron_user``, which
returns the owner's display name, so a fired ``cron_example`` tick records
``User <id>`` where a server without the resolver records ``cron-scheduler``.

Requires an auth-enabled server with ``CRON_ALLOW_SECONDS_SCHEDULE=true`` and a
short ``CRON_POLL_INTERVAL_SECONDS``; for ``make e2e-auth``, set both in ``.env``.
Run with: make e2e-auth  or  pytest -m auth_only
"""

import asyncio
from typing import Any
from uuid import uuid4

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import elog

# The scheduler's default poll interval is 60 s, so wait past one full interval.
FIRING_TIMEOUT_SECONDS = 75


async def _ticks(http: httpx.AsyncClient, thread_id: str) -> list[dict[str, Any]]:
    response = await http.get(f"/threads/{thread_id}/state")
    assert response.status_code == 200, response.text
    messages: list[dict[str, Any]] = response.json()["values"].get("messages", [])
    return [m for m in messages if str(m.get("content", "")).startswith("Tick #")]


@pytest.mark.e2e
@pytest.mark.auth_only
@pytest.mark.asyncio
async def test_scheduled_firing_runs_as_the_resolved_user() -> None:
    user_id = f"cronowner{uuid4().hex[:8]}"
    headers = {"Authorization": f"Bearer mock-jwt-{user_id}-admin-team1"}

    async with httpx.AsyncClient(base_url=settings.app.SERVER_URL or "", headers=headers, timeout=30.0) as http:
        thread_id = (await http.post("/threads", json={})).json()["thread_id"]
        cron_id: str | None = None
        try:
            response = await http.post(
                f"/threads/{thread_id}/runs/crons",
                json={"assistant_id": "cron_example", "schedule": "*/2 * * * * *", "input": {"messages": []}},
            )
            if response.status_code == 422 and "CRON_ALLOW_SECONDS_SCHEDULE" in response.text:
                pytest.skip("requires CRON_ALLOW_SECONDS_SCHEDULE=true on the server")
            assert response.status_code == 200, response.text
            search = await http.post("/runs/crons/search", json={"thread_id": thread_id})
            assert search.status_code == 200, search.text
            cron_id = search.json()[0]["cron_id"]

            ticks: list[dict[str, Any]] = []
            for _ in range(FIRING_TIMEOUT_SECONDS):
                ticks = await _ticks(http, thread_id)
                if len(ticks) >= 2:
                    break
                await asyncio.sleep(1)
            elog("cron_example ticks", ticks)

            assert len(ticks) >= 2, "the scheduler fired no run; check CRON_POLL_INTERVAL_SECONDS on the server"
            assert ticks[0]["additional_kwargs"]["run_by"] == f"User {user_id}"
            assert ticks[1]["additional_kwargs"]["run_by"] == f"User {user_id}"
        finally:
            if cron_id is not None:
                await http.delete(f"/runs/crons/{cron_id}")
            await http.delete(f"/threads/{thread_id}")
