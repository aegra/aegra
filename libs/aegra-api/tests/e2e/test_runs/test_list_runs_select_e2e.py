"""E2E: GET /threads/{thread_id}/runs honors `select`, so a client can list runs without their outputs."""

import json

import httpx
import pytest

from tests.e2e._utils import elog, get_e2e_client


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_runs_select_returns_only_requested_fields() -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    message = {"role": "user", "content": json.dumps({"delay": 0, "steps": 1})}
    for _ in range(2):
        await client.runs.wait(thread_id, assistant["assistant_id"], input={"messages": [message]})

    full = await client.runs.list(thread_id)
    selected = await client.runs.list(thread_id, select=["run_id", "status"])
    elog("runs.list select", selected)

    assert len(full) == len(selected) == 2
    assert all("output" in run for run in full)
    assert [set(run) for run in selected] == [{"run_id", "status"}] * 2
    assert [run["run_id"] for run in selected] == [run["run_id"] for run in full]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_runs_rejects_unknown_select_field() -> None:
    client = get_e2e_client()
    thread = await client.threads.create()

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.runs.list(thread["thread_id"], select=["run_id", "not_a_field"])

    assert exc_info.value.response.status_code == 422
