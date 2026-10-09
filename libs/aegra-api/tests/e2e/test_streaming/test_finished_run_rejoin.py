"""Terminal v1 rejoin coverage with a deterministic graph and real storage."""

import asyncio
import json

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import await_terminal_run, check_and_skip_if_geo_blocked, elog, get_e2e_client


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_finished_run_rejoin_closes_after_replay_and_acknowledged_end(fail: bool) -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test")
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant["assistant_id"],
        input={"messages": [{"role": "user", "content": json.dumps({"delay": 0.01, "steps": 1, "fail": fail})}]},
        stream_mode=["values"],
    )
    run_id = run["run_id"]
    try:
        final = await await_terminal_run(client, thread_id, run_id)
        check_and_skip_if_geo_blocked(final)
        expected_status = "error" if fail else "success"
        assert final["status"] == expected_status
        url = f"{settings.app.SERVER_URL}/threads/{thread_id}/runs/{run_id}/stream"
        async with httpx.AsyncClient(timeout=3.0) as http:
            for _ in range(2):
                deadline = asyncio.get_running_loop().time() + 5.0
                while True:
                    response = await asyncio.wait_for(http.get(url, headers={"Last-Event-ID": "-1"}), timeout=3.0)
                    response.raise_for_status()
                    body = response.text.replace("\r\n", "\n")
                    assert body.count("event: end") == 1
                    assert f'"status":"{expected_status}"' in body
                    end_frame = next(frame for frame in body.split("\n\n") if frame.startswith("event: end"))
                    end_id = next(
                        (line.removeprefix("id: ") for line in end_frame.splitlines() if line.startswith("id: ")),
                        None,
                    )
                    if end_id is not None:
                        break
                    # Database finalization precedes publication of the buffered end.
                    assert asyncio.get_running_loop().time() < deadline, "Buffered end was never published"
                    await asyncio.sleep(0.05)

                acknowledged = await asyncio.wait_for(http.get(url, headers={"Last-Event-ID": end_id}), timeout=3.0)
                acknowledged.raise_for_status()
                assert acknowledged.text.count("event: end") == 1
                assert f'"status":"{expected_status}"' in acknowledged.text
                assert "event: values" not in acknowledged.text
        elog("Finished run rejoin", {"run_id": run_id, "status": expected_status})
    finally:
        await client.threads.delete(thread_id)
        await client.assistants.delete(assistant["assistant_id"])
