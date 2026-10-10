"""E2E: an A2A send round trip through the real a2a-sdk client, in both dialects."""

import json
from uuid import uuid4

import pytest
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, SendMessageRequest, StreamResponse, TaskState
from httpx import AsyncClient, Request

from aegra_api.settings import settings
from tests.e2e._utils import elog

# A graph that answers without calling an LLM, so the round trip needs no provider key.
ASSISTANT_ID = "cron_example"
ASSISTANT_URL = f"{settings.app.SERVER_URL}/a2a/{ASSISTANT_ID}"


async def _send_through_sdk(*, legacy: bool) -> tuple[list[str], StreamResponse]:
    """Send one message with the SDK; return the JSON-RPC methods it used and its parsed response."""
    methods: list[str] = []

    async def record_method(request: Request) -> None:
        if request.method == "POST":
            methods.append(json.loads(request.content)["method"])

    async with AsyncClient(timeout=60.0, event_hooks={"request": [record_method]}) as http_client:
        card = await A2ACardResolver(http_client, ASSISTANT_URL).get_agent_card()
        if legacy:
            # Stands in for a deployed 0.3.x client: the SDK picks its dialect from this field.
            for interface in card.supported_interfaces:
                interface.protocol_version = "0.3.0"
        client = ClientFactory(ClientConfig(httpx_client=http_client, streaming=False)).create(card)
        request = SendMessageRequest(
            message=Message(message_id=str(uuid4()), role=Role.ROLE_USER, parts=[Part(text="hello")])
        )
        responses = [response async for response in client.send_message(request)]

    assert len(responses) == 1
    return methods, responses[0]


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("legacy", "method"),
    [pytest.param(False, "SendMessage", id="modern"), pytest.param(True, "message/send", id="legacy")],
)
async def test_sdk_send_round_trip_returns_a_completed_task(legacy: bool, method: str) -> None:
    """The SDK refuses a response in the wrong dialect, so parsing it at all proves the shape."""
    methods, response = await _send_through_sdk(legacy=legacy)
    task = response.task
    elog("task via a2a-sdk", {"methods": methods, "id": task.id, "state": TaskState.Name(task.status.state)})

    assert methods == [method]
    assert task.status.state == TaskState.TASK_STATE_COMPLETED
    assert task.id.startswith(f"{task.context_id}:")
    assert [artifact.name for artifact in task.artifacts] == ["Assistant Response"]
    assert task.artifacts[0].parts[0].text


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_a2a_context_and_task_are_a_thread_and_run_in_the_langgraph_api() -> None:
    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=60.0) as http_client:
        first = await _post_send(http_client, context_id=None)
        context_id = first["contextId"]
        second = await _post_send(http_client, context_id=context_id)

        thread = await http_client.get(f"/threads/{context_id}")
        runs = await http_client.get(f"/threads/{context_id}/runs")
    elog("a2a thread", {"context": context_id, "runs": [run["run_id"] for run in runs.json()]})

    assert second["contextId"] == context_id
    assert second["id"] != first["id"]
    assert thread.status_code == 200
    assert {f"{context_id}:{run['run_id']}" for run in runs.json()} == {first["id"], second["id"]}


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_send_accepts_a_context_id_that_is_not_a_uuid() -> None:
    """The conformance suite sends non-UUID context ids; thread ids here are not required to be UUIDs."""
    context_id = f"a2a-context-{uuid4().hex[:8]}"

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=60.0) as http_client:
        task = await _post_send(http_client, context_id=context_id)

    assert task["contextId"] == context_id
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"


async def _post_send(http_client: AsyncClient, *, context_id: str | None) -> dict:
    message: dict = {"messageId": str(uuid4()), "role": "ROLE_USER", "parts": [{"text": "hello"}]}
    if context_id is not None:
        message["contextId"] = context_id
    resp = await http_client.post(
        f"/a2a/{ASSISTANT_ID}",
        json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {"message": message}},
    )
    assert resp.status_code == 200
    return resp.json()["result"]["task"]
