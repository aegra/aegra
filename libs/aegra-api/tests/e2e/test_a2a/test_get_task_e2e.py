"""E2E: reading a task back through the real a2a-sdk client, in both dialects."""

from uuid import uuid4

import pytest
from a2a.client import A2ACardResolver, Client, ClientConfig, ClientFactory
from a2a.types import CancelTaskRequest, GetTaskRequest, Message, Part, Role, SendMessageRequest, TaskState
from a2a.utils.errors import TaskNotFoundError
from httpx import AsyncClient

from aegra_api.settings import settings
from tests.e2e._utils import elog

# A graph that answers without calling an LLM, so the round trip needs no provider key.
ASSISTANT_URL = f"{settings.app.SERVER_URL}/a2a/cron_example"


async def _sdk_client(http_client: AsyncClient, *, legacy: bool) -> Client:
    card = await A2ACardResolver(http_client, ASSISTANT_URL).get_agent_card()
    if legacy:
        # Stands in for a deployed 0.3.x client: the SDK picks its dialect from this field.
        for interface in card.supported_interfaces:
            interface.protocol_version = "0.3.0"
    return ClientFactory(ClientConfig(httpx_client=http_client, streaming=False)).create(card)


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [pytest.param(False, id="modern"), pytest.param(True, id="legacy")])
async def test_sdk_reads_back_the_task_it_just_created(legacy: bool) -> None:
    async with AsyncClient(timeout=60.0) as http_client:
        client = await _sdk_client(http_client, legacy=legacy)
        request = SendMessageRequest(
            message=Message(message_id=str(uuid4()), role=Role.ROLE_USER, parts=[Part(text="hello")])
        )
        sent = [response async for response in client.send_message(request)][0].task

        fetched = await client.get_task(GetTaskRequest(id=sent.id))
    elog("task read back", {"id": fetched.id, "state": TaskState.Name(fetched.status.state)})

    assert (fetched.id, fetched.context_id) == (sent.id, sent.context_id)
    assert fetched.status.state == TaskState.TASK_STATE_COMPLETED
    assert fetched.status.message.parts[0].text == "Task completed successfully"
    assert not fetched.artifacts


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [pytest.param(False, id="modern"), pytest.param(True, id="legacy")])
async def test_sdk_raises_its_typed_error_for_an_unknown_task(legacy: bool) -> None:
    """The client only maps -32001 to its typed error when it arrives with HTTP 200."""
    async with AsyncClient(timeout=60.0) as http_client:
        client = await _sdk_client(http_client, legacy=legacy)

        with pytest.raises(TaskNotFoundError):
            await client.get_task(GetTaskRequest(id=f"{uuid4()}:{uuid4()}"))


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [pytest.param(False, id="modern"), pytest.param(True, id="legacy")])
async def test_sdk_cancel_of_a_finished_task_is_acknowledged(legacy: bool) -> None:
    async with AsyncClient(timeout=60.0) as http_client:
        client = await _sdk_client(http_client, legacy=legacy)
        request = SendMessageRequest(
            message=Message(message_id=str(uuid4()), role=Role.ROLE_USER, parts=[Part(text="hello")])
        )
        sent = [response async for response in client.send_message(request)][0].task

        canceled = await client.cancel_task(CancelTaskRequest(id=sent.id))
    elog("task canceled", {"id": canceled.id, "text": canceled.status.message.parts[0].text})

    assert canceled.id == sent.id
    assert canceled.status.state == TaskState.TASK_STATE_CANCELED
    assert canceled.status.message.parts[0].text == "Task cancel acknowledged (was: success)"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sdk_raises_its_typed_error_when_canceling_an_unknown_task() -> None:
    async with AsyncClient(timeout=60.0) as http_client:
        client = await _sdk_client(http_client, legacy=False)

        with pytest.raises(TaskNotFoundError):
            await client.cancel_task(CancelTaskRequest(id=f"{uuid4()}:{uuid4()}"))
