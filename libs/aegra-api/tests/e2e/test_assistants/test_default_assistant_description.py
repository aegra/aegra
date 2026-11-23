from uuid import uuid5

import pytest

from aegra_api.constants import ASSISTANT_NAMESPACE_UUID
from tests.e2e._utils import elog, get_e2e_client


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("graph_id", "expected"),
    [("agent", "ReAct agent with tools"), ("agent_hitl", "Default assistant for graph 'agent_hitl'")],
)
async def test_sdk_reads_default_assistant_description(graph_id: str, expected: str) -> None:
    client = get_e2e_client()

    assistant_id = str(uuid5(ASSISTANT_NAMESPACE_UUID, graph_id))
    assistant = await client.assistants.get(assistant_id)
    versions = await client.assistants.get_versions(assistant["assistant_id"])

    assert assistant["graph_id"] == graph_id
    assert assistant["description"] == expected
    active_version = next(version for version in versions if version["version"] == assistant["version"])
    assert active_version["description"] == expected
    elog("Default assistant description", assistant)
