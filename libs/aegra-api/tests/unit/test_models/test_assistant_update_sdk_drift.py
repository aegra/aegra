"""Pin AssistantUpdate to what ``langgraph_sdk`` actually PATCHes.

The SDK builds a sparse body: a rename sends ``{"name": ...}`` and nothing
else, and documents metadata as merged rather than replaced. Driving the real
client here means an SDK that starts sending a field (or stops omitting one)
fails this test instead of silently changing what the server writes.
"""

from typing import Any

import pytest
from langgraph_sdk.client import AssistantsClient

from aegra_api.models.assistants import AssistantUpdate


class RecordingHttp:
    """Stands in for the SDK's HttpClient, capturing the request body."""

    def __init__(self) -> None:
        self.path: str | None = None
        self.payload: dict[str, Any] | None = None

    async def patch(self, path: str, *, json: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.path = path
        self.payload = json
        return {}


@pytest.fixture
def http() -> RecordingHttp:
    return RecordingHttp()


@pytest.fixture
def assistants(http: RecordingHttp) -> AssistantsClient:
    return AssistantsClient(http)


async def test_sdk_rename_sends_only_the_name(http: RecordingHttp, assistants: AssistantsClient) -> None:
    await assistants.update("asst-1", name="Renamed")

    assert http.path == "/assistants/asst-1"
    assert http.payload == {"name": "Renamed"}


async def test_sdk_rename_leaves_every_other_field_unset(http: RecordingHttp, assistants: AssistantsClient) -> None:
    """The server must not be able to tell a rename apart from a rename plus
    'and reset everything else', which is what a defaulted field would mean."""
    await assistants.update("asst-1", name="Renamed")

    request = AssistantUpdate.model_validate(http.payload)

    assert request.model_dump(exclude_unset=True) == {"name": "Renamed"}
    assert request.graph_id is None
    assert request.config is None
    assert request.context is None
    assert request.metadata is None


async def test_every_sdk_field_is_declared_on_the_model(http: RecordingHttp, assistants: AssistantsClient) -> None:
    await assistants.update(
        "asst-1",
        graph_id="other-graph",
        config={"recursion_limit": 7},
        context={"model_name": "anthropic"},
        metadata={"number": 2},
        name="Renamed",
        description="Described",
    )

    assert http.payload is not None
    assert set(http.payload) <= set(AssistantUpdate.model_fields)
    assert AssistantUpdate.model_validate(http.payload).model_dump(exclude_unset=True) == http.payload
