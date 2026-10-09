from collections.abc import AsyncIterator
from unittest.mock import MagicMock
from uuid import uuid5

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.constants import ASSISTANT_NAMESPACE_UUID
from aegra_api.core.orm import Assistant, AssistantVersion
from aegra_api.services import langgraph_service
from aegra_api.services.langgraph_service import LangGraphService


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_session = MagicMock(spec=AsyncSession)

    async def get_session() -> AsyncIterator[AsyncSession]:
        yield mock_session

    monkeypatch.setattr(langgraph_service, "get_session", get_session)
    return mock_session


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("./agent.py:graph", "Default assistant for graph 'agent'"),
        ({"path": "./agent.py:graph"}, "Default assistant for graph 'agent'"),
        ({"path": "./agent.py:graph", "description": None}, "Default assistant for graph 'agent'"),
        ({"path": "./agent.py:graph", "description": "Configured description"}, "Configured description"),
        ({"path": "./agent.py:graph", "description": ""}, ""),
    ],
)
async def test_creates_default_assistant_and_version_with_description(
    session: MagicMock, entry: object, expected: str
) -> None:
    service = LangGraphService()
    service.config = {"graphs": {"agent": entry}}
    service._load_graph_registry()
    session.scalar.return_value = None

    await service._ensure_default_assistants()

    assistant, version = [call.args[0] for call in session.add.call_args_list]
    assert isinstance(assistant, Assistant)
    assert isinstance(version, AssistantVersion)
    assert assistant.assistant_id == version.assistant_id == str(uuid5(ASSISTANT_NAMESPACE_UUID, "agent"))
    assert assistant.description == version.description == expected
    session.commit.assert_awaited_once()
    session.close.assert_awaited_once()


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ({"path": "./agent.py:graph", "description": "New description"}, "New description"),
        ({"path": "./agent.py:graph", "description": ""}, ""),
        ({"path": "./agent.py:graph", "description": None}, "Default assistant for graph 'agent'"),
        ({"path": "./agent.py:graph"}, "Default assistant for graph 'agent'"),
        ("./agent.py:graph", "Default assistant for graph 'agent'"),
    ],
)
async def test_syncs_changed_description_without_changing_other_assistant_fields(
    session: MagicMock, entry: object, expected: str
) -> None:
    assistant_id = str(uuid5(ASSISTANT_NAMESPACE_UUID, "agent"))
    existing = Assistant(
        assistant_id=assistant_id,
        graph_id="agent",
        name="Custom name",
        description="Old description",
        version=3,
        config={"configurable": {"temperature": 0.5}},
        context={"temperature": 0.5},
        metadata_dict={"created_by": "system", "custom": True},
        user_id="system",
    )
    service = LangGraphService()
    service.config = {"graphs": {"agent": entry}}
    service._load_graph_registry()
    session.scalar.side_effect = [existing, 5]

    await service._ensure_default_assistants()

    assert existing.description == expected
    assert existing.name == "Custom name"
    assert existing.version == 6
    assert existing.config == {"configurable": {"temperature": 0.5}}
    assert existing.metadata_dict == {"created_by": "system", "custom": True}
    session.add.assert_called_once()
    version = session.add.call_args.args[0]
    assert isinstance(version, AssistantVersion)
    assert version.assistant_id == assistant_id
    assert version.version == 6
    assert version.description == expected
    assert version.name == existing.name
    assert version.graph_id == existing.graph_id
    assert version.config == existing.config
    assert version.context == existing.context
    assert version.metadata_dict == existing.metadata_dict
    assert version.created_at == existing.updated_at
    session.execute.assert_not_awaited()
    session.commit.assert_awaited_once()
    session.close.assert_awaited_once()


async def test_sync_allocates_version_when_history_is_missing(session: MagicMock) -> None:
    service = LangGraphService()
    service.config = {"graphs": {"agent": {"path": "./agent.py:graph", "description": "New"}}}
    service._load_graph_registry()
    existing = Assistant(assistant_id="agent", graph_id="agent", description="Old", version=3)
    session.scalar.side_effect = [existing, None]

    await service._ensure_default_assistants()

    assert existing.version == 4
    assert session.add.call_args.args[0].version == 4


async def test_unchanged_description_does_not_rewrite_version(session: MagicMock) -> None:
    service = LangGraphService()
    service.config = {"graphs": {"agent": {"path": "./agent.py:graph", "description": "Same"}}}
    service._load_graph_registry()
    session.scalar.return_value = Assistant(description="Same", version=1)

    await service._ensure_default_assistants()

    session.add.assert_not_called()
    session.execute.assert_not_awaited()
    session.commit.assert_awaited_once()


async def test_closes_session_when_commit_fails(session: MagicMock) -> None:
    service = LangGraphService()
    session.commit.side_effect = RuntimeError("Commit failed")

    with pytest.raises(RuntimeError, match="Commit failed"):
        await service._ensure_default_assistants()

    session.close.assert_awaited_once()
