from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid5

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.api.assistants import router
from aegra_api.constants import ASSISTANT_NAMESPACE_UUID
from aegra_api.core.orm import Assistant, get_session
from aegra_api.services import langgraph_service
from aegra_api.services.langgraph_service import LangGraphService, get_langgraph_service
from tests.fixtures.clients import create_test_app, make_client


@pytest.mark.parametrize("description", ["Updated configured description", "", None])
async def test_http_reads_description_after_startup_sync(
    monkeypatch: pytest.MonkeyPatch, description: str | None
) -> None:
    assistant_id = str(uuid5(ASSISTANT_NAMESPACE_UUID, "agent"))
    existing = Assistant(
        assistant_id=assistant_id,
        graph_id="agent",
        name="agent",
        description="Original description",
        config={},
        context={},
        metadata_dict={"created_by": "system"},
        user_id="system",
        version=1,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session = MagicMock(spec=AsyncSession)
    session.scalar.side_effect = [existing, 1, existing]

    async def session_dependency() -> AsyncIterator[AsyncSession]:
        yield session

    monkeypatch.setattr(langgraph_service, "get_session", session_dependency)
    service = LangGraphService()
    service.config = {"graphs": {"agent": {"path": "./agent.py:graph", "description": description}}}
    service._load_graph_registry()
    app = create_test_app(include_runs=False, include_threads=False)
    app.include_router(router)
    app.dependency_overrides[get_session] = session_dependency
    app.dependency_overrides[get_langgraph_service] = lambda: service

    await service._ensure_default_assistants()
    with make_client(app) as client:
        response = client.get(f"/assistants/{assistant_id}")

    assert response.status_code == 200
    assert response.json()["description"] == (
        description if description is not None else "Default assistant for graph 'agent'"
    )
    assert response.json()["user_id"] == "system"
    assert response.json()["version"] == 2
