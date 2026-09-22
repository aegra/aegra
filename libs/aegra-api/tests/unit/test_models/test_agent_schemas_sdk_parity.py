"""``AgentSchemas`` is the wire shape of ``GET /assistants/{id}/schemas``.

The service derives six values and answers ``None`` for any it could not
produce. FastAPI serialises through this model, so a field it does not declare
is dropped and a ``None`` in a non-nullable field fails response validation and
becomes a 500 — both silent, since the service-level tests never see the wire.
"""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph_sdk.schema import GraphSchema

from aegra_api.models.assistants import AgentSchemas

_DERIVED: dict[str, Any] = {
    "graph_id": "agent",
    "input_schema": {"type": "object"},
    "output_schema": {"type": "object"},
    "state_schema": {"type": "object"},
    "config_schema": {"type": "object"},
    "context_schema": {"properties": {"prompt_version": {"type": "integer"}}},
}


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    served: dict[str, Any] = {}

    @app.get("/schemas", response_model=AgentSchemas)
    async def schemas() -> dict[str, Any]:
        return served

    app.state.served = served
    return TestClient(app, raise_server_exceptions=False)


def test_the_model_declares_exactly_what_the_sdk_does() -> None:
    """Drift either way should fail here rather than silently drop a field."""
    assert set(AgentSchemas.model_fields) == set(GraphSchema.__annotations__)


def test_every_derived_value_reaches_the_client(client: TestClient) -> None:
    client.app.state.served.update(_DERIVED)

    body = client.get("/schemas").json()

    assert body == _DERIVED


def test_a_schema_that_could_not_be_derived_serialises_as_null(client: TestClient) -> None:
    """The service answers None by design; that must not become a 500."""
    client.app.state.served.update({**_DERIVED, "input_schema": None, "context_schema": None})

    response = client.get("/schemas")

    assert response.status_code == 200
    assert response.json()["input_schema"] is None
    assert response.json()["context_schema"] is None


def test_every_schema_may_be_null_at_once(client: TestClient) -> None:
    """``_extract_graph_schemas`` returns all-None when the graph cannot be introspected."""
    client.app.state.served.update({key: None for key in _DERIVED if key != "graph_id"})
    client.app.state.served["graph_id"] = "agent"

    assert client.get("/schemas").status_code == 200


def test_graph_id_is_still_required() -> None:
    """Nullable schemas must not make the identifier optional too."""
    with pytest.raises(ValueError):
        AgentSchemas.model_validate({key: value for key, value in _DERIVED.items() if key != "graph_id"})
