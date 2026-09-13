"""Regression: AssistantCreate must not share mutable default dicts, and
AssistantUpdate must leave every field unset by default.

Pydantic v2 currently deep-copies on assignment so the historical
`Field({})` shape hasn't bitten us on create, but the pattern is brittle. These
tests pin the safe `default_factory=dict` behavior so a future revert can't
sneak shared state across instances.

AssistantUpdate is a patch document, where a default is not merely brittle but
wrong: a defaulted `graph_id` or `config` is written to the row as though the
caller had asked for it. Its fields default to None and must stay out of
`model_dump(exclude_unset=True)`, which is what the service resolves against.
"""

from aegra_api.models.assistants import AssistantCreate, AssistantUpdate


def test_assistant_create_defaults_do_not_share_state() -> None:
    a = AssistantCreate(graph_id="agent")
    b = AssistantCreate(graph_id="agent")
    assert a.config is not None
    assert a.context is not None
    assert a.metadata is not None

    a.config["x"] = 1
    a.context["y"] = 2
    a.metadata["z"] = 3

    assert b.config == {}
    assert b.context == {}
    assert b.metadata == {}


def test_assistant_create_defaults_are_distinct_instances() -> None:
    a = AssistantCreate(graph_id="agent")
    b = AssistantCreate(graph_id="agent")

    assert a.config is not b.config
    assert a.context is not b.context
    assert a.metadata is not b.metadata


def test_assistant_update_defaults_every_field_to_none() -> None:
    request = AssistantUpdate()

    assert request.name is None
    assert request.description is None
    assert request.graph_id is None
    assert request.config is None
    assert request.context is None
    assert request.metadata is None


def test_assistant_update_omits_unsent_fields() -> None:
    request = AssistantUpdate.model_validate({"name": "Renamed"})

    assert request.model_dump(exclude_unset=True) == {"name": "Renamed"}


def test_assistant_update_keeps_explicit_empties() -> None:
    """An empty dict is a request to clear, not the absence of a request."""
    request = AssistantUpdate.model_validate({"config": {}, "context": {}, "metadata": {}})

    assert request.model_dump(exclude_unset=True) == {"config": {}, "context": {}, "metadata": {}}
