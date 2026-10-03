"""Validation tests for client-provided AssistantCreate.assistant_id."""

import secrets
from uuid import uuid4

import pytest
from pydantic import ValidationError

from aegra_api.models.assistants import AssistantCreate
from aegra_api.models.entity_ids import MAX_ENTITY_ID_LENGTH


def _payload(**overrides: object) -> dict[str, object]:
    return {"graph_id": "agent", **overrides}


class TestAssistantCreateAssistantId:
    """Provided assistant_id must be non-blank and fit PostgreSQL btree keys."""

    def test_omitted_assistant_id_is_none(self) -> None:
        request = AssistantCreate.model_validate(_payload())

        assert request.assistant_id is None

    def test_explicit_null_is_none(self) -> None:
        request = AssistantCreate.model_validate(_payload(assistant_id=None))

        assert request.assistant_id is None

    def test_accepts_uuid(self) -> None:
        assistant_id = str(uuid4())
        request = AssistantCreate.model_validate(_payload(assistant_id=assistant_id))

        assert request.assistant_id == assistant_id

    def test_accepts_max_length(self) -> None:
        assistant_id = "a" * MAX_ENTITY_ID_LENGTH
        request = AssistantCreate.model_validate(_payload(assistant_id=assistant_id))

        assert request.assistant_id == assistant_id

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValidationError):
            AssistantCreate.model_validate(_payload(assistant_id=""))

    def test_rejects_blank(self) -> None:
        with pytest.raises(ValidationError):
            AssistantCreate.model_validate(_payload(assistant_id="   "))

    def test_rejects_oversized_random_id(self) -> None:
        with pytest.raises(ValidationError):
            AssistantCreate.model_validate(_payload(assistant_id=secrets.token_hex(2500)))

    def test_rejects_one_over_max_length(self) -> None:
        with pytest.raises(ValidationError):
            AssistantCreate.model_validate(_payload(assistant_id="a" * (MAX_ENTITY_ID_LENGTH + 1)))
