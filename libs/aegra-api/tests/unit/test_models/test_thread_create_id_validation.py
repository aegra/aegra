"""Validation tests for client-provided ThreadCreate.thread_id."""

import secrets
from uuid import uuid4

import pytest
from pydantic import ValidationError

from aegra_api.models.threads import MAX_THREAD_ID_LENGTH, ThreadCreate


class TestThreadCreateThreadId:
    """Provided thread_id must be non-blank and fit PostgreSQL btree keys."""

    def test_omitted_thread_id_is_none(self) -> None:
        request = ThreadCreate.model_validate({})

        assert request.thread_id is None

    def test_explicit_null_is_none(self) -> None:
        request = ThreadCreate.model_validate({"thread_id": None})

        assert request.thread_id is None

    def test_accepts_uuid(self) -> None:
        thread_id = str(uuid4())
        request = ThreadCreate.model_validate({"thread_id": thread_id})

        assert request.thread_id == thread_id

    def test_accepts_max_length(self) -> None:
        thread_id = "a" * MAX_THREAD_ID_LENGTH
        request = ThreadCreate.model_validate({"thread_id": thread_id})

        assert request.thread_id == thread_id

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValidationError):
            ThreadCreate.model_validate({"thread_id": ""})

    def test_rejects_blank(self) -> None:
        with pytest.raises(ValidationError):
            ThreadCreate.model_validate({"thread_id": "   "})

    def test_rejects_oversized_random_id(self) -> None:
        with pytest.raises(ValidationError):
            ThreadCreate.model_validate({"thread_id": secrets.token_hex(2500)})

    def test_rejects_one_over_max_length(self) -> None:
        with pytest.raises(ValidationError):
            ThreadCreate.model_validate({"thread_id": "a" * (MAX_THREAD_ID_LENGTH + 1)})
