"""Validation tests for the per-thread TTL request models."""

import pytest
from pydantic import ValidationError

from aegra_api.models import ThreadCreate, ThreadTTLSpec


class TestThreadTTLSpec:
    """ThreadTTLSpec accepts both spec and SDK key spellings."""

    def test_accepts_default_ttl_key(self) -> None:
        spec = ThreadTTLSpec.model_validate({"default_ttl": 1440, "strategy": "delete"})

        assert spec.default_ttl == 1440
        assert spec.strategy == "delete"

    def test_accepts_sdk_ttl_key(self) -> None:
        """langgraph-sdk sends the minutes value under 'ttl', not 'default_ttl'."""
        spec = ThreadTTLSpec.model_validate({"ttl": 60, "strategy": "keep_latest"})

        assert spec.default_ttl == 60
        assert spec.strategy == "keep_latest"

    def test_all_fields_optional(self) -> None:
        spec = ThreadTTLSpec.model_validate({})

        assert spec.default_ttl is None
        assert spec.strategy is None

    @pytest.mark.parametrize("value", [0, -1, -0.5])
    def test_rejects_non_positive_ttl(self, value: float) -> None:
        with pytest.raises(ValidationError):
            ThreadTTLSpec.model_validate({"default_ttl": value})

    def test_rejects_unknown_strategy(self) -> None:
        with pytest.raises(ValidationError):
            ThreadTTLSpec.model_validate({"strategy": "purge"})


class TestThreadCreateTTL:
    """ThreadCreate carries an optional nested ttl object."""

    def test_defaults_to_no_ttl(self) -> None:
        request = ThreadCreate.model_validate({})

        assert request.ttl is None

    def test_nested_ttl_parsed(self) -> None:
        request = ThreadCreate.model_validate({"ttl": {"ttl": 5, "strategy": "delete"}})

        assert request.ttl is not None
        assert request.ttl.default_ttl == 5
        assert request.ttl.strategy == "delete"

    def test_invalid_nested_ttl_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ThreadCreate.model_validate({"ttl": {"default_ttl": 0}})
