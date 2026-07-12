"""Unit tests for ThreadState binary and arbitrary-type serialization (issue #451).

Verifies that the JSON-only field serializer on ThreadState produces wire
output matching langgraph-api's serde convention for bytes, models,
dataclasses, NamedTuples, sets, and all other supported types. Expected
values are derived from the executed reference orjson invocation, not from
the production helper.
"""

import json
from base64 import b64encode
from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Any, NamedTuple
from uuid import UUID
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, TypeAdapter

from aegra_api.models.threads import ThreadCheckpoint, ThreadState

NON_UTF8 = b"\x89PNG\r\n\x1a\n\xff\xfe"
ALPHABET_DISC = b"\xfb\xff"
VALID_UTF8 = b"hello"
NON_UTF8_B64 = b64encode(NON_UTF8).decode("ascii")
ALPHABET_B64 = b64encode(ALPHABET_DISC).decode("ascii")
VALID_UTF8_B64 = b64encode(VALID_UTF8).decode("ascii")


def _make_state(**overrides: Any) -> ThreadState:
    defaults: dict[str, Any] = {
        "values": {},
        "next": [],
        "tasks": [],
        "interrupts": [],
        "metadata": {},
        "checkpoint": ThreadCheckpoint(checkpoint_id="c1", thread_id="t1", checkpoint_ns=""),
    }
    defaults.update(overrides)
    return ThreadState(**defaults)


def _dump_list(state: ThreadState) -> dict:
    """Serialize via list adapter (history endpoint path) and parse."""
    return json.loads(TypeAdapter(list[ThreadState]).dump_json([state]))[0]


def _dump_scalar(state: ThreadState) -> dict:
    """Serialize via scalar adapter and parse."""
    return json.loads(TypeAdapter(ThreadState).dump_json(state))


class TestNonUtf8Bytes:
    """Non-UTF-8 bytes no longer raise during JSON serialization."""

    def test_non_utf8_bytes_in_values_via_list_adapter(self):
        state = _make_state(values={"blob": NON_UTF8})
        result = _dump_list(state)
        assert result["values"]["blob"] == NON_UTF8_B64

    def test_non_utf8_bytes_in_values_via_scalar_adapter(self):
        state = _make_state(values={"blob": NON_UTF8})
        result = _dump_scalar(state)
        assert result["values"]["blob"] == NON_UTF8_B64


class TestNoDoubleEncoding:
    """The field serializer returns a parsed Python structure, not a JSON string."""

    def test_values_is_dict_not_string(self):
        state = _make_state(values={"blob": NON_UTF8})
        payload = json.loads(TypeAdapter(list[ThreadState]).dump_json([state]))
        assert isinstance(payload[0]["values"], dict)

    def test_metadata_is_dict_not_string(self):
        state = _make_state(metadata={"snapshot": NON_UTF8})
        payload = json.loads(TypeAdapter(list[ThreadState]).dump_json([state]))
        assert isinstance(payload[0]["metadata"], dict)


class TestValidUtf8BytesAreBase64:
    """Valid UTF-8 bytes are Base64, not plain text."""

    def test_valid_utf8_bytes_encoded_as_base64(self):
        state = _make_state(values={"text": VALID_UTF8})
        result = _dump_list(state)
        assert result["values"]["text"] == VALID_UTF8_B64
        assert result["values"]["text"] != "hello"


class TestStandardBase64Alphabet:
    """b'\\xfb\\xff' produces exactly '+/8=' (standard, not URL-safe)."""

    def test_alphabet_discriminator_is_standard(self):
        state = _make_state(values={"disc": ALPHABET_DISC})
        result = _dump_list(state)
        assert result["values"]["disc"] == "+/8="

    def test_alphabet_discriminator_is_not_url_safe(self):
        state = _make_state(values={"disc": ALPHABET_DISC})
        result = _dump_list(state)
        assert result["values"]["disc"] != "-_8="


class TestBytearrayParity:
    """bytearray behaves identically to bytes."""

    def test_bytearray_produces_same_base64_as_bytes(self):
        state_bytes = _make_state(values={"data": bytes(ALPHABET_DISC)})
        state_bytearray = _make_state(values={"data": bytearray(ALPHABET_DISC)})
        assert _dump_list(state_bytes)["values"]["data"] == ALPHABET_B64
        assert _dump_list(state_bytearray)["values"]["data"] == ALPHABET_B64


class TestSetAndFrozenset:
    """set/frozenset with bytes are serialized as JSON arrays with Base64."""

    @pytest.mark.parametrize("container_type", [set, frozenset])
    def test_set_like_containers_with_bytes(self, container_type):
        original = container_type({ALPHABET_DISC})
        state = _make_state(values={"items": original})
        result = _dump_list(state)
        assert ALPHABET_B64 in result["values"]["items"]
        assert state.values["items"] == original


class TestDeque:
    """deque with bytes are serialized as JSON arrays with Base64."""

    def test_deque_with_bytes(self):
        state = _make_state(values={"items": deque([ALPHABET_DISC])})
        result = _dump_list(state)
        assert result["values"]["items"] == [ALPHABET_B64]


class TestNamedTupleAsObject:
    """NamedTuple serializes as a JSON object (matching orjson/reference), not an array."""

    class _Payload(NamedTuple):
        name: str
        blob: bytes

    def test_named_tuple_serializes_as_object(self):
        state = _make_state(values={"nt": self._Payload(name="example", blob=ALPHABET_DISC)})
        result = _dump_list(state)
        assert isinstance(result["values"]["nt"], dict)
        assert result["values"]["nt"]["name"] == "example"
        assert result["values"]["nt"]["blob"] == ALPHABET_B64


class TestDataclassAsObject:
    """dataclass serializes as a JSON object (orjson handles natively)."""

    @dataclass
    class _DCPayload:
        blob: bytes
        name: str

    def test_dataclass_serializes_as_object(self):
        state = _make_state(values={"dc": self._DCPayload(blob=ALPHABET_DISC, name="test")})
        result = _dump_list(state)
        assert isinstance(result["values"]["dc"], dict)
        assert result["values"]["dc"]["blob"] == ALPHABET_B64
        assert result["values"]["dc"]["name"] == "test"


class TestNestedPydanticModel:
    """Nested Pydantic model containing bytes serializes correctly."""

    class _BinaryModel(BaseModel):
        blob: bytes

    def test_nested_pydantic_model_with_bytes(self):
        state = _make_state(values={"model": self._BinaryModel(blob=ALPHABET_DISC)})
        result = _dump_list(state)
        assert result["values"]["model"]["blob"] == ALPHABET_B64


class TestNonFiniteFloats:
    """NaN, Infinity, and -Infinity become null (matching orjson reference)."""

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_floats_become_null(self, value):
        state = _make_state(values={"val": value})
        result = _dump_list(state)
        assert result["values"]["val"] is None


class TestUnknownObjectBecomesNull:
    """Unknown objects become null (matching reference default returning None)."""

    class _Unsupported:
        pass

    def test_unknown_object_becomes_null(self):
        state = _make_state(values={"u": self._Unsupported()})
        result = _dump_list(state)
        assert result["values"]["u"] is None


class TestDictionaryKeys:
    """Dictionary key behavior with OPT_NON_STR_KEYS (derived from reference)."""

    def test_uuid_key_supported(self):
        uid = UUID(int=1)
        state = _make_state(values={"data": {uid: "val"}})
        result = _dump_list(state)
        assert result["values"]["data"]["00000000-0000-0000-0000-000000000001"] == "val"

    def test_int_key_supported(self):
        state = _make_state(values={"data": {42: "val"}})
        result = _dump_list(state)
        assert result["values"]["data"]["42"] == "val"

    def test_str_key_supported(self):
        state = _make_state(values={"data": {"key": "val"}})
        result = _dump_list(state)
        assert result["values"]["data"]["key"] == "val"

    def test_enum_key_supported(self):
        class Color(Enum):
            RED = "red"

        state = _make_state(values={"data": {Color.RED: "val"}})
        result = _dump_list(state)
        assert result["values"]["data"]["red"] == "val"


class TestDatetimeAndTimezoneFormatting:
    """datetime and timezone formatting match reference output."""

    def test_datetime_with_zoneinfo(self):
        dt = type("dt", (), {})  # placeholder - use actual datetime
        from datetime import datetime

        dt = datetime(2024, 1, 15, 12, 30, 45, tzinfo=ZoneInfo("America/New_York"))
        state = _make_state(values={"dt": dt})
        result = _dump_list(state)
        assert result["values"]["dt"] == "2024-01-15T12:30:45-05:00"

    def test_naive_datetime(self):
        from datetime import datetime

        dt = datetime(2024, 1, 15, 12, 30, 45)
        state = _make_state(values={"dt": dt})
        result = _dump_list(state)
        assert result["values"]["dt"] == "2024-01-15T12:30:45"

    def test_timedelta(self):
        state = _make_state(values={"td": timedelta(seconds=30)})
        result = _dump_list(state)
        assert result["values"]["td"] == 30.0

    def test_zoneinfo_becomes_null(self):
        state = _make_state(values={"tz": ZoneInfo("America/New_York")})
        result = _dump_list(state)
        assert result["values"]["tz"] is None


class TestPythonModeRetainsRawValues:
    """Python-mode dumps retain raw bytes and original types."""

    def test_model_dump_python_retains_bytes(self):
        state = _make_state(values={"blob": NON_UTF8, "text": VALID_UTF8})
        dumped = state.model_dump(mode="python")
        assert dumped["values"]["blob"] == NON_UTF8
        assert dumped["values"]["text"] == VALID_UTF8

    def test_model_dump_python_retains_bytearray(self):
        state = _make_state(values={"data": bytearray(ALPHABET_DISC)})
        dumped = state.model_dump(mode="python")
        assert dumped["values"]["data"] == bytearray(ALPHABET_DISC)

    def test_model_dump_python_retains_set(self):
        state = _make_state(values={"items": {ALPHABET_DISC}})
        dumped = state.model_dump(mode="python")
        assert dumped["values"]["items"] == {ALPHABET_DISC}
        assert isinstance(dumped["values"]["items"], set)

    def test_model_dump_json_encodes_bytes(self):
        state = _make_state(values={"blob": NON_UTF8})
        dumped = state.model_dump(mode="json")
        assert dumped["values"]["blob"] == NON_UTF8_B64


class TestOriginalContainersUnchanged:
    """Original containers are not mutated by JSON serialization."""

    def test_values_dict_not_mutated(self):
        original = {"blob": NON_UTF8, "nested": ["a", b"\xff"]}
        state = _make_state(values=original)
        _dump_list(state)
        assert state.values["blob"] == NON_UTF8
        assert state.values["nested"][1] == b"\xff"
        assert original["blob"] == NON_UTF8

    def test_metadata_dict_not_mutated(self):
        state = _make_state(metadata={"snapshot": NON_UTF8})
        _dump_list(state)
        assert state.metadata["snapshot"] == NON_UTF8

    def test_set_not_mutated(self):
        original = {ALPHABET_DISC}
        state = _make_state(values={"items": original})
        _dump_list(state)
        assert state.values["items"] == original


class TestAllFourArbitraryFields:
    """All four arbitrary fields have direct model-level coverage."""

    def test_values_field(self):
        state = _make_state(values={"blob": NON_UTF8})
        assert _dump_list(state)["values"]["blob"] == NON_UTF8_B64

    def test_metadata_field(self):
        state = _make_state(metadata={"snapshot": NON_UTF8})
        assert _dump_list(state)["metadata"]["snapshot"] == NON_UTF8_B64

    def test_tasks_field(self):
        state = _make_state(tasks=[{"id": "t1", "payload": NON_UTF8}])
        assert _dump_list(state)["tasks"][0]["payload"] == NON_UTF8_B64

    def test_interrupts_field(self):
        state = _make_state(interrupts=[{"value": NON_UTF8, "id": "i1"}])
        assert _dump_list(state)["interrupts"][0]["value"] == NON_UTF8_B64


class TestNestedContainers:
    """Bytes in nested dict/list/tuple/set structures are encoded."""

    def test_bytes_in_nested_list(self):
        state = _make_state(values={"items": ["text", NON_UTF8, 42]})
        result = _dump_list(state)
        assert result["values"]["items"][1] == NON_UTF8_B64
        assert result["values"]["items"][0] == "text"
        assert result["values"]["items"][2] == 42

    def test_bytes_in_nested_dict(self):
        state = _make_state(values={"outer": {"inner": NON_UTF8, "keep": "str"}})
        result = _dump_list(state)
        assert result["values"]["outer"]["inner"] == NON_UTF8_B64
        assert result["values"]["outer"]["keep"] == "str"

    def test_deeply_nested_structure(self):
        state = _make_state(values={"deep": [{"level": [NON_UTF8, [VALID_UTF8]]}]})
        result = _dump_list(state)
        assert result["values"]["deep"][0]["level"][0] == NON_UTF8_B64
        assert result["values"]["deep"][0]["level"][1][0] == VALID_UTF8_B64


class TestSubgraphStateEncoding:
    """Nested subgraph ThreadState in tasks[*].state is encoded."""

    def test_subgraph_state_bytes_encoded(self):
        subgraph = _make_state(values={"sub_blob": NON_UTF8})
        parent = _make_state(tasks=[{"id": "t1", "state": subgraph}])
        result = _dump_list(parent)
        assert result["tasks"][0]["state"]["values"]["sub_blob"] == NON_UTF8_B64


class TestOrdinaryValuesUnchanged:
    """Ordinary strings, numbers, bools, and None are not affected."""

    def test_ordinary_values_unchanged(self):
        state = _make_state(values={"msg": "hello", "num": 42, "flag": True, "none": None})
        result = _dump_list(state)
        assert result["values"]["msg"] == "hello"
        assert result["values"]["num"] == 42
        assert result["values"]["flag"] is True
        assert result["values"]["none"] is None


class TestScalarAndListAdapter:
    """Both scalar and list adapter serialization paths work."""

    def test_scalar_adapter(self):
        state = _make_state(values={"blob": NON_UTF8})
        assert _dump_scalar(state)["values"]["blob"] == NON_UTF8_B64

    def test_list_adapter(self):
        state1 = _make_state(values={"blob": NON_UTF8})
        state2 = _make_state(values={"disc": ALPHABET_DISC})
        payload = json.loads(TypeAdapter(list[ThreadState]).dump_json([state1, state2]))
        assert payload[0]["values"]["blob"] == NON_UTF8_B64
        assert payload[1]["values"]["disc"] == ALPHABET_B64


class TestEdgeCases:
    """Empty and no-bytes cases."""

    def test_empty_values(self):
        state = _make_state()
        result = _dump_list(state)
        assert result["values"] == {}

    def test_values_without_bytes_unchanged(self):
        state = _make_state(values={"messages": ["hello", "world"], "count": 3})
        result = _dump_list(state)
        assert result["values"]["messages"] == ["hello", "world"]
        assert result["values"]["count"] == 3
