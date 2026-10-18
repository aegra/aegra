"""Tests for converting A2A message parts into graph input."""

from typing import Any

import pytest

from aegra_api.services.a2a.jsonrpc import JsonRpcError, JsonRpcErrorCode
from aegra_api.services.a2a.parts import ConvertedParts, convert_parts_to_graph_input

NOT_AN_OBJECT_ERROR = "Each message part must be an object. A2A agents support 'text', 'data', and 'file' parts only."
UNSUPPORTED_ERROR = "Unsupported part type. A2A agents support 'text', 'data', and 'file' parts only."
FILE_ERROR = "File parts are not supported yet. Send 'text' or 'data' parts."
DATA_ERROR = "DataPart must contain a JSON object in the 'data' field"


def _multiple(keys: str) -> str:
    return f"Each message part must contain exactly one of 'text', 'file', or 'data'. Got multiple: {keys}."


def _convert(parts: list[Any], role: str = "ROLE_USER") -> ConvertedParts:
    return convert_parts_to_graph_input(parts, "msg-1", role)


@pytest.mark.parametrize(
    "part",
    [
        pytest.param({"text": "hi"}, id="modern"),
        pytest.param({"kind": "text", "text": "hi"}, id="legacy"),
    ],
)
def test_text_part_becomes_a_message_in_either_dialect(part: dict[str, Any]) -> None:
    """Parts are classified by key presence, so the legacy ``kind`` label changes nothing."""
    assert _convert([part]) == ConvertedParts(
        graph_input={"messages": [{"role": "user", "content": "hi", "id": "msg-1"}]},
        has_conversational_content=True,
    )


def test_each_text_part_becomes_its_own_message_sharing_the_message_id() -> None:
    converted = _convert([{"text": "first"}, {"text": "second"}])

    assert converted.graph_input == {
        "messages": [
            {"role": "user", "content": "first", "id": "msg-1"},
            {"role": "user", "content": "second", "id": "msg-1"},
        ]
    }


@pytest.mark.parametrize(
    ("role", "message_role"),
    [
        ("ROLE_USER", "user"),
        ("user", "user"),
        ("ROLE_AGENT", "assistant"),
        ("agent", "assistant"),
        ("wizard", "assistant"),
    ],
)
def test_any_role_that_is_not_user_becomes_an_assistant_message(role: str, message_role: str) -> None:
    assert _convert([{"text": "hi"}], role).graph_input["messages"][0]["role"] == message_role


def test_data_part_merges_into_the_input_without_producing_a_message() -> None:
    assert _convert([{"data": {"city": "Paris"}}]) == ConvertedParts(
        graph_input={"city": "Paris"}, has_conversational_content=False
    )


def test_later_data_parts_overwrite_earlier_keys() -> None:
    assert _convert([{"data": {"k": 1, "kept": True}}, {"data": {"k": 2}}]).graph_input == {"k": 2, "kept": True}


def test_text_and_data_parts_sit_side_by_side() -> None:
    converted = _convert([{"text": "weather?"}, {"data": {"city": "Paris"}}])

    assert converted.graph_input == {
        "messages": [{"role": "user", "content": "weather?", "id": "msg-1"}],
        "city": "Paris",
    }


@pytest.mark.parametrize(
    "parts",
    [
        pytest.param([{"text": "hi"}, {"data": {"messages": []}}], id="data-after-text"),
        pytest.param([{"data": {"messages": []}}, {"text": "hi"}], id="data-before-text"),
    ],
)
def test_data_key_named_messages_overrides_the_converted_messages(parts: list[Any]) -> None:
    """The text still counts as conversational content even though its message was replaced."""
    assert _convert(parts) == ConvertedParts(graph_input={"messages": []}, has_conversational_content=True)


@pytest.mark.parametrize(
    ("parts", "message"),
    [
        pytest.param(["oops"], NOT_AN_OBJECT_ERROR, id="string-part"),
        pytest.param([None], NOT_AN_OBJECT_ERROR, id="null-part"),
        pytest.param([{"text": "ok"}, "oops"], NOT_AN_OBJECT_ERROR, id="bad-part-after-a-good-one"),
        pytest.param([{"data": {}, "text": "a"}], _multiple("text, data"), id="text-and-data"),
        pytest.param([{"text": "a", "raw": "AAEC"}], _multiple("text, file"), id="text-and-modern-file"),
        pytest.param([{"data": {}, "file": {}, "text": "a"}], _multiple("text, file, data"), id="all-three"),
        pytest.param([{"foo": 1}], UNSUPPORTED_ERROR, id="unknown-key"),
        pytest.param([{"filename": "a.png"}], UNSUPPORTED_ERROR, id="file-metadata-without-content"),
        pytest.param([{}], UNSUPPORTED_ERROR, id="empty-object"),
        pytest.param([{"url": "https://ex/f.png", "mediaType": "image/png"}], FILE_ERROR, id="modern-file-url"),
        pytest.param([{"raw": "AAEC", "mediaType": "application/octet-stream"}], FILE_ERROR, id="modern-file-bytes"),
        pytest.param([{"kind": "file", "file": {"uri": "https://ex/f.png"}}], FILE_ERROR, id="legacy-file"),
        pytest.param([{"text": "see attached"}, {"url": "https://ex/f.png"}], FILE_ERROR, id="file-after-text"),
        pytest.param([{"data": "x"}], DATA_ERROR, id="data-string"),
        pytest.param([{"data": [1]}], DATA_ERROR, id="data-list"),
    ],
)
def test_invalid_part_raises_content_type_not_supported(parts: list[Any], message: str) -> None:
    """Keys in the 'multiple' message follow the fixed text, file, data order, not the order sent."""
    with pytest.raises(JsonRpcError) as exc_info:
        _convert(parts)

    assert exc_info.value.code == JsonRpcErrorCode.CONTENT_TYPE_NOT_SUPPORTED
    assert exc_info.value.message == message
