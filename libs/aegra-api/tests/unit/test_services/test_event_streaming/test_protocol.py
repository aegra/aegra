"""Tests for the v2 wire envelope builders."""

from aegra_api.services.event_streaming.channels import is_supported_channel
from aegra_api.services.event_streaming.protocol import (
    build_error,
    build_event,
    build_success,
)


class TestBuildEvent:
    def test_event_nests_payload_under_params_data(self) -> None:
        evt = build_event(
            "messages", {"event": "message-start", "role": "ai", "id": "m1"}, seq=7, event_id="run_event_7"
        )
        assert evt == {
            "type": "event",
            "seq": 7,
            "method": "messages",
            "event_id": "run_event_7",
            "params": {"data": {"event": "message-start", "role": "ai", "id": "m1"}, "namespace": []},
        }

    def test_namespace_is_carried(self) -> None:
        evt = build_event("values", {"x": 1}, namespace=["sub", "graph"], seq=2)
        assert evt["params"]["namespace"] == ["sub", "graph"]

    def test_event_id_omitted_when_none(self) -> None:
        evt = build_event("values", {"x": 1}, seq=1)
        assert "event_id" not in evt
        assert evt["seq"] == 1
        assert evt["method"] == "values"
        assert evt["params"] == {"data": {"x": 1}, "namespace": []}


class TestBuildSuccess:
    def test_success_without_meta(self) -> None:
        assert build_success(3, {"run_id": "r1"}) == {"type": "success", "id": 3, "result": {"run_id": "r1"}}

    def test_success_with_applied_through_seq(self) -> None:
        resp = build_success(3, {}, applied_through_seq=42)
        assert resp["meta"] == {"applied_through_seq": 42}


class TestBuildError:
    def test_error_shape(self) -> None:
        assert build_error(5, "not_supported", "nope") == {
            "type": "error",
            "id": 5,
            "error": "not_supported",
            "message": "nope",
        }

    def test_error_allows_null_id(self) -> None:
        assert build_error(None, "invalid_argument", "bad")["id"] is None


class TestChannels:
    def test_known_channels_supported(self) -> None:
        for ch in ("values", "updates", "messages", "tools", "lifecycle", "custom"):
            assert is_supported_channel(ch)

    def test_custom_namespaced_channel_supported(self) -> None:
        assert is_supported_channel("custom:my_event")

    def test_empty_custom_channel_rejected(self) -> None:
        assert not is_supported_channel("custom:")

    def test_unknown_channel_rejected(self) -> None:
        assert not is_supported_channel("bogus")
