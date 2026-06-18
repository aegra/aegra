"""Unit tests for client-checkpoint sanitization on thread state/history routes."""

from aegra_api.api.threads import _client_checkpoint


class TestClientCheckpoint:
    def test_strips_thread_id_and_run_id(self) -> None:
        """Server-authoritative identity keys must be dropped from client input.

        A client checkpoint carrying thread_id would otherwise override the
        ownership-verified thread and redirect state reads/writes (GHSA cross
        -tenant class).
        """
        cleaned = _client_checkpoint({"thread_id": "victim", "run_id": "victim-run", "checkpoint_id": "cp-1"})

        assert "thread_id" not in cleaned
        assert "run_id" not in cleaned

    def test_preserves_legitimate_checkpoint_keys(self) -> None:
        cleaned = _client_checkpoint({"checkpoint_id": "cp-1", "checkpoint_ns": "ns"})

        assert cleaned == {"checkpoint_id": "cp-1", "checkpoint_ns": "ns"}

    def test_empty_dict_returns_empty(self) -> None:
        assert _client_checkpoint({}) == {}
