"""Unit tests for the structlog secret-redaction processor."""

from pydantic import SecretStr

from aegra_api.utils.setup_logging import _REDACTED, redact_secrets


def _run(event_dict: dict) -> dict:
    return redact_secrets(None, "info", event_dict)


class TestRedactSecrets:
    def test_masks_secret_keys(self) -> None:
        out = _run({"event": "x", "api_key": "sk-1", "authorization": "Bearer t", "access_token": "at"})
        assert out["api_key"] == _REDACTED
        assert out["authorization"] == _REDACTED
        assert out["access_token"] == _REDACTED

    def test_masks_nested_dict(self) -> None:
        out = _run({"event": "x", "data": {"password": "p", "keep": "v"}})
        assert out["data"]["password"] == _REDACTED
        assert out["data"]["keep"] == "v"

    def test_unwraps_secretstr(self) -> None:
        out = _run({"event": "x", "creds": SecretStr("supersecret")})
        assert out["creds"] == "**********"
        assert "supersecret" not in str(out)

    def test_preserves_non_secret_and_counts(self) -> None:
        # "total_tokens" must survive (bare 'token' is intentionally not matched).
        out = _run({"event": "x", "total_tokens": 42, "run_id": "r1"})
        assert out["total_tokens"] == 42
        assert out["run_id"] == "r1"

    def test_preserves_tuple_type(self) -> None:
        # structlog's positional_args is a tuple consumed by ``event % args``.
        out = _run({"event": "hi %s", "positional_args": ("world",)})
        assert isinstance(out["positional_args"], tuple)

    def test_depth_cap_does_not_crash(self) -> None:
        nested: dict = {"event": "x"}
        cursor = nested
        for _ in range(10):
            child: dict = {}
            cursor["child"] = child
            cursor = child
        cursor["api_key"] = "deep"  # deeper than _MAX_REDACT_DEPTH; must not raise
        _run(nested)  # no assertion — just must not raise


class TestRedactLazy:
    """Copy-on-first-change: clean values are returned by identity (zero allocation)."""

    def test_returns_same_object_when_clean(self) -> None:
        from aegra_api.utils.setup_logging import _redact

        inner = {"env": "prod"}
        event = {"event": "hi", "run_id": "r1", "meta": inner, "positional_args": ("x",)}
        out = _redact(event, 0)
        assert out is event  # no allocation when nothing matches
        assert out["meta"] is inner  # nested clean dict kept by identity

    def test_copies_only_on_change_keeping_clean_nested_identity(self) -> None:
        from aegra_api.utils.setup_logging import _REDACTED, _redact

        clean = {"env": "prod"}
        event = {"event": "hi", "api_key": "sk", "clean": clean}
        out = _redact(event, 0)
        assert out is not event  # copied because api_key changed
        assert out["api_key"] == _REDACTED
        assert out["clean"] is clean  # unchanged nested dict not re-copied

    def test_tuple_identity_preserved_when_clean(self) -> None:
        from aegra_api.utils.setup_logging import _redact

        args = ("world",)
        out = _redact({"event": "hi %s", "positional_args": args}, 0)
        assert out["positional_args"] is args  # same tuple object, still a tuple
