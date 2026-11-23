"""Tests for RunCreate model validation."""

from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from aegra_api.models import runs as runs_models
from aegra_api.models.runs import RunCreate

_CHECKPOINT_ID = "1ef4f797-8335-6428-8001-8a1503f9b875"


class TestRunCreateValidation:
    """Tests for RunCreate input/command validation."""

    def test_checkpoint_only_payload_preserves_none_input(self):
        """Checkpoint-only payloads must keep input as None so LangGraph resumes
        from the checkpoint instead of restarting the graph from __start__.

        Regression test: previously the validator coerced input to ``{}`` which
        LangGraph Pregel treats as "new input" and re-enters __start__, ignoring
        the checkpoint's ``next=[...]``.
        """
        run_create = RunCreate(
            assistant_id="agent",
            checkpoint={"checkpoint_id": "chk-1", "checkpoint_ns": ""},
        )

        assert run_create.input is None
        assert run_create.command is None
        assert run_create.checkpoint == {"checkpoint_id": "chk-1", "checkpoint_ns": ""}

    def test_rejects_payload_without_input_command_or_checkpoint(self):
        """Ensure payloads with no input, command, or checkpoint are rejected."""
        with pytest.raises(ValueError, match="Must specify at least one of 'input', 'command', or 'checkpoint'"):
            RunCreate(assistant_id="agent")

    def test_top_level_checkpoint_id_counts_as_a_checkpoint(self) -> None:
        run_create = RunCreate(assistant_id="agent", checkpoint_id=_CHECKPOINT_ID)

        assert run_create.checkpoint_id == UUID(_CHECKPOINT_ID)
        assert run_create.input is None

    @pytest.mark.parametrize("checkpoint_id", ["", "not-a-uuid", 123, ["a"]])
    def test_rejects_malformed_checkpoint_id(self, checkpoint_id: Any) -> None:
        with pytest.raises(ValidationError):
            RunCreate(assistant_id="agent", input={"x": 1}, checkpoint_id=checkpoint_id)


class TestRunCreateDurability:
    """``durability`` and its deprecated ``checkpoint_during`` alias are declared, not dropped."""

    def test_defaults_leave_both_unset(self) -> None:
        run_create = RunCreate(assistant_id="agent", input={"x": 1})

        assert run_create.durability is None
        assert run_create.checkpoint_during is None

    @pytest.mark.parametrize("mode", ["sync", "async", "exit"])
    def test_accepts_every_langgraph_mode(self, mode: str) -> None:
        run_create = RunCreate(assistant_id="agent", input={"x": 1}, durability=mode)

        assert run_create.durability == mode

    @pytest.mark.parametrize("mode", ["", "SYNC", "eventual", 1, True])
    def test_rejects_unknown_mode(self, mode: Any) -> None:
        with pytest.raises(ValidationError, match="durability"):
            RunCreate(assistant_id="agent", input={"x": 1}, durability=mode)

    @pytest.mark.parametrize("value", [True, False])
    def test_accepts_checkpoint_during_bool(self, value: bool) -> None:
        run_create = RunCreate(assistant_id="agent", input={"x": 1}, checkpoint_during=value)

        assert run_create.checkpoint_during is value

    def test_rejects_non_bool_checkpoint_during(self) -> None:
        with pytest.raises(ValidationError, match="checkpoint_during"):
            RunCreate(assistant_id="agent", input={"x": 1}, checkpoint_during="sometimes")


class TestRunCreateMetadataValidation:
    """Tests for ``RunCreate.metadata`` shape enforcement.

    The schema constrains user-supplied metadata to OTEL-attribute
    primitives at request time so the contract is honest in the OpenAPI
    schema.  Previously ``dict[str, Any]`` accepted nested values that
    were silently dropped downstream by ``merge_run_metadata``; surfacing
    the rejection as a 422 means clients can fix the payload upstream
    rather than wonder why their metadata never reaches Langfuse.
    """

    def _payload(self, **overrides):
        base = {"assistant_id": "agent", "input": {"x": 1}}
        base.update(overrides)
        return base

    def test_none_metadata_accepted(self):
        run_create = RunCreate(**self._payload(metadata=None))
        assert run_create.metadata is None

    def test_empty_dict_metadata_accepted(self):
        run_create = RunCreate(**self._payload(metadata={}))
        assert run_create.metadata == {}

    def test_all_primitive_types_accepted(self):
        run_create = RunCreate(**self._payload(metadata={"tenant": "acme", "retries": 3, "ratio": 0.5, "flag": True}))
        assert run_create.metadata == {
            "tenant": "acme",
            "retries": 3,
            "ratio": 0.5,
            "flag": True,
        }

    def test_nested_dict_value_rejected(self):
        with pytest.raises(ValidationError):
            RunCreate(**self._payload(metadata={"k": {"nested": 1}}))

    def test_list_value_rejected(self):
        with pytest.raises(ValidationError):
            RunCreate(**self._payload(metadata={"k": [1, 2, 3]}))

    def test_too_many_keys_rejected(self):
        with pytest.raises(ValidationError, match="exceeds 32 keys"):
            RunCreate(**self._payload(metadata={f"k{i}": i for i in range(33)}))

    def test_dotted_key_rejected(self):
        """Reject keys containing ``.`` so users cannot land bare attributes
        like ``langfuse.user.id`` next to the system ones via the
        ``langfuse.trace.metadata.<key>`` channel."""
        with pytest.raises(ValidationError, match="must match"):
            RunCreate(**self._payload(metadata={"langfuse.user.id": "spoof"}))

    def test_empty_key_rejected(self):
        with pytest.raises(ValidationError, match="must match"):
            RunCreate(**self._payload(metadata={"": "v"}))

    def test_too_long_key_rejected(self):
        with pytest.raises(ValidationError, match="must match"):
            RunCreate(**self._payload(metadata={"k" * 65: "v"}))

    def test_non_ascii_key_rejected(self):
        """Reject non-ASCII keys to prevent visual-spoof of join keys
        (e.g. ``run_id`` vs ``run_id​`` zero-width-space variant)."""
        with pytest.raises(ValidationError, match="must match"):
            RunCreate(**self._payload(metadata={"run_id​": "v"}))

    def test_too_long_string_value_rejected(self):
        with pytest.raises(ValidationError, match="exceeds 512 characters"):
            RunCreate(**self._payload(metadata={"k": "v" * 513}))

    def test_max_keys_exactly_32_accepted(self):
        """Boundary: exactly 32 keys is allowed."""
        run_create = RunCreate(**self._payload(metadata={f"k{i}": i for i in range(32)}))
        assert len(run_create.metadata) == 32

    def test_max_value_length_exactly_512_accepted(self):
        """Boundary: exactly 512 chars is allowed."""
        run_create = RunCreate(**self._payload(metadata={"k": "v" * 512}))
        assert len(run_create.metadata["k"]) == 512


class TestRunCreateWebhook:
    """``webhook`` is a declared field; the URL policy runs when the run is prepared."""

    def test_accepts_a_webhook_url(self) -> None:
        run_create = RunCreate(assistant_id="agent", input={"x": 1}, webhook="https://hooks.example.com/runs?t=1")

        assert run_create.webhook == "https://hooks.example.com/runs?t=1"

    def test_defaults_to_no_webhook(self) -> None:
        assert RunCreate(assistant_id="agent", input={"x": 1}).webhook is None

    def test_rejects_non_string_webhook(self) -> None:
        with pytest.raises(ValidationError, match="webhook"):
            RunCreate(assistant_id="agent", input={"x": 1}, webhook={"url": "https://hooks.example.com"})

    def test_the_sdk_webhook_key_is_not_dropped(self) -> None:
        """The SDK sends ``webhook`` in the body; it must survive into the dumped request."""
        body = {"assistant_id": "agent", "input": {"x": 1}, "webhook": "https://hooks.example.com/runs"}

        assert RunCreate.model_validate(body).model_dump()["webhook"] == "https://hooks.example.com/runs"


class TestRunCreateInput:
    """``input`` is any JSON value, as on LangGraph Platform; null still means no input."""

    @pytest.mark.parametrize("value", [["a", "b"], "hello", 7, 1.5, True, {"x": 1}, [], "", 0, False])
    def test_accepts_any_json_value(self, value: Any) -> None:
        run_create = RunCreate(assistant_id="agent", input=value)

        assert run_create.input == value
        assert type(run_create.input) is type(value)

    def test_non_dict_input_is_still_exclusive_with_command(self) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            RunCreate(assistant_id="agent", input=["a"], command={"resume": "ok"})


class TestRunCreateIfNotExists:
    def test_defaults_to_unset(self) -> None:
        assert RunCreate(assistant_id="agent", input={"x": 1}).if_not_exists is None

    @pytest.mark.parametrize("value", ["create", "reject"])
    def test_accepts_both_sdk_values(self, value: str) -> None:
        assert RunCreate(assistant_id="agent", input={"x": 1}, if_not_exists=value).if_not_exists == value

    @pytest.mark.parametrize("value", ["ignore", "", True])
    def test_rejects_any_other_value(self, value: Any) -> None:
        with pytest.raises(ValidationError, match="if_not_exists"):
            RunCreate(assistant_id="agent", input={"x": 1}, if_not_exists=value)


class TestRunCreateAfterSeconds:
    """Delayed start is not implemented, so a delay is an error rather than an immediate run."""

    @pytest.mark.parametrize("value", [None, 0])
    def test_accepts_an_immediate_start(self, value: int | None) -> None:
        assert RunCreate(assistant_id="agent", input={"x": 1}, after_seconds=value).after_seconds == value

    @pytest.mark.parametrize("value", [1, 3600])
    def test_rejects_a_delayed_start(self, value: int) -> None:
        with pytest.raises(ValidationError, match="after_seconds is not supported"):
            RunCreate(assistant_id="agent", input={"x": 1}, after_seconds=value)

    def test_rejects_a_negative_delay(self) -> None:
        with pytest.raises(ValidationError, match="after_seconds"):
            RunCreate(assistant_id="agent", input={"x": 1}, after_seconds=-1)


class TestRunCreateInertSdkFields:
    """SDK fields that change nothing here; docs/feature-support.mdx lists each with the reason."""

    @pytest.mark.parametrize("value", [True, False])
    def test_accepts_stream_resumable(self, value: bool) -> None:
        assert RunCreate(assistant_id="agent", input={"x": 1}, stream_resumable=value).stream_resumable is value

    def test_rejects_non_bool_stream_resumable(self) -> None:
        with pytest.raises(ValidationError, match="stream_resumable"):
            RunCreate(assistant_id="agent", input={"x": 1}, stream_resumable="always")

    def test_accepts_the_langsmith_fields(self) -> None:
        run_create = RunCreate(
            assistant_id="agent",
            input={"x": 1},
            feedback_keys=["quality"],
            langsmith_tracer={"project_name": "p", "example_id": "e"},
        )

        assert run_create.feedback_keys == ["quality"]
        assert run_create.langsmith_tracer == {"project_name": "p", "example_id": "e"}


@pytest.fixture
def logged_warnings(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Warnings the models log; structlog output does not reach caplog."""
    recorded: list[dict[str, Any]] = []

    def record(event: str, **fields: Any) -> None:
        recorded.append({"event": event, **fields})

    monkeypatch.setattr(runs_models.logger, "warning", record)
    return recorded


class TestRunCreateUnknownKeys:
    """An undeclared key is still dropped, but the drop is logged with its name."""

    def test_logs_the_unknown_key_names(self, logged_warnings: list[dict[str, Any]]) -> None:
        RunCreate.model_validate({"assistant_id": "agent", "input": {"x": 1}, "zeta": 1, "alpha": 2})

        assert logged_warnings == [
            {"event": "Ignoring unknown request fields", "model": "RunCreate", "keys": ["alpha", "zeta"], "count": 2}
        ]

    def test_drops_the_unknown_key_without_an_error(self, logged_warnings: list[dict[str, Any]]) -> None:
        run_create = RunCreate.model_validate({"assistant_id": "agent", "input": {"x": 1}, "not_a_real_field": True})

        assert "not_a_real_field" not in run_create.model_dump()

    def test_default_python_sdk_body_logs_nothing(self, logged_warnings: list[dict[str, Any]]) -> None:
        """The Python SDK sends these default-valued keys on every create and stream call."""
        body = {
            "assistant_id": "agent",
            "input": {"x": 1},
            "stream_mode": "values",
            "stream_subgraphs": False,
            "stream_resumable": False,
        }

        RunCreate.model_validate(body)

        assert logged_warnings == []

    def test_logs_at_most_twenty_names_with_the_full_count(self, logged_warnings: list[dict[str, Any]]) -> None:
        body = {"assistant_id": "agent", "input": {"x": 1}, **{f"k{i:03d}": i for i in range(50)}}

        RunCreate.model_validate(body)

        assert logged_warnings[0]["count"] == 50
        assert logged_warnings[0]["keys"] == [f"k{i:03d}" for i in range(20)]

    def test_truncates_a_long_key_name(self, logged_warnings: list[dict[str, Any]]) -> None:
        RunCreate.model_validate({"assistant_id": "agent", "input": {"x": 1}, "k" * 500: 1})

        assert logged_warnings[0]["keys"] == ["k" * 64]
