"""Tests for RunCreate model validation."""

import pytest
from pydantic import ValidationError

from aegra_api.models.runs import RunCreate


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

    def test_configurable_checkpoint_id_counts_as_checkpoint(self):
        """config.configurable.checkpoint_id is the LangGraph SDK's spelling of a
        checkpoint-only resume (ThreadStream folds forkFrom there). Execution
        already honors it — create_run_config merges the top-level checkpoint
        into configurable — so validation must accept it too, keeping input None.
        """
        run_create = RunCreate(
            assistant_id="agent",
            config={"configurable": {"checkpoint_id": "chk-1"}},
        )

        assert run_create.input is None
        assert run_create.command is None
        assert run_create.checkpoint is None

    def test_empty_configurable_checkpoint_id_still_rejected(self):
        """An empty/absent checkpoint_id in configurable must not satisfy the
        input/command/checkpoint requirement."""
        with pytest.raises(ValidationError, match="Must specify at least one"):
            RunCreate(assistant_id="agent", config={"configurable": {"checkpoint_id": ""}})

    def test_rejects_payload_without_input_command_or_checkpoint(self):
        """Ensure payloads with no input, command, or checkpoint are rejected."""
        with pytest.raises(ValueError, match="Must specify at least one of 'input', 'command', or 'checkpoint'"):
            RunCreate(assistant_id="agent")


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
