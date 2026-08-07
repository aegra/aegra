"""Tests for RunJob Pydantic model and serialization."""

import pytest
from pydantic import ValidationError

from aegra_api.models.auth import User
from aegra_api.models.run_job import RunBehavior, RunExecution, RunIdentity, RunJob


class TestRunIdentity:
    def test_frozen(self) -> None:
        identity = RunIdentity(run_id="r1", thread_id="t1", graph_id="g1")
        with pytest.raises(ValidationError):
            identity.run_id = "r2"  # type: ignore[misc]

    def test_fields(self) -> None:
        identity = RunIdentity(run_id="r1", thread_id="t1", graph_id="g1")
        assert identity.run_id == "r1"
        assert identity.thread_id == "t1"
        assert identity.graph_id == "g1"


class TestRunExecution:
    def test_defaults(self) -> None:
        execution = RunExecution()
        assert execution.input_data is None
        assert execution.config == {}
        assert execution.context == {}
        assert execution.stream_mode is None
        assert execution.checkpoint is None
        assert execution.command is None

    def test_with_values(self) -> None:
        execution = RunExecution(
            input_data={"key": "value"},
            config={"configurable": {}},
            stream_mode=["values", "updates"],
            command={"resume": True},
        )
        assert execution.input_data == {"key": "value"}
        assert execution.stream_mode == ["values", "updates"]
        assert execution.command == {"resume": True}


class TestRunBehavior:
    def test_defaults(self) -> None:
        behavior = RunBehavior()
        assert behavior.interrupt_before is None
        assert behavior.interrupt_after is None
        assert behavior.multitask_strategy is None
        assert behavior.subgraphs is False


class TestRunJob:
    @pytest.fixture()
    def sample_job(self) -> RunJob:
        return RunJob(
            identity=RunIdentity(run_id="run-1", thread_id="thread-1", graph_id="graph-1"),
            user=User(identity="user-1", is_authenticated=True, permissions=["read"]),
            execution=RunExecution(
                input_data={"message": "hello"},
                config={"configurable": {"model": "gpt-4"}},
                context={"tenant": "acme"},
                stream_mode="values",
                checkpoint={"thread_ts": "123"},
                command=None,
            ),
            behavior=RunBehavior(
                interrupt_before=["review"],
                interrupt_after=None,
                subgraphs=True,
            ),
        )

    def test_roundtrip_model_dump_validate(self, sample_job: RunJob) -> None:
        """model_dump -> model_validate produces identical job."""
        data = sample_job.model_dump()
        restored = RunJob.model_validate(data)
        assert restored == sample_job

    def test_execution_params_roundtrip(self, sample_job: RunJob) -> None:
        """to_execution_params -> from_run_orm produces identical job."""

        class FakeORM:
            run_id = "run-1"
            thread_id = "thread-1"
            execution_params = sample_job.to_execution_params()

        restored = RunJob.from_run_orm(FakeORM())
        assert restored.identity == sample_job.identity
        assert restored.user.identity == sample_job.user.identity
        assert restored.execution == sample_job.execution
        assert restored.behavior == sample_job.behavior

    def test_execution_params_includes_graph_id(self, sample_job: RunJob) -> None:
        params = sample_job.to_execution_params()
        assert params["graph_id"] == "graph-1"
        assert "run_id" not in params
        assert "thread_id" not in params

    def test_extra_user_fields_preserved(self) -> None:
        """User model allows extra fields (ConfigDict extra='allow')."""
        job = RunJob(
            identity=RunIdentity(run_id="r1", thread_id="t1", graph_id="g1"),
            user=User(identity="u1", is_authenticated=True, permissions=[], team_id="team-42"),
        )
        params = job.to_execution_params()
        assert params["user"]["team_id"] == "team-42"

        class FakeORM:
            run_id = "r1"
            thread_id = "t1"
            execution_params = params

        restored = RunJob.from_run_orm(FakeORM())
        assert restored.user.model_extra.get("team_id") == "team-42"

    def test_defaults_when_optional_fields_missing(self) -> None:
        """RunJob with minimal required fields."""
        job = RunJob(
            identity=RunIdentity(run_id="r1", thread_id="t1", graph_id="g1"),
            user=User(identity="u1"),
        )
        assert job.execution.input_data is None
        assert job.behavior.subgraphs is False
        assert job.run_metadata == {}

    def test_run_metadata_roundtrip(self) -> None:
        """run_metadata round-trips through to_execution_params / from_run_orm."""
        job = RunJob(
            identity=RunIdentity(run_id="r1", thread_id="t1", graph_id="g1"),
            user=User(identity="u1"),
            run_metadata={"tenant": "acme", "feature_flag": True, "retry_count": 2},
        )
        params = job.to_execution_params()
        assert params["run_metadata"] == {"tenant": "acme", "feature_flag": True, "retry_count": 2}

        class FakeORM:
            run_id = "r1"
            thread_id = "t1"
            execution_params = params

        restored = RunJob.from_run_orm(FakeORM())
        assert restored.run_metadata == job.run_metadata

    def test_from_run_orm_legacy_row_without_run_metadata(self) -> None:
        """Rows persisted before run_metadata existed deserialize with an empty dict."""

        class LegacyORM:
            run_id = "r1"
            thread_id = "t1"
            execution_params = {
                "graph_id": "g1",
                "user": {"identity": "u1", "is_authenticated": True, "permissions": []},
                "execution": {
                    "input_data": {},
                    "config": {},
                    "context": {},
                    "stream_mode": None,
                    "checkpoint": None,
                    "command": None,
                },
                "behavior": {
                    "interrupt_before": None,
                    "interrupt_after": None,
                    "multitask_strategy": None,
                    "subgraphs": False,
                },
                # No "run_metadata" key — pre-PR row.
            }

        restored = RunJob.from_run_orm(LegacyORM())
        assert restored.run_metadata == {}


class TestRunExecutionWebhook:
    """webhook round-trips through execution_params in both executor modes."""

    def test_webhook_roundtrip(self) -> None:
        from aegra_api.models.webhooks import WebhookConfig

        job = RunJob(
            identity=RunIdentity(run_id="r1", thread_id="t1", graph_id="g1", assistant_id="a1"),
            user=User(identity="u1"),
            execution=RunExecution(
                webhook=WebhookConfig(url="https://h.io/x", headers={"Authorization": "Bearer t"}, secret="s")
            ),
        )
        params = job.to_execution_params()
        assert params["execution"]["webhook"]["url"] == "https://h.io/x"

        class FakeORM:
            run_id = "r1"
            thread_id = "t1"
            assistant_id = "a1"
            execution_params = params

        restored = RunJob.from_run_orm(FakeORM())
        assert restored.execution.webhook == job.execution.webhook
        assert restored.identity.assistant_id == "a1"

    def test_no_webhook_defaults_none(self) -> None:
        execution = RunExecution()
        assert execution.webhook is None
