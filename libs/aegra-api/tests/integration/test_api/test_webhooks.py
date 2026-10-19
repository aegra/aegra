"""Integration tests: ``webhook`` on run and cron requests, through the HTTP layer."""

from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from aegra_api.api import crons as crons_module
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import get_session
from aegra_api.models.auth import User
from aegra_api.models.run_job import RunBehavior, RunIdentity, RunJob
from aegra_api.services import webhook_policy
from aegra_api.services.cron_service import CronService, get_cron_service
from tests.fixtures.clients import create_test_app, make_client
from tests.fixtures.database import DummySessionBase
from tests.fixtures.session_fixtures import override_session_dependency
from tests.fixtures.test_helpers import DummyRun, DummyThread

_WEBHOOK = "https://hooks.example.com/runs?token=secret"

# (webhook, 422 detail) pairs the default policy rejects.
_REJECTED = [
    ("http://169.254.169.254/latest/meta-data", "Webhook host blocked: cloud metadata endpoint"),
    ("http://localhost:9000/hook", "Webhook host blocked: localhost address"),
    ("http://127.0.0.1:9000/hook", "Webhook host blocked: loopback IP range"),
    ("/custom/webhook", "Relative webhook URLs are not supported; use an absolute http or https URL"),
    ("ftp://hooks.example.com/x", "Webhook must use http or https"),
]


def _assistant_row() -> MagicMock:
    assistant = MagicMock()
    assistant.assistant_id = "test-assistant-123"
    assistant.graph_id = "test-graph"
    assistant.config = {}
    assistant.context = {}
    assistant.user_id = "test-user"
    return assistant


class TestRunWebhook:
    """``webhook`` on POST /threads/{id}/runs reaches the queued job, or 422s before anything is created."""

    @staticmethod
    def _create_run(body: dict[str, Any]) -> tuple[Any, MagicMock, list[Any]]:
        app = create_test_app(include_runs=True, include_threads=False)
        thread = DummyThread("test-thread-123", "idle", {}, "test-user")
        assistant = _assistant_row()
        added: list[Any] = []

        class Session(DummySessionBase):
            async def scalar(self, stmt: Any) -> Any:
                stmt_str = str(stmt).lower()
                if "from thread" in stmt_str:
                    return thread
                if "from assistant" in stmt_str:
                    return assistant
                return None

            def add(self, obj: Any) -> None:
                added.append(obj)

            async def execute(self, _stmt: Any) -> Any:
                return MagicMock(rowcount=1)

        override_session_dependency(app, Session)
        client = make_client(app)
        mock_executor = MagicMock()
        mock_executor.submit = AsyncMock(return_value=None)

        with (
            patch("aegra_api.services.run_preparation.executor", mock_executor),
            patch("aegra_api.services.run_preparation.get_langgraph_service") as mock_service,
        ):
            mock_service.return_value.list_graphs.return_value = ["test-graph"]
            resp = client.post(
                "/threads/test-thread-123/runs",
                json={"assistant_id": "test-assistant-123", "input": {"message": "test"}, **body},
            )
        return resp, mock_executor, added

    def test_webhook_reaches_the_job_and_execution_params(self) -> None:
        resp, mock_executor, added = self._create_run({"webhook": _WEBHOOK})

        assert resp.status_code == 200
        job = mock_executor.submit.await_args.args[0]
        assert job.behavior.webhook == _WEBHOOK
        assert job.identity.assistant_id == "test-assistant-123"
        assert job.identity.created_at is not None
        run_orm = next(obj for obj in added if isinstance(obj, RunORM))
        assert run_orm.execution_params["behavior"]["webhook"] == _WEBHOOK
        assert run_orm.created_at == job.identity.created_at
        # The URL can carry the receiver's token; the Run response never echoes it.
        assert "secret" not in resp.text

    def test_run_without_webhook_has_none(self) -> None:
        resp, mock_executor, _added = self._create_run({})

        assert resp.status_code == 200
        assert mock_executor.submit.await_args.args[0].behavior.webhook is None

    @pytest.mark.parametrize(("webhook", "detail"), _REJECTED)
    def test_rejected_webhook_is_422_and_nothing_runs(self, webhook: str, detail: str) -> None:
        resp, mock_executor, added = self._create_run({"webhook": webhook})

        assert resp.status_code == 422
        assert resp.json()["detail"] == detail
        mock_executor.submit.assert_not_awaited()
        assert added == []

    def test_disabled_webhooks_are_422(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(webhook_policy, "load_http_config", lambda: {"disable_webhooks": True})
        webhook_policy.webhooks_enabled.cache_clear()

        resp, mock_executor, _added = self._create_run({"webhook": _WEBHOOK})

        assert resp.status_code == 422
        assert "http.disable_webhooks" in resp.json()["detail"]
        mock_executor.submit.assert_not_awaited()

    def test_disabled_webhooks_still_accept_runs_without_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(webhook_policy, "load_http_config", lambda: {"disable_webhooks": True})
        webhook_policy.webhooks_enabled.cache_clear()

        resp, _mock_executor, _added = self._create_run({})

        assert resp.status_code == 200

    def test_configured_url_policy_applies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            webhook_policy, "load_webhooks_config", lambda: {"url": {"allowed_domains": ["hooks.example.com"]}}
        )
        webhook_policy.get_webhook_settings.cache_clear()

        allowed, _executor, _added = self._create_run({"webhook": _WEBHOOK})
        insecure, _executor, _added = self._create_run({"webhook": "http://hooks.example.com/runs"})
        elsewhere, _executor, _added = self._create_run({"webhook": "https://other.example.com/runs"})

        assert allowed.status_code == 200
        assert insecure.status_code == 422
        assert insecure.json()["detail"] == "Webhook must use https"
        assert elsewhere.status_code == 422
        assert elsewhere.json()["detail"] == "Webhook domain not allowed"

    def test_non_string_webhook_is_422(self) -> None:
        resp, mock_executor, _added = self._create_run({"webhook": {"url": _WEBHOOK}})

        assert resp.status_code == 422
        mock_executor.submit.assert_not_awaited()


class TestCancelledRunWebhook:
    """A cancel that makes the run terminal itself reports it; one the executor finishes does not."""

    @staticmethod
    def _cancel(*, reconciled: bool, webhook: str | None = _WEBHOOK) -> MagicMock:
        app = create_test_app(include_runs=True, include_threads=False)
        run = DummyRun(status="running")
        run.execution_params = RunJob(
            identity=RunIdentity(run_id="test-run-123", thread_id="test-thread-123", graph_id="test-graph"),
            user=User(identity="test-user"),
            behavior=RunBehavior(webhook=webhook),
        ).to_execution_params()

        class Session(DummySessionBase):
            async def scalar(self, _stmt: Any) -> Any:
                return run

            async def refresh(self, _obj: Any) -> None:
                return None

        override_session_dependency(app, Session)
        client = make_client(app)

        with (
            patch("aegra_api.api.runs.streaming_service") as mock_streaming,
            patch("aegra_api.api.runs.interrupt_unowned_run", new_callable=AsyncMock, return_value=reconciled),
            patch("aegra_api.api.runs.notify_run_finished") as mock_notify,
        ):
            mock_streaming.cancel_run = AsyncMock()
            mock_streaming.interrupt_run = AsyncMock()
            mock_streaming.signal_run_cancelled = AsyncMock()
            resp = client.post("/threads/test-thread-123/runs/test-run-123/cancel")

        assert resp.status_code == 200
        return mock_notify

    def test_cancel_that_reconciles_the_run_reports_interrupted(self) -> None:
        # Dev mode never sets claimed_by, so the endpoint, not the executor, commits the cancel.
        mock_notify = self._cancel(reconciled=True)

        mock_notify.assert_called_once()
        job, outcome = mock_notify.call_args.args
        assert job.identity.run_id == "test-run-123"
        assert job.identity.assistant_id == "test-assistant-123"
        assert job.behavior.webhook == _WEBHOOK
        assert outcome.status == "interrupted"

    def test_cancel_of_a_worker_owned_run_leaves_the_report_to_the_worker(self) -> None:
        assert self._cancel(reconciled=False).call_count == 0

    def test_cancel_of_a_run_without_webhook_reports_nothing(self) -> None:
        assert self._cancel(reconciled=True, webhook=None).call_count == 0


class TestStatelessRunWebhook:
    def test_rejected_webhook_on_stateless_wait_is_422_and_cleans_up(self) -> None:
        app = create_test_app(include_runs=True, include_threads=False)
        client = make_client(app)
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=None)
        maker = MagicMock()
        maker.return_value.__aenter__ = AsyncMock(return_value=session)
        maker.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("aegra_api.api.runs._get_session_maker", return_value=maker),
            patch("aegra_api.api.stateless_runs.delete_thread_by_id", new_callable=AsyncMock) as mock_delete,
            patch("aegra_api.services.run_preparation.executor") as mock_executor,
        ):
            resp = client.post(
                "/runs/wait",
                json={"assistant_id": "agent", "input": {"message": "hi"}, "webhook": "http://169.254.169.254/"},
            )

        assert resp.status_code == 422
        assert resp.json()["detail"] == "Webhook host blocked: cloud metadata endpoint"
        mock_executor.submit.assert_not_called()
        mock_delete.assert_awaited_once()


class TestCronWebhook:
    """Cron create/update run the webhook policy in the real CronService."""

    @pytest.fixture
    def cron_client(self) -> Iterator[tuple[TestClient, AsyncMock]]:
        app = create_test_app(include_runs=False, include_threads=False)
        app.include_router(crons_module.router)
        session = AsyncMock()
        owned_thread = Mock()
        owned_thread.user_id = "test-user"
        session.scalar.return_value = owned_thread
        langgraph_service = MagicMock()
        langgraph_service.list_graphs.return_value = {"test-graph": {}}
        app.dependency_overrides[get_cron_service] = lambda: CronService(session, langgraph_service)

        async def _session() -> AsyncIterator[AsyncMock]:
            yield session

        app.dependency_overrides[get_session] = _session
        with TestClient(app) as client:
            yield client, session

    @pytest.mark.parametrize(("webhook", "detail"), _REJECTED[:3])
    def test_create_rejects_blocked_webhooks(
        self, cron_client: tuple[TestClient, AsyncMock], webhook: str, detail: str
    ) -> None:
        client, session = cron_client

        resp = client.post(
            "/runs/crons",
            json={"assistant_id": "agent", "schedule": "0 * * * *", "input": {"q": 1}, "webhook": webhook},
        )

        assert resp.status_code == 422
        assert resp.json()["detail"] == detail
        session.add.assert_not_called()

    def test_create_for_thread_rejects_blocked_webhooks(self, cron_client: tuple[TestClient, AsyncMock]) -> None:
        client, session = cron_client

        resp = client.post(
            "/threads/thread-1/runs/crons",
            json={
                "assistant_id": "agent",
                "schedule": "0 * * * *",
                "input": {"q": 1},
                "webhook": "http://169.254.169.254/",
            },
        )

        assert resp.status_code == 422
        session.add.assert_not_called()

    def test_update_rejects_blocked_webhooks(self, cron_client: tuple[TestClient, AsyncMock]) -> None:
        client, session = cron_client

        resp = client.patch("/runs/crons/cron-001", json={"webhook": "http://localhost:9000/hook"})

        assert resp.status_code == 422
        assert resp.json()["detail"] == "Webhook host blocked: localhost address"
        session.execute.assert_not_awaited()

    def test_create_rejects_webhooks_when_disabled(
        self, cron_client: tuple[TestClient, AsyncMock], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(webhook_policy, "load_http_config", lambda: {"disable_webhooks": True})
        webhook_policy.webhooks_enabled.cache_clear()
        client, _session = cron_client

        resp = client.post(
            "/runs/crons",
            json={"assistant_id": "agent", "schedule": "0 * * * *", "input": {"q": 1}, "webhook": _WEBHOOK},
        )

        assert resp.status_code == 422
        assert "http.disable_webhooks" in resp.json()["detail"]
