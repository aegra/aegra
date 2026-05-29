"""Additional unit tests added during PR review.

These cover behaviours introduced for the cron PR fix-ups: per-user quota,
6-field schedule gating, ownership enforcement on update/delete, and the
``advance_next_run`` claim release.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from aegra_api.models.crons import CronCreate, CronUpdate
from aegra_api.services.cron_service import CronService


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.add = Mock()
    return session


@pytest.fixture
def mock_langgraph_service() -> Mock:
    svc = Mock()
    svc.list_graphs.return_value = {"test-graph": {}}
    return svc


@pytest.fixture
def cron_service(mock_session: AsyncMock, mock_langgraph_service: Mock) -> CronService:
    return CronService(mock_session, mock_langgraph_service)


def _make_assistant_orm() -> Mock:
    a = Mock()
    a.assistant_id = "asst-001"
    a.graph_id = "test-graph"
    return a


def _make_cron_orm(
    *,
    cron_id: str = "cron-x",
    user_id: str = "tenant-A",
    on_run_completed: str | None = None,
    payload: dict[str, Any] | None = None,
) -> Mock:
    cron = Mock()
    cron.cron_id = cron_id
    cron.user_id = user_id
    cron.assistant_id = "asst-001"
    cron.thread_id = None
    cron.schedule = "*/5 * * * *"
    cron.payload = payload or {}
    cron.metadata_dict = {}
    cron.on_run_completed = on_run_completed
    cron.enabled = True
    cron.end_time = None
    cron.next_run_date = None
    cron.created_at = None
    cron.updated_at = None
    return cron


@pytest.fixture
def sample_create() -> CronCreate:
    return CronCreate(
        assistant_id="asst-001",
        schedule="*/5 * * * *",
        input={"messages": [{"role": "user", "content": "hi"}]},
    )


# ---------------------------------------------------------------------------
# Per-user quota
# ---------------------------------------------------------------------------


class TestPerUserQuota:
    """Per-user cap so a single tenant cannot exhaust the scheduler."""

    @pytest.mark.asyncio
    async def test_returns_429_when_user_at_cap(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
        sample_create: CronCreate,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from aegra_api.services import cron_service as _cs

        monkeypatch.setattr(_cs.settings.cron, "CRON_MAX_PER_USER", 2)
        # First (and only) scalar() call is the COUNT, returning the cap.
        mock_session.scalar.return_value = 2

        with pytest.raises(HTTPException) as exc:
            await cron_service.create_cron(sample_create, "tenant-A")
        assert exc.value.status_code == 429
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_zero_cap_disables_check(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
        sample_create: CronCreate,
    ) -> None:
        # Default fixture sets cap to 0 -> no COUNT(*) issued.
        mock_session.scalar.return_value = _make_assistant_orm()
        await cron_service.create_cron(sample_create, "tenant-A")
        mock_session.add.assert_called_once()


# ---------------------------------------------------------------------------
# 6-field (seconds) schedule gating
# ---------------------------------------------------------------------------


class TestSecondsScheduleGate:
    """6-field (seconds) cron schedules must be opt-in."""

    @pytest.mark.asyncio
    async def test_rejected_by_default(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = _make_assistant_orm()
        req = CronCreate(assistant_id="asst-001", schedule="*/30 * * * * *")
        with pytest.raises(HTTPException) as exc:
            await cron_service.create_cron(req, "tenant-A")
        assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_accepted_when_gate_open(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from aegra_api.services import cron_service as _cs

        monkeypatch.setattr(_cs.settings.cron, "CRON_ALLOW_SECONDS_SCHEDULE", True)
        mock_session.scalar.return_value = _make_assistant_orm()
        req = CronCreate(assistant_id="asst-001", schedule="*/30 * * * * *")
        await cron_service.create_cron(req, "tenant-A")
        mock_session.add.assert_called_once()


# ---------------------------------------------------------------------------
# Ownership enforcement
# ---------------------------------------------------------------------------


class TestOwnershipEnforcement:
    """Update/delete must filter by user_id so tenants cannot touch each other's crons."""

    @pytest.mark.asyncio
    async def test_update_404_when_other_user_owns_row(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        # _get_cron_or_404(lock=True) returns None when the WHERE
        # (cron_id AND user_id) does not match.
        mock_session.scalar.return_value = None
        with pytest.raises(HTTPException) as exc:
            await cron_service.update_cron("cron-x", CronUpdate(enabled=False), "tenant-B")
        assert exc.value.status_code == 404
        mock_session.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_404_when_other_user_owns_row(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = None
        with pytest.raises(HTTPException) as exc:
            await cron_service.delete_cron("cron-x", "tenant-B")
        assert exc.value.status_code == 404
        mock_session.delete.assert_not_called()


# ---------------------------------------------------------------------------
# advance_next_run releases the claim
# ---------------------------------------------------------------------------


class TestAdvanceClearsClaim:
    @pytest.mark.asyncio
    async def test_advance_sets_claimed_until_to_null(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        cron = _make_cron_orm()
        cron.end_time = None
        await cron_service.advance_next_run(cron)
        # Inspect the UPDATE statement we issued and assert claimed_until is set.
        stmt = mock_session.execute.call_args[0][0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": False}))
        assert "claimed_until" in compiled
        assert "next_run_date" in compiled


# ---------------------------------------------------------------------------
# Webhook credential masking in CronResponse
# ---------------------------------------------------------------------------


class TestWebhookCredentialMasking:
    """Webhook userinfo (https://user:tok@host/) must never round-trip in payload."""

    def test_strips_userinfo_from_webhook(self) -> None:
        from aegra_api.services.cron_service import _cron_to_response

        cron = _make_cron_orm(payload={"webhook": "https://user:secret@hooks.example.com/x"})
        from datetime import UTC, datetime

        cron.created_at = cron.updated_at = datetime.now(UTC)
        resp = _cron_to_response(cron)
        assert resp.payload["webhook"] == "https://hooks.example.com/x"

    def test_preserves_port_and_path(self) -> None:
        from aegra_api.services.cron_service import _cron_to_response

        cron = _make_cron_orm(payload={"webhook": "https://u:p@host.example:8443/a/b?q=1"})
        from datetime import UTC, datetime

        cron.created_at = cron.updated_at = datetime.now(UTC)
        resp = _cron_to_response(cron)
        assert resp.payload["webhook"] == "https://host.example:8443/a/b?q=1"

    def test_webhook_without_credentials_unchanged(self) -> None:
        from aegra_api.services.cron_service import _cron_to_response

        cron = _make_cron_orm(payload={"webhook": "https://hooks.example.com/x"})
        from datetime import UTC, datetime

        cron.created_at = cron.updated_at = datetime.now(UTC)
        resp = _cron_to_response(cron)
        assert resp.payload["webhook"] == "https://hooks.example.com/x"

    def test_empty_payload_returns_empty_dict(self) -> None:
        from datetime import UTC, datetime

        from aegra_api.services.cron_service import _cron_to_response

        # _make_cron_orm normalizes payload=None to {}, so set it explicitly to
        # exercise the real None branch in _cron_to_response.
        cron = _make_cron_orm()
        cron.payload = None
        cron.created_at = cron.updated_at = datetime.now(UTC)
        resp = _cron_to_response(cron)
        assert resp.payload == {}


# ---------------------------------------------------------------------------
# Re-enabling an expired cron must recompute next_run_date
# ---------------------------------------------------------------------------


class TestReenableRecomputesNextRun:
    """PATCH {enabled: true} on a cron with stale next_run_date must not fire on next tick."""

    @pytest.mark.asyncio
    async def test_recomputes_when_reenabling_expired_cron(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from datetime import UTC, datetime, timedelta

        from aegra_api.services import cron_service as _cs

        cron = _make_cron_orm()
        cron.enabled = False
        cron.next_run_date = datetime.now(UTC) - timedelta(days=7)
        cron.created_at = cron.updated_at = datetime.now(UTC)
        mock_session.scalar.return_value = cron
        mock_session.execute.return_value = Mock(rowcount=1)

        sentinel = datetime.now(UTC) + timedelta(hours=1)
        monkeypatch.setattr(_cs, "_compute_next_run", lambda *a, **kw: sentinel)

        await cron_service.update_cron("cron-x", CronUpdate(enabled=True), "tenant-A")

        # Find the UPDATE call and confirm next_run_date is in the values.
        update_stmt = mock_session.execute.call_args[0][0]
        compiled_params = update_stmt.compile().params
        assert "next_run_date" in compiled_params
        assert compiled_params["next_run_date"] == sentinel

    @pytest.mark.asyncio
    async def test_no_recompute_when_reenabling_with_future_next_run(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from datetime import UTC, datetime, timedelta

        from aegra_api.services import cron_service as _cs

        cron = _make_cron_orm()
        cron.enabled = False
        cron.next_run_date = datetime.now(UTC) + timedelta(hours=1)
        cron.created_at = cron.updated_at = datetime.now(UTC)
        mock_session.scalar.return_value = cron
        mock_session.execute.return_value = Mock(rowcount=1)

        called = {"n": 0}

        def _track(*_a: Any, **_kw: Any) -> Any:
            called["n"] += 1
            return datetime.now(UTC)

        monkeypatch.setattr(_cs, "_compute_next_run", _track)

        await cron_service.update_cron("cron-x", CronUpdate(enabled=True), "tenant-A")
        assert called["n"] == 0

    @pytest.mark.asyncio
    async def test_no_recompute_when_disabling(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from datetime import UTC, datetime, timedelta

        from aegra_api.services import cron_service as _cs

        cron = _make_cron_orm()
        cron.enabled = True
        cron.next_run_date = datetime.now(UTC) - timedelta(days=1)
        cron.created_at = cron.updated_at = datetime.now(UTC)
        mock_session.scalar.return_value = cron
        mock_session.execute.return_value = Mock(rowcount=1)

        called = {"n": 0}

        def _track(*_a: Any, **_kw: Any) -> Any:
            called["n"] += 1
            return datetime.now(UTC)

        monkeypatch.setattr(_cs, "_compute_next_run", _track)

        await cron_service.update_cron("cron-x", CronUpdate(enabled=False), "tenant-A")
        assert called["n"] == 0
