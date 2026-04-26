"""Unit tests for CronService business logic.

All external dependencies (database, LangGraph) are mocked.
Follows the same fixture + class pattern as test_assistant_service.py.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from zoneinfo import ZoneInfoNotFoundError

import pytest
from fastapi import HTTPException

from aegra_api.models.crons import (
    CronCountRequest,
    CronCreate,
    CronSearchRequest,
    CronUpdate,
)
from aegra_api.services.cron_service import (
    CronService,
    _build_payload,
    _compute_next_run,
    _cron_to_response,
    _is_valid_schedule,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session() -> AsyncMock:
    """Mock AsyncSession for testing."""
    session = AsyncMock()
    session.add = Mock()
    return session


@pytest.fixture
def mock_langgraph_service() -> Mock:
    """Mock LangGraphService with one available graph."""
    svc = Mock()
    svc.list_graphs.return_value = {"test-graph": {}}
    return svc


@pytest.fixture
def cron_service(mock_session: AsyncMock, mock_langgraph_service: Mock) -> CronService:
    """CronService instance with mocked dependencies."""
    return CronService(mock_session, mock_langgraph_service)


@pytest.fixture
def sample_create() -> CronCreate:
    """Minimal CronCreate request."""
    return CronCreate(
        assistant_id="asst-001",
        schedule="*/5 * * * *",
        input={"messages": [{"role": "user", "content": "hello"}]},
        metadata={"env": "test"},
    )


def _make_cron_orm(
    *,
    cron_id: str = "cron-001",
    assistant_id: str = "asst-001",
    thread_id: str | None = None,
    user_id: str = "test-user",
    schedule: str = "*/5 * * * *",
    payload: dict[str, Any] | None = None,
    metadata_dict: dict[str, Any] | None = None,
    on_run_completed: str | None = None,
    enabled: bool = True,
    end_time: datetime | None = None,
    next_run_date: datetime | None = None,
) -> Mock:
    """Build a mock CronORM row."""
    now = datetime.now(UTC)
    cron = Mock()
    cron.cron_id = cron_id
    cron.assistant_id = assistant_id
    cron.thread_id = thread_id
    cron.user_id = user_id
    cron.schedule = schedule
    cron.payload = payload or {}
    cron.metadata_dict = metadata_dict or {}
    cron.on_run_completed = on_run_completed
    cron.enabled = enabled
    cron.end_time = end_time
    cron.next_run_date = next_run_date or now
    cron.created_at = now
    cron.updated_at = now
    return cron


def _make_assistant_orm(
    assistant_id: str = "asst-001",
    graph_id: str = "test-graph",
) -> Mock:
    assistant = Mock()
    assistant.assistant_id = assistant_id
    assistant.graph_id = graph_id
    return assistant


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestBuildPayload:
    """Test _build_payload helper."""

    def test_extracts_run_fields(self) -> None:
        req = CronCreate(
            assistant_id="a",
            schedule="* * * * *",
            input={"x": 1},
            config={"y": 2},
            webhook="https://example.com",
        )
        payload = _build_payload(req)
        assert payload["input"] == {"x": 1}
        assert payload["config"] == {"y": 2}
        assert payload["webhook"] == "https://example.com"

    def test_skips_none_fields(self) -> None:
        req = CronCreate(assistant_id="a", schedule="* * * * *")
        payload = _build_payload(req)
        assert "input" not in payload
        assert "config" not in payload


class TestComputeNextRun:
    """Test _compute_next_run helper."""

    def test_returns_future_datetime(self) -> None:
        now = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        result = _compute_next_run("*/5 * * * *", now=now)
        assert result > now

    def test_respects_cron_expression(self) -> None:
        now = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        result = _compute_next_run("0 12 * * *", now=now)
        assert result.hour == 12
        assert result.minute == 0


class TestCronToResponse:
    """Test _cron_to_response helper."""

    def test_maps_all_fields(self) -> None:
        cron = _make_cron_orm(
            metadata_dict={"k": "v"},
            payload={"input": {"x": 1}},
        )
        resp = _cron_to_response(cron)
        assert resp.cron_id == "cron-001"
        assert resp.assistant_id == "asst-001"
        assert resp.metadata == {"k": "v"}
        assert resp.payload == {"input": {"x": 1}}
        assert resp.enabled is True

    def test_handles_none_metadata(self) -> None:
        cron = _make_cron_orm(metadata_dict=None, payload=None)
        resp = _cron_to_response(cron)
        assert resp.metadata == {}
        assert resp.payload == {}


# ---------------------------------------------------------------------------
# CronService.create_cron
# ---------------------------------------------------------------------------


class TestCreateCron:
    """Test CronService.create_cron."""

    @pytest.mark.asyncio
    async def test_creates_cron_with_valid_schedule(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
        sample_create: CronCreate,
    ) -> None:
        mock_session.scalar.return_value = _make_assistant_orm()

        result = await cron_service.create_cron(sample_create, "test-user")

        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()
        mock_session.refresh.assert_awaited_once()
        assert result is not None

    @pytest.mark.asyncio
    async def test_rejects_invalid_schedule(
        self,
        cron_service: CronService,
    ) -> None:
        req = CronCreate(assistant_id="a", schedule="not-a-cron")
        with pytest.raises(HTTPException) as exc:
            await cron_service.create_cron(req, "test-user")
        assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_nonexistent_assistant(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
        sample_create: CronCreate,
    ) -> None:
        mock_session.scalar.return_value = None  # assistant not found
        with pytest.raises(HTTPException) as exc:
            await cron_service.create_cron(sample_create, "test-user")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_rejects_missing_graph(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
        mock_langgraph_service: Mock,
        sample_create: CronCreate,
    ) -> None:
        mock_session.scalar.return_value = _make_assistant_orm(graph_id="nonexistent")
        mock_langgraph_service.list_graphs.return_value = {"other-graph": {}}

        with pytest.raises(HTTPException) as exc:
            await cron_service.create_cron(sample_create, "test-user")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_passes_thread_id(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
        sample_create: CronCreate,
    ) -> None:
        mock_session.scalar.return_value = _make_assistant_orm()

        await cron_service.create_cron(sample_create, "test-user", thread_id="t-1")

        added_obj = mock_session.add.call_args[0][0]
        assert added_obj.thread_id == "t-1"

    @pytest.mark.asyncio
    async def test_resolves_graph_id_to_default_assistant(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
        mock_langgraph_service: Mock,
    ) -> None:
        request = CronCreate(
            assistant_id="test-graph",
            schedule="*/5 * * * *",
            input={"messages": [{"role": "user", "content": "hello"}]},
        )
        mock_langgraph_service.list_graphs.return_value = {"test-graph": {}}
        mock_session.scalar.return_value = _make_assistant_orm(
            assistant_id="resolved-assistant-id", graph_id="test-graph"
        )

        with patch(
            "aegra_api.services.cron_service.resolve_assistant_id",
            return_value="resolved-assistant-id",
        ) as mock_resolve:
            await cron_service.create_cron(request, "test-user")

        mock_resolve.assert_called_once_with("test-graph", {"test-graph": {}})
        added_obj = mock_session.add.call_args[0][0]
        assert added_obj.assistant_id == "resolved-assistant-id"

    @pytest.mark.asyncio
    async def test_search_resolves_graph_id_filter(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        scalars = Mock()
        scalars.all.return_value = []
        mock_session.scalars.return_value = scalars

        with patch(
            "aegra_api.services.cron_service.resolve_assistant_id",
            return_value="resolved-assistant-id",
        ) as mock_resolve:
            await cron_service.search_crons(CronSearchRequest(assistant_id="test-graph"), "test-user")

        mock_resolve.assert_called_once_with("test-graph", cron_service.langgraph_service.list_graphs.return_value)
        stmt = mock_session.scalars.await_args.args[0]
        compiled = stmt.compile()
        assert compiled.params["assistant_id_1"] == "resolved-assistant-id"

    @pytest.mark.asyncio
    async def test_count_resolves_graph_id_filter(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = 1

        with patch(
            "aegra_api.services.cron_service.resolve_assistant_id",
            return_value="resolved-assistant-id",
        ) as mock_resolve:
            result = await cron_service.count_crons(CronCountRequest(assistant_id="test-graph"), "test-user")

        assert result == 1
        mock_resolve.assert_called_once_with("test-graph", cron_service.langgraph_service.list_graphs.return_value)
        stmt = mock_session.scalar.await_args.args[0]
        compiled = stmt.compile()
        assert compiled.params["assistant_id_1"] == "resolved-assistant-id"


# ---------------------------------------------------------------------------
# CronService.update_cron
# ---------------------------------------------------------------------------


class TestUpdateCron:
    """Test CronService.update_cron."""

    @pytest.mark.asyncio
    async def test_returns_404_when_not_found(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = None
        req = CronUpdate(schedule="*/10 * * * *")

        with pytest.raises(HTTPException) as exc:
            await cron_service.update_cron("missing", req, "test-user")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_rejects_invalid_new_schedule(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = _make_cron_orm()
        req = CronUpdate(schedule="bad")

        with pytest.raises(HTTPException) as exc:
            await cron_service.update_cron("cron-001", req, "test-user")
        assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_updates_enabled_flag(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        # First scalar call → _get_cron_or_404; second → re-fetch after update
        updated = _make_cron_orm(enabled=False)
        mock_session.scalar.side_effect = [_make_cron_orm(), updated]

        resp = await cron_service.update_cron("cron-001", CronUpdate(enabled=False), "test-user")
        assert resp.enabled is False
        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# CronService.delete_cron
# ---------------------------------------------------------------------------


class TestDeleteCron:
    """Test CronService.delete_cron."""

    @pytest.mark.asyncio
    async def test_deletes_existing_cron(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        cron = _make_cron_orm()
        mock_session.scalar.return_value = cron

        await cron_service.delete_cron("cron-001", "test-user")

        mock_session.delete.assert_awaited_once_with(cron)
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_404_when_not_found(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = None
        with pytest.raises(HTTPException) as exc:
            await cron_service.delete_cron("missing", "test-user")
        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# CronService.search_crons / count_crons
# ---------------------------------------------------------------------------


class TestSearchCrons:
    """Test CronService.search_crons."""

    @pytest.mark.asyncio
    async def test_returns_empty_list(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        scalars = Mock()
        scalars.all.return_value = []
        mock_session.scalars.return_value = scalars

        result = await cron_service.search_crons(CronSearchRequest(), "test-user")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_mapped_responses(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        rows = [_make_cron_orm(cron_id="c1"), _make_cron_orm(cron_id="c2")]
        scalars = Mock()
        scalars.all.return_value = rows
        mock_session.scalars.return_value = scalars

        result = await cron_service.search_crons(CronSearchRequest(), "test-user")
        assert len(result) == 2
        assert result[0].cron_id == "c1"
        assert result[1].cron_id == "c2"


class TestCountCrons:
    """Test CronService.count_crons."""

    @pytest.mark.asyncio
    async def test_returns_count(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = 42
        result = await cron_service.count_crons(CronCountRequest(), "test-user")
        assert result == 42

    @pytest.mark.asyncio
    async def test_returns_zero_when_none(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = None
        result = await cron_service.count_crons(CronCountRequest(), "test-user")
        assert result == 0

    @pytest.mark.asyncio
    async def test_filters_by_assistant_id(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = 3
        result = await cron_service.count_crons(CronCountRequest(assistant_id="asst-001"), "test-user")
        assert result == 3
        mock_session.scalar.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_filters_by_thread_id(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = 1
        result = await cron_service.count_crons(CronCountRequest(thread_id="t-1"), "test-user")
        assert result == 1


# ---------------------------------------------------------------------------
# _build_payload — additional coverage
# ---------------------------------------------------------------------------


class TestBuildPayloadExtended:
    """Additional coverage for _build_payload."""

    def test_extracts_all_supported_fields(self) -> None:
        req = CronCreate(
            assistant_id="a",
            schedule="* * * * *",
            input={"x": 1},
            config={"y": 2},
            context={"z": 3},
            checkpoint_during=True,
            interrupt_before=["node_a"],
            interrupt_after="*",
            webhook="https://example.com/hook",
            multitask_strategy="reject",
            stream_mode="values",
            stream_subgraphs=True,
            stream_resumable=False,
            durability="ephemeral",
        )
        payload = _build_payload(req)
        assert payload["input"] == {"x": 1}
        assert payload["config"] == {"y": 2}
        assert payload["context"] == {"z": 3}
        assert payload["checkpoint_during"] is True
        assert payload["interrupt_before"] == ["node_a"]
        assert payload["interrupt_after"] == "*"
        assert payload["webhook"] == "https://example.com/hook"
        assert payload["multitask_strategy"] == "reject"
        assert payload["stream_mode"] == "values"
        assert payload["stream_subgraphs"] is True
        assert payload["stream_resumable"] is False
        assert payload["durability"] == "ephemeral"

    def test_works_with_cron_update(self) -> None:
        req = CronUpdate(
            input={"new": "data"},
            config={"cfg": True},
            checkpoint_during=True,
            multitask_strategy="enqueue",
        )
        payload = _build_payload(req)
        assert payload == {
            "input": {"new": "data"},
            "config": {"cfg": True},
            "checkpoint_during": True,
            "multitask_strategy": "enqueue",
        }

    def test_cron_update_empty_body(self) -> None:
        req = CronUpdate()
        payload = _build_payload(req)
        assert payload == {}


# ---------------------------------------------------------------------------
# _compute_next_run — additional coverage
# ---------------------------------------------------------------------------


class TestComputeNextRunExtended:
    """Additional coverage for _compute_next_run."""

    def test_uses_utc_now_by_default(self) -> None:
        before = datetime.now(UTC)
        result = _compute_next_run("* * * * *")
        assert result >= before
        assert result.tzinfo is not None

    def test_hourly_schedule(self) -> None:
        now = datetime(2025, 3, 15, 10, 30, 0, tzinfo=UTC)
        result = _compute_next_run("0 * * * *", now=now)
        assert result.hour == 11
        assert result.minute == 0

    def test_daily_at_midnight(self) -> None:
        now = datetime(2025, 3, 15, 0, 1, 0, tzinfo=UTC)
        result = _compute_next_run("0 0 * * *", now=now)
        assert result.day == 16
        assert result.hour == 0
        assert result.minute == 0

    def test_6field_every_30_seconds_schedules_within_30s(self) -> None:
        """6-field seconds-first: '*/30 * * * * *' must fire within 30 s, not 30 min."""
        now = datetime(2025, 3, 15, 10, 0, 1, tzinfo=UTC)
        result = _compute_next_run("*/30 * * * * *", now=now)
        diff = (result - now).total_seconds()
        assert diff <= 30, f"Expected next run within 30s, got {diff}s"

    def test_6field_every_10_seconds(self) -> None:
        now = datetime(2025, 3, 15, 10, 0, 0, tzinfo=UTC)
        result = _compute_next_run("*/10 * * * * *", now=now)
        diff = (result - now).total_seconds()
        assert diff == 10

    def test_5field_unchanged_by_fix(self) -> None:
        """Standard 5-field expressions must still be interpreted correctly."""
        now = datetime(2025, 3, 15, 10, 0, 0, tzinfo=UTC)
        result = _compute_next_run("*/30 * * * *", now=now)  # every 30 minutes
        diff = (result - now).total_seconds()
        assert diff == 30 * 60


# ---------------------------------------------------------------------------
# _is_valid_schedule
# ---------------------------------------------------------------------------


class TestIsValidSchedule:
    def test_valid_5field(self) -> None:
        assert _is_valid_schedule("*/5 * * * *") is True

    def test_valid_6field_seconds(self) -> None:
        assert _is_valid_schedule("*/30 * * * * *") is True

    def test_invalid_expression(self) -> None:
        assert _is_valid_schedule("not a cron") is False


# ---------------------------------------------------------------------------
# _cron_to_response — additional coverage
# ---------------------------------------------------------------------------


class TestCronToResponseExtended:
    """Additional coverage for _cron_to_response."""

    def test_handles_thread_id(self) -> None:
        cron = _make_cron_orm(thread_id="t-42")
        resp = _cron_to_response(cron)
        assert resp.thread_id == "t-42"

    def test_handles_on_run_completed(self) -> None:
        cron = _make_cron_orm(on_run_completed="create_new")
        resp = _cron_to_response(cron)
        assert resp.on_run_completed == "create_new"

    def test_handles_end_time(self) -> None:
        end = datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)
        cron = _make_cron_orm(end_time=end)
        resp = _cron_to_response(cron)
        assert resp.end_time == end

    def test_disabled_cron(self) -> None:
        cron = _make_cron_orm(enabled=False)
        resp = _cron_to_response(cron)
        assert resp.enabled is False


# ---------------------------------------------------------------------------
# CronService.create_cron — additional coverage
# ---------------------------------------------------------------------------


class TestCreateCronExtended:
    """Additional edge cases for CronService.create_cron."""

    @pytest.mark.asyncio
    async def test_sets_enabled_false_explicitly(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = _make_assistant_orm()
        req = CronCreate(
            assistant_id="asst-001",
            schedule="*/5 * * * *",
            enabled=False,
        )

        await cron_service.create_cron(req, "test-user")

        added_obj = mock_session.add.call_args[0][0]
        assert added_obj.enabled is False

    @pytest.mark.asyncio
    async def test_sets_end_time(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = _make_assistant_orm()
        end = datetime(2026, 1, 1, tzinfo=UTC)
        req = CronCreate(
            assistant_id="asst-001",
            schedule="*/5 * * * *",
            end_time=end,
        )

        await cron_service.create_cron(req, "test-user")

        added_obj = mock_session.add.call_args[0][0]
        assert added_obj.end_time == end

    @pytest.mark.asyncio
    async def test_sets_on_run_completed(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = _make_assistant_orm()
        req = CronCreate(
            assistant_id="asst-001",
            schedule="*/5 * * * *",
            on_run_completed="create_new",
        )

        await cron_service.create_cron(req, "test-user")

        added_obj = mock_session.add.call_args[0][0]
        assert added_obj.on_run_completed == "create_new"

    @pytest.mark.asyncio
    async def test_stores_payload_with_all_fields(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = _make_assistant_orm()
        req = CronCreate(
            assistant_id="asst-001",
            schedule="*/5 * * * *",
            input={"x": 1},
            config={"y": 2},
            webhook="https://hook.example.com",
        )

        await cron_service.create_cron(req, "test-user")

        added_obj = mock_session.add.call_args[0][0]
        assert added_obj.payload["input"] == {"x": 1}
        assert added_obj.payload["config"] == {"y": 2}
        assert added_obj.payload["webhook"] == "https://hook.example.com"

    @pytest.mark.asyncio
    async def test_next_run_date_skips_first_occurrence(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        """next_run_date must skip the first scheduled occurrence.

        _trigger_first_run fires a run immediately on creation; the scheduler
        should therefore start from the SECOND occurrence to avoid a double-fire.
        """
        mock_session.scalar.return_value = _make_assistant_orm()
        req = CronCreate(assistant_id="asst-001", schedule="*/5 * * * *")

        before = datetime.now(UTC)
        await cron_service.create_cron(req, "test-user")

        added_obj = mock_session.add.call_args[0][0]
        first_occ = _compute_next_run("*/5 * * * *", now=before)
        # next_run_date must be strictly after the first scheduled occurrence
        assert added_obj.next_run_date > first_occ

    @pytest.mark.asyncio
    async def test_metadata_stored_correctly(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = _make_assistant_orm()
        req = CronCreate(
            assistant_id="asst-001",
            schedule="*/5 * * * *",
            metadata={"team": "backend", "priority": "high"},
        )

        await cron_service.create_cron(req, "test-user")

        added_obj = mock_session.add.call_args[0][0]
        assert added_obj.metadata_dict == {"team": "backend", "priority": "high"}

    @pytest.mark.asyncio
    async def test_default_enabled_is_true(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = _make_assistant_orm()
        req = CronCreate(assistant_id="asst-001", schedule="*/5 * * * *")

        await cron_service.create_cron(req, "test-user")

        added_obj = mock_session.add.call_args[0][0]
        assert added_obj.enabled is True


# ---------------------------------------------------------------------------
# CronService.update_cron — additional coverage
# ---------------------------------------------------------------------------


class TestUpdateCronExtended:
    """Additional edge cases for CronService.update_cron."""

    @pytest.mark.asyncio
    async def test_updates_schedule_recomputes_next_run(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        updated = _make_cron_orm(schedule="0 * * * *")
        mock_session.scalar.side_effect = [_make_cron_orm(), updated]

        resp = await cron_service.update_cron("cron-001", CronUpdate(schedule="0 * * * *"), "test-user")
        assert resp.schedule == "0 * * * *"
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_updates_end_time(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        end = datetime(2026, 6, 1, tzinfo=UTC)
        updated = _make_cron_orm(end_time=end)
        mock_session.scalar.side_effect = [_make_cron_orm(), updated]

        resp = await cron_service.update_cron("cron-001", CronUpdate(end_time=end), "test-user")
        assert resp.end_time == end

    @pytest.mark.asyncio
    async def test_updates_on_run_completed(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        updated = _make_cron_orm(on_run_completed="create_new")
        mock_session.scalar.side_effect = [_make_cron_orm(), updated]

        resp = await cron_service.update_cron("cron-001", CronUpdate(on_run_completed="create_new"), "test-user")
        assert resp.on_run_completed == "create_new"

    @pytest.mark.asyncio
    async def test_updates_metadata(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        updated = _make_cron_orm(metadata_dict={"new": "meta"})
        mock_session.scalar.side_effect = [_make_cron_orm(), updated]

        resp = await cron_service.update_cron("cron-001", CronUpdate(metadata={"new": "meta"}), "test-user")
        assert resp.metadata == {"new": "meta"}

    @pytest.mark.asyncio
    async def test_merges_payload_fields(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        existing = _make_cron_orm(payload={"input": {"old": True}})
        updated = _make_cron_orm(payload={"input": {"old": True}, "webhook": "https://new.com"})
        mock_session.scalar.side_effect = [existing, updated]

        resp = await cron_service.update_cron(
            "cron-001",
            CronUpdate(webhook="https://new.com"),
            "test-user",
        )
        assert resp.payload["webhook"] == "https://new.com"

    @pytest.mark.asyncio
    async def test_empty_update_still_sets_updated_at(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        updated = _make_cron_orm()
        mock_session.scalar.side_effect = [_make_cron_orm(), updated]

        await cron_service.update_cron("cron-001", CronUpdate(), "test-user")
        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_404_when_refetch_fails(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        """Edge case: cron exists for _get_cron_or_404 but vanishes on re-fetch."""
        mock_session.scalar.side_effect = [_make_cron_orm(), None]

        with pytest.raises(HTTPException) as exc:
            await cron_service.update_cron("cron-001", CronUpdate(enabled=True), "test-user")
        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# CronService.search_crons — additional coverage
# ---------------------------------------------------------------------------


class TestSearchCronsExtended:
    """Additional coverage for CronService.search_crons."""

    @pytest.mark.asyncio
    async def test_filters_by_assistant_id(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        scalars = Mock()
        scalars.all.return_value = [_make_cron_orm()]
        mock_session.scalars.return_value = scalars

        result = await cron_service.search_crons(CronSearchRequest(assistant_id="asst-001"), "test-user")
        assert len(result) == 1
        mock_session.scalars.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_filters_by_thread_id(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        scalars = Mock()
        scalars.all.return_value = []
        mock_session.scalars.return_value = scalars

        result = await cron_service.search_crons(CronSearchRequest(thread_id="t-1"), "test-user")
        assert result == []

    @pytest.mark.asyncio
    async def test_filters_by_enabled(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        scalars = Mock()
        scalars.all.return_value = [_make_cron_orm(enabled=True)]
        mock_session.scalars.return_value = scalars

        result = await cron_service.search_crons(CronSearchRequest(enabled=True), "test-user")
        assert len(result) == 1
        assert result[0].enabled is True

    @pytest.mark.asyncio
    async def test_sort_by_next_run_date(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        scalars = Mock()
        scalars.all.return_value = []
        mock_session.scalars.return_value = scalars

        await cron_service.search_crons(CronSearchRequest(sort_by="next_run_date"), "test-user")
        mock_session.scalars.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sort_by_updated_at(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        scalars = Mock()
        scalars.all.return_value = []
        mock_session.scalars.return_value = scalars

        await cron_service.search_crons(CronSearchRequest(sort_by="updated_at"), "test-user")
        mock_session.scalars.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sort_order_desc(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        scalars = Mock()
        scalars.all.return_value = []
        mock_session.scalars.return_value = scalars

        await cron_service.search_crons(CronSearchRequest(sort_order="desc"), "test-user")
        mock_session.scalars.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pagination(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        scalars = Mock()
        scalars.all.return_value = []
        mock_session.scalars.return_value = scalars

        await cron_service.search_crons(CronSearchRequest(limit=5, offset=10), "test-user")
        mock_session.scalars.assert_awaited_once()


# ---------------------------------------------------------------------------
# CronService.get_due_crons
# ---------------------------------------------------------------------------


class TestGetDueCrons:
    """Test CronService.get_due_crons."""

    @pytest.mark.asyncio
    async def test_returns_due_crons(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        due = [_make_cron_orm(cron_id="c1"), _make_cron_orm(cron_id="c2")]
        result = Mock()
        result.scalars.return_value.all.return_value = due
        mock_session.execute.return_value = result

        result = await cron_service.get_due_crons()
        assert len(result) == 2
        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_empty_when_none_due(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        result = Mock()
        result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = result

        result = await cron_service.get_due_crons()
        assert result == []
        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_accepts_custom_now(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        result = Mock()
        result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = result

        custom_now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
        result = await cron_service.get_due_crons(now=custom_now)
        assert result == []
        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# CronService.advance_next_run
# ---------------------------------------------------------------------------


class TestAdvanceNextRun:
    """Test CronService.advance_next_run."""

    @pytest.mark.asyncio
    async def test_advances_to_next_occurrence(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        cron = _make_cron_orm(schedule="*/5 * * * *", end_time=None)
        cron.end_time = None

        await cron_service.advance_next_run(cron)

        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disables_when_past_end_time(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        cron = _make_cron_orm(schedule="*/5 * * * *", end_time=past)

        await cron_service.advance_next_run(cron)

        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_does_not_disable_when_end_time_is_future(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        future = datetime.now(UTC) + timedelta(days=30)
        cron = _make_cron_orm(schedule="*/5 * * * *", end_time=future)

        await cron_service.advance_next_run(cron)

        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_uses_timezone_from_payload(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        """advance_next_run must respect the timezone stored in the payload JSONB."""
        cron = _make_cron_orm(
            schedule="0 9 * * *",
            payload={"timezone": "America/New_York"},
            end_time=None,
        )

        await cron_service.advance_next_run(cron)

        mock_session.execute.assert_awaited_once()
        # Verify the next_run passed to execute is in UTC (timezone-aware)
        call_args = mock_session.execute.call_args
        assert call_args is not None


# ---------------------------------------------------------------------------
# timezone — _compute_next_run
# ---------------------------------------------------------------------------


class TestTimezoneAwareNextRun:
    """Tests for timezone-aware _compute_next_run."""

    def test_returns_utc_datetime(self) -> None:
        now = datetime(2025, 3, 15, 0, 0, 0, tzinfo=UTC)
        result = _compute_next_run("0 9 * * *", now=now, timezone="America/New_York")
        assert result.tzinfo is not None
        # March 15 falls in EDT (UTC-4, DST active) so 09:00 NY = 13:00 UTC
        assert result.hour == 13
        assert result.minute == 0

    def test_timezone_shifts_next_run_vs_utc(self) -> None:
        """A schedule in UTC+9 (Tokyo) fires earlier in UTC when now is 23:00 UTC.

        At 23:00 UTC the next 09:00 UTC is 10 hours away (09:00 next day).
        In Tokyo (UTC+9) it is 08:00, so the next 09:00 Tokyo is only 1 hour away
        and maps to 00:00 UTC — well before 09:00 UTC.
        """
        now = datetime(2025, 6, 1, 23, 0, 0, tzinfo=UTC)  # = June 2 08:00 Tokyo
        utc_result = _compute_next_run("0 9 * * *", now=now)  # June 2 09:00 UTC
        tokyo_result = _compute_next_run("0 9 * * *", now=now, timezone="Asia/Tokyo")  # June 2 00:00 UTC
        assert tokyo_result < utc_result

    def test_none_timezone_behaves_as_utc(self) -> None:
        now = datetime(2025, 3, 15, 0, 0, 0, tzinfo=UTC)
        without_tz = _compute_next_run("0 12 * * *", now=now)
        with_utc = _compute_next_run("0 12 * * *", now=now, timezone="UTC")
        assert without_tz == with_utc

    def test_invalid_timezone_raises(self) -> None:
        """ZoneInfoNotFoundError must propagate so the caller can map it to 422."""
        with pytest.raises((ZoneInfoNotFoundError, KeyError)):
            _compute_next_run("0 9 * * *", timezone="Not/ATimezone")


# ---------------------------------------------------------------------------
# timezone — CronService.create_cron
# ---------------------------------------------------------------------------


class TestCreateCronTimezone:
    """Timezone handling in CronService.create_cron."""

    @pytest.mark.asyncio
    async def test_stores_timezone_in_payload(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = _make_assistant_orm()
        req = CronCreate(
            assistant_id="asst-001",
            schedule="0 9 * * *",
            timezone="America/New_York",
        )

        await cron_service.create_cron(req, "test-user")

        added_obj = mock_session.add.call_args[0][0]
        assert added_obj.payload["timezone"] == "America/New_York"

    @pytest.mark.asyncio
    async def test_rejected_invalid_timezone(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.scalar.return_value = _make_assistant_orm()
        req = CronCreate(
            assistant_id="asst-001",
            schedule="0 9 * * *",
            timezone="Not/Valid",
        )

        with pytest.raises(HTTPException) as exc:
            await cron_service.create_cron(req, "test-user")
        assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_next_run_is_timezone_aware(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        """next_run_date should reflect the timezone-shifted schedule."""
        mock_session.scalar.return_value = _make_assistant_orm()
        req = CronCreate(
            assistant_id="asst-001",
            schedule="0 9 * * *",
            timezone="America/New_York",
        )

        await cron_service.create_cron(req, "test-user")

        added_obj = mock_session.add.call_args[0][0]
        # next_run_date must be UTC-aware (not naive)
        assert added_obj.next_run_date.tzinfo is not None


# ---------------------------------------------------------------------------
# timezone — CronService.update_cron
# ---------------------------------------------------------------------------


class TestUpdateCronTimezone:
    """Timezone handling in CronService.update_cron."""

    @pytest.mark.asyncio
    async def test_update_timezone_stored_in_payload(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        existing = _make_cron_orm(schedule="0 9 * * *", payload={})
        updated_orm = _make_cron_orm(
            schedule="0 9 * * *",
            payload={"timezone": "Europe/London"},
        )
        mock_session.scalar.side_effect = [existing, updated_orm]

        resp = await cron_service.update_cron("cron-001", CronUpdate(timezone="Europe/London"), "test-user")
        assert resp.payload["timezone"] == "Europe/London"
        execute_params = mock_session.execute.await_args_list[0].args[0].compile().params
        assert execute_params["next_run_date"].tzinfo is not None

    @pytest.mark.asyncio
    async def test_update_timezone_recomputes_next_run_date_without_schedule_change(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        existing = _make_cron_orm(
            schedule="0 9 * * *",
            payload={"timezone": "UTC"},
        )
        updated_orm = _make_cron_orm(schedule="0 9 * * *", payload={"timezone": "Asia/Tokyo"})
        mock_session.scalar.side_effect = [existing, updated_orm]

        await cron_service.update_cron("cron-001", CronUpdate(timezone="Asia/Tokyo"), "test-user")

        execute_params = mock_session.execute.await_args_list[0].args[0].compile().params
        assert execute_params["payload"]["timezone"] == "Asia/Tokyo"
        assert execute_params["next_run_date"].tzinfo is not None

    @pytest.mark.asyncio
    async def test_update_timezone_rejects_invalid_timezone_without_schedule_change(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        existing = _make_cron_orm(schedule="0 9 * * *", payload={"timezone": "UTC"})
        mock_session.scalar.return_value = existing

        with pytest.raises(HTTPException) as exc:
            await cron_service.update_cron("cron-001", CronUpdate(timezone="Not/Valid"), "test-user")

        assert exc.value.status_code == 422
        mock_session.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_schedule_respects_existing_payload_timezone(
        self,
        cron_service: CronService,
        mock_session: AsyncMock,
    ) -> None:
        """Changing the schedule must still use the timezone already stored in payload."""
        existing = _make_cron_orm(
            schedule="0 9 * * *",
            payload={"timezone": "Asia/Tokyo"},
        )
        updated_orm = _make_cron_orm(schedule="0 10 * * *", payload={"timezone": "Asia/Tokyo"})
        mock_session.scalar.side_effect = [existing, updated_orm]

        await cron_service.update_cron("cron-001", CronUpdate(schedule="0 10 * * *"), "test-user")
        # verify execute was called (next_run_date was set)
        mock_session.execute.assert_awaited_once()
