"""Unit tests for cron API helpers."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException, Response

from aegra_api.api.crons import _authorize_cron_create, _trigger_first_run, get_cron
from aegra_api.models import Run, User
from aegra_api.models.crons import CronCreate


def _make_cron(
    *,
    assistant_id: str = "agent",
    thread_id: str | None = None,
    on_run_completed: str | None = None,
    payload: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        cron_id="cron-001",
        assistant_id=assistant_id,
        thread_id=thread_id,
        on_run_completed=on_run_completed,
        payload=payload
        if payload is not None
        else {"input": {"messages": [{"role": "user", "content": "hi"}]}, "config": {"k": "v"}},
    )


class TestTriggerFirstRun:
    """Test initial run triggering for cron creation."""

    @pytest.fixture
    def mock_user(self) -> User:
        return User(identity="test-user", scopes=[])

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def mock_run(self, mock_user: User) -> Run:
        return Run(
            run_id="run-001",
            thread_id="thread-001",
            assistant_id="agent",
            status="pending",
            input={"messages": [{"role": "user", "content": "hi"}]},
            user_id=mock_user.identity,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    @pytest.mark.asyncio
    async def test_schedules_cleanup_for_stateless_cron(
        self, mock_user: User, mock_session: AsyncMock, mock_run: Run
    ) -> None:
        cron = _make_cron(thread_id=None, on_run_completed=None)

        with (
            patch("aegra_api.api.crons.uuid4", return_value="eph-thread-1"),
            patch(
                "aegra_api.api.crons._prepare_run",
                new_callable=AsyncMock,
                return_value=("run-001", mock_run, Mock()),
            ) as mock_prepare,
            patch("aegra_api.api.crons.schedule_background_cleanup") as mock_schedule,
        ):
            result = await _trigger_first_run(mock_session, cron, mock_user)

        assert result is mock_run
        mock_prepare.assert_awaited_once()
        assert mock_prepare.await_args.args[1] == "eph-thread-1"
        mock_schedule.assert_called_once_with("run-001", "eph-thread-1", mock_user.identity)

    @pytest.mark.asyncio
    async def test_skips_cleanup_for_thread_bound_cron(
        self, mock_user: User, mock_session: AsyncMock, mock_run: Run
    ) -> None:
        cron = _make_cron(thread_id="thread-bound-1")

        with (
            patch(
                "aegra_api.api.crons._prepare_run",
                new_callable=AsyncMock,
                return_value=("run-001", mock_run, Mock()),
            ),
            patch("aegra_api.api.crons.schedule_background_cleanup") as mock_schedule,
        ):
            await _trigger_first_run(mock_session, cron, mock_user)

        mock_schedule.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_cleanup_when_keep_requested(
        self, mock_user: User, mock_session: AsyncMock, mock_run: Run
    ) -> None:
        cron = _make_cron(thread_id=None, on_run_completed="keep")

        with (
            patch("aegra_api.api.crons.uuid4", return_value="eph-thread-keep"),
            patch(
                "aegra_api.api.crons._prepare_run",
                new_callable=AsyncMock,
                return_value=("run-001", mock_run, Mock()),
            ),
            patch("aegra_api.api.crons.schedule_background_cleanup") as mock_schedule,
        ):
            await _trigger_first_run(mock_session, cron, mock_user)

        mock_schedule.assert_not_called()

    @pytest.mark.asyncio
    async def test_deletes_ephemeral_thread_when_initial_run_setup_fails(
        self, mock_user: User, mock_session: AsyncMock
    ) -> None:
        cron = _make_cron(thread_id=None, on_run_completed=None)

        with (
            patch("aegra_api.api.crons.uuid4", return_value="eph-thread-fail"),
            patch(
                "aegra_api.api.crons._prepare_run",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            patch("aegra_api.api.crons.delete_thread_by_id", new_callable=AsyncMock) as mock_delete,
            pytest.raises(RuntimeError, match="boom"),
        ):
            await _trigger_first_run(mock_session, cron, mock_user)

        mock_delete.assert_awaited_once_with("eph-thread-fail", mock_user.identity)


class TestAuthorizeCronCreate:
    """Regression tests for the multi-resource auth chain on cron creation.

    Spec contract: cron creation must dispatch ``crons.create`` plus the
    underlying ``assistants.read`` and either ``threads.read`` (thread-bound)
    or ``threads.search`` (stateless). A handler can deny at any layer.
    """

    @pytest.fixture
    def user(self) -> User:
        return User(identity="alice", scopes=[])

    @pytest.fixture
    def request_body(self) -> CronCreate:
        return CronCreate(input={"q": 1}, assistant_id="agent-1", schedule="*/5 * * * *")

    @pytest.mark.asyncio
    async def test_stateless_create_fires_full_chain(self, user: User, request_body: CronCreate) -> None:
        with patch("aegra_api.api.crons.handle_event", new_callable=AsyncMock) as mock_handle:
            await _authorize_cron_create(user, request_body, thread_id=None)

        # Three events: crons.create, assistants.read, threads.search.
        assert mock_handle.await_count == 3
        contexts = [call.args[0] for call in mock_handle.await_args_list]
        values = [call.args[1] for call in mock_handle.await_args_list]

        assert (contexts[0].resource, contexts[0].action) == ("crons", "create")
        assert (contexts[1].resource, contexts[1].action) == ("assistants", "read")
        assert (contexts[2].resource, contexts[2].action) == ("threads", "search")

        assert values[1] == {"assistant_id": "agent-1"}
        assert values[2] == {}

    @pytest.mark.asyncio
    async def test_thread_bound_create_fires_threads_read_instead_of_search(
        self, user: User, request_body: CronCreate
    ) -> None:
        with patch("aegra_api.api.crons.handle_event", new_callable=AsyncMock) as mock_handle:
            await _authorize_cron_create(user, request_body, thread_id="t-42")

        assert mock_handle.await_count == 3
        contexts = [call.args[0] for call in mock_handle.await_args_list]
        values = [call.args[1] for call in mock_handle.await_args_list]

        assert (contexts[0].resource, contexts[0].action) == ("crons", "create")
        assert (contexts[1].resource, contexts[1].action) == ("assistants", "read")
        assert (contexts[2].resource, contexts[2].action) == ("threads", "read")

        # Crons.create value carries thread_id; threads.read value targets that thread.
        assert values[0]["thread_id"] == "t-42"
        assert values[2] == {"thread_id": "t-42"}

    @pytest.mark.asyncio
    async def test_chain_stops_when_assistants_read_denies(self, user: User, request_body: CronCreate) -> None:
        """A 403 from assistants.read must short-circuit before threads.read fires."""
        from fastapi import HTTPException

        async def fake_handle(ctx, _value: dict[str, object]) -> None:
            if ctx.resource == "assistants":
                raise HTTPException(status_code=403, detail="denied")

        with (
            patch("aegra_api.api.crons.handle_event", side_effect=fake_handle) as mock_handle,
            pytest.raises(HTTPException) as exc_info,
        ):
            await _authorize_cron_create(user, request_body, thread_id="t-42")

        assert exc_info.value.status_code == 403
        # Only crons.create and assistants.read should have run.
        assert mock_handle.await_count == 2


class TestGetCron:
    """Regression tests for get-by-ID authorization and ownership filtering."""

    @pytest.mark.asyncio
    async def test_applies_auth_filters_to_owned_cron_query(self) -> None:
        user = User(identity="alice", scopes=[])
        session = AsyncMock()
        cron = Mock()
        session.scalar.return_value = cron
        expected = object()
        response = Response()

        with (
            patch(
                "aegra_api.api.crons.handle_event",
                new_callable=AsyncMock,
                return_value={"team_id": "eng"},
            ) as mock_handle,
            patch("aegra_api.api.crons._cron_to_response", return_value=expected),
        ):
            result = await get_cron("cron-42", response, user, session)

        context, value = mock_handle.await_args.args
        assert (context.resource, context.action) == ("crons", "read")
        assert value == {"cron_id": "cron-42"}
        stmt = session.scalar.await_args.args[0]
        compiled = stmt.compile()
        assert "crons.cron_id" in str(compiled)
        assert "crons.user_id" in str(compiled)
        assert "crons.metadata" in str(compiled)
        assert compiled.params["cron_id_1"] == "cron-42"
        assert compiled.params["user_id_1"] == "alice"
        assert compiled.params["metadata_1"] == {"team_id": "eng"}
        assert result is expected

    @pytest.mark.asyncio
    async def test_returns_404_when_cron_is_missing_or_filtered_out(self) -> None:
        user = User(identity="alice", scopes=[])
        session = AsyncMock()
        session.scalar.return_value = None
        response = Response()

        with (
            patch(
                "aegra_api.api.crons.handle_event",
                new_callable=AsyncMock,
                return_value={"team_id": "eng"},
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_cron("cron-42", response, user, session)

        assert exc_info.value.status_code == 404
