"""Unit tests for cron API helpers."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aegra_api.api.crons import _authorize_cron_create, create_cron, create_cron_for_thread
from aegra_api.models import User
from aegra_api.models.crons import CronCreate, CronResponse


def _make_cron_row(*, thread_id: str | None = None, enabled: bool = True) -> SimpleNamespace:
    now = datetime.now(UTC)
    return SimpleNamespace(
        cron_id="cron-001",
        assistant_id="agent",
        thread_id=thread_id,
        on_run_completed=None,
        end_time=None,
        schedule="*/5 * * * *",
        created_at=now,
        updated_at=now,
        payload={"input": {"messages": [{"role": "user", "content": "hi"}]}},
        user_id="test-user",
        next_run_date=now,
        metadata_dict={},
        enabled=enabled,
    )


class TestCreateCronReturnsCron:
    """Create endpoints return the persisted Cron, matching LangSmith Deployments."""

    @pytest.fixture
    def user(self) -> User:
        return User(identity="test-user", scopes=[])

    @pytest.fixture
    def request_body(self) -> CronCreate:
        return CronCreate(input={"q": 1}, assistant_id="agent", schedule="*/5 * * * *")

    @pytest.mark.asyncio
    async def test_stateless_create_returns_cron(self, user: User, request_body: CronCreate) -> None:
        service = Mock(create_cron=AsyncMock(return_value=_make_cron_row()))
        with patch("aegra_api.api.crons.handle_event", new_callable=AsyncMock):
            result = await create_cron(request_body, user, service)

        assert isinstance(result, CronResponse)
        assert result.cron_id == "cron-001"
        service.create_cron.assert_awaited_once_with(request_body, "test-user")

    @pytest.mark.asyncio
    async def test_thread_create_returns_cron(self, user: User, request_body: CronCreate) -> None:
        service = Mock(create_cron=AsyncMock(return_value=_make_cron_row(thread_id="thread-001")))
        session = AsyncMock()
        session.scalar.return_value = SimpleNamespace(user_id="test-user")
        with patch("aegra_api.api.crons.handle_event", new_callable=AsyncMock):
            result = await create_cron_for_thread("thread-001", request_body, user, service, session)

        assert isinstance(result, CronResponse)
        assert result.thread_id == "thread-001"
        service.create_cron.assert_awaited_once_with(request_body, "test-user", thread_id="thread-001")


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
