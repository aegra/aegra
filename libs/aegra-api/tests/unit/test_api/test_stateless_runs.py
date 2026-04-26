"""Unit tests for stateless (thread-free) run endpoints."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.responses import StreamingResponse
from sse_starlette import EventSourceResponse

from aegra_api.api.stateless_runs import (
    _cleanup_after_background_run,
    _delete_thread_by_id,
    stateless_create_run,
    stateless_stream_run,
    stateless_wait_for_run,
)
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.models import Run, RunCreate, User


class TestDeleteThreadById:
    """Tests for the _delete_thread_by_id helper."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        session = AsyncMock()
        session.delete = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_deletes_thread_with_cascade(self, mock_session: AsyncMock) -> None:
        """Thread and its runs are deleted via cascade."""
        thread_id = str(uuid4())
        user_id = "test-user"

        thread_orm = ThreadORM(
            thread_id=thread_id,
            user_id=user_id,
            status="idle",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        # No active runs
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_session.scalars.return_value = mock_scalars

        # Thread lookup returns the thread
        mock_session.scalar.return_value = thread_orm

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("aegra_api.api.stateless_runs._get_session_maker", return_value=mock_maker):
            await _delete_thread_by_id(thread_id, user_id)

        mock_session.delete.assert_called_once_with(thread_orm)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancels_active_runs_before_delete(self, mock_session: AsyncMock) -> None:
        """Active runs are cancelled before thread deletion."""
        thread_id = str(uuid4())
        user_id = "test-user"
        run_id = str(uuid4())

        active_run = RunORM(
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            status="running",
            input={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [active_run]
        mock_session.scalars.return_value = mock_scalars
        mock_session.scalar.return_value = None  # Thread already gone

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_task = MagicMock()
        mock_task.done.return_value = True

        with (
            patch("aegra_api.api.stateless_runs._get_session_maker", return_value=mock_maker),
            patch(
                "aegra_api.api.stateless_runs.streaming_service.cancel_run",
                new_callable=AsyncMock,
            ) as mock_cancel,
            patch("aegra_api.api.stateless_runs.active_runs", {run_id: mock_task}),
        ):
            await _delete_thread_by_id(thread_id, user_id)

        mock_cancel.assert_called_once_with(run_id)

    @pytest.mark.asyncio
    async def test_noop_when_thread_not_found(self, mock_session: AsyncMock) -> None:
        """No error when thread doesn't exist (idempotent)."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_session.scalars.return_value = mock_scalars
        mock_session.scalar.return_value = None

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("aegra_api.api.stateless_runs._get_session_maker", return_value=mock_maker):
            await _delete_thread_by_id("nonexistent", "user")

        mock_session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_cancelled_error_on_task_await(self, mock_session: AsyncMock) -> None:
        """CancelledError from awaiting a cancelled task is silently absorbed."""
        thread_id = str(uuid4())
        user_id = "test-user"
        run_id = str(uuid4())

        active_run = RunORM(
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            status="running",
            input={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [active_run]
        mock_session.scalars.return_value = mock_scalars
        mock_session.scalar.return_value = None

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

        # Use a Future that raises CancelledError when awaited
        fut = asyncio.get_event_loop().create_future()
        fut.cancel()

        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel.return_value = None
        mock_task.__await__ = fut.__await__

        with (
            patch("aegra_api.api.stateless_runs._get_session_maker", return_value=mock_maker),
            patch(
                "aegra_api.api.stateless_runs.streaming_service.cancel_run",
                new_callable=AsyncMock,
            ),
            patch("aegra_api.api.stateless_runs.active_runs", {run_id: mock_task}),
        ):
            # Should not raise — CancelledError is caught
            await _delete_thread_by_id(thread_id, user_id)

    @pytest.mark.asyncio
    async def test_logs_exception_on_task_await_error(self, mock_session: AsyncMock) -> None:
        """Generic exception from awaiting a task is logged but not re-raised."""
        thread_id = str(uuid4())
        user_id = "test-user"
        run_id = str(uuid4())

        active_run = RunORM(
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            status="running",
            input={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [active_run]
        mock_session.scalars.return_value = mock_scalars
        mock_session.scalar.return_value = None

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

        # Use a Future that raises RuntimeError when awaited
        fut = asyncio.get_event_loop().create_future()
        fut.set_exception(RuntimeError("task exploded"))

        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel.return_value = None
        mock_task.__await__ = fut.__await__

        with (
            patch("aegra_api.api.stateless_runs._get_session_maker", return_value=mock_maker),
            patch(
                "aegra_api.api.stateless_runs.streaming_service.cancel_run",
                new_callable=AsyncMock,
            ),
            patch("aegra_api.api.stateless_runs.active_runs", {run_id: mock_task}),
        ):
            # Should not raise — exception is logged
            await _delete_thread_by_id(thread_id, user_id)


class TestCleanupAfterBackgroundRun:
    """Tests for the _cleanup_after_background_run helper."""

    @pytest.mark.asyncio
    async def test_awaits_task_then_deletes(self) -> None:
        """Waits for the background task to finish, then deletes the thread."""
        run_id = str(uuid4())
        thread_id = str(uuid4())
        user_id = "test-user"

        task_awaited = False

        async def _fake_task() -> None:
            nonlocal task_awaited
            task_awaited = True

        # Create a real asyncio.Task so `await task` works
        task = asyncio.create_task(_fake_task())
        await task  # let it finish before test to avoid timing issues

        with (
            patch("aegra_api.api.stateless_runs.active_runs", {run_id: task}),
            patch(
                "aegra_api.api.stateless_runs._delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
        ):
            await _cleanup_after_background_run(run_id, thread_id, user_id)

        assert task_awaited
        mock_delete.assert_called_once_with(thread_id, user_id)

    @pytest.mark.asyncio
    async def test_deletes_thread_when_no_task_in_active_runs(self) -> None:
        """Cleanup proceeds directly to thread deletion when run_id is not in active_runs."""
        run_id = str(uuid4())
        thread_id = str(uuid4())
        user_id = "test-user"

        with (
            patch("aegra_api.api.stateless_runs.active_runs", {}),
            patch(
                "aegra_api.api.stateless_runs._delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
        ):
            await _cleanup_after_background_run(run_id, thread_id, user_id)

        mock_delete.assert_called_once_with(thread_id, user_id)


class TestStatelessWaitForRun:
    """Tests for POST /runs/wait."""

    @pytest.fixture
    def mock_user(self) -> User:
        return User(identity="test-user", scopes=[])

    @pytest.mark.asyncio
    async def test_delegates_and_deletes_thread(self, mock_user: User) -> None:
        """Delegates to wait_for_run and deletes ephemeral thread after stream."""
        import json

        expected_output = {"result": "done"}
        request = RunCreate(assistant_id="agent", input={"msg": "hi"})

        # wait_for_run now returns a StreamingResponse, so mock it accordingly
        mock_response = StreamingResponse(
            iter([json.dumps(expected_output).encode()]),
            media_type="application/json",
        )

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-1"),
            patch(
                "aegra_api.api.stateless_runs.wait_for_run",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_wait,
            patch(
                "aegra_api.api.stateless_runs._delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
        ):
            result = await stateless_wait_for_run(request, mock_user)

            # Result is a StreamingResponse; consume to trigger cleanup
            assert isinstance(result, StreamingResponse)
            body = b""
            async for chunk in result.body_iterator:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()

            assert json.loads(body) == expected_output
            mock_wait.assert_called_once_with("eph-thread-1", request, mock_user)
            mock_delete.assert_called_once_with("eph-thread-1", mock_user.identity)

    @pytest.mark.asyncio
    async def test_keeps_thread_when_requested(self, mock_user: User) -> None:
        """Thread is preserved when on_completion='keep'."""
        import json

        request = RunCreate(assistant_id="agent", input={"msg": "hi"}, on_completion="keep")

        mock_response = StreamingResponse(
            iter([json.dumps({}).encode()]),
            media_type="application/json",
        )

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-2"),
            patch(
                "aegra_api.api.stateless_runs.wait_for_run",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch(
                "aegra_api.api.stateless_runs._delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
        ):
            result = await stateless_wait_for_run(request, mock_user)

        # Returns original response unchanged (no wrapper)
        assert result is mock_response
        mock_delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleans_up_on_error(self, mock_user: User) -> None:
        """Thread is deleted even when wait_for_run raises."""
        request = RunCreate(assistant_id="agent", input={"msg": "hi"})

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-3"),
            patch(
                "aegra_api.api.stateless_runs.wait_for_run",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "aegra_api.api.stateless_runs._delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
            pytest.raises(RuntimeError, match="boom"),
        ):
            await stateless_wait_for_run(request, mock_user)

        mock_delete.assert_called_once_with("eph-thread-3", mock_user.identity)

    @pytest.mark.asyncio
    async def test_cleanup_failure_does_not_mask_original_error(self, mock_user: User) -> None:
        """If _delete_thread_by_id raises during cleanup, the original error propagates."""
        request = RunCreate(assistant_id="agent", input={"msg": "hi"})

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-err"),
            patch(
                "aegra_api.api.stateless_runs.wait_for_run",
                new_callable=AsyncMock,
                side_effect=RuntimeError("original"),
            ),
            patch(
                "aegra_api.api.stateless_runs._delete_thread_by_id",
                new_callable=AsyncMock,
                side_effect=OSError("cleanup failed"),
            ),
            pytest.raises(RuntimeError, match="original"),
        ):
            await stateless_wait_for_run(request, mock_user)


class TestStatelessStreamRun:
    """Tests for POST /runs/stream."""

    @pytest.fixture
    def mock_user(self) -> User:
        return User(identity="test-user", scopes=[])

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        session = AsyncMock()
        session.refresh = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.mark.asyncio
    async def test_delegates_and_wraps_body_for_cleanup(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Delegates to create_and_stream_run and wraps iterator for cleanup."""
        request = RunCreate(assistant_id="agent", input={"msg": "hi"})

        async def _fake_body() -> AsyncIterator[bytes]:
            yield b"event: data\n\n"

        inner_close_handler = AsyncMock()
        mock_response = EventSourceResponse(
            _fake_body(),
            headers={"Location": "/threads/t/runs/r/stream"},
            client_close_handler_callable=inner_close_handler,
        )

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-4"),
            patch(
                "aegra_api.api.stateless_runs.create_and_stream_run",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_stream,
            patch(
                "aegra_api.api.stateless_runs._delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
        ):
            result = await stateless_stream_run(request, mock_user, mock_session)

            assert isinstance(result, EventSourceResponse)
            mock_stream.assert_called_once_with("eph-thread-4", request, mock_user, mock_session)
            # Outer response must re-expose the inner close handler so real
            # http.disconnect still cancels the run.
            assert result.client_close_handler_callable is inner_close_handler

            # Consume the iterator to trigger cleanup (must be inside mock context)
            chunks: list[bytes] = []
            async for chunk in result.body_iterator:
                chunks.append(chunk)

            assert len(chunks) > 0
            mock_delete.assert_called_once_with("eph-thread-4", mock_user.identity)

    @pytest.mark.asyncio
    async def test_passes_through_when_keep(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Returns original response unchanged when on_completion='keep'."""
        request = RunCreate(assistant_id="agent", input={"msg": "hi"}, on_completion="keep")

        async def _fake_body() -> AsyncIterator[bytes]:
            yield b"event: data\n\n"

        mock_response = EventSourceResponse(_fake_body())

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-5"),
            patch(
                "aegra_api.api.stateless_runs.create_and_stream_run",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch(
                "aegra_api.api.stateless_runs._delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
        ):
            result = await stateless_stream_run(request, mock_user, mock_session)

        # Should return original response, not wrapped
        assert result is mock_response
        mock_delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleans_up_thread_when_delegation_raises(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Thread is deleted if create_and_stream_run raises (e.g. assistant not found)."""
        request = RunCreate(assistant_id="agent", input={"msg": "hi"})

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-err"),
            patch(
                "aegra_api.api.stateless_runs.create_and_stream_run",
                new_callable=AsyncMock,
                side_effect=RuntimeError("setup failed"),
            ),
            patch(
                "aegra_api.api.stateless_runs._delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
            pytest.raises(RuntimeError, match="setup failed"),
        ):
            await stateless_stream_run(request, mock_user, mock_session)

        mock_delete.assert_called_once_with("eph-thread-err", mock_user.identity)

    @pytest.mark.asyncio
    async def test_early_disconnect_keeps_thread(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Client disconnect before stream completion must NOT delete the thread.

        Regression: deleting the thread here would cancel active runs via
        ``_delete_thread_by_id`` and break the ``on_disconnect="continue"``
        contract. The wrapper must mirror ``stateless_wait_for_run`` and only
        delete on normal completion.
        """
        request = RunCreate(assistant_id="agent", input={"msg": "hi"}, on_disconnect="continue")

        async def _fake_body() -> AsyncIterator[bytes]:
            yield b"event: metadata\n\n"
            # Client disconnect — outer EventSourceResponse cancels the iterator
            raise asyncio.CancelledError

        mock_response = EventSourceResponse(
            _fake_body(),
            headers={"Location": "/threads/t/runs/r/stream"},
        )

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-disc"),
            patch(
                "aegra_api.api.stateless_runs.create_and_stream_run",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch(
                "aegra_api.api.stateless_runs._delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
        ):
            result = await stateless_stream_run(request, mock_user, mock_session)

            # Drain until CancelledError bubbles up from the iterator
            with contextlib.suppress(asyncio.CancelledError):
                async for _chunk in result.body_iterator:
                    pass

        mock_delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_cleanup_failure_is_logged_not_raised(self, mock_user: User, mock_session: AsyncMock) -> None:
        """If _delete_thread_by_id raises during stream cleanup, it is logged but not propagated."""
        request = RunCreate(assistant_id="agent", input={"msg": "hi"})

        async def _fake_body() -> AsyncIterator[bytes]:
            yield b"event: data\n\n"

        mock_response = EventSourceResponse(_fake_body())

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-cleanup"),
            patch(
                "aegra_api.api.stateless_runs.create_and_stream_run",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch(
                "aegra_api.api.stateless_runs._delete_thread_by_id",
                new_callable=AsyncMock,
                side_effect=OSError("cleanup failed"),
            ),
        ):
            result = await stateless_stream_run(request, mock_user, mock_session)

            # Consuming the iterator should not raise despite cleanup failure
            chunks: list[bytes] = []
            async for chunk in result.body_iterator:
                chunks.append(chunk)

            assert len(chunks) > 0


class TestStatelessCreateRun:
    """Tests for POST /runs."""

    @pytest.fixture
    def mock_user(self) -> User:
        return User(identity="test-user", scopes=[])

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        session = AsyncMock()
        session.refresh = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.mark.asyncio
    async def test_delegates_and_schedules_cleanup(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Delegates to create_run and schedules background cleanup."""
        run_id = str(uuid4())
        request = RunCreate(assistant_id="agent", input={"msg": "hi"})

        mock_run = Run(
            run_id=run_id,
            thread_id="eph-thread-6",
            assistant_id="agent",
            status="pending",
            input={"msg": "hi"},
            user_id=mock_user.identity,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-6"),
            patch(
                "aegra_api.api.stateless_runs.create_run",
                new_callable=AsyncMock,
                return_value=mock_run,
            ) as mock_create,
            patch("aegra_api.api.stateless_runs.asyncio.create_task") as mock_create_task,
        ):
            result = await stateless_create_run(request, mock_user, mock_session)

        assert result.run_id == run_id
        mock_create.assert_called_once_with("eph-thread-6", request, mock_user, mock_session)
        mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_cleanup_when_keep(self, mock_user: User, mock_session: AsyncMock) -> None:
        """No background cleanup task when on_completion='keep'."""
        run_id = str(uuid4())
        request = RunCreate(assistant_id="agent", input={"msg": "hi"}, on_completion="keep")

        mock_run = Run(
            run_id=run_id,
            thread_id="eph-thread-7",
            assistant_id="agent",
            status="pending",
            input={"msg": "hi"},
            user_id=mock_user.identity,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-7"),
            patch(
                "aegra_api.api.stateless_runs.create_run",
                new_callable=AsyncMock,
                return_value=mock_run,
            ),
            patch("aegra_api.api.stateless_runs.asyncio.create_task") as mock_create_task,
        ):
            result = await stateless_create_run(request, mock_user, mock_session)

        assert result.run_id == run_id
        mock_create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleans_up_thread_when_delegation_raises(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Thread is deleted if create_run raises after auto-creating the thread."""
        request = RunCreate(assistant_id="agent", input={"msg": "hi"})

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-err"),
            patch(
                "aegra_api.api.stateless_runs.create_run",
                new_callable=AsyncMock,
                side_effect=RuntimeError("create failed"),
            ),
            patch(
                "aegra_api.api.stateless_runs._delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
            pytest.raises(RuntimeError, match="create failed"),
        ):
            await stateless_create_run(request, mock_user, mock_session)

        mock_delete.assert_called_once_with("eph-thread-err", mock_user.identity)
