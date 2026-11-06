"""Messages sent to a running run with the Agent Protocol v2 ``input.inject`` command.

The command stores each message in ``run_inputs``; the graph reads them through the
``InjectedInput`` reader Aegra pins on every run's config under ``INJECTED_INPUT_KEY``:

* ``accept()`` declares that the run takes live input. Until a pending or running
  run of the thread has declared it, ``input.inject`` is refused.
* ``pending()`` returns the messages this attempt has not taken, oldest first, each
  a message dict a ``messages`` reducer accepts (``id`` is the message id).
* ``take(ids)`` claims messages for this run and returns the ids it claimed.

A claim holds while the run is pending or running, so a run requeued after a
worker drain or crash sees again what its earlier attempt took. When the graph
runs to its end, its claims become delivered. A run that pauses at an interrupt,
fails, or is cancelled leaves them to the thread's next run, the resume included,
which may therefore see a message a node before the pause already added to the
state: the graph takes it again, so it settles, and skips it, because adding it
under the same id would only replace it in the state while a model called with
it would read it twice. A message no run took expires ``INPUT_INJECT_TTL_SECONDS``
after it was sent; one a run took stays until a run delivers it, and a delivered
one is kept as long after it was sent, so a repeat of its id is still stored once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Final, Literal
from uuid import uuid4

from sqlalchemy import ColumnElement, and_, delete, exists, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import RunInput as RunInputORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.settings import settings

INJECTED_INPUT_KEY: Final = "aegra_injected_input"

_ACTIVE_RUN_STATES = ("pending", "running")

Admission = Literal["accepting", "not_accepting", "no_active_run"]


@dataclass(frozen=True, slots=True)
class InjectedMessage:
    """One injected message as the graph receives it."""

    message_id: str
    message: dict[str, Any]


def _expired() -> ColumnElement[bool]:
    """Sent longer than the TTL ago and held by no run: never taken, or already delivered."""
    ttl = timedelta(seconds=settings.event_streaming.INPUT_INJECT_TTL_SECONDS)
    return and_(
        RunInputORM.created_at <= func.now() - ttl,
        or_(RunInputORM.taken_by_run.is_(None), RunInputORM.delivered.is_(True)),
    )


def _available_to(run_id: str, attempt: str) -> ColumnElement[bool]:
    """Undelivered and unexpired; unclaimed, claimed by an earlier attempt of ``run_id``, or by a run that ended."""
    claimant_active = exists().where(
        RunORM.run_id == RunInputORM.taken_by_run,
        RunORM.status.in_(_ACTIVE_RUN_STATES),
    )
    return and_(
        RunInputORM.delivered.is_(False),
        ~_expired(),
        or_(
            RunInputORM.taken_by_run.is_(None),
            and_(RunInputORM.taken_by_run == run_id, RunInputORM.taken_by_attempt != attempt),
            ~claimant_active,
        ),
    )


@dataclass(frozen=True, slots=True)
class InjectedInput:
    """A run's handle on its thread's injected messages; pinned by the server, never built by clients."""

    run_id: str
    thread_id: str
    user_id: str
    # One per execution: a requeued attempt of the run gets a new reader.
    attempt: str = field(default_factory=lambda: uuid4().hex)

    def _scope(self) -> tuple[ColumnElement[bool], ...]:
        return (RunInputORM.thread_id == self.thread_id, RunInputORM.user_id == self.user_id)

    async def accept(self) -> None:
        """Declare that this run takes input.inject."""
        maker = _get_session_maker()
        async with maker() as session:
            await session.execute(
                update(RunORM)
                .where(RunORM.run_id == self.run_id, RunORM.user_id == self.user_id)
                .values(accepts_input_inject=True)
            )
            await session.commit()

    async def pending(self) -> list[InjectedMessage]:
        """Messages this run may take, in the order they were sent."""
        maker = _get_session_maker()
        async with maker() as session:
            rows = await session.execute(
                select(RunInputORM.message_id, RunInputORM.message)
                .where(*self._scope(), _available_to(self.run_id, self.attempt))
                .order_by(RunInputORM.seq)
            )
            return [InjectedMessage(message_id=message_id, message=message) for message_id, message in rows.all()]

    async def take(self, message_ids: Sequence[str]) -> list[str]:
        """Claim messages for this run; returns the ids claimed, which excludes any another run holds."""
        if not message_ids:
            return []
        maker = _get_session_maker()
        async with maker() as session:
            result = await session.execute(
                update(RunInputORM)
                .where(
                    *self._scope(),
                    RunInputORM.message_id.in_(message_ids),
                    _available_to(self.run_id, self.attempt),
                )
                .values(taken_by_run=self.run_id, taken_by_attempt=self.attempt)
                .returning(RunInputORM.message_id)
            )
            taken = set(result.scalars().all())
            await session.commit()
        return [message_id for message_id in message_ids if message_id in taken]


async def thread_admission(session: AsyncSession, *, thread_id: str, user_id: str) -> Admission:
    """Whether the caller's thread has an active run that takes input.inject."""
    rows = await session.scalars(
        select(RunORM.accepts_input_inject).where(
            RunORM.thread_id == thread_id,
            RunORM.user_id == user_id,
            RunORM.status.in_(_ACTIVE_RUN_STATES),
        )
    )
    declarations = rows.all()
    if not declarations:
        return "no_active_run"
    return "accepting" if any(declarations) else "not_accepting"


async def store_message(
    session: AsyncSession, *, thread_id: str, user_id: str, message_id: str, message: dict[str, Any]
) -> bool:
    """Store a message for the thread's runs; commits.

    An identical repeat of a message the thread still holds is stored once and returns True.
    Returns False when the id already names a different message.
    """
    owned = (RunInputORM.thread_id == thread_id, RunInputORM.user_id == user_id)
    await session.execute(delete(RunInputORM).where(*owned, _expired()))
    inserted = await session.scalar(
        insert(RunInputORM)
        .values(thread_id=thread_id, message_id=message_id, message=message, user_id=user_id)
        .on_conflict_do_nothing(index_elements=["thread_id", "message_id"])
        .returning(RunInputORM.seq)
    )
    existing = None
    if inserted is None:
        existing = await session.scalar(select(RunInputORM.message).where(*owned, RunInputORM.message_id == message_id))
    await session.commit()
    return inserted is not None or existing == message


async def mark_delivered(*, run_id: str, thread_id: str, user_id: str) -> None:
    """Mark what a finished graph took as delivered and drop the thread's expired messages."""
    owned = (RunInputORM.thread_id == thread_id, RunInputORM.user_id == user_id)
    maker = _get_session_maker()
    async with maker() as session:
        await session.execute(
            update(RunInputORM).where(*owned, RunInputORM.taken_by_run == run_id).values(delivered=True)
        )
        await session.execute(delete(RunInputORM).where(*owned, _expired()))
        await session.commit()
