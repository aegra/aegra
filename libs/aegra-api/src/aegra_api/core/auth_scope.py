"""Apply a handler's returned filter on routes that reach a thread by id without querying it."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.auth_filters import build_metadata_filter
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import get_session_maker
from aegra_api.models.auth import User


async def scalar_in_session(stmt: Select[Any], session: AsyncSession | None) -> Any:
    """Run on the caller's session when it has one: a second checkout per request can drain the pool."""
    if session is not None:
        return await session.scalar(stmt)
    async with get_session_maker()() as own:
        return await own.scalar(stmt)


async def ensure_thread_visible(
    thread_id: str, user: User, filters: dict[str, Any] | None, *, session: AsyncSession | None = None
) -> None:
    """404 unless the caller's thread matches the handler filter, the answer GET /threads/{id} gives.

    Without this, a filter that hides a thread from GET /threads/{id} still lets its state,
    history and runs be read by anyone who knows the id.
    """
    auth_filter = build_metadata_filter(ThreadORM.metadata_json, filters)
    if auth_filter is None:
        return
    stmt = select(ThreadORM.thread_id).where(
        ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity, auth_filter
    )
    if await scalar_in_session(stmt, session) is None:
        raise HTTPException(404, f"Thread '{thread_id}' not found")
