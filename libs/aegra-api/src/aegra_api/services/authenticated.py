"""Service base that makes `@auth.on.*` dispatch mandatory at the call site.

Putting dispatch on the service (not the route) means an endpoint cannot reach
the database without authorizing first. `handle_event` is default-allow when no
handler matches; the SQL-layer `user_id == user.identity` filter is the tenant
boundary (GHSA-m98r-6667-4wq7).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.auth_handlers import build_auth_context, handle_event
from aegra_api.models.auth import User


class Authenticated:
    """Carries the request identity and the single `_dispatch` entry point.

    Subclasses set the `resource` class attribute (e.g. "threads", "assistants")
    and call `await self._dispatch(action, value)` at the top of each public
    method, applying the returned filter dict to their query.
    """

    resource: str

    def __init__(self, session: AsyncSession, user: User) -> None:
        self.session = session
        self.user = user

    async def _dispatch(self, action: str, value: dict[str, Any]) -> dict[str, Any] | None:
        """Authorize `action` on this resource. Returns handler filters, or None.

        Mutates `value` in place if the handler injects metadata. Raises
        HTTPException(403) when a registered handler denies.
        """
        ctx = build_auth_context(self.user, self.resource, action)
        return await handle_event(ctx, value)
