"""Base class that makes authorization dispatch mandatory at the service layer.

Every tenant-scoped service inherits from `Authenticated` and calls `_dispatch`
at the entry of each public method. This hoists the `@auth.on.*` dispatch out of
route handlers so no route can forget to authorize (closing the gaps tracked in
the auth dispatch spec). Dispatch policy itself is unchanged: `handle_event`
stays default-allow when no handler matches; the SQL-layer `user_id ==
user.identity` filter remains the tenant boundary (GHSA-m98r-6667-4wq7).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.auth_handlers import build_auth_context, handle_event
from aegra_api.models.auth import User


class Authenticated:
    """Service base carrying the request identity and a single dispatch entry point.

    Subclasses set the `resource` class attribute (e.g. "threads", "assistants")
    and call `await self._dispatch(action, value)` at the top of each public
    method. The returned filter dict (or None) is applied to the query the same
    way the route layer applied it before.
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
