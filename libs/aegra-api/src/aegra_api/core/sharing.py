"""Assistant sharing visibility — owner composite isolation + shared grants.

Read/execute paths use visible_assistant_filter; write operations stay on
owner_filter — grantees must not modify someone else's assistant.
"""

from sqlalchemy import ColumnElement, or_, select

from aegra_api.core.authz import owner_filter
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import AssistantShare as AssistantShareORM
from aegra_api.models.auth import User


def grantees(user: User) -> list[str]:
    """Grant targets matching this user: public, the user, and their tenant."""
    values = ["public", f"user:{user.user_id}"]
    if user.tenant_id is not None:
        values.append(f"tenant:{user.tenant_id}")
    return values


def visible_assistant_filter(user: User) -> ColumnElement[bool]:
    """Assistant read/execute visibility: owner composite isolation OR system OR shared."""
    shared = select(AssistantShareORM.assistant_id).where(AssistantShareORM.grantee.in_(grantees(user)))
    return or_(
        owner_filter(AssistantORM, user, allow_system=True),
        AssistantORM.assistant_id.in_(shared),
    )
