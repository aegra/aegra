"""Assistant sharing visibility — owner composite isolation + shared (user/tenant/public).

Read/execute paths use visible_assistant_filter; write operations
(update/delete/set_latest) stay on owner_filter — grantees must not modify
someone else's assistant.
"""

from sqlalchemy import ColumnElement, and_, or_, select

from aegra_api.core.authz import owner_filter
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import AssistantShare as AssistantShareORM
from aegra_api.models.auth import User


def shared_to_user(user: User) -> ColumnElement[bool]:
    """Match assistant_share rows shared to this user: public / this user / this tenant."""
    conds: list[ColumnElement[bool]] = [
        AssistantShareORM.share_type == "public",
        and_(
            AssistantShareORM.share_type == "user",
            AssistantShareORM.target_user_id == user.user_id,
        ),
    ]
    if getattr(user, "tenant_id", None) is not None:
        conds.append(
            and_(
                AssistantShareORM.share_type == "tenant",
                AssistantShareORM.target_tenant_id == user.tenant_id,
            )
        )
    return or_(*conds)


def visible_assistant_filter(user: User) -> ColumnElement[bool]:
    """Assistant read/execute visibility: owner composite isolation OR system OR shared."""
    shared_ids = select(AssistantShareORM.assistant_id).where(shared_to_user(user))
    return or_(
        owner_filter(AssistantORM, user, allow_system=True),
        AssistantORM.assistant_id.in_(shared_ids),
    )
