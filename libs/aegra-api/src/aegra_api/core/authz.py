"""Tenant/user ownership filtering — the single source of truth for isolation.

Every request-facing tenant query MUST build its WHERE via owner_filter; never
hand-write user_id/tenant_id filters (one miss is a privilege escalation — see
GHSA-m98r-6667-4wq7). Switching isolation semantics (composite <-> tenant-shared)
means changing only this one function; call sites stay put.
"""

from typing import Any

from sqlalchemy import ColumnElement, and_, or_

from aegra_api.models.auth import User


def scope(model: Any, user_id: str, tenant_id: str | None, *, allow_system: bool = False) -> ColumnElement[bool]:
    """Low-level ownership filter built directly from user_id + tenant_id.

    For the service layer, which holds a string id rather than a User. tenant_id
    uses IS NOT DISTINCT FROM: None matches rows whose tenant_id is NULL, a value
    matches only equal rows (the hard tenant boundary). allow_system=True also
    admits shared rows where user_id == "system" (e.g. built-in assistants);
    keep it False for writes.
    """
    own = and_(
        model.user_id == user_id,
        model.tenant_id.is_not_distinct_from(tenant_id),
    )
    if allow_system:
        return or_(own, model.user_id == "system")
    return own


def owner_filter(model: Any, user: User, *, allow_system: bool = False) -> ColumnElement[bool]:
    """User-object wrapper around scope, for the api layer that holds a User."""
    return scope(model, user.user_id, getattr(user, "tenant_id", None), allow_system=allow_system)


def owns(obj: Any, user: User, *, allow_system: bool = False) -> bool:
    """Python-level ownership check, matching owner_filter (for fetched ORM rows).

    tenant is compared with ==: in Python None == None is True, equivalent to SQL
    IS NOT DISTINCT FROM. Used when a row is fetched by id first, then checked for
    ownership (e.g. an optionally-existing thread).
    """
    if obj.user_id == user.user_id and getattr(obj, "tenant_id", None) == getattr(user, "tenant_id", None):
        return True
    return bool(allow_system and obj.user_id == "system")


def query_scope(
    model: Any,
    *,
    user_id: str | None = None,
    tenant_id: str | None = None,
    allow_system: bool = False,
) -> ColumnElement[bool]:
    """Build a query filter by user_id, tenant_id, or both (requirement B).

    Unlike scope, here a None user_id/tenant_id means "do not filter on that
    dimension" (omit), for explicit searches; scope's tenant_id=None means "match
    NULL tenant" (hard isolation). Request-facing isolation still goes through
    scope/owner_filter — do not use this as a replacement. At least one dimension
    is required, else ValueError — no unbounded full-table scan.
    """
    conds: list[ColumnElement[bool]] = []
    if user_id is not None:
        conds.append(model.user_id == user_id)
    if tenant_id is not None:
        conds.append(model.tenant_id == tenant_id)
    if not conds:
        raise ValueError("query_scope requires at least one filter dimension (user_id or tenant_id)")
    own = and_(*conds) if len(conds) > 1 else conds[0]
    if allow_system:
        return or_(own, model.user_id == "system")
    return own
