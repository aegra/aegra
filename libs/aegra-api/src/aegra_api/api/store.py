"""Store endpoints for Agent Protocol"""

from collections.abc import Mapping
from functools import cache

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from aegra_api.config import load_store_config
from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.auth_handlers import build_auth_context, handle_event
from aegra_api.core.database import db_manager
from aegra_api.models import (
    StoreDeleteRequest,
    StoreGetResponse,
    StoreItem,
    StoreListNamespacesRequest,
    StoreListNamespacesResponse,
    StorePutRequest,
    StoreSearchRequest,
    StoreSearchResponse,
    User,
)
from aegra_api.models.errors import BAD_REQUEST, NOT_FOUND

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Store"], dependencies=auth_dependency)


@router.put("/store/items", status_code=204)
async def put_store_item(request: StorePutRequest, user: User = Depends(get_current_user)) -> Response:
    """Create or update an item in the store.

    If an item with the same namespace and key already exists, its value is
    overwritten. Values must be JSON objects (dictionaries).
    """
    # Authorization check
    ctx = build_auth_context(user, "store", "put")
    value = request.model_dump()
    filters = await handle_event(ctx, value)

    # If handler modified namespace/key/value, update request
    if filters:
        if "namespace" in filters:
            request.namespace = filters["namespace"]
        if "key" in filters:
            request.key = filters["key"]
        if "value" in filters:
            request.value = filters["value"]

    # Apply user namespace scoping
    scoped_namespace = apply_namespace_scoping(request.namespace, user)

    store = db_manager.get_store()

    await store.aput(namespace=tuple(scoped_namespace), key=request.key, value=request.value)

    return Response(status_code=204)


@router.get("/store/items", response_model=StoreGetResponse, responses={**BAD_REQUEST, **NOT_FOUND})
async def get_store_item(
    key: str = Query(..., description="Key of the item to retrieve."),
    namespace: str | list[str] | None = Query(
        None, description="Namespace path. Use dot-separated string or repeated query params."
    ),
    user: User = Depends(get_current_user),
) -> StoreGetResponse:
    """Get an item from the store by key.

    Returns 404 if no item exists at the given namespace and key.
    """
    # Authorization check
    ctx = build_auth_context(user, "store", "get")
    value = {"key": key, "namespace": namespace}
    filters = await handle_event(ctx, value)

    # If handler modified namespace/key, update
    if filters:
        if "namespace" in filters:
            namespace = filters["namespace"]
        if "key" in filters:
            key = filters["key"]

    # Apply user namespace scoping
    scoped_namespace = apply_namespace_scoping(_normalize_namespace(namespace), user)

    store = db_manager.get_store()

    item = await store.aget(tuple(scoped_namespace), key)

    if not item:
        raise HTTPException(404, "Item not found")

    return StoreGetResponse(key=key, value=item.value, namespace=list(scoped_namespace))


@router.delete("/store/items", status_code=204)
async def delete_store_item(
    body: StoreDeleteRequest | None = None,
    key: str | None = Query(None, description="Key of the item to delete (query param alternative)."),
    namespace: list[str] | None = Query(None, description="Namespace path (query param alternative)."),
    user: User = Depends(get_current_user),
) -> Response:
    """Delete an item from the store.

    Accepts parameters via JSON body (`namespace` + `key`) or query
    parameters. The JSON body takes precedence when both are provided.
    """
    # Determine source of parameters
    ns = None
    k = None
    if body is not None:
        ns = _normalize_namespace(body.namespace)
        k = body.key
    else:
        if key is None:
            raise HTTPException(422, "Missing 'key' parameter")
        ns = _normalize_namespace(namespace)
        k = key

    # Authorization check
    ctx = build_auth_context(user, "store", "delete")
    value = {"namespace": ns, "key": k}
    filters = await handle_event(ctx, value)

    # If handler modified namespace/key, update
    if filters:
        if "namespace" in filters:
            ns = filters["namespace"]
        if "key" in filters:
            k = filters["key"]

    # Apply user namespace scoping
    scoped_namespace = apply_namespace_scoping(ns, user)

    store = db_manager.get_store()

    await store.adelete(tuple(scoped_namespace), k)

    return Response(status_code=204)


@router.post("/store/items/search", response_model=StoreSearchResponse)
async def search_store_items(
    request: StoreSearchRequest, user: User = Depends(get_current_user)
) -> StoreSearchResponse:
    """Search items in the store.

    Filter items by namespace prefix, key-value metadata filters, or semantic
    query. Results are paginated via `limit` and `offset`.
    """
    # Authorization check
    ctx = build_auth_context(user, "store", "search")
    value = request.model_dump()
    filters = await handle_event(ctx, value)

    # Merge handler filters with request filters
    if filters:
        if "namespace_prefix" in filters:
            request.namespace_prefix = filters["namespace_prefix"]

        handler_filters = {k: v for k, v in filters.items() if k != "namespace_prefix"}
        if handler_filters:
            request.filter = {**(request.filter or {}), **handler_filters}

    # Apply user namespace scoping
    scoped_prefix = apply_namespace_scoping(request.namespace_prefix, user)

    store = db_manager.get_store()

    # Search with LangGraph store
    # asearch takes namespace_prefix as a positional-only argument
    results = await store.asearch(
        tuple(scoped_prefix),
        query=request.query,
        filter=request.filter,
        limit=request.limit or 20,
        offset=request.offset or 0,
    )

    items = [StoreItem(key=r.key, value=r.value, namespace=list(r.namespace)) for r in results]

    return StoreSearchResponse(
        items=items,
        total=len(items),  # LangGraph store doesn't provide total count
        limit=request.limit or 20,
        offset=request.offset or 0,
    )


@router.post("/store/namespaces", response_model=StoreListNamespacesResponse)
async def list_namespaces(
    request: StoreListNamespacesRequest,
    user: User = Depends(get_current_user),
) -> StoreListNamespacesResponse:
    """List namespaces in the store.

    Returns the namespace paths that contain items. Filter by prefix, suffix,
    or maximum depth.
    """
    # Authorization check
    ctx = build_auth_context(user, "store", "search")
    value = request.model_dump()
    filters = await handle_event(ctx, value)

    # Apply authorization filters if handler provided any
    if filters:
        if "prefix" in filters:
            request.prefix = filters["prefix"]
        if "suffix" in filters:
            request.suffix = filters["suffix"]

    # Apply user namespace scoping to prefix
    scoped_prefix = apply_namespace_scoping(request.prefix or [], user)
    prefix: tuple[str, ...] = tuple(scoped_prefix)
    suffix: tuple[str, ...] | None = tuple(request.suffix) if request.suffix else None

    store = db_manager.get_store()

    result = await store.alist_namespaces(
        prefix=prefix,
        suffix=suffix,
        max_depth=request.max_depth,
        limit=request.limit,
        offset=request.offset,
    )

    return StoreListNamespacesResponse(namespaces=[list(ns) for ns in result])


def _normalize_namespace(value: str | list[str] | None) -> list[str]:
    """Normalize namespace input to a clean list, filtering out empty parts."""
    if isinstance(value, str):
        return [part for part in value.split(".") if part]
    if isinstance(value, list):
        return [part for part in value if part]
    return []


def _scope(prefix: str, scope_ids: list[str], namespace: list[str]) -> list[str]:
    """Bury a namespace under [prefix, *scope_ids] unless it already starts with exactly that."""
    head = [prefix, *scope_ids]
    if namespace[: len(head)] == head:
        return namespace
    return [*head, *namespace]


_USER_SCOPE_PREFIX = "users"


@cache
def _scope_attr_map() -> Mapping[str, list[str]]:
    """Map of namespace prefix -> list of User attributes, from aegra.json store.scopes.

    Empty unless configured — configurable scopes are entirely opt-in. The
    reserved "users" prefix is dropped so it can never be remapped away from
    per-user isolation.
    """
    store_config = load_store_config()
    configured = store_config.get("scopes") if store_config else None
    if not configured:
        return {}

    scopes: dict[str, list[str]] = {}
    for prefix, attrs in configured.items():
        if prefix == _USER_SCOPE_PREFIX:
            logger.warning("store.scopes key %r is reserved and was ignored", _USER_SCOPE_PREFIX)
            continue
        if not isinstance(attrs, list) or not attrs or not all(isinstance(a, str) and a for a in attrs):
            logger.warning("store.scopes[%r] must be a non-empty list of attribute names; ignoring", prefix)
            continue
        scopes[prefix] = attrs
    return scopes


def apply_namespace_scoping(
    namespace: list[str], user: User, *, scopes: Mapping[str, list[str]] | None = None
) -> list[str]:
    """Scope store namespaces for data isolation.

    User scope is the default. A leading element matching a configured scope
    prefix opts into that scope, isolating data under [prefix, *<user attrs>].
    Items land in the same scope for every user sharing those attribute values,
    so mapping a shared attribute (e.g. org_id) shares data across that group.
    """
    scopes = scopes if scopes is not None else _scope_attr_map()

    if namespace and (prefix := namespace[0]) in scopes:
        values: list[str] = []
        for attr in scopes[prefix]:
            value = getattr(user, attr, None)
            if value is None or value == "":
                raise HTTPException(403, f"User has no {attr!r} required for {prefix!r}-scoped store access")
            values.append(str(value))
        return _scope(prefix, values, namespace)

    return _scope(_USER_SCOPE_PREFIX, [user.identity], namespace)
