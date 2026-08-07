"""Authorization handler support for @auth.on.* decorators.

This module provides integration with authorization handlers,
allowing users to define fine-grained access control rules using @auth.on.*
decorators in their auth.py files.
"""

from typing import Any

from fastapi import HTTPException
from langgraph_sdk import Auth
from langgraph_sdk.auth.types import AuthContext as LangGraphAuthContext

from aegra_api.core.auth_middleware import get_auth_instance
from aegra_api.models.auth import User


class AuthContextWrapper:
    """Wrapper to convert Aegra User model to AuthContext.

    AuthContext expects a BaseUser-compatible object. Our User model
    implements the BaseUser protocol (identity, permissions, display_name, __getitem__),
    so we can use it directly after ensuring compatibility.
    """

    def __init__(
        self,
        user: User,
        resource: str,
        action: str,
    ):
        """Initialize auth context wrapper.

        Args:
            user: Authenticated user from Aegra's User model
            resource: Resource being accessed (e.g., "threads", "assistants")
            action: Action being performed (e.g., "create", "read", "update")
        """
        self.user = user
        self.resource = resource
        self.action = action
        self.permissions = user.permissions or []

    def to_langgraph_context(self) -> LangGraphAuthContext:
        """Convert to LangGraph AuthContext.

        Our User model implements the BaseUser protocol (identity, permissions,
        display_name, __getitem__, __contains__, __iter__), so it's compatible
        with LangGraph's AuthContext.

        Returns:
            AuthContext instance compatible with @auth.on handlers
        """
        return LangGraphAuthContext(
            user=self.user,  # Our User model implements BaseUser protocol
            resource=self.resource,  # type: ignore
            action=self.action,  # type: ignore
            permissions=self.permissions,
        )


async def handle_event(
    ctx: AuthContextWrapper | None,
    value: dict[str, Any],
) -> dict[str, Any] | None:
    """Call the appropriate @auth.on.* handler for authorization.

    This function resolves the most specific handler for the given resource
    and action, calls it with the auth context and value, and interprets
    the result.

    **Default Behavior (Non-Interruptive):**
    - If no auth is configured → allows by default (returns None)
    - If no handlers are defined → allows by default (returns None)
    - If handler returns None/True → allows (returns None)
    - If handler returns dict → allows with filters applied (returns dict)
    - **Only interrupts if handler returns False or raises exception**

    This ensures that developers using raw Aegra without custom authorization
    handlers will have a working system out-of-the-box. Handlers are purely
    additive - they can inject metadata, apply filters, or deny access, but
    they don't interrupt the flow unless explicitly configured to do so.

    Handler resolution priority (most specific first):
    1. Resource+action specific (e.g., "threads", "create")
    2. Resource-specific (e.g., "threads", "*")
    3. Action-specific (e.g., "*", "create")
    4. Global handler ("*", "*")

    Args:
        ctx: Auth context wrapper with user, resource, and action
        value: The data being authorized (request body, search filters, etc.)
               This dict may be modified by the handler (e.g., injecting metadata)

    Returns:
        None: Request allowed, no filters to apply
        dict: Filter dict to apply to queries (e.g., {"user_id": "123"})

    Raises:
        HTTPException(403): If handler returns False or raises AssertionError
        HTTPException(500): If handler raises unexpected exception or returns invalid type
    """
    if ctx is None:
        # No auth context means no authorization check needed
        # This allows the request to proceed normally
        return None

    auth = get_auth_instance()
    if auth is None:
        # No auth configured, allow by default
        # This ensures raw Aegra works out-of-the-box without interruption
        return None

    # Convert to AuthContext
    auth_ctx = ctx.to_langgraph_context()

    # Find the most specific handler
    handler = _get_handler(auth, auth_ctx.resource, auth_ctx.action)
    if handler is None:
        # No handler for this resource/action, allow by default
        # Developers can use Aegra without defining handlers - it won't break
        return None

    try:
        # Call the handler with context and value
        result = await handler(ctx=auth_ctx, value=value)
    except Auth.exceptions.HTTPException as e:
        # Handler raised HTTP exception, convert to FastAPI HTTPException
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail,
            headers=dict(e.headers) if hasattr(e, "headers") and e.headers else None,
        ) from e
    except AssertionError as e:
        # Handler used assert for authorization check
        raise HTTPException(status_code=403, detail=str(e)) from e
    # Programmer errors (TypeError, AttributeError, ...) propagate so the
    # standard error handler logs the stack and returns a generic 500 — we
    # don't want a handler bug to leak its exception text to API clients.

    # Interpret handler result
    if result in (None, True):
        # Allow request, no filters
        return None

    if result is False:
        # Deny request
        raise HTTPException(status_code=403, detail="Forbidden")

    if isinstance(result, dict):
        # Return filter dict to apply to queries
        return result

    # Invalid return type
    raise HTTPException(
        status_code=500,
        detail=f"Auth handler returned invalid type: {type(result)}. Expected dict, None, True, or False.",
    )


def _get_handler(
    auth: Auth,
    resource: str,
    action: str,
) -> Any | None:
    """Find the most specific handler for resource+action.

    Handler resolution follows this priority order (most specific first):
    1. (resource, action) - e.g., ("threads", "create")
    2. (resource, "*") - e.g., ("threads", "*")
    3. ("*", action) - e.g., ("*", "create")
    4. ("*", "*") - global handler

    Args:
        auth: Auth instance with registered handlers
        resource: Resource name (e.g., "threads", "assistants")
        action: Action name (e.g., "create", "read", "update")

    Returns:
        Handler function or None if no handler found
    """
    # Check cache first
    key = (resource, action)
    if key in auth._handler_cache:
        return auth._handler_cache[key]

    # Priority order (most specific first)
    keys = [
        (resource, action),  # Most specific: exact resource+action
        (resource, "*"),  # Resource-specific: all actions on resource
        ("*", action),  # Action-specific: all resources for action
        ("*", "*"),  # Global: all resources and actions
    ]

    # Find first matching handler
    for check_key in keys:
        if check_key in auth._handlers and auth._handlers[check_key]:
            # Get the last registered handler (most recent wins)
            handler = auth._handlers[check_key][-1]
            # Cache the result
            auth._handler_cache[key] = handler
            return handler

    # Check global handlers (fallback for backward compatibility)
    if auth._global_handlers:
        handler = auth._global_handlers[-1]
        auth._handler_cache[key] = handler
        return handler

    return None


def build_auth_context(
    user: User,
    resource: str,
    action: str,
) -> AuthContextWrapper:
    """Build AuthContextWrapper from user and operation info.

    This is a convenience function to create an AuthContextWrapper with
    the correct resource and action for a given operation.

    Args:
        user: Authenticated user from require_auth dependency
        resource: Resource being accessed (e.g., "threads", "assistants")
        action: Action being performed (e.g., "create", "read", "update")

    Returns:
        AuthContextWrapper instance ready to pass to handle_event()

    Example:
        ```python
        @router.post("/threads")
        async def create_thread(
            request: ThreadCreate,
            user: User = Depends(require_auth),
        ):
            ctx = build_auth_context(user, "threads", "create")
            value = request.model_dump()
            filters = await handle_event(ctx, value)
            # Use filters or modified value...
        ```
    """
    return AuthContextWrapper(
        user=user,
        resource=resource,
        action=action,
    )
