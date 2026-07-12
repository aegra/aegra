"""Authentication dependencies for FastAPI endpoints"""

from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request

from aegra_api.core.auth_middleware import get_auth_backend
from aegra_api.models.auth import User


def _extract_user_data(user_obj: Any) -> dict[str, Any]:
    """Extract user data from various object types.

    Handles dict, objects with to_dict(), and objects with dict() methods.

    Args:
        user_obj: User object from authentication middleware

    Returns:
        Dictionary containing user data
    """
    if isinstance(user_obj, dict):
        return user_obj
    if hasattr(user_obj, "to_dict"):
        return user_obj.to_dict()
    if hasattr(user_obj, "dict"):
        return user_obj.dict()
    # Fallback: extract known attributes (user_id first, identity as alias)
    return {
        "user_id": getattr(user_obj, "user_id", None) or getattr(user_obj, "identity", str(user_obj)),
        "is_authenticated": getattr(user_obj, "is_authenticated", True),
    }


def _to_user_model(user: Any) -> User:
    """Convert auth result to User model.

    Args:
        user: User object from auth backend (LangGraphUser, dict, etc.)

    Returns:
        User model instance with all fields preserved
    """
    user_data = _extract_user_data(user)

    # user_id is primary; accept the LangGraph-style identity key as an alias
    if "user_id" not in user_data and "identity" not in user_data:
        raise HTTPException(status_code=401, detail="User id not provided")

    # Set display_name default if not provided
    if user_data.get("display_name") is None:
        user_data["display_name"] = user_data.get("user_id") or user_data["identity"]

    # Pass all fields through to User model (extra fields allowed via ConfigDict)
    return User(**user_data)


async def require_auth(request: Request) -> User:
    """FastAPI dependency for authentication.

    Replaces Starlette AuthenticationMiddleware by calling the auth backend directly.
    This allows FastAPI to properly track dependencies for OpenAPI generation.

    Args:
        request: FastAPI request object

    Returns:
        User object with authentication context including any extra fields

    Raises:
        HTTPException: If user is not authenticated
    """
    backend = get_auth_backend()

    try:
        result = await backend.authenticate(request)
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e)) from e

    if result is None:
        # No auth configured - for now, require auth if backend exists
        # (This will be handled by noop auth in the backend)
        raise HTTPException(status_code=401, detail="Authentication required")

    credentials, user = result

    # Set request.scope for backward compatibility
    # (Some code might still read request.scope["user"] or request.user)
    request.scope["user"] = user
    request.scope["auth"] = credentials
    # Also set request.user for Starlette compatibility
    if not hasattr(request, "user"):
        request.user = user

    # Convert to User model
    return _to_user_model(user)


# Type alias for cleaner route signatures
AuthenticatedUser = Annotated[User, Depends(require_auth)]

# For applying to entire routers
auth_dependency = [Depends(require_auth)]


def get_current_user(request: Request) -> User:
    """
    Legacy: Extract current user from request context set by middleware or dependency.

    This function reads from request.scope["user"] which is set by either:
    - The new require_auth() dependency (preferred)
    - The old AuthenticationMiddleware (for backward compatibility)

    This function passes ALL fields from auth handlers through to the User model,
    allowing custom auth handlers to return extra fields (e.g., subscription_tier,
    team_id) that will be accessible on the User object.

    Args:
        request: FastAPI request object

    Returns:
        User object with authentication context including any extra fields

    Raises:
        HTTPException: If user is not authenticated
    """
    # Try reading from request.scope first (set by require_auth dependency)
    user = request.scope.get("user")
    if user is None:
        # Fallback to request.user (set by middleware)
        if not hasattr(request, "user") or request.user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        user = request.user

    if hasattr(user, "is_authenticated") and not user.is_authenticated:
        raise HTTPException(status_code=401, detail="Invalid authentication")

    # Convert to User model
    return _to_user_model(user)


def get_user_id(user: User = Depends(get_current_user)) -> str:
    """
    Helper dependency to get user ID safely.

    Args:
        user: User object from get_current_user dependency

    Returns:
        User identity string
    """
    return user.user_id


def require_permission(permission: str):
    """
    Create a dependency that requires a specific permission.

    Args:
        permission: Required permission string

    Returns:
        Dependency function that checks for the permission

    Example:
        @app.get("/admin")
        def admin_endpoint(user: User = Depends(require_permission("admin"))):
            return {"message": "Admin access granted"}
    """

    def permission_dependency(user: User = Depends(get_current_user)) -> User:
        if permission not in user.permissions:
            raise HTTPException(status_code=403, detail=f"Permission '{permission}' required")
        return user

    return permission_dependency


def require_authenticated(request: Request) -> User:
    """
    Simplified dependency that just ensures user is authenticated.

    This is equivalent to get_current_user but with a clearer name
    for endpoints that just need any authenticated user.
    """
    return get_current_user(request)
