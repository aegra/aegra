"""Example JWT authentication handler for Aegra.

This demonstrates how to implement authentication and authorization using
LangGraph SDK's Auth system with @auth.authenticate and @auth.on.* handlers.

Token format: mock-jwt-<user_id>-<role>-<team_id>
Example: mock-jwt-alice-admin-team123

Configuration:
Add this to your aegra.json or langgraph.json:

{
  "graphs": {
    "agent": "./graphs/react_agent/graph.py:graph"
  },
  "auth": {
    "path": "./jwt_mock_auth_example.py:auth"
  }
}

This example includes:
- Authentication handler (@auth.authenticate)
- Authorization handlers (@auth.on.*) for fine-grained access control
- Custom user fields (role, team_id, subscription_tier)
- Metadata injection in create operations
- Filter application in search operations
"""

from langgraph_sdk import Auth
from langgraph_sdk.auth.types import (
    AssistantsCreate,
    AssistantsDelete,
    AuthContext,
    HandlerResult,
    ThreadsCreate,
    ThreadsSearch,
)

auth = Auth()


@auth.authenticate
async def authenticate(headers: dict[str, str]) -> dict[str, str | bool | list[str]]:
    """Mock JWT authentication that simulates real JWT behavior.

    Expects: Authorization: Bearer <token>
    Token format: mock-jwt-<user_id>-<role>-<team_id>

    Returns user data with custom fields that flow through to routes.

    Args:
        headers: Request headers dict

    Returns:
        User data dict with identity, display_name, permissions, and custom fields

    Raises:
        HTTPException: If token is missing or invalid
    """
    auth_header = headers.get("authorization", "") or headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise Auth.exceptions.HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header[7:]  # Strip "Bearer "

    # Parse mock token: mock-jwt-userid-role-teamid
    if not token.startswith("mock-jwt-"):
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token format")

    parts = token.split("-")[2:]  # Skip "mock-jwt"
    if len(parts) < 2:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Token missing required fields")

    user_id = parts[0]
    role = parts[1]
    has_team = len(parts) > 2
    team_id = parts[2] if has_team else "team_default"

    # Determine subscription tier based on role
    subscription_tier = "premium" if role in ("admin", "premium") else "free"

    return {
        "identity": user_id,
        "display_name": f"User {user_id}",
        "is_authenticated": True,
        "permissions": [role, f"{role}:read", f"{role}:write"],
        # Custom fields that flow through to User model
        "role": role,
        "subscription_tier": subscription_tier,
        "team_id": team_id,
        # A team doubles as the user's org for org-scoped store sharing; no team => no org.
        "org_id": team_id if has_team else None,
        "email": f"{user_id}@example.com",
    }


# Authorization handlers (@auth.on.*)


# Global fallback handler - runs for all resources/actions that don't have specific handlers
# This provides default filtering behavior (e.g., user-scoped access)
@auth.on
async def authorize(ctx: AuthContext, value: dict) -> HandlerResult:
    """Global authorization handler - fallback for all requests.

    This handler runs for any resource/action that doesn't have a more specific
    handler. It provides default filtering to ensure users only see their own data.

    Returns:
        Filter dict to apply to queries, or True to allow without filtering
    """
    # Default behavior: allow but don't apply filters
    # Specific handlers (like @auth.on.threads.create) will override this
    return True


@auth.on.threads.create
async def allow_thread_create(ctx: AuthContext, value: ThreadsCreate) -> HandlerResult:
    """Allow thread creation and inject team_id into metadata.

    This handler automatically injects the user's team_id into thread metadata,
    ensuring data isolation at the team level.
    """
    # Ensure metadata exists (handle None case)
    if value.get("metadata") is None:
        value["metadata"] = {}
    # Inject team_id from user custom fields
    try:
        team_id = ctx.user["team_id"] if "team_id" in ctx.user else getattr(ctx.user, "team_id", None)
        if team_id:
            value["metadata"]["team_id"] = team_id
    except (KeyError, AttributeError):
        pass
    return True


@auth.on.threads.search
async def filter_threads_by_team(ctx: AuthContext, value: ThreadsSearch) -> HandlerResult:
    """Filter thread searches by team_id.

    This handler ensures users only see threads from their team,
    providing automatic data filtering for search operations.
    """
    try:
        team_id = ctx.user["team_id"] if "team_id" in ctx.user else getattr(ctx.user, "team_id", None)
        if team_id:
            return {"metadata": {"team_id": team_id}}
    except (KeyError, AttributeError):
        pass
    return {"user_id": ctx.user.identity}


@auth.on.assistants.delete
async def restrict_assistant_deletion(ctx: AuthContext, value: AssistantsDelete) -> HandlerResult:
    """Only admins can delete assistants.

    This demonstrates role-based authorization - only users with
    role="admin" can delete assistants.
    """
    try:
        role = ctx.user["role"] if "role" in ctx.user else getattr(ctx.user, "role", None)
        if role == "admin":
            return True
    except (KeyError, AttributeError):
        pass
    return False


@auth.on.assistants.create
async def allow_assistant_create(ctx: AuthContext, value: AssistantsCreate) -> HandlerResult:
    """Allow assistant creation and inject creator info.

    This handler injects metadata about who created the assistant
    and which team it belongs to.
    """
    # Ensure metadata exists (handle None case)
    if value.get("metadata") is None:
        value["metadata"] = {}
    value["metadata"]["created_by"] = ctx.user.identity
    try:
        team_id = ctx.user["team_id"] if "team_id" in ctx.user else getattr(ctx.user, "team_id", None)
        if team_id:
            value["metadata"]["team_id"] = team_id
    except (KeyError, AttributeError):
        pass
    return True
