"""Authentication and user context models"""

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class User(BaseModel):
    """User model that accepts any auth fields.

    This model uses ConfigDict(extra="allow") to accept any additional fields
    from auth handlers (e.g., subscription_tier, team_id) while maintaining
    type hints for common fields.
    """

    model_config = ConfigDict(extra="allow")

    # Required
    user_id: str

    # Optional with defaults
    is_authenticated: bool = True
    permissions: list[str] = []
    display_name: str | None = None

    # Common optional fields (for IDE hints)
    org_id: str | None = None
    email: str | None = None
    # Multi-tenant isolation field. Returned by the auth handler, server-authoritative, not client-forgeable.
    # None falls back to pure user_id isolation (tenant optional).
    tenant_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_identity_alias(cls, data: Any) -> Any:
        """Auth handlers may return identity (LangGraph convention); converge on user_id."""
        if isinstance(data, dict) and not data.get("user_id") and data.get("identity"):
            data = {**data, "user_id": data["identity"]}
        return data

    @property
    def identity(self) -> str:
        """Only to satisfy the LangGraph/starlette BaseUser protocol (matched by attribute name); business code should use user_id."""
        return self.user_id

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict including all extra fields."""
        return self.model_dump()

    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to extra fields."""
        try:
            extra = object.__getattribute__(self, "__pydantic_extra__") or {}
        except AttributeError:
            extra = {}
        if name in extra:
            return extra[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class AuthContext(BaseModel):
    """Authentication context for request processing"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user: User
    request_id: str | None = None


class TokenPayload(BaseModel):
    """JWT token payload structure"""

    sub: str  # subject (user ID)
    name: str | None = None
    scopes: list[str] = []
    org: str | None = None
    exp: int | None = None
    iat: int | None = None
