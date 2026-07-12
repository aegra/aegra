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
    identity: str

    # Optional with defaults
    is_authenticated: bool = True
    permissions: list[str] = []
    display_name: str | None = None

    # Common optional fields (for IDE hints)
    org_id: str | None = None
    email: str | None = None
    # 多租户隔离字段。由 auth handler 返回,服务端权威,客户端不可伪造。
    # 为 None 时退回纯 user_id 隔离(tenant 可选)。
    tenant_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_user_id_alias(cls, data: Any) -> Any:
        """接受 `user_id` 作为 `identity` 的输入别名(auth handler 可返回任一)。"""
        if isinstance(data, dict) and not data.get("identity") and data.get("user_id"):
            data = {**data, "identity": data["user_id"]}
        return data

    @property
    def user_id(self) -> str:
        """`identity` 的对外一等别名;identity 保留以兼容 LangGraph BaseUser 协议。"""
        return self.identity

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
