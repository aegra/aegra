"""Shared search pagination cap for protocol search request models.

Read at validation time so MAX_SEARCH_LIMIT can change without reimporting models.
"""

from typing import Any

from pydantic_core import PydanticKnownError

from aegra_api.settings import settings

DEFAULT_SEARCH_LIMIT: int = 20


def effective_search_limit() -> int:
    return min(DEFAULT_SEARCH_LIMIT, settings.app.MAX_SEARCH_LIMIT)


def resolve_search_limit(value: int | None) -> int:
    # Omitted fields and JSON null both arrive as None.
    if value is None:
        return effective_search_limit()
    cap = settings.app.MAX_SEARCH_LIMIT
    if value > cap:
        # Same type/msg/ctx as Field(le=cap) so 422 payloads stay Pydantic-shaped.
        raise PydanticKnownError("less_than_equal", {"le": cap})
    return value


def search_limit_json_schema_extra(schema: dict[str, Any]) -> None:
    cap = settings.app.MAX_SEARCH_LIMIT
    schema["default"] = effective_search_limit()
    for option in schema.get("anyOf", []):
        if isinstance(option, dict) and option.get("type") == "integer":
            option["maximum"] = cap
            return
    schema["maximum"] = cap
