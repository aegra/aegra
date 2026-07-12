"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from react_agent import prompts

# context 字段名 → 环境变量名的回退映射(用户要求 OPENAI_ 前缀)。
_ENV_ALIASES = {
    "model": "OPENAI_MODEL",
    "base_url": "OPENAI_BASE_URL",
    "api_key": "OPENAI_API_KEY",
}


@dataclass(kw_only=True)
class Context:
    """The context for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    base_url: str | None = field(
        default=None,
        metadata={"description": "OpenAI 兼容端点的 base URL;覆盖 OPENAI_BASE_URL 环境变量。"},
    )

    api_key: str | None = field(
        default=None,
        metadata={
            "description": "覆盖 OPENAI_API_KEY 的 API key。注意:存入 assistant 会被持久化,"
            "敏感场景建议改用环境变量。"
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={"description": "The maximum number of search results to return for each search query."},
    )

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                env_name = _ENV_ALIASES.get(f.name, f.name.upper())
                setattr(self, f.name, os.environ.get(env_name, f.default))
