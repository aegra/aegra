"""Utility & helper functions."""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(
    fully_specified_name: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name: String in the format 'provider/model'.
        base_url: Optional OpenAI-compatible base URL (overrides OPENAI_BASE_URL).
        api_key: Optional API key (overrides OPENAI_API_KEY).
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    kwargs: dict[str, Any] = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return init_chat_model(model, model_provider=provider, **kwargs)
