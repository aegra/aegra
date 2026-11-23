"""Unit tests for _extract_thread_name helper in run_preparation.py."""

from typing import Any

import pytest

from aegra_api.services.run_preparation import _extract_thread_name


class TestExtractThreadName:
    """Test _extract_thread_name helper."""

    def test_returns_empty_for_no_messages(self) -> None:
        assert _extract_thread_name({}) == ""

    def test_returns_empty_for_empty_messages_list(self) -> None:
        assert _extract_thread_name({"messages": []}) == ""

    def test_returns_empty_for_non_list_messages(self) -> None:
        assert _extract_thread_name({"messages": "not a list"}) == ""

    def test_extracts_content_from_dict_message(self) -> None:
        result = _extract_thread_name({"messages": [{"role": "human", "content": "Hello world"}]})
        assert result == "Hello world"

    def test_extracts_first_message_with_content(self) -> None:
        result = _extract_thread_name(
            {
                "messages": [
                    {"role": "human", "content": "First message"},
                    {"role": "ai", "content": "Response"},
                ]
            }
        )
        assert result == "First message"

    def test_skips_messages_without_content(self) -> None:
        result = _extract_thread_name(
            {
                "messages": [
                    {"role": "system"},
                    {"role": "human", "content": "Actual question"},
                ]
            }
        )
        assert result == "Actual question"

    def test_skips_empty_string_content(self) -> None:
        result = _extract_thread_name(
            {
                "messages": [
                    {"role": "human", "content": "   "},
                    {"role": "human", "content": "Real content"},
                ]
            }
        )
        assert result == "Real content"

    def test_truncates_long_content_at_word_boundary(self) -> None:
        long_content = "word " * 30  # 148 chars after strip
        result = _extract_thread_name({"messages": [{"role": "human", "content": long_content}]})
        assert result.endswith("...")
        assert len(result) <= 104  # 100 chars + "..."

    def test_handles_langchain_message_objects(self) -> None:
        class FakeMessage:
            type = "human"
            content = "Object message"

        result = _extract_thread_name({"messages": [FakeMessage()]})
        assert result == "Object message"

    def test_skips_non_human_messages(self) -> None:
        """Only human/user messages should be used for thread name."""
        result = _extract_thread_name(
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "human", "content": "What is Python?"},
                ]
            }
        )
        assert result == "What is Python?"

    def test_accepts_user_role(self) -> None:
        """OpenAI-style 'user' role should work same as 'human'."""
        result = _extract_thread_name({"messages": [{"role": "user", "content": "Hi there"}]})
        assert result == "Hi there"

    def test_accepts_type_human_in_dict_messages(self) -> None:
        """Aegra docs/examples commonly send message type instead of role."""
        result = _extract_thread_name({"messages": [{"type": "human", "content": "Hello from docs"}]})
        assert result == "Hello from docs"

    def test_returns_empty_for_none_content(self) -> None:
        result = _extract_thread_name({"messages": [{"role": "human", "content": None}]})
        assert result == ""

    def test_returns_empty_for_non_string_content(self) -> None:
        result = _extract_thread_name({"messages": [{"role": "human", "content": 42}]})
        assert result == ""

    def test_extracts_from_list_content_blocks(self) -> None:
        """OpenAI-compatible SDKs send content as list of blocks."""
        result = _extract_thread_name(
            {
                "messages": [
                    {
                        "role": "human",
                        "content": [{"type": "text", "text": "Hello from blocks"}],
                    }
                ]
            }
        )
        assert result == "Hello from blocks"

    def test_joins_multiple_text_blocks(self) -> None:
        result = _extract_thread_name(
            {
                "messages": [
                    {
                        "role": "human",
                        "content": [
                            {"type": "text", "text": "Part one"},
                            {"type": "image_url", "image_url": {"url": "..."}},
                            {"type": "text", "text": "Part two"},
                        ],
                    }
                ]
            }
        )
        assert result == "Part one Part two"

    def test_skips_list_blocks_without_text_type(self) -> None:
        result = _extract_thread_name(
            {
                "messages": [
                    {
                        "role": "human",
                        "content": [{"type": "image_url", "image_url": {"url": "..."}}],
                    },
                    {"role": "human", "content": "Fallback"},
                ]
            }
        )
        assert result == "Fallback"

    @pytest.mark.parametrize("input_data", [["hello"], "hello", 7, True, None])
    def test_returns_empty_for_non_dict_input(self, input_data: Any) -> None:
        """Run input may be any JSON value; only an object can carry messages."""
        assert _extract_thread_name(input_data) == ""
