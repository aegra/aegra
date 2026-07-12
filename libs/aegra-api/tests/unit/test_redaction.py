"""Tests for secret redaction in API responses (utils/redaction.py)."""

from aegra_api.utils.redaction import redact_secrets


class TestRedactSecrets:
    def test_masks_api_key(self) -> None:
        assert redact_secrets({"api_key": "sk-x", "model": "gpt"}) == {"api_key": "***", "model": "gpt"}

    def test_masks_nested(self) -> None:
        assert redact_secrets({"configurable": {"api_key": "x"}}) == {"configurable": {"api_key": "***"}}

    def test_key_name_variants(self) -> None:
        result = redact_secrets(
            {"apiKey": "a", "access_token": "b", "Authorization": "c", "password": "d", "client_secret": "e"}
        )
        assert result == {
            "apiKey": "***",
            "access_token": "***",
            "Authorization": "***",
            "password": "***",
            "client_secret": "***",
        }

    def test_non_sensitive_untouched(self) -> None:
        data = {"base_url": "https://x/v1", "model": "openai/gpt-4o"}
        assert redact_secrets(data) == data

    def test_does_not_mutate_input(self) -> None:
        original = {"api_key": "secret"}
        redact_secrets(original)
        assert original == {"api_key": "secret"}

    def test_masks_inside_lists(self) -> None:
        assert redact_secrets({"items": [{"token": "t"}, {"model": "gpt"}]}) == {
            "items": [{"token": "***"}, {"model": "gpt"}]
        }
