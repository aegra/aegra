"""Unit tests for the shared webhook model: coercion, validation, signing, redaction."""

import pytest
from pydantic import ValidationError

from aegra_api.models.webhooks import WebhookConfig, redact, sign


class TestUrlValidation:
    @pytest.mark.parametrize(
        "url",
        ["ftp://example.com/h", "javascript:alert(1)", "file:///etc/passwd", "//example.com", "http:///no-host"],
    )
    def test_rejects_non_http_or_hostless(self, url: str) -> None:
        with pytest.raises(ValidationError):
            WebhookConfig(url=url)

    @pytest.mark.parametrize("url", ["http://example.com/h", "https://example.com/h"])
    def test_accepts_http_https(self, url: str) -> None:
        assert WebhookConfig(url=url).url == url

    def test_rejects_bad_method(self) -> None:
        with pytest.raises(ValidationError):
            WebhookConfig(url="https://x.io/h", method="DELETE")  # type: ignore[arg-type]

    def test_rejects_oversized_url(self) -> None:
        with pytest.raises(ValidationError):
            WebhookConfig(url="https://x.io/" + "a" * 4096)


class TestToPayload:
    def test_bare_url_collapses_to_string(self) -> None:
        assert WebhookConfig(url="https://x.io/h").to_payload() == "https://x.io/h"

    def test_rich_returns_dict(self) -> None:
        payload = WebhookConfig(url="https://x.io/h", headers={"A": "b"}, secret="s").to_payload()
        assert isinstance(payload, dict)
        assert payload["url"] == "https://x.io/h"
        assert payload["headers"] == {"A": "b"}
        assert payload["secret"] == "s"

    def test_non_post_method_stays_dict(self) -> None:
        assert isinstance(WebhookConfig(url="https://x.io/h", method="PUT").to_payload(), dict)


class TestSign:
    def test_matches_standard_webhooks_vector(self) -> None:
        """Known vector from the Standard Webhooks spec."""
        secret = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"
        msg_id = "msg_p5jXN8AQM9LWM0D4loKWxJek"
        timestamp = 1614265330
        body = b'{"test": 2432232314}'
        assert sign(secret, msg_id, timestamp, body) == "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE="

    def test_raw_secret_without_prefix(self) -> None:
        sig = sign("plainsecret", "id1", 1000, b"{}")
        assert sig.startswith("v1,")

    def test_unpadded_whsec_secret_does_not_crash(self) -> None:
        # Regression: unpadded base64 whsec_ key must not raise "Incorrect padding".
        assert sign("whsec_abc", "id1", 1000, b"{}").startswith("v1,")


class TestRedact:
    def test_strips_userinfo_from_string(self) -> None:
        assert redact("https://user:secret@host.io/x") == "https://host.io/x"

    def test_preserves_port_and_path(self) -> None:
        assert redact("https://u:p@host.io:8443/a/b?q=1") == "https://host.io:8443/a/b?q=1"

    def test_preserves_ipv6_brackets(self) -> None:
        # Regression: rebuilding netloc from parsed.hostname dropped IPv6 brackets.
        assert redact("http://[::1]:9000/hook") == "http://[::1]:9000/hook"
        assert redact("https://u:p@[::1]:8443/x") == "https://[::1]:8443/x"

    def test_masks_secret_headers_and_params(self) -> None:
        masked = redact(
            {
                "url": "https://u:p@host.io/x",
                "secret": "whsec_x",
                "headers": {"Authorization": "Bearer t"},
                "params": {"access_token": "SECRET"},
            }
        )
        assert masked["url"] == "https://host.io/x"
        assert masked["secret"] == "***"
        assert masked["headers"] == {"Authorization": "***"}
        assert masked["params"] == {"access_token": "***"}  # tokens ride in query params

    def test_leaves_non_credential_fields(self) -> None:
        masked = redact({"url": "https://host.io/x", "method": "POST"})
        assert masked["method"] == "POST"
