"""Webhook configuration, signing, and redaction — shared by crons and runs.

A webhook is either a bare URL string (the LangGraph SDK contract) or a rich
object carrying method/headers/params/body/secret. This module is deliberately
import-cycle-free (only stdlib + pydantic) so both ``models`` and ``services``
can depend on it. Delivery lives in ``services.webhook_service``.
"""

import base64
import hashlib
import hmac
from typing import Annotated, Any, Literal
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

_URL_MAX_LEN = 2048
_SECRET_MAX_LEN = 256
_MASK = "***"


class WebhookConfig(BaseModel):
    """Where and how to POST a run-completion callback."""

    model_config = ConfigDict(frozen=True)

    url: str = Field(..., max_length=_URL_MAX_LEN, description="Callback URL (http/https).")
    method: Literal["POST", "PUT", "PATCH"] = Field("POST", description="HTTP method for the callback.")
    headers: dict[str, str] | None = Field(None, description="Extra request headers (may carry auth; masked on read).")
    params: dict[str, str] | None = Field(None, description="Query-string parameters appended to the URL.")
    body: dict[str, Any] | None = Field(None, description="Fields merged over the run payload in the request body.")
    secret: str | None = Field(
        None,
        max_length=_SECRET_MAX_LEN,
        description="HMAC signing key. When set, Standard Webhooks signature headers are added.",
    )

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        """Reject non-http(s) or host-less URLs — this is the SSRF entry point."""
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("webhook url must use http or https scheme")
        if not parsed.netloc:
            raise ValueError("webhook url must include a host")
        return value

    def to_payload(self) -> str | dict[str, Any]:
        """Serialize for JSONB storage: bare URL collapses to a str (SDK-compatible), else a dict."""
        if self.method == "POST" and not (self.headers or self.params or self.body or self.secret):
            return self.url
        return self.model_dump(exclude_none=True)


def normalize(value: Any) -> Any:
    """BeforeValidator hook: turn a bare URL string into a WebhookConfig-shaped dict."""
    return {"url": value} if isinstance(value, str) else value


# Reusable field type: accepts a URL string or a webhook object, yields WebhookConfig.
WebhookField = Annotated[WebhookConfig | None, BeforeValidator(normalize)]


def _decode_secret(secret: str) -> bytes:
    """Standard Webhooks keys are base64 behind a ``whsec_`` prefix; otherwise raw utf-8.

    Tolerates missing base64 padding — Standard Webhooks keys are commonly stored
    unpadded, and an unpadded key must not crash signing (silent non-delivery).
    """
    if secret.startswith("whsec_"):
        raw = secret[len("whsec_") :]
        return base64.b64decode(raw + "=" * (-len(raw) % 4))
    return secret.encode()


def sign(secret: str, msg_id: str, timestamp: int, body: bytes) -> str:
    """Return a Standard Webhooks signature: ``v1,<base64(HMAC-SHA256(secret, id.ts.body))>``."""
    signed = b".".join((msg_id.encode(), str(timestamp).encode(), body))
    digest = hmac.new(_decode_secret(secret), signed, hashlib.sha256).digest()
    return "v1," + base64.b64encode(digest).decode()


def _strip_userinfo(url: str) -> str:
    """Drop ``user:pass@`` from a URL so credentials never round-trip on read.

    Strips only the userinfo segment (before the last ``@``) to preserve the host
    verbatim — rebuilding from ``parsed.hostname`` would drop IPv6 ``[...]`` brackets.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    if "@" not in parsed.netloc:
        return url
    host_port = parsed.netloc.rsplit("@", 1)[1]
    return urlunparse(parsed._replace(netloc=host_port))


def redact(webhook: str | dict[str, Any]) -> str | dict[str, Any]:
    """Mask credentials in a stored webhook: URL userinfo, secret, header + query values."""
    if isinstance(webhook, str):
        return _strip_userinfo(webhook)
    masked = dict(webhook)
    if isinstance(masked.get("url"), str):
        masked["url"] = _strip_userinfo(masked["url"])
    if masked.get("secret") is not None:
        masked["secret"] = _MASK
    # headers/params commonly carry auth tokens (e.g. ?access_token=...).
    for field in ("headers", "params"):
        if isinstance(masked.get(field), dict):
            masked[field] = dict.fromkeys(masked[field], _MASK)
    return masked
