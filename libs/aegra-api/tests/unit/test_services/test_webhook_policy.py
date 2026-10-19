"""Unit tests for the webhook config and URL policy."""

import asyncio
import ipaddress
import socket
from typing import Any

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from aegra_api.config import HttpConfig, WebhooksConfig
from aegra_api.services import webhook_policy
from aegra_api.services.webhook_policy import (
    WebhookDNSError,
    WebhookSettings,
    WebhookUrlError,
    WebhookUrlPolicy,
    _resolve,
    check_webhook_url,
    ensure_webhook_allowed,
    get_webhook_settings,
    resolve_webhook_address,
    validate_webhook_url,
    webhooks_enabled,
)

# Captured at import, before the autouse fixture swaps in the offline resolver.
_real_resolve = _resolve

_OPEN_POLICY = WebhookUrlPolicy(require_https=False, max_url_length=4096)


def _use_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    webhooks: WebhooksConfig | dict[str, Any] | None = None,
    http: HttpConfig | dict[str, Any] | None = None,
) -> None:
    monkeypatch.setattr(webhook_policy, "load_webhooks_config", lambda: webhooks)
    monkeypatch.setattr(webhook_policy, "load_http_config", lambda: http)
    get_webhook_settings.cache_clear()
    webhooks_enabled.cache_clear()


def _resolve_to(monkeypatch: pytest.MonkeyPatch, *addresses: str) -> None:
    async def _fake(host: str, port: int) -> list[webhook_policy.IPAddress]:
        return [ipaddress.ip_address(addr) for addr in addresses]

    monkeypatch.setattr(webhook_policy, "_resolve", _fake)


class TestWebhookUrlPolicy:
    def test_block_defaults_require_https_and_cap_length_at_2048(self) -> None:
        policy = WebhookUrlPolicy()

        assert policy.require_https is True
        assert policy.max_url_length == 2048
        assert policy.disable_loopback is True
        assert policy.disable_private_ips is False

    def test_normalizes_domains(self) -> None:
        policy = WebhookUrlPolicy(allowed_domains=[" Hooks.Example.com. ", "*.Corp.com"])

        assert policy.allowed_domains == ["hooks.example.com", "*.corp.com"]

    @pytest.mark.parametrize("domain", ["", "   ", "corp.*", "*corp.com", "*.*.corp.com", "a.*.corp.com"])
    def test_rejects_malformed_domains(self, domain: str) -> None:
        with pytest.raises(ValidationError, match="allowed_domains"):
            WebhookUrlPolicy(allowed_domains=[domain])

    @pytest.mark.parametrize("port", [0, -1, 65536])
    def test_rejects_out_of_range_ports(self, port: int) -> None:
        with pytest.raises(ValidationError, match="allowed_ports"):
            WebhookUrlPolicy(allowed_ports=[port])

    def test_wildcard_matches_subdomains_only(self) -> None:
        policy = WebhookUrlPolicy(allowed_domains=["*.corp.com"])

        assert policy.allows_domain("hooks.corp.com")
        assert policy.allows_domain("a.b.corp.com")
        assert not policy.allows_domain("corp.com")
        assert not policy.allows_domain("evilcorp.com")

    def test_only_exact_entries_are_trusted(self) -> None:
        policy = WebhookUrlPolicy(allowed_domains=["hooks.corp.com", "*.corp.com"])

        assert policy.trusts("hooks.corp.com")
        assert not policy.trusts("other.corp.com")


class TestWebhookSettings:
    def test_missing_url_block_uses_langgraph_open_defaults(self) -> None:
        settings = WebhookSettings.model_validate({})

        assert settings.env_prefix == "LG_WEBHOOK_"
        assert settings.url.require_https is False
        assert settings.url.max_url_length == 4096
        assert settings.url.disable_loopback is True

    def test_null_url_block_counts_as_missing(self) -> None:
        assert WebhookSettings.model_validate({"url": None}).url.require_https is False

    def test_url_block_fills_unset_keys_with_block_defaults(self) -> None:
        settings = WebhookSettings.model_validate({"url": {"allowed_domains": ["*.corp.com"]}})

        assert settings.url.require_https is True
        assert settings.url.max_url_length == 2048

    def test_resolves_header_templates_from_prefixed_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LG_WEBHOOK_TOKEN", "s3cret")
        monkeypatch.setenv("LG_WEBHOOK_ENV", "prod")

        settings = WebhookSettings.model_validate(
            {"headers": {"Authorization": "Bearer ${{ env.LG_WEBHOOK_TOKEN }}", "X-Env": "${{env.LG_WEBHOOK_ENV}}-1"}}
        )

        assert settings.headers == {"Authorization": "Bearer s3cret", "X-Env": "prod-1"}

    def test_custom_env_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOOK_TOKEN", "abc")

        settings = WebhookSettings.model_validate(
            {"env_prefix": "HOOK_", "headers": {"Authorization": "${{ env.HOOK_TOKEN }}"}}
        )

        assert settings.headers["Authorization"] == "abc"

    def test_rejects_variable_outside_the_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql://secret")

        with pytest.raises(ValidationError, match="must start with env_prefix"):
            WebhookSettings.model_validate({"headers": {"X-Leak": "${{ env.DATABASE_URL }}"}})

    def test_rejects_unset_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LG_WEBHOOK_MISSING", raising=False)

        with pytest.raises(ValidationError, match="LG_WEBHOOK_MISSING"):
            WebhookSettings.model_validate({"headers": {"Authorization": "${{ env.LG_WEBHOOK_MISSING }}"}})

    @pytest.mark.parametrize("template", ["${{ secrets.TOKEN }}", "${{ env.LG_WEBHOOK_A.B }}", "${{ env. }}"])
    def test_rejects_unsupported_templates(self, template: str) -> None:
        with pytest.raises(ValidationError, match="webhooks.headers"):
            WebhookSettings.model_validate({"headers": {"Authorization": template}})

    def test_rejects_invalid_header_name(self) -> None:
        with pytest.raises(ValidationError, match="invalid header name"):
            WebhookSettings.model_validate({"headers": {"Bad Header": "x"}})

    def test_rejects_line_breaks_in_resolved_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LG_WEBHOOK_TOKEN", "abc\r\nX-Injected: 1")

        with pytest.raises(ValidationError, match="line break"):
            WebhookSettings.model_validate({"headers": {"Authorization": "${{ env.LG_WEBHOOK_TOKEN }}"}})

    def test_rejects_non_string_header_values(self) -> None:
        with pytest.raises(ValidationError):
            WebhookSettings.model_validate({"headers": {"X-Retries": 3}})

    def test_allowed_fields_accepts_payload_keys(self) -> None:
        settings = WebhookSettings.model_validate({"allowed_fields": ["run_id", "status", "values"]})

        assert settings.allowed_fields == frozenset({"run_id", "status", "values"})

    def test_empty_allowed_fields_means_every_field(self) -> None:
        assert WebhookSettings.model_validate({"allowed_fields": []}).allowed_fields is None

    def test_rejects_unknown_allowed_fields(self) -> None:
        with pytest.raises(ValidationError, match="unknown fields"):
            WebhookSettings.model_validate({"allowed_fields": ["run_id", "secrets"]})

    @pytest.mark.parametrize(
        "config",
        [
            {"url": {"disable_private_ip": True}},
            {"header": {"Authorization": "x"}},
        ],
    )
    def test_rejects_misspelled_keys(self, config: dict[str, Any]) -> None:
        # A typo in a security setting must fail the boot, not leave the default in place.
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            WebhookSettings.model_validate(config)


class TestConfigLoading:
    def test_reads_the_webhooks_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(monkeypatch, webhooks={"url": {"allowed_domains": ["hooks.example.com"]}})

        assert get_webhook_settings().url.allowed_domains == ["hooks.example.com"]

    def test_invalid_block_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(monkeypatch, webhooks={"url": {"allowed_ports": [70000]}})

        with pytest.raises(ValidationError):
            get_webhook_settings()

    def test_enabled_without_http_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(monkeypatch, http=None)

        assert webhooks_enabled() is True

    def test_http_disable_webhooks_turns_them_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(monkeypatch, http={"disable_webhooks": True})

        assert webhooks_enabled() is False


class TestCheckWebhookUrl:
    def test_accepts_absolute_https_url(self) -> None:
        parsed = check_webhook_url("https://hooks.example.com/run?token=abc", WebhookUrlPolicy())

        assert parsed.host == "hooks.example.com"

    def test_rejects_url_over_the_length_cap(self) -> None:
        policy = WebhookUrlPolicy(max_url_length=30)

        with pytest.raises(WebhookUrlError, match="Webhook URL too long"):
            check_webhook_url("https://hooks.example.com/" + "x" * 10, policy)

    def test_rejects_relative_url(self) -> None:
        with pytest.raises(WebhookUrlError, match="Relative webhook URLs are not supported"):
            check_webhook_url("/custom/webhook", _OPEN_POLICY)

    @pytest.mark.parametrize("url", ["ftp://hooks.example.com/x", "file:///etc/passwd", "hooks.example.com/x"])
    def test_rejects_non_http_schemes(self, url: str) -> None:
        with pytest.raises(WebhookUrlError, match="Webhook must use http or https"):
            check_webhook_url(url, _OPEN_POLICY)

    def test_require_https_rejects_http(self) -> None:
        with pytest.raises(WebhookUrlError, match="Webhook must use https"):
            check_webhook_url("http://hooks.example.com/x", WebhookUrlPolicy(require_https=True))

    def test_rejects_url_without_host(self) -> None:
        with pytest.raises(WebhookUrlError, match="Invalid webhook hostname"):
            check_webhook_url("http:///path", _OPEN_POLICY)

    @pytest.mark.parametrize(
        "url", ["https://hooks.example.com:99999/x", "https://hooks.example.com:abc/x", "http://hooks.exa\x00mple.com/"]
    )
    def test_rejects_unparseable_urls(self, url: str) -> None:
        with pytest.raises(WebhookUrlError, match="Invalid webhook URL"):
            check_webhook_url(url, _OPEN_POLICY)

    @pytest.mark.parametrize(
        ("url", "allowed"),
        [
            ("https://hooks.example.com/x", True),
            ("https://hooks.example.com:443/x", True),
            ("http://hooks.example.com/x", False),
            ("https://hooks.example.com:8443/x", False),
        ],
    )
    def test_allowed_ports_use_the_scheme_default(self, url: str, allowed: bool) -> None:
        policy = WebhookUrlPolicy(require_https=False, allowed_ports=[443])

        if allowed:
            check_webhook_url(url, policy)
        else:
            with pytest.raises(WebhookUrlError, match=r"Webhook port \d+ not allowed"):
                check_webhook_url(url, policy)

    @pytest.mark.parametrize(
        ("url", "allowed"),
        [
            ("https://hooks.corp.com/x", True),
            ("https://HOOKS.corp.com./x", True),
            ("https://corp.com/x", False),
            ("https://evilcorp.com/x", False),
            ("https://hooks.corp.com.evil.com/x", False),
        ],
    )
    def test_allowed_domains(self, url: str, allowed: bool) -> None:
        policy = WebhookUrlPolicy(allowed_domains=["*.corp.com"])

        if allowed:
            check_webhook_url(url, policy)
        else:
            with pytest.raises(WebhookUrlError, match="Webhook domain not allowed"):
                check_webhook_url(url, policy)


class TestResolveWebhookAddress:
    @pytest.mark.parametrize(
        ("host", "reason"),
        [
            ("127.0.0.1", "loopback IP range"),
            ("::1", "loopback IP range"),
            ("::ffff:127.0.0.1", "loopback IP range"),
            ("169.254.169.254", "cloud metadata endpoint"),
            ("100.100.100.200", "cloud metadata endpoint"),
            ("fd00:ec2::254", "cloud metadata endpoint"),
            ("64:ff9b::a9fe:a9fe", "cloud metadata endpoint"),
            ("169.254.10.10", "blocked IP range"),
            ("0.0.0.0", "blocked IP range"),
            ("224.0.0.1", "blocked IP range"),
            ("fe80::1", "blocked IP range"),
            ("::", "blocked IP range"),
        ],
    )
    async def test_blocks_ip_literals_by_default(self, host: str, reason: str) -> None:
        with pytest.raises(WebhookUrlError, match=f"Webhook host blocked: {reason}"):
            await resolve_webhook_address(host, 443, WebhookUrlPolicy())

    @pytest.mark.parametrize("host", ["10.0.0.5", "172.16.0.1", "192.168.1.10", "fc00::1", "8.8.8.8"])
    async def test_private_and_public_literals_pass_by_default(self, host: str) -> None:
        assert await resolve_webhook_address(host, 443, WebhookUrlPolicy()) is None

    @pytest.mark.parametrize("host", ["10.0.0.5", "100.64.0.1", "fc00::1"])
    async def test_disable_private_ips_blocks_private_ranges(self, host: str) -> None:
        with pytest.raises(WebhookUrlError, match="private IP range"):
            await resolve_webhook_address(host, 443, WebhookUrlPolicy(disable_private_ips=True))

    async def test_loopback_allowed_when_disable_loopback_is_false(self) -> None:
        assert await resolve_webhook_address("127.0.0.1", 80, WebhookUrlPolicy(disable_loopback=False)) is None

    @pytest.mark.parametrize("host", ["localhost", "host.docker.internal", "localhost.localdomain"])
    async def test_blocks_localhost_names(self, host: str) -> None:
        with pytest.raises(WebhookUrlError, match="localhost address"):
            await resolve_webhook_address(host, 80, WebhookUrlPolicy())

    async def test_localhost_names_connect_by_name_when_loopback_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Docker Desktop maps host.docker.internal outside 127.0.0.0/8; the name is what was allowed.
        _resolve_to(monkeypatch, "169.254.1.1")

        assert (
            await resolve_webhook_address("host.docker.internal", 80, WebhookUrlPolicy(disable_loopback=False)) is None
        )

    @pytest.mark.parametrize(
        ("host", "reason"),
        [
            ("metadata.google.internal", "cloud metadata endpoint"),
            ("metadata", "cloud metadata endpoint"),
            ("billing.default.svc.cluster.local", "Kubernetes internal DNS"),
        ],
    )
    async def test_blocks_internal_hostnames(self, host: str, reason: str) -> None:
        with pytest.raises(WebhookUrlError, match=reason):
            await resolve_webhook_address(host, 443, WebhookUrlPolicy())

    async def test_exact_allow_list_entry_skips_address_checks(self) -> None:
        policy = WebhookUrlPolicy(allowed_domains=["billing.default.svc.cluster.local"])

        assert await resolve_webhook_address("billing.default.svc.cluster.local", 80, policy) is None

    async def test_wildcard_allow_list_entry_still_checks_addresses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _resolve_to(monkeypatch, "127.0.0.1")
        policy = WebhookUrlPolicy(allowed_domains=["*.corp.com"])

        with pytest.raises(WebhookUrlError, match="loopback IP range"):
            await resolve_webhook_address("hooks.corp.com", 443, policy)

    async def test_returns_the_first_resolved_address_to_pin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _resolve_to(monkeypatch, "93.184.215.14", "2606:2800:21f:cb07:6820:80da:af6b:8b2c")

        assert await resolve_webhook_address("hooks.example.com", 443, WebhookUrlPolicy()) == "93.184.215.14"

    async def test_any_blocked_address_rejects_the_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A rebinding-style answer mixing a public and an internal address must not slip through.
        _resolve_to(monkeypatch, "93.184.215.14", "169.254.169.254")

        with pytest.raises(WebhookUrlError, match="cloud metadata endpoint"):
            await resolve_webhook_address("hooks.example.com", 443, WebhookUrlPolicy())

    async def test_resolved_private_address_blocked_when_private_ips_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _resolve_to(monkeypatch, "10.1.2.3")

        with pytest.raises(WebhookUrlError, match="private IP range"):
            await resolve_webhook_address("internal.example.com", 443, WebhookUrlPolicy(disable_private_ips=True))


class TestResolve:
    async def test_returns_every_address(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _getaddrinfo(host: str, port: int, **_kwargs: Any) -> list[tuple[Any, ...]]:
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", port)),
                (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:2800::1", port, 0, 0)),
            ]

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", _getaddrinfo)

        addresses = await _real_resolve("hooks.example.com", 443)

        assert addresses == [ipaddress.ip_address("93.184.215.14"), ipaddress.ip_address("2606:2800::1")]

    async def test_lookup_failure_is_a_policy_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _getaddrinfo(host: str, port: int, **_kwargs: Any) -> list[tuple[Any, ...]]:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", _getaddrinfo)

        # A DNS error is retryable at delivery time, unlike a policy block.
        with pytest.raises(WebhookDNSError, match="DNS resolution failed"):
            await _real_resolve("missing.example.com", 443)

    async def test_empty_answer_is_a_policy_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _getaddrinfo(host: str, port: int, **_kwargs: Any) -> list[tuple[Any, ...]]:
            return []

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", _getaddrinfo)

        with pytest.raises(WebhookUrlError, match="DNS resolution failed"):
            await _real_resolve("empty.example.com", 443)

    async def test_slow_lookup_times_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _getaddrinfo(host: str, port: int, **_kwargs: Any) -> list[tuple[Any, ...]]:
            await asyncio.sleep(1)
            return []

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", _getaddrinfo)
        monkeypatch.setattr(webhook_policy, "_DNS_TIMEOUT_SECONDS", 0.01)

        with pytest.raises(WebhookUrlError, match="DNS resolution failed"):
            await _real_resolve("slow.example.com", 443)


class TestValidateWebhookUrl:
    async def test_applies_the_configured_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(monkeypatch, webhooks={"url": {"allowed_domains": ["hooks.example.com"]}})

        await validate_webhook_url("https://hooks.example.com/x")
        with pytest.raises(WebhookUrlError, match="Webhook must use https"):
            await validate_webhook_url("http://hooks.example.com/x")

    async def test_resolves_the_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _resolve_to(monkeypatch, "127.0.0.1")

        with pytest.raises(WebhookUrlError, match="loopback IP range"):
            await validate_webhook_url("http://rebind.example.com/x")


class TestEnsureWebhookAllowed:
    @pytest.mark.parametrize("webhook", [None, ""])
    async def test_no_webhook_is_a_no_op(self, monkeypatch: pytest.MonkeyPatch, webhook: str | None) -> None:
        _use_config(monkeypatch, http={"disable_webhooks": True})

        await ensure_webhook_allowed(webhook)

    async def test_allowed_url_passes(self) -> None:
        await ensure_webhook_allowed("https://hooks.example.com/x?token=abc")

    async def test_disabled_webhooks_are_a_422(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(monkeypatch, http={"disable_webhooks": True})

        with pytest.raises(HTTPException) as exc_info:
            await ensure_webhook_allowed("https://hooks.example.com/x")

        assert exc_info.value.status_code == 422
        assert "http.disable_webhooks" in exc_info.value.detail

    async def test_policy_violation_is_a_422_with_the_reason(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            await ensure_webhook_allowed("http://169.254.169.254/latest/meta-data")

        assert exc_info.value.status_code == 422
        assert exc_info.value.detail == "Webhook host blocked: cloud metadata endpoint"
