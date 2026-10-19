"""Webhook configuration and URL policy.

Reads the ``webhooks`` block of aegra.json (the same keys LangGraph's
langgraph.json uses) and decides which user-supplied webhook URLs the server
agrees to call. The checks run when a run or cron is created and again right
before every call, so a policy change also covers webhooks stored earlier.
"""

import asyncio
import ipaddress
import os
import re
import socket
from functools import cache
from typing import Self

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from aegra_api.config import load_http_config, load_webhooks_config

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

DEFAULT_ENV_PREFIX = "LG_WEBHOOK_"

# Top-level payload keys; the same vocabulary as LangGraph's webhooks.allowed_fields.
WEBHOOK_PAYLOAD_FIELDS: frozenset[str] = frozenset(
    {
        "run_id",
        "thread_id",
        "assistant_id",
        "created_at",
        "updated_at",
        "metadata",
        "kwargs",
        "multitask_strategy",
        "status",
        "run_started_at",
        "run_ended_at",
        "webhook_sent_at",
        "values",
        "error",
    }
)

_ENV_TEMPLATE_RE = re.compile(r"\$\{\{\s*(.*?)\s*\}\}")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")

_DNS_TIMEOUT_SECONDS = 10.0

_LOCALHOST_NAMES = frozenset({"localhost", "localhost.localdomain", "host.docker.internal"})
_METADATA_HOSTNAMES = frozenset({"metadata", "metadata.google.internal", "metadata.amazonaws.com", "instance-data"})
_K8S_SERVICE_SUFFIX = ".svc.cluster.local"

_LOOPBACK_NETWORKS = tuple(ipaddress.ip_network(net) for net in ("127.0.0.0/8", "::1/128"))
_PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(net) for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "fc00::/7")
)
# Link-local, "this network", multicast, reserved, benchmarking and documentation ranges.
_ALWAYS_BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(net)
    for net in (
        "0.0.0.0/8",
        "169.254.0.0/16",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::/96",
        "64:ff9b:1::/48",
        "fe80::/10",
        "ff00::/8",
    )
)
# Some providers serve instance metadata from addresses inside the private ranges.
_METADATA_ADDRESSES = frozenset(
    ipaddress.ip_address(addr) for addr in ("169.254.169.254", "169.254.170.2", "100.100.100.200", "fd00:ec2::254")
)
_NAT64_NETWORK = ipaddress.IPv6Network("64:ff9b::/96")


class WebhookUrlError(ValueError):
    """A webhook URL the server refuses to call. The message is safe to return to the caller."""


class WebhookUrlPolicy(BaseModel):
    """The ``webhooks.url`` block. Field defaults apply to keys a configured block leaves out."""

    require_https: bool = True
    allowed_domains: list[str] = Field(default_factory=list)
    allowed_ports: list[int] = Field(default_factory=list)
    max_url_length: int = Field(2048, gt=0)
    disable_loopback: bool = True
    disable_private_ips: bool = False

    @field_validator("allowed_domains")
    @classmethod
    def _normalize_domains(cls, domains: list[str]) -> list[str]:
        """Lowercase entries and only allow ``*.`` as a leading wildcard."""
        normalized: list[str] = []
        for domain in domains:
            entry = domain.strip().lower().rstrip(".")
            if not entry:
                raise ValueError("webhooks.url.allowed_domains entries must be non-empty strings")
            if "*" in entry and (not entry.startswith("*.") or "*" in entry[2:]):
                raise ValueError(f"webhooks.url.allowed_domains wildcards must be a '*.' prefix, got {domain!r}")
            normalized.append(entry)
        return normalized

    @field_validator("allowed_ports")
    @classmethod
    def _check_ports(cls, ports: list[int]) -> list[int]:
        """Reject port numbers no URL can carry."""
        for port in ports:
            if not 0 < port < 65536:
                raise ValueError(f"webhooks.url.allowed_ports entries must be between 1 and 65535, got {port}")
        return ports

    def allows_domain(self, host: str) -> bool:
        """True when *host* equals an entry or is a subdomain of a ``*.`` entry."""
        return any(
            host.endswith(entry[1:]) if entry.startswith("*.") else host == entry for entry in self.allowed_domains
        )

    def trusts(self, host: str) -> bool:
        """Exact allow-list entries are vouched for by the operator and skip the address checks."""
        return host in self.allowed_domains


def _unconfigured_url_policy() -> WebhookUrlPolicy:
    # Without a webhooks.url block LangGraph accepts plain http and URLs up to 4096 characters.
    return WebhookUrlPolicy(require_https=False, max_url_length=4096)


def _render_header(value: str, env_prefix: str) -> str:
    """Resolve ``${{ env.VAR }}`` templates; VAR must start with *env_prefix* and be set."""

    def _substitute(match: re.Match[str]) -> str:
        expression = match.group(1)
        if not expression.startswith("env."):
            raise ValueError(f"webhooks.headers only supports ${{{{ env.VAR }}}} templates, got {match.group(0)!r}")
        name = expression.removeprefix("env.")
        if not _ENV_NAME_RE.match(name):
            raise ValueError(f"webhooks.headers references an invalid environment variable name {name!r}")
        if not name.startswith(env_prefix):
            raise ValueError(f"webhooks.headers variable {name!r} must start with env_prefix {env_prefix!r}")
        resolved = os.environ.get(name)
        if resolved is None:
            raise ValueError(f"webhooks.headers references {name!r}, which is not set")
        return resolved

    return _ENV_TEMPLATE_RE.sub(_substitute, value)


class WebhookSettings(BaseModel):
    """The validated ``webhooks`` block, with header templates already resolved."""

    env_prefix: str = DEFAULT_ENV_PREFIX
    url: WebhookUrlPolicy = Field(default_factory=_unconfigured_url_policy)
    headers: dict[str, str] = Field(default_factory=dict)
    allowed_fields: frozenset[str] | None = None

    @field_validator("url", mode="before")
    @classmethod
    def _default_url_policy(cls, value: object) -> object:
        """Treat an explicit null like a missing block."""
        return _unconfigured_url_policy() if value is None else value

    @field_validator("allowed_fields")
    @classmethod
    def _check_allowed_fields(cls, fields: frozenset[str] | None) -> frozenset[str] | None:
        """Reject unknown payload keys; an empty list means every key, as in LangGraph."""
        if not fields:
            return None
        unknown = sorted(fields - WEBHOOK_PAYLOAD_FIELDS)
        if unknown:
            raise ValueError(
                f"webhooks.allowed_fields has unknown fields {unknown}; valid fields are {sorted(WEBHOOK_PAYLOAD_FIELDS)}"
            )
        return fields

    @model_validator(mode="after")
    def _render_headers(self) -> Self:
        """Resolve header templates once, so a missing variable fails the boot instead of every call."""
        rendered: dict[str, str] = {}
        for name, value in self.headers.items():
            if not _HEADER_NAME_RE.match(name):
                raise ValueError(f"webhooks.headers has an invalid header name {name!r}")
            resolved = _render_header(value, self.env_prefix)
            if "\r" in resolved or "\n" in resolved:
                raise ValueError(f"webhooks.headers value for {name!r} contains a line break")
            rendered[name] = resolved
        self.headers = rendered
        return self


@cache
def get_webhook_settings() -> WebhookSettings:
    """Validated ``webhooks`` config. The lifespan resolves it so bad config fails the boot."""
    return WebhookSettings.model_validate(load_webhooks_config() or {})


@cache
def webhooks_enabled() -> bool:
    """False when aegra.json sets ``http.disable_webhooks``."""
    http_config = load_http_config() or {}
    return not http_config.get("disable_webhooks", False)


def webhook_host(url: httpx.URL) -> str:
    """The ASCII host a request to *url* connects to, without a trailing dot."""
    return url.raw_host.decode("ascii").rstrip(".")


def webhook_port(url: httpx.URL) -> int:
    """The port a request to *url* connects to."""
    if url.port is not None:
        return url.port
    return 443 if url.scheme == "https" else 80


def check_webhook_url(url: str, policy: WebhookUrlPolicy) -> httpx.URL:
    """Apply the parts of *policy* that need no DNS lookup and return the parsed URL.

    Parsed with httpx, the client that later sends the request, so both agree on the host.
    """
    if len(url) > policy.max_url_length:
        raise WebhookUrlError("Webhook URL too long")
    if url.startswith("/"):
        raise WebhookUrlError("Relative webhook URLs are not supported; use an absolute http or https URL")
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL:
        raise WebhookUrlError("Invalid webhook URL") from None
    if parsed.scheme not in ("http", "https"):
        raise WebhookUrlError("Webhook must use http or https")
    if policy.require_https and parsed.scheme != "https":
        raise WebhookUrlError("Webhook must use https")
    host = webhook_host(parsed)
    if not host:
        raise WebhookUrlError("Invalid webhook hostname")
    port = webhook_port(parsed)
    if not 0 < port < 65536:
        raise WebhookUrlError("Invalid webhook URL")
    if policy.allowed_ports and port not in policy.allowed_ports:
        raise WebhookUrlError(f"Webhook port {port} not allowed")
    if policy.allowed_domains and not policy.allows_domain(host):
        raise WebhookUrlError("Webhook domain not allowed")
    return parsed


def _nat64_ipv4(address: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address embedded in a NAT64 well-known-prefix address."""
    if address not in _NAT64_NETWORK:
        return None
    return ipaddress.IPv4Address(address.packed[-4:])


def _blocked_reason(address: IPAddress, policy: WebhookUrlPolicy) -> str | None:
    """Why *policy* forbids connecting to *address*, or None when it allows it."""
    if isinstance(address, ipaddress.IPv6Address):
        embedded = address.ipv4_mapped or _nat64_ipv4(address)
        if embedded is not None:
            return _blocked_reason(embedded, policy)
    if any(address in net for net in _LOOPBACK_NETWORKS):
        return "loopback IP range" if policy.disable_loopback else None
    if address in _METADATA_ADDRESSES:
        return "cloud metadata endpoint"
    if any(address in net for net in _ALWAYS_BLOCKED_NETWORKS):
        return "blocked IP range"
    if policy.disable_private_ips and any(address in net for net in _PRIVATE_NETWORKS):
        return "private IP range"
    return None


def _raise_if_blocked(address: IPAddress, policy: WebhookUrlPolicy) -> None:
    reason = _blocked_reason(address, policy)
    if reason is not None:
        raise WebhookUrlError(f"Webhook host blocked: {reason}")


def _check_hostname(host: str, policy: WebhookUrlPolicy) -> None:
    """Names that point inside the deployment no matter what DNS says."""
    if policy.disable_loopback and host in _LOCALHOST_NAMES:
        raise WebhookUrlError("Webhook host blocked: localhost address")
    if host in _METADATA_HOSTNAMES:
        raise WebhookUrlError("Webhook host blocked: cloud metadata endpoint")
    if host.endswith(_K8S_SERVICE_SUFFIX):
        raise WebhookUrlError("Webhook host blocked: Kubernetes internal DNS")


async def _resolve(host: str, port: int) -> list[IPAddress]:
    loop = asyncio.get_running_loop()
    try:
        async with asyncio.timeout(_DNS_TIMEOUT_SECONDS):
            infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, TimeoutError):
        raise WebhookUrlError("Webhook host blocked: DNS resolution failed") from None
    try:
        addresses: list[IPAddress] = [ipaddress.ip_address(info[4][0]) for info in infos]
    except ValueError:
        raise WebhookUrlError("Webhook host blocked: invalid IP address") from None
    if not addresses:
        raise WebhookUrlError("Webhook host blocked: DNS resolution failed")
    return addresses


async def resolve_webhook_address(host: str, port: int, policy: WebhookUrlPolicy) -> str | None:
    """Check where *host* points and return the resolved address to pin the connection to.

    None means use the URL's host as-is: an IP literal, a trusted host, or an allowed
    localhost alias (Docker can map those outside the loopback range).
    """
    if policy.trusts(host):
        return None
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        _raise_if_blocked(literal, policy)
        return None
    _check_hostname(host, policy)
    if host in _LOCALHOST_NAMES:
        return None
    addresses = await _resolve(host, port)
    for address in addresses:
        _raise_if_blocked(address, policy)
    return str(addresses[0])


async def validate_webhook_url(url: str) -> None:
    """Raise WebhookUrlError unless the server's policy allows calling *url*."""
    policy = get_webhook_settings().url
    parsed = check_webhook_url(url, policy)
    await resolve_webhook_address(webhook_host(parsed), webhook_port(parsed), policy)


async def ensure_webhook_allowed(webhook: str | None) -> None:
    """422 when a request sets a webhook this server will not call."""
    if not webhook:
        return
    if not webhooks_enabled():
        raise HTTPException(422, "Webhooks are disabled on this server (http.disable_webhooks)")
    try:
        await validate_webhook_url(webhook)
    except WebhookUrlError as exc:
        raise HTTPException(422, str(exc)) from None
