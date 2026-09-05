import logging
import re
from typing import Annotated, NamedTuple
from urllib.parse import SplitResult, parse_qsl, quote_plus, unquote, urlencode, urlsplit

from pydantic import BeforeValidator, Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aegra_api import __version__
from aegra_api.constants import MULTIHOST_URL_RE

_logger = logging.getLogger(__name__)

# libpq sslmode → asyncpg ssl query param. asyncpg's ssl param validates
# via SSLMode.parse(), which accepts libpq spellings only — "true"/"false"
# raise ClientConfigurationError. asyncpg has no "allow"; map it to "prefer"
# (the closest try-TLS-then-fallback mode).
_SSLMODE_TO_ASYNCPG: dict[str, str] = {
    "disable": "disable",
    "allow": "prefer",
    "prefer": "prefer",
    "require": "require",
    "verify-ca": "verify-ca",
    "verify-full": "verify-full",
}

# libpq params that asyncpg rejects as unknown kwargs. We strip these from
# the async URL — users who need them must use PG* env vars or a custom
# SSLContext, neither of which fits the URL-only fast path.
_LIBPQ_ONLY_PARAMS: frozenset[str] = frozenset(
    {
        "sslmode",
        "sslcert",
        "sslkey",
        "sslrootcert",
        "sslcrl",
        "channel_binding",
        "gssencmode",
        "target_session_attrs",
    }
)


def parse_lower(v: str) -> str:
    """Converts to lowercase and strips whitespace."""
    return v.strip().lower() if isinstance(v, str) else v


def parse_upper(v: str) -> str:
    """Converts to uppercase and strips whitespace."""
    return v.strip().upper() if isinstance(v, str) else v


# Custom types for automatic formatting
LowerStr = Annotated[str, BeforeValidator(parse_lower)]
UpperStr = Annotated[str, BeforeValidator(parse_upper)]


class EnvBase(BaseSettings):
    """Base settings model that ignores unknown environment variables."""

    model_config = SettingsConfigDict(
        extra="ignore",
    )


class AppSettings(EnvBase):
    """General application settings."""

    PROJECT_NAME: str = "Aegra"
    VERSION: str = __version__

    # Server config
    HOST: str = "0.0.0.0"  # nosec B104
    PORT: int = 2026
    SERVER_URL: str | None = None

    @model_validator(mode="after")
    def _validate_keepalive_interval(self) -> "AppSettings":
        """Reject non-positive keepalive intervals during settings validation."""
        if self.KEEPALIVE_INTERVAL_SECS <= 0:
            raise ValueError(f"KEEPALIVE_INTERVAL_SECS must be greater than 0, got {self.KEEPALIVE_INTERVAL_SECS}")
        return self

    @model_validator(mode="after")
    def _derive_server_url(self) -> "AppSettings":
        """Derive SERVER_URL from HOST/PORT when not explicitly set."""
        if self.SERVER_URL is None:
            host = "localhost" if self.HOST in ("0.0.0.0", "127.0.0.1") else self.HOST  # nosec B104
            object.__setattr__(self, "SERVER_URL", f"http://{host}:{self.PORT}")
        return self

    # App logic
    AEGRA_CONFIG: str = "aegra.json"  # Default config file path
    KEEPALIVE_INTERVAL_SECS: float = 5  # Heartbeat interval for join/wait endpoints
    AUTH_TYPE: LowerStr = "noop"
    ENV_MODE: UpperStr = "LOCAL"
    DEBUG: bool = False
    # Default 1000 matches LangGraph Platform threads.search (Agent Server OpenAPI max).
    MAX_SEARCH_LIMIT: int = Field(default=1000, ge=1)

    # Run alembic upgrade head on startup. Default True (dev / single-pod).
    # Set False for multi-pod K8s to avoid advisory-lock probe timeouts;
    # run migrations out-of-band via `aegra db upgrade`.
    RUN_MIGRATIONS_ON_STARTUP: bool = True

    # Logging
    LOG_LEVEL: UpperStr = "INFO"
    LOG_VERBOSITY: LowerStr = "verbose"
    LOG_EXCLUDE_PATHS: str = ""  # Comma-separated path prefixes to skip in access logs

    @computed_field
    @property
    def log_exclude_paths(self) -> tuple[str, ...]:
        """Parse LOG_EXCLUDE_PATHS into a tuple of path prefixes."""
        if not self.LOG_EXCLUDE_PATHS:
            return ()
        return tuple(part.strip() for part in self.LOG_EXCLUDE_PATHS.split(",") if part.strip())

    @computed_field
    @property
    def sse_ping_interval_secs(self) -> int:
        """Integer ping interval for ``EventSourceResponse``.

        sse-starlette accepts only ``int`` seconds; the underlying setting is
        ``float`` to support sub-second heartbeats in the legacy JSON-wait
        endpoints and in tests. Clamp to ``>= 1`` so 0/negative floats can't
        produce a zero ping interval.
        """
        return max(1, int(self.KEEPALIVE_INTERVAL_SECS))


class DatabaseSettings(EnvBase):
    """Database connection settings.

    Supports two configuration modes:
    1. DATABASE_URL (standard for containerized deployments) — parsed into individual fields
    2. Individual POSTGRES_* vars — used when DATABASE_URL is not set
    """

    DATABASE_URL: str | None = None

    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "aegra"
    DB_ECHO_LOG: bool = False

    @staticmethod
    def _normalize_scheme(url: str, target_scheme: str) -> str:
        """Replace the URL scheme/driver prefix with the target scheme."""
        return re.sub(r"^postgres(?:ql)?(\+\w+)?://", f"{target_scheme}://", url)

    @staticmethod
    def _translate_libpq_params_for_asyncpg(url: str) -> str:
        """Strip libpq-only query params from an asyncpg URL.

        SQLAlchemy's asyncpg dialect forwards every URL query param as a
        kwarg to ``asyncpg.connect()``. asyncpg rejects libpq spellings
        (``sslmode``, ``channel_binding``, ``sslcert``, …) as unknown
        kwargs, so a URL copied from any libpq-aware tool crashes at
        startup. We translate ``sslmode`` to asyncpg's ``ssl`` query param
        and drop the rest with a warning.

        psycopg (sync) accepts libpq syntax natively — ``database_url_sync``
        is not affected.
        """
        # String-splice on "?" rather than urlsplit/urlunsplit: stdlib drops
        # the "//" authority marker when netloc is empty (e.g. multi-host
        # URLs with no userinfo), corrupting ``postgresql+asyncpg:///db``
        # into ``postgresql+asyncpg:/db``.
        head, sep, query = url.partition("?")
        if not sep:
            return url

        rewritten: list[tuple[str, str]] = []
        dropped: list[str] = []

        for key, value in parse_qsl(query, keep_blank_values=True):
            if key == "sslmode":
                mapped = _SSLMODE_TO_ASYNCPG.get(value.lower())
                if mapped is None:
                    _logger.warning("Unknown sslmode=%r in DATABASE_URL; ignoring", value)
                    continue
                if value.lower() in ("verify-ca", "verify-full"):
                    _logger.warning(
                        "DATABASE_URL sslmode=%s requires an SSLContext for cert verification; "
                        "asyncpg will negotiate TLS but skip the verify-* check. "
                        "Use PGSSLMODE + PGSSLROOTCERT env vars for full verification.",
                        value,
                    )
                rewritten.append(("ssl", mapped))
            elif key in _LIBPQ_ONLY_PARAMS:
                dropped.append(key)
            else:
                rewritten.append((key, value))

        if dropped:
            _logger.warning(
                "DATABASE_URL contains libpq-only params %s that asyncpg cannot accept; "
                "set them via PG* env vars instead.",
                sorted(dropped),
            )

        if not rewritten:
            return head
        # safe=",[]:" preserves the comma-separated host/port lists and
        # IPv6 literals (``[::1]``) produced by _to_sqlalchemy_multihost —
        # asyncpg's URL parser expects these raw, not percent-encoded.
        return f"{head}?{urlencode(rewritten, safe=',[]:')}"

    @staticmethod
    def _to_sqlalchemy_multihost(url: str) -> str:
        """Convert a libpq multi-host URL to SQLAlchemy query-param format.

        PostgreSQL libpq and psycopg accept comma-separated hosts in the
        URL authority (``host1:5432,host2:5433``).  SQLAlchemy's asyncpg
        dialect requires hosts and ports as query parameters instead.

        Single-host URLs are returned unchanged.
        """
        m = MULTIHOST_URL_RE.match(url)
        if not m:
            return url

        hostlist = m.group("hostlist")
        if "," not in hostlist:
            return url

        scheme = m.group("scheme")
        userinfo = m.group("userinfo") or ""
        path = m.group("path") or ""
        query = m.group("query") or ""

        hosts: list[str] = []
        ports: list[str] = []
        for spec in hostlist.split(","):
            if spec.startswith("["):
                # IPv6 literal: [::1]:5432 or [::1]
                if "]" not in spec:
                    msg = f"Malformed IPv6 in DATABASE_URL: `{spec}` — missing closing bracket"
                    raise ValueError(msg)
                bracket_end = spec.index("]")
                host = spec[: bracket_end + 1]
                rest = spec[bracket_end + 1 :]
                port = rest[1:] if rest.startswith(":") else ""
            else:
                host, _, port = spec.rpartition(":")
            if host and port:
                if not port.isdigit():
                    msg = f"Non-integer port in DATABASE_URL: `{spec}` — port must be a number, got `{port}`"
                    raise ValueError(msg)
                hosts.append(host)
                ports.append(port)
            else:
                hosts.append(host if host else spec)
                ports.append("5432")

        auth = f"{userinfo}@" if userinfo else ""
        ha_params = f"host={','.join(hosts)}&port={','.join(ports)}"
        all_params = f"{ha_params}&{query}" if query else ha_params

        return f"{scheme}{auth}/{path}?{all_params}"

    @computed_field
    @property
    def database_url(self) -> str:
        """Async URL for SQLAlchemy (asyncpg).

        When ``DATABASE_URL`` contains multiple comma-separated hosts
        (e.g. ``postgresql://h1:5432,h2:5432/db``), the URL is rewritten
        into SQLAlchemy's query-param multi-host format so that asyncpg
        receives hosts as a list and can fail over natively.
        """
        if self.DATABASE_URL:
            url = self._normalize_scheme(self.DATABASE_URL, "postgresql+asyncpg")
            url = self._to_sqlalchemy_multihost(url)
            return self._translate_libpq_params_for_asyncpg(url)
        return (
            f"postgresql+asyncpg://{quote_plus(self.POSTGRES_USER)}:{quote_plus(self.POSTGRES_PASSWORD)}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def database_url_sync(self) -> str:
        """Sync URL for LangGraph/Psycopg (postgresql://)."""
        if self.DATABASE_URL:
            return self._normalize_scheme(self.DATABASE_URL, "postgresql")
        return (
            f"postgresql://{quote_plus(self.POSTGRES_USER)}:{quote_plus(self.POSTGRES_PASSWORD)}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


class PoolSettings(EnvBase):
    """Connection pool settings for SQLAlchemy and LangGraph."""

    SQLALCHEMY_POOL_SIZE: int = 10
    SQLALCHEMY_MAX_OVERFLOW: int = 20

    LANGGRAPH_MIN_POOL_SIZE: int = 5
    LANGGRAPH_MAX_POOL_SIZE: int = 20


class ObservabilitySettings(EnvBase):
    """
    Unified settings for OpenTelemetry and Vendor targets.
    Supports Fan-out configuration via OTEL_TARGETS.
    """

    # General OTEL Config
    OTEL_SERVICE_NAME: str = "aegra-backend"
    OTEL_TARGETS: str = ""  # Comma-separated: "LANGFUSE,PHOENIX"
    OTEL_CONSOLE_EXPORT: bool = False  # For local debugging

    # --- Generic OTLP Target (Default/Custom) ---
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None
    OTEL_EXPORTER_OTLP_HEADERS: str | None = None

    # --- Prometheus Metrics ---
    ENABLE_PROMETHEUS_METRICS: bool = False

    # --- Langfuse Specifics ---
    LANGFUSE_BASE_URL: str = "http://localhost:3000"
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None

    # --- Phoenix Specifics ---
    PHOENIX_COLLECTOR_ENDPOINT: str = "http://127.0.0.1:6006/v1/traces"
    PHOENIX_API_KEY: str | None = None


SENTINEL_SCHEME = "redis+sentinel"
SENTINEL_TLS_SCHEME = "rediss+sentinel"
DEFAULT_SENTINEL_PORT = 26379
MAX_PORT = 65535
_SENTINEL_AUTH_PARAMS = frozenset({"sentinel_username", "sentinel_password"})
# TLS options, named as redis-py names them on a rediss:// URL so the two
# schemes stay consistent. Only accepted on the TLS scheme.
_SENTINEL_TLS_PARAMS = frozenset(
    {
        "ssl_cert_reqs",
        "ssl_ca_certs",
        "ssl_ca_path",
        "ssl_certfile",
        "ssl_keyfile",
        "ssl_password",
        "ssl_check_hostname",
    }
)
_SSL_CERT_REQS = frozenset({"none", "optional", "required"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class SentinelConfig(NamedTuple):
    """A parsed ``redis+sentinel://`` or ``rediss+sentinel://`` URL."""

    hosts: tuple[tuple[str, int], ...]
    master_name: str
    db: int
    username: str | None
    password: str | None
    sentinel_username: str | None
    sentinel_password: str | None
    ssl: bool
    ssl_options: dict[str, str | bool]


def _parse_ssl_options(query: dict[str, str]) -> dict[str, str | bool]:
    """Validate the ssl_* query parameters of a rediss+sentinel:// URL."""
    options: dict[str, str | bool] = {}
    for key in sorted(_SENTINEL_TLS_PARAMS & set(query)):
        value = query[key]
        if key == "ssl_check_hostname":
            lowered = value.strip().lower()
            if lowered in _TRUE_VALUES:
                options[key] = True
            elif lowered in _FALSE_VALUES:
                options[key] = False
            else:
                msg = f"REDIS_URL ssl_check_hostname must be a boolean, got `{value}`"
                raise ValueError(msg)
        elif key == "ssl_cert_reqs":
            if value not in _SSL_CERT_REQS:
                msg = f"REDIS_URL ssl_cert_reqs must be one of {sorted(_SSL_CERT_REQS)}, got `{value}`"
                raise ValueError(msg)
            options[key] = value
        else:
            options[key] = value
    return options


def _split_redis_url(url: str) -> SplitResult:
    """urlsplit REDIS_URL, naming the setting when the URL is unparseable.

    Stdlib raises a bare "Invalid IPv6 URL" for an unbalanced bracket, which
    does not say which setting is at fault.
    """
    try:
        return urlsplit(url)
    except ValueError as exc:
        msg = f"REDIS_URL is not a valid URL: {exc}"
        raise ValueError(msg) from exc


def _split_sentinel_hosts(netloc_hosts: str) -> tuple[tuple[str, int], ...]:
    """Split the comma-separated host list of a sentinel URL authority.

    Accepts "host:port", bare "host" (defaulting to 26379), and bracketed IPv6
    literals such as "[::1]:26379".
    """
    hosts: list[tuple[str, int]] = []
    for entry in netloc_hosts.split(","):
        spec = entry.strip()
        if not spec:
            continue
        if spec.startswith("["):
            if "]" not in spec:
                msg = f"Malformed IPv6 sentinel endpoint in REDIS_URL: `{spec}` -- missing closing bracket"
                raise ValueError(msg)
            bracket_end = spec.index("]")
            host = spec[1:bracket_end]
            rest = spec[bracket_end + 1 :]
            if rest and not rest.startswith(":"):
                # urlsplit rejects unbalanced brackets before we get here, so
                # this only guards direct callers of this helper.
                msg = f"Malformed sentinel endpoint in REDIS_URL: `{spec}`"
                raise ValueError(msg)
            port_str = rest[1:] if rest.startswith(":") else ""
        else:
            host, separator, port_str = spec.rpartition(":")
            if not separator:
                host, port_str = port_str, ""
        if not host:
            msg = f"Sentinel endpoint in REDIS_URL is missing a host: `{spec}`"
            raise ValueError(msg)
        if port_str and not port_str.isdigit():
            msg = f"Non-integer port in REDIS_URL sentinel endpoint: `{spec}` -- got `{port_str}`"
            raise ValueError(msg)
        port = int(port_str) if port_str else DEFAULT_SENTINEL_PORT
        if not 1 <= port <= MAX_PORT:
            msg = f"Port out of range in REDIS_URL sentinel endpoint: `{spec}` -- got `{port}`"
            raise ValueError(msg)
        hosts.append((host, port))
    if not hosts:
        raise ValueError("REDIS_URL names no sentinel endpoints")
    return tuple(hosts)


def _parse_sentinel_url(url: str) -> SentinelConfig:
    """Parse ``redis+sentinel://[user:pass@]host:port[,host:port]/master[/db]``.

    Userinfo is the *data node's* credentials, matching what it means in a
    plain ``redis://`` URL. The sentinels' own AUTH, which is usually
    different, comes from the ``sentinel_username`` / ``sentinel_password``
    query parameters.

    The ``rediss+sentinel://`` scheme wraps both the sentinel and the data
    node connections in TLS, configured by the ``ssl_*`` query parameters.
    """
    split = _split_redis_url(url)
    ssl_enabled = split.scheme == SENTINEL_TLS_SCHEME

    authority = split.netloc
    userinfo, _, hostpart = authority.rpartition("@")
    username: str | None = None
    password: str | None = None
    if userinfo:
        raw_user, _, raw_password = userinfo.partition(":")
        username = unquote(raw_user) or None
        password = unquote(raw_password) or None

    hosts = _split_sentinel_hosts(hostpart)

    segments = [segment for segment in split.path.split("/") if segment]
    if not segments:
        raise ValueError(
            f"REDIS_URL is missing the master name. Expected {SENTINEL_SCHEME}://host:port/<master_name>[/<db>]"
        )
    if len(segments) > 2:
        msg = f"REDIS_URL has too many path segments: `{split.path}`. Expected /<master_name>[/<db>]"
        raise ValueError(msg)
    master_name = unquote(segments[0])
    db = 0
    if len(segments) == 2:
        if not segments[1].isdigit():
            msg = f"Non-integer database index in REDIS_URL: `{segments[1]}`"
            raise ValueError(msg)
        db = int(segments[1])

    query = dict(parse_qsl(split.query, keep_blank_values=True))
    supported = _SENTINEL_AUTH_PARAMS | (_SENTINEL_TLS_PARAMS if ssl_enabled else frozenset())
    unknown = sorted(set(query) - supported)
    if unknown:
        if not ssl_enabled and set(unknown) & _SENTINEL_TLS_PARAMS:
            msg = (
                f"TLS options {sorted(set(unknown) & _SENTINEL_TLS_PARAMS)} need the "
                f"{SENTINEL_TLS_SCHEME}:// scheme; {SENTINEL_SCHEME}:// does not use TLS"
            )
            raise ValueError(msg)
        msg = f"Unsupported query parameters in REDIS_URL: {unknown}. Supported: {sorted(supported)}"
        raise ValueError(msg)

    ssl_options = _parse_ssl_options(query)
    if ssl_enabled:
        # redis-py's *async* SSLConnection defaulted check_hostname to False
        # before 6.0, and our floor is >=5.0.0. Pin it instead of inheriting.
        ssl_options.setdefault("ssl_check_hostname", True)

    return SentinelConfig(
        hosts=hosts,
        master_name=master_name,
        db=db,
        username=username,
        password=password,
        sentinel_username=query.get("sentinel_username") or None,
        sentinel_password=query.get("sentinel_password") or None,
        ssl=ssl_enabled,
        ssl_options=ssl_options,
    )


class RedisSettings(EnvBase):
    """Redis settings for the event broker.

    When REDIS_BROKER_ENABLED is True, SSE streaming uses Redis pub/sub
    instead of in-memory queues, enabling multi-instance deployments.

    REDIS_URL selects how the client connects. ``redis://`` and ``rediss://``
    dial a single endpoint, as before. ``redis+sentinel://`` instead resolves
    the current master through Sentinel and re-resolves it after a failover,
    and ``rediss+sentinel://`` does the same over TLS:

        redis+sentinel://sentinel-a:26379,sentinel-b:26379/mymaster/0
        rediss+sentinel://sentinel-a:26379/mymaster/0?ssl_ca_certs=/certs/ca.pem

    Sentinel is entirely opt-in -- keep a ``redis://`` URL and nothing about
    the existing behaviour changes.
    """

    REDIS_BROKER_ENABLED: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CHANNEL_PREFIX: str = "aegra:run:"
    REDIS_MAX_CONNECTIONS: int = 250
    # PING pooled connections idle longer than this (seconds) before reuse, so
    # server-side idle disconnects don't surface as ConnectionError. 0 disables.
    REDIS_HEALTH_CHECK_INTERVAL: int = Field(default=30, ge=0)
    # Non-negative retries after the initial attempt (up to 4 calls total) on
    # connection and timeout errors, with exponential backoff 50ms..1s.
    REDIS_RETRY_ATTEMPTS: int = Field(default=3, ge=0)

    @property
    def sentinel(self) -> SentinelConfig | None:
        """The parsed sentinel URL, or None for a direct connection."""
        if _split_redis_url(self.REDIS_URL).scheme not in (SENTINEL_SCHEME, SENTINEL_TLS_SCHEME):
            return None
        return _parse_sentinel_url(self.REDIS_URL)

    @model_validator(mode="after")
    def _validate_redis_url(self) -> "RedisSettings":
        """Parse a sentinel URL at startup so a typo fails before first connect."""
        if _split_redis_url(self.REDIS_URL).scheme in (SENTINEL_SCHEME, SENTINEL_TLS_SCHEME):
            _parse_sentinel_url(self.REDIS_URL)
        return self


class WorkerSettings(EnvBase):
    """Worker configuration for background graph execution.

    When REDIS_BROKER_ENABLED is True, runs are dispatched to worker
    coroutines via a Redis List job queue instead of local asyncio tasks.
    Each worker loop dequeues run_ids from Redis and spawns up to
    N_JOBS_PER_WORKER concurrent asyncio tasks for graph execution.
    """

    WORKER_COUNT: int = 3
    N_JOBS_PER_WORKER: int = 10
    WORKER_QUEUE_KEY: str = "aegra:jobs"
    WORKER_DRAIN_TIMEOUT: float = 30.0
    BG_JOB_TIMEOUT_SECS: int = 3600
    BG_JOB_MAX_RETRIES: int = 3

    # Lease-based crash recovery.
    # The lease must be long enough that a healthy worker NEVER loses it.
    # Safety margin = LEASE / HEARTBEAT = 30/10 = 3 missed heartbeats
    # before expiry (industry standard — matches Kubernetes liveness probes).
    # Worst-case recovery: ~30s lease expiry + ~20s reaper interval = ~50s.
    LEASE_DURATION_SECONDS: int = 30
    HEARTBEAT_INTERVAL_SECONDS: int = 10
    REAPER_INTERVAL_SECONDS: int = 15
    STUCK_PENDING_THRESHOLD_SECONDS: int = 120
    POSTGRES_POLL_INTERVAL_SECONDS: int = 5

    @model_validator(mode="after")
    def _validate_lease_timing(self) -> "WorkerSettings":
        """Ensure the worker lease safely outlives missed heartbeat intervals."""
        if self.LEASE_DURATION_SECONDS <= 2 * self.HEARTBEAT_INTERVAL_SECONDS:
            raise ValueError(
                f"LEASE_DURATION_SECONDS ({self.LEASE_DURATION_SECONDS}) must be "
                f"greater than 2 * HEARTBEAT_INTERVAL_SECONDS ({self.HEARTBEAT_INTERVAL_SECONDS}). "
                f"A worker must survive at least 2 missed heartbeats before its lease expires."
            )
        return self


class CronSettings(EnvBase):
    """Cron scheduler configuration.

    Controls the background scheduler that fires cron jobs.
    """

    CRON_ENABLED: bool = True
    CRON_POLL_INTERVAL_SECONDS: int = 60
    # Maximum lease duration for an in-flight cron firing. Once a cron is
    # claimed by ``get_due_crons`` its ``claimed_until`` is set to
    # ``now + CRON_CLAIM_DURATION_SECONDS`` so concurrent pollers and
    # subsequent ticks don't double-fire it. Should comfortably exceed the
    # worst-case ``_fire_cron`` duration. Defaults to 5 minutes.
    CRON_CLAIM_DURATION_SECONDS: int = 300
    # Cap on how many crons a single user may own. Set to 0 to disable.
    CRON_MAX_PER_USER: int = 100
    # Allow 6-field (seconds-first) cron schedules. Sub-minute schedules
    # multiply scheduler load and DB writes; off by default.
    CRON_ALLOW_SECONDS_SCHEDULE: bool = False
    # Cap on how many crons a single tick will fire (prevents one slow
    # poll from queuing up unbounded work).
    CRON_TICK_BATCH_SIZE: int = 100
    # Soft cap on JSONB payload size (input + config + context + checkpoint
    # + metadata combined) accepted on create/update.
    CRON_MAX_PAYLOAD_BYTES: int = 64 * 1024

    @model_validator(mode="after")
    def _validate_poll_interval(self) -> "CronSettings":
        """Reject non-positive cron poll intervals during settings validation."""
        if self.CRON_POLL_INTERVAL_SECONDS <= 0:
            raise ValueError(
                f"CRON_POLL_INTERVAL_SECONDS must be greater than 0, got {self.CRON_POLL_INTERVAL_SECONDS}"
            )
        if self.CRON_CLAIM_DURATION_SECONDS <= 0:
            raise ValueError(
                f"CRON_CLAIM_DURATION_SECONDS must be greater than 0, got {self.CRON_CLAIM_DURATION_SECONDS}"
            )
        if self.CRON_MAX_PER_USER < 0:
            raise ValueError(f"CRON_MAX_PER_USER must be >= 0, got {self.CRON_MAX_PER_USER}")
        if self.CRON_TICK_BATCH_SIZE <= 0:
            raise ValueError(f"CRON_TICK_BATCH_SIZE must be greater than 0, got {self.CRON_TICK_BATCH_SIZE}")
        if self.CRON_MAX_PAYLOAD_BYTES <= 0:
            raise ValueError(f"CRON_MAX_PAYLOAD_BYTES must be greater than 0, got {self.CRON_MAX_PAYLOAD_BYTES}")
        return self


class ThreadTTLSettings(EnvBase):
    """Thread TTL sweeper configuration.

    AEGRA_THREAD_TTL is either a bare number (default_ttl in minutes) or a
    JSON object with any of: strategy, default_ttl, sweep_interval_minutes,
    sweep_limit. When set it replaces the aegra.json checkpointer.ttl block
    entirely. LANGGRAPH_THREAD_TTL is accepted as a fallback alias so env
    files migrated from LangGraph Platform work unchanged; AEGRA_THREAD_TTL
    wins when both are set. Parsed and validated in services.thread_ttl.
    """

    AEGRA_THREAD_TTL: str | None = None
    LANGGRAPH_THREAD_TTL: str | None = None


class EventStreamingSettings(EnvBase):
    """Agent Protocol v2 event streaming (/threads/{id}/stream/events + /commands).

    On by default — it's a new endpoint set the LangGraph SDK targets and
    has no v1 to break. The flag is a kill switch: set false to disable v2
    serving (requests return 503 with an enable hint) and roll back without
    a redeploy. Also requires a langgraph/langchain-core new enough to emit
    native v3 events (enforced by event_streaming.capabilities; otherwise 503).
    """

    FF_V2_EVENT_STREAMING: bool = True


class Settings:
    """Container object that instantiates all application settings groups."""

    def __init__(self) -> None:
        """Build the settings tree from environment-backed settings models."""
        self.app = AppSettings()
        self.db = DatabaseSettings()
        self.pool = PoolSettings()
        self.observability = ObservabilitySettings()
        self.redis = RedisSettings()
        self.worker = WorkerSettings()
        self.cron = CronSettings()
        self.thread_ttl = ThreadTTLSettings()
        self.event_streaming = EventStreamingSettings()


settings = Settings()
