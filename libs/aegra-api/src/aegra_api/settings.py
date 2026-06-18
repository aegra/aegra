import logging
import re
from typing import Annotated
from urllib.parse import parse_qsl, quote_plus, urlencode

from pydantic import BeforeValidator, computed_field, model_validator
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


class RedisSettings(EnvBase):
    """Redis settings for the event broker.

    When REDIS_BROKER_ENABLED is True, SSE streaming uses Redis pub/sub
    instead of in-memory queues, enabling multi-instance deployments.
    """

    REDIS_BROKER_ENABLED: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CHANNEL_PREFIX: str = "aegra:run:"
    REDIS_MAX_CONNECTIONS: int = 250


class WorkerSettings(EnvBase):
    """Worker configuration for background graph execution.

    When REDIS_BROKER_ENABLED is True, runs are dispatched to worker
    coroutines via a Redis List job queue instead of local asyncio tasks.
    Each worker loop dequeues run_ids from Redis and spawns up to
    N_JOBS_PER_WORKER concurrent asyncio tasks for graph execution.
    """
    ASSISTANT_ID: str | None = None
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


settings = Settings()
