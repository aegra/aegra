"""Tests for AppSettings, DatabaseSettings, and WorkerSettings."""

from urllib.parse import quote_plus

import pytest
from pydantic import ValidationError
from sqlalchemy.engine import make_url

from aegra_api.settings import AppSettings, CronSettings, DatabaseSettings, WorkerSettings


class TestAppSettingsServerURL:
    """Test SERVER_URL derivation from HOST/PORT."""

    def _clear_app_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Remove env vars that affect AppSettings."""
        for var in (
            "HOST",
            "PORT",
            "SERVER_URL",
            "PROJECT_NAME",
            "AUTH_TYPE",
            "ENV_MODE",
            "DEBUG",
            "LOG_LEVEL",
            "LOG_VERBOSITY",
            "AEGRA_CONFIG",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_derives_localhost_when_host_is_all_interfaces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SERVER_URL uses localhost when HOST=0.0.0.0."""
        self._clear_app_env(monkeypatch)
        app = AppSettings(_env_file=None)

        assert app.HOST == "0.0.0.0"
        assert app.PORT == 2026
        assert app.SERVER_URL == "http://localhost:2026"

    def test_derives_localhost_when_host_is_loopback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SERVER_URL uses localhost when HOST=127.0.0.1."""
        self._clear_app_env(monkeypatch)
        monkeypatch.setenv("HOST", "127.0.0.1")
        app = AppSettings(_env_file=None)

        assert app.SERVER_URL == "http://localhost:2026"

    def test_derives_with_custom_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SERVER_URL uses the literal HOST when it's a custom address."""
        self._clear_app_env(monkeypatch)
        monkeypatch.setenv("HOST", "192.168.1.5")
        app = AppSettings(_env_file=None)

        assert app.SERVER_URL == "http://192.168.1.5:2026"

    def test_derives_with_custom_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SERVER_URL uses the configured PORT."""
        self._clear_app_env(monkeypatch)
        monkeypatch.setenv("PORT", "9090")
        app = AppSettings(_env_file=None)

        assert app.SERVER_URL == "http://localhost:9090"

    def test_explicit_server_url_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit SERVER_URL is not overridden by derivation."""
        self._clear_app_env(monkeypatch)
        monkeypatch.setenv("SERVER_URL", "https://api.example.com")
        app = AppSettings(_env_file=None)

        assert app.SERVER_URL == "https://api.example.com"

    def test_default_port_is_2026(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default PORT is 2026."""
        self._clear_app_env(monkeypatch)
        app = AppSettings(_env_file=None)

        assert app.PORT == 2026


class TestSsePingIntervalSecs:
    """``sse_ping_interval_secs`` derives an int ping value from the float setting."""

    def _clear(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KEEPALIVE_INTERVAL_SECS", raising=False)

    def test_default_matches_default_keepalive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear(monkeypatch)
        app = AppSettings(_env_file=None)

        assert app.KEEPALIVE_INTERVAL_SECS == 5
        assert app.sse_ping_interval_secs == 5

    def test_truncates_floats_to_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """sse-starlette requires int seconds; fractional part is dropped."""
        self._clear(monkeypatch)
        monkeypatch.setenv("KEEPALIVE_INTERVAL_SECS", "10.9")
        app = AppSettings(_env_file=None)

        assert app.sse_ping_interval_secs == 10

    def test_clamps_sub_second_to_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sub-second heartbeats (used by legacy JSON endpoints and tests)
        must not yield a zero ping interval that would flood the client."""
        self._clear(monkeypatch)
        monkeypatch.setenv("KEEPALIVE_INTERVAL_SECS", "0.5")
        app = AppSettings(_env_file=None)

        assert app.sse_ping_interval_secs == 1


class TestDatabaseURLSupport:
    """Test that DATABASE_URL is used directly for computed URLs."""

    def test_defaults_when_no_database_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Individual defaults are used when DATABASE_URL is not set."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("POSTGRES_USER", raising=False)
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
        monkeypatch.delenv("POSTGRES_HOST", raising=False)
        monkeypatch.delenv("POSTGRES_PORT", raising=False)
        monkeypatch.delenv("POSTGRES_DB", raising=False)

        db = DatabaseSettings(_env_file=None)

        assert db.POSTGRES_USER == "postgres"
        assert db.POSTGRES_HOST == "localhost"
        assert db.POSTGRES_PORT == "5432"
        assert db.POSTGRES_DB == "aegra"
        assert "postgres:postgres@localhost:5432/aegra" in db.database_url

    def test_database_url_used_directly_in_computed_urls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DATABASE_URL is used directly with correct driver prefix."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://rdsuser:rdspass@rds.aws.com:5432/prod")

        db = DatabaseSettings(_env_file=None)

        assert db.database_url == "postgresql+asyncpg://rdsuser:rdspass@rds.aws.com:5432/prod"
        assert db.database_url_sync == "postgresql://rdsuser:rdspass@rds.aws.com:5432/prod"

    def test_query_params_preserved_for_sync_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sync URL (psycopg) preserves libpq query params verbatim — psycopg
        speaks libpq natively."""
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql://user:pass@host:5432/db?sslmode=require&connect_timeout=10",
        )

        db = DatabaseSettings(_env_file=None)

        assert "sslmode=require" in db.database_url_sync
        assert "connect_timeout=10" in db.database_url_sync
        assert db.database_url_sync.startswith("postgresql://")

    def test_async_url_translates_sslmode_require_to_ssl_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """asyncpg rejects ``sslmode`` as an unknown kwarg. We translate it
        to ``ssl=true`` so the async engine starts cleanly when users paste
        a libpq-style URL (the common shape from RDS/Azure/GCP consoles)."""
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql://user:pass@host:5432/db?sslmode=require",
        )

        db = DatabaseSettings(_env_file=None)

        assert "sslmode" not in db.database_url
        assert "ssl=true" in db.database_url
        assert db.database_url.startswith("postgresql+asyncpg://")

    @pytest.mark.parametrize(
        ("sslmode", "expected_ssl"),
        [
            ("disable", "false"),
            ("allow", "false"),
            ("prefer", "false"),
            ("require", "true"),
            ("verify-ca", "true"),
            ("verify-full", "true"),
        ],
    )
    def test_async_url_translates_all_sslmode_values(
        self, monkeypatch: pytest.MonkeyPatch, sslmode: str, expected_ssl: str
    ) -> None:
        monkeypatch.setenv("DATABASE_URL", f"postgresql://u:p@h:5432/db?sslmode={sslmode}")

        db = DatabaseSettings(_env_file=None)

        assert f"ssl={expected_ssl}" in db.database_url
        assert "sslmode" not in db.database_url

    def test_async_url_strips_other_libpq_only_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``channel_binding`` / ``sslcert`` etc. also crash asyncpg as unknown
        kwargs. Strip them from the async URL; users needing them must set
        PG* env vars instead (psycopg sync path still sees the libpq form)."""
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql://u:p@h:5432/db?channel_binding=require&sslcert=/x.pem&connect_timeout=10",
        )

        db = DatabaseSettings(_env_file=None)

        assert "channel_binding" not in db.database_url
        assert "sslcert" not in db.database_url
        # Non-libpq-only params are preserved.
        assert "connect_timeout=10" in db.database_url

    def test_async_url_unknown_sslmode_is_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unknown sslmode values are dropped (with a warning) rather than
        forwarded to asyncpg, which would crash on the unknown kwarg."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/db?sslmode=bogus")

        db = DatabaseSettings(_env_file=None)

        assert "sslmode" not in db.database_url
        assert "ssl=" not in db.database_url

    def test_async_url_no_query_params_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/db")

        db = DatabaseSettings(_env_file=None)

        assert "?" not in db.database_url
        assert db.database_url.endswith("/db")

    def test_driver_prefix_normalized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Driver prefix is always normalized regardless of input."""
        monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@host:5432/db")

        db = DatabaseSettings(_env_file=None)

        assert db.database_url.startswith("postgresql+asyncpg://")
        assert db.database_url_sync.startswith("postgresql://")
        assert not db.database_url_sync.startswith("postgresql+")

    def test_legacy_postgres_scheme_normalized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Legacy postgres:// scheme (Heroku/Render) is normalized."""
        monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@host:5432/db")

        db = DatabaseSettings(_env_file=None)

        assert db.database_url.startswith("postgresql+asyncpg://")
        assert db.database_url_sync.startswith("postgresql://")
        assert "user:pass@host:5432/db" in db.database_url

    def test_individual_vars_still_work(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Individual POSTGRES_* vars work when DATABASE_URL is not set."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("POSTGRES_USER", "custom_user")
        monkeypatch.setenv("POSTGRES_PASSWORD", "custom_pass")
        monkeypatch.setenv("POSTGRES_HOST", "custom-host")
        monkeypatch.setenv("POSTGRES_PORT", "5555")
        monkeypatch.setenv("POSTGRES_DB", "custom_db")

        db = DatabaseSettings(_env_file=None)

        assert db.POSTGRES_USER == "custom_user"
        assert "custom_user:custom_pass@custom-host:5555/custom_db" in db.database_url

    def test_malformed_database_url_does_not_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Malformed DATABASE_URL doesn't crash — regex just won't match."""
        monkeypatch.setenv("DATABASE_URL", "not-a-url")

        db = DatabaseSettings(_env_file=None)

        # _normalize_scheme won't match, so URL passes through as-is
        assert db.DATABASE_URL == "not-a-url"


class TestSpecialCharactersInCredentials:
    """Test that special characters in POSTGRES_USER/POSTGRES_PASSWORD are URL-encoded."""

    def _clear_db_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "DATABASE_URL",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DB",
        ):
            monkeypatch.delenv(var, raising=False)

    @pytest.mark.parametrize(
        ("password", "expected_encoded"),
        [
            pytest.param("p@ssword", quote_plus("p@ssword"), id="at_sign"),
            pytest.param("p/ss#word", quote_plus("p/ss#word"), id="slash_and_hash"),
            pytest.param("pass%word", quote_plus("pass%word"), id="percent"),
            pytest.param("p?ss=word", quote_plus("p?ss=word"), id="question_and_equals"),
            pytest.param("p:ss+word", quote_plus("p:ss+word"), id="colon_and_plus"),
            pytest.param("p@ss/w#rd%2B", quote_plus("p@ss/w#rd%2B"), id="multiple_special"),
        ],
    )
    def test_password_url_encoded_in_database_url(
        self, monkeypatch: pytest.MonkeyPatch, password: str, expected_encoded: str
    ) -> None:
        """Special characters in POSTGRES_PASSWORD are URL-encoded in database_url."""
        self._clear_db_env(monkeypatch)
        monkeypatch.setenv("POSTGRES_USER", "user")
        monkeypatch.setenv("POSTGRES_PASSWORD", password)

        db = DatabaseSettings(_env_file=None)

        assert f"user:{expected_encoded}@" in db.database_url
        assert db.database_url.startswith("postgresql+asyncpg://")

    @pytest.mark.parametrize(
        ("password", "expected_encoded"),
        [
            pytest.param("p@ssword", quote_plus("p@ssword"), id="at_sign"),
            pytest.param("p/ss#word", quote_plus("p/ss#word"), id="slash_and_hash"),
        ],
    )
    def test_password_url_encoded_in_database_url_sync(
        self, monkeypatch: pytest.MonkeyPatch, password: str, expected_encoded: str
    ) -> None:
        """Special characters in POSTGRES_PASSWORD are URL-encoded in database_url_sync."""
        self._clear_db_env(monkeypatch)
        monkeypatch.setenv("POSTGRES_USER", "user")
        monkeypatch.setenv("POSTGRES_PASSWORD", password)

        db = DatabaseSettings(_env_file=None)

        assert f"user:{expected_encoded}@" in db.database_url_sync
        assert db.database_url_sync.startswith("postgresql://")

    def test_user_with_special_chars_url_encoded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Special characters in POSTGRES_USER are URL-encoded."""
        self._clear_db_env(monkeypatch)
        monkeypatch.setenv("POSTGRES_USER", "user@domain")
        monkeypatch.setenv("POSTGRES_PASSWORD", "pass")

        db = DatabaseSettings(_env_file=None)

        expected_user = quote_plus("user@domain")
        assert f"{expected_user}:pass@" in db.database_url
        assert f"{expected_user}:pass@" in db.database_url_sync

    def test_both_user_and_password_with_special_chars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both POSTGRES_USER and POSTGRES_PASSWORD with special chars are encoded."""
        self._clear_db_env(monkeypatch)
        monkeypatch.setenv("POSTGRES_USER", "u@er")
        monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss/word")

        db = DatabaseSettings(_env_file=None)

        expected_user = quote_plus("u@er")
        expected_pass = quote_plus("p@ss/word")
        assert f"{expected_user}:{expected_pass}@" in db.database_url
        assert f"{expected_user}:{expected_pass}@" in db.database_url_sync

    def test_plain_credentials_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Credentials without special characters produce the same URL as before."""
        self._clear_db_env(monkeypatch)
        monkeypatch.setenv("POSTGRES_USER", "admin")
        monkeypatch.setenv("POSTGRES_PASSWORD", "secret123")

        db = DatabaseSettings(_env_file=None)

        assert "admin:secret123@" in db.database_url
        assert "admin:secret123@" in db.database_url_sync


class TestMultiHostDatabaseURL:
    """Test multi-host DATABASE_URL support for native PostgreSQL HA."""

    def test_multihost_converted_for_asyncpg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multi-host libpq URL is converted to SQLAlchemy query-param format for asyncpg."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@h1:5432,h2:5433/db")

        db = DatabaseSettings(_env_file=None)

        assert db.database_url == "postgresql+asyncpg://user:pass@/db?host=h1,h2&port=5432,5433"

    def test_multihost_preserved_for_psycopg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multi-host URL is kept in libpq format for psycopg (database_url_sync)."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@h1:5432,h2:5433/db")

        db = DatabaseSettings(_env_file=None)

        assert db.database_url_sync == "postgresql://user:pass@h1:5432,h2:5433/db"

    def test_multihost_preserves_query_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multi-host conversion plays nicely with the libpq → asyncpg
        translation layer: hosts/ports preserved verbatim, ``sslmode`` rewritten
        to ``ssl``, and libpq-only kwargs (``target_session_attrs``) stripped
        because asyncpg rejects them."""
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql://user:pass@h1:5432,h2:5432/db?target_session_attrs=read-write&sslmode=require",
        )

        db = DatabaseSettings(_env_file=None)

        url = db.database_url
        assert url.startswith("postgresql+asyncpg://user:pass@/db?")
        assert "host=h1,h2" in url
        assert "port=5432,5432" in url
        assert "ssl=true" in url
        # libpq-only kwargs that asyncpg would reject are stripped.
        assert "target_session_attrs" not in url
        assert "sslmode" not in url
        # Sync (psycopg) URL keeps the libpq spelling — psycopg accepts it.
        assert "target_session_attrs=read-write" in db.database_url_sync
        assert "sslmode=require" in db.database_url_sync

    @pytest.mark.parametrize(
        ("env_url", "expected_hosts", "expected_ports"),
        [
            pytest.param(
                "postgresql://user:pass@h1,h2/db",
                "host=h1,h2",
                "port=5432,5432",
                id="defaults_port_when_omitted",
            ),
            pytest.param(
                "postgresql://user:pass@h1:5432,h2/db",
                "host=h1,h2",
                "port=5432,5432",
                id="mixed_ports",
            ),
            pytest.param(
                "postgresql://u:p@h1:5432,h2:5432,h3:5432/db",
                "host=h1,h2,h3",
                "port=5432,5432,5432",
                id="three_hosts",
            ),
            pytest.param(
                "postgresql://u:p@h1:,h2:5433/db",
                "host=h1,h2",
                "port=5432,5433",
                id="trailing_colon_defaults_port",
            ),
            pytest.param(
                "postgresql://u:p@[::1]:5432,[::2]:5433/db",
                "host=[::1],[::2]",
                "port=5432,5433",
                id="ipv6_hosts",
            ),
            pytest.param(
                "postgresql://u:p@[::1],[::2]/db",
                "host=[::1],[::2]",
                "port=5432,5432",
                id="ipv6_without_port",
            ),
        ],
    )
    def test_multihost_host_port_matrix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_url: str,
        expected_hosts: str,
        expected_ports: str,
    ) -> None:
        """Multi-host URLs are parsed into correct host/port query params."""
        monkeypatch.setenv("DATABASE_URL", env_url)

        db = DatabaseSettings(_env_file=None)

        assert expected_hosts in db.database_url
        assert expected_ports in db.database_url

    def test_malformed_ipv6_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Malformed bracketed IPv6 (missing closing bracket) raises at startup."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@[::1:5432,[::2]:5433/db")

        db = DatabaseSettings(_env_file=None)

        with pytest.raises(ValueError, match="missing closing bracket"):
            _ = db.database_url

    def test_non_integer_port_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-integer port in multi-host URL raises at startup."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h1:abc,h2:5433/db")

        db = DatabaseSettings(_env_file=None)

        with pytest.raises(ValueError, match="Non-integer port"):
            _ = db.database_url

    def test_single_host_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Single-host URL is not rewritten."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@h1:5432/db")

        db = DatabaseSettings(_env_file=None)

        assert db.database_url == "postgresql+asyncpg://user:pass@h1:5432/db"

    def test_multihost_no_userinfo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multi-host URL without userinfo is converted correctly."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://h1:5432,h2:5432/db")

        db = DatabaseSettings(_env_file=None)

        assert db.database_url == "postgresql+asyncpg:///db?host=h1,h2&port=5432,5432"

    def test_multihost_rewritten_url_parses_with_sqlalchemy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Asyncpg multi-host URL must parse via SQLAlchemy (alembic env path).

        ``target_session_attrs`` is libpq-only and stripped from the async URL
        (asyncpg would reject it as an unknown kwarg). The sync URL retains it.
        """
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql://user:pass@h1:5432,h2:5432/db?target_session_attrs=read-write",
        )

        db = DatabaseSettings(_env_file=None)
        url = make_url(db.database_url)

        assert url.drivername == "postgresql+asyncpg"
        assert url.database == "db"
        # Multi-host details live in query params, not in url.host.
        assert url.query["host"] == "h1,h2"
        assert url.query["port"] == "5432,5432"
        assert "target_session_attrs" not in url.query

    def test_database_url_sync_preserves_libpq_multihost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """database_url_sync must preserve libpq comma-host syntax (psycopg consumers)."""
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql://user:pass@h1:5432,h2:5432/db?target_session_attrs=read-write",
        )

        db = DatabaseSettings(_env_file=None)

        # Raw libpq form: comma-separated hosts in the authority, no
        # ``host=`` query param rewrite, no async driver suffix.
        assert "h1:5432,h2:5432" in db.database_url_sync
        assert "host=" not in db.database_url_sync
        assert db.database_url_sync.startswith("postgresql://")


class TestWorkerSettingsLeaseValidation:
    """Test that lease timing invariants are enforced at startup."""

    def _clear_worker_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "LEASE_DURATION_SECONDS",
            "HEARTBEAT_INTERVAL_SECONDS",
            "REAPER_INTERVAL_SECONDS",
            "WORKER_COUNT",
            "N_JOBS_PER_WORKER",
            "WORKER_QUEUE_KEY",
            "WORKER_DRAIN_TIMEOUT",
            "BG_JOB_TIMEOUT_SECS",
            "BG_JOB_MAX_RETRIES",
            "POSTGRES_POLL_INTERVAL_SECONDS",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_defaults_pass_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_worker_env(monkeypatch)
        ws = WorkerSettings(_env_file=None)
        assert ws.LEASE_DURATION_SECONDS == 30
        assert ws.HEARTBEAT_INTERVAL_SECONDS == 10

    def test_rejects_lease_equal_to_two_heartbeats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_worker_env(monkeypatch)
        monkeypatch.setenv("LEASE_DURATION_SECONDS", "20")
        monkeypatch.setenv("HEARTBEAT_INTERVAL_SECONDS", "10")
        with pytest.raises(ValueError, match="LEASE_DURATION_SECONDS.*must be greater"):
            WorkerSettings(_env_file=None)

    def test_rejects_lease_less_than_two_heartbeats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_worker_env(monkeypatch)
        monkeypatch.setenv("LEASE_DURATION_SECONDS", "10")
        monkeypatch.setenv("HEARTBEAT_INTERVAL_SECONDS", "10")
        with pytest.raises(ValueError, match="LEASE_DURATION_SECONDS.*must be greater"):
            WorkerSettings(_env_file=None)

    def test_accepts_lease_greater_than_two_heartbeats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_worker_env(monkeypatch)
        monkeypatch.setenv("LEASE_DURATION_SECONDS", "21")
        monkeypatch.setenv("HEARTBEAT_INTERVAL_SECONDS", "10")
        ws = WorkerSettings(_env_file=None)
        assert ws.LEASE_DURATION_SECONDS == 21


class TestCronSettingsValidation:
    """Test that cron scheduler timing is validated at startup."""

    def _clear_cron_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("CRON_ENABLED", "CRON_POLL_INTERVAL_SECONDS"):
            monkeypatch.delenv(var, raising=False)

    def test_defaults_pass_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_cron_env(monkeypatch)
        cron = CronSettings(_env_file=None)
        assert cron.CRON_POLL_INTERVAL_SECONDS == 60

    def test_rejects_non_positive_poll_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_cron_env(monkeypatch)
        monkeypatch.setenv("CRON_POLL_INTERVAL_SECONDS", "0")

        with pytest.raises((ValueError, ValidationError), match="CRON_POLL_INTERVAL_SECONDS"):
            CronSettings(_env_file=None)
