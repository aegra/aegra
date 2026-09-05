"""Alembic migration helpers.

Resolves the bundled alembic.ini from the installed package. Two entry points:
- ``run_migrations()``: unconditional upgrade. Online ``env.py`` holds a
  session ``pg_advisory_lock`` so concurrent upgrades serialize. Alembic itself
  does not take this lock. For ``aegra db upgrade``.
- ``run_migrations_if_needed()``: lock-free precheck, skips upgrade when
  already at head. FastAPI startup uses this to avoid multi-pod lock contention.
"""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import LiteralString

import psycopg
import structlog
from alembic.config import Config
from alembic.script import ScriptDirectory

from aegra_api.settings import settings
from alembic import command

logger = structlog.get_logger(__name__)

# Session-level pair (not pg_advisory_xact_lock): GIN rebuilds use
# autocommit_block() / CREATE INDEX CONCURRENTLY, which would drop an xact lock.
_AEGRA_MIGRATION_LOCK_KEY1: int = 0xAE6A
_AEGRA_MIGRATION_LOCK_KEY2: int = 1
_ADVISORY_LOCK_SQL: LiteralString = "SELECT pg_advisory_lock(%s, %s)"
_ADVISORY_UNLOCK_SQL: LiteralString = "SELECT pg_advisory_unlock(%s, %s)"


def find_alembic_ini() -> Path:
    """Find the bundled alembic.ini, never a foreign one in CWD.

    Resolution order:
    1. Bundled with aegra_api package (pip install)
    2. Development layout (repo/editable install)

    Resolving CWD first would match a host project's own alembic.ini and
    silently skip our migrations, so a fresh DB crashes with relation
    "assistant" does not exist (GH #306). Both branches resolve relative to
    this module, so CWD is irrelevant — including Docker, where the package
    branch wins regardless of workdir.

    Returns:
        Absolute path to alembic.ini

    Raises:
        FileNotFoundError: If alembic.ini cannot be found
    """
    # 1. Package bundled (pip install aegra-api)
    # In installed package: site-packages/aegra_api/alembic.ini
    package_dir = Path(__file__).resolve().parent.parent  # aegra_api/
    package_ini = package_dir / "alembic.ini"
    if package_ini.exists():
        return package_ini

    # 2. Development layout (src layout: libs/aegra-api/src/aegra_api/ → libs/aegra-api/)
    dev_root = package_dir.parent.parent  # Up from src/aegra_api/ to libs/aegra-api/
    dev_ini = dev_root / "alembic.ini"
    if dev_ini.exists():
        return dev_ini

    raise FileNotFoundError(
        "Could not find alembic.ini. Ensure aegra-api is properly installed or run from the project root."
    )


def get_alembic_config() -> Config:
    """Create Alembic Config with correct paths.

    Works in both development (repo) and production (pip install) environments.
    Resolves relative script_location to absolute path so migrations work
    regardless of CWD.

    Returns:
        Configured Alembic Config object
    """
    ini_path = find_alembic_ini()
    cfg = Config(str(ini_path))

    # Resolve script_location to absolute path so it works from any CWD
    script_location = cfg.get_main_option("script_location")
    if script_location and not Path(script_location).is_absolute():
        abs_script_location = str((ini_path.parent / script_location).resolve())
        cfg.set_main_option("script_location", abs_script_location)

    return cfg


def _is_database_up_to_date(cfg: Config) -> bool:
    """Lock-free check: True iff DB revision matches script head."""
    script = ScriptDirectory.from_config(cfg)
    head = script.get_current_head()

    # Empty script directory: nothing to apply.
    if head is None:
        return True

    # Read alembic_version directly via psycopg. MigrationContext.configure
    # requires a SQLAlchemy Connection (accesses conn.dialect), and SQLAlchemy's
    # URL parser breaks on libpq comma-host syntax — so we bypass both,
    # preserving multi-host failover from PR #299.
    with psycopg.connect(settings.db.database_url_sync) as conn, conn.cursor() as cur:
        try:
            cur.execute("SELECT version_num FROM alembic_version LIMIT 1")
            row = cur.fetchone()
            current = row[0] if row else None
        except psycopg.errors.UndefinedTable:
            conn.rollback()
            current = None

    return current == head


@contextmanager
def migration_advisory_lock() -> Iterator[None]:
    """Hold a session ``pg_advisory_lock`` on a dedicated psycopg connection.

    Uses ``database_url_sync`` so SQLAlchemy never parses libpq comma-hosts.
    """
    lock_keys = (_AEGRA_MIGRATION_LOCK_KEY1, _AEGRA_MIGRATION_LOCK_KEY2)
    conn = psycopg.connect(settings.db.database_url_sync, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute(_ADVISORY_LOCK_SQL, lock_keys)
            cur.fetchone()
        body_error: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            try:
                with conn.cursor() as cur:
                    cur.execute(_ADVISORY_UNLOCK_SQL, lock_keys)
                    cur.fetchone()
            except Exception:
                if body_error is None:
                    raise
                logger.warning("failed to release migration advisory lock after upgrade error")
    finally:
        # Best-effort: a close error must not replace an upgrade or unlock error.
        try:
            conn.close()
        except Exception:
            logger.warning("failed to close migration lock connection")


def run_migrations() -> None:
    """Unconditional upgrade to head. Online env.py takes the session advisory lock."""
    cfg = get_alembic_config()
    logger.info("running database migrations")
    command.upgrade(cfg, "head")
    logger.info("database migrations completed")


def run_migrations_if_needed() -> None:
    """Skip upgrade when already at head; otherwise fall through to upgrade.

    Precheck failure (e.g. fresh install with no alembic_version yet) also
    falls through so bootstrap works.
    """
    cfg = get_alembic_config()
    try:
        if _is_database_up_to_date(cfg):
            logger.debug("database already at migration head; skipping upgrade")
            return
    except Exception as exc:
        logger.debug("revision precheck failed; falling back to full upgrade", error=str(exc))

    logger.info("running database migrations")
    command.upgrade(cfg, "head")
    logger.info("database migrations completed")


async def run_migrations_async() -> None:
    """Async wrapper over the lock-free fast path. Alembic's env.py owns
    its own event loop, so we hand off to a thread."""
    await asyncio.to_thread(run_migrations_if_needed)
