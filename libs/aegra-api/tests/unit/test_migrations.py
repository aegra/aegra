"""Tests for aegra_api.core.migrations module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aegra_api.core import migrations as migrations_mod
from aegra_api.core.migrations import find_alembic_ini, get_alembic_config


class TestFindAlembicIni:
    """Tests for find_alembic_ini() resolution order."""

    def test_finds_alembic_ini_in_package(self, tmp_path, monkeypatch):
        """Should find alembic.ini bundled with the aegra_api package."""
        # CWD has no alembic.ini
        monkeypatch.chdir(tmp_path)

        # The package directory should have alembic.ini (from force-include)
        # We verify it finds the file relative to migrations.py's parent.parent
        result = find_alembic_ini()
        # Should be in the package directory (aegra_api/)
        assert result.name == "alembic.ini"
        assert result.exists()

    def test_ignores_foreign_cwd_ini(self, tmp_path, monkeypatch):
        """A host project's own alembic.ini in CWD must never shadow ours.

        Resolving CWD first matched a foreign alembic.ini and skipped our
        migrations, crashing a fresh DB with relation "assistant" does not
        exist (GH #306). Discovery must be CWD-independent.
        """
        decoy = tmp_path / "alembic.ini"
        decoy.write_text("[alembic]\nscript_location = not-ours\n")
        monkeypatch.chdir(tmp_path)

        result = find_alembic_ini()

        assert result.resolve() != decoy.resolve()
        assert result.parent != tmp_path
        assert result.exists()

    def test_discovery_is_independent_of_cwd(self, tmp_path, monkeypatch):
        """The resolved ini is identical regardless of process CWD."""
        baseline = find_alembic_ini()

        monkeypatch.chdir(tmp_path)
        assert find_alembic_ini() == baseline

    def test_raises_when_not_found(self, tmp_path, monkeypatch):
        """Should raise FileNotFoundError when alembic.ini is nowhere."""
        monkeypatch.chdir(tmp_path)

        # Create a fake module location where no alembic.ini exists
        # at any resolution level (CWD, package dir, or dev layout)
        fake_core = tmp_path / "fake" / "pkg" / "core"
        fake_core.mkdir(parents=True)
        (fake_core / "migrations.py").touch()

        monkeypatch.setattr(
            migrations_mod,
            "__file__",
            str(fake_core / "migrations.py"),
        )

        with pytest.raises(FileNotFoundError, match="Could not find alembic.ini"):
            find_alembic_ini()


class TestGetAlembicConfig:
    """Tests for get_alembic_config()."""

    def test_returns_config_object(self, tmp_path, monkeypatch):
        """Should return a valid Alembic Config object."""
        ini_file = tmp_path / "alembic.ini"
        ini_file.write_text("[alembic]\nscript_location = alembic\n")
        alembic_dir = tmp_path / "alembic"
        alembic_dir.mkdir()
        monkeypatch.setattr(migrations_mod, "find_alembic_ini", lambda: ini_file)

        cfg = get_alembic_config()
        assert cfg is not None

        # script_location should be resolved to absolute path
        script_loc = cfg.get_main_option("script_location")
        assert script_loc is not None
        assert Path(script_loc).is_absolute()
        assert script_loc == str(alembic_dir.resolve())

    def test_resolves_relative_script_location(self, tmp_path, monkeypatch):
        """Should resolve relative script_location to absolute path."""
        ini_file = tmp_path / "alembic.ini"
        ini_file.write_text("[alembic]\nscript_location = alembic\n")
        (tmp_path / "alembic").mkdir()
        monkeypatch.setattr(migrations_mod, "find_alembic_ini", lambda: ini_file)

        cfg = get_alembic_config()
        script_loc = cfg.get_main_option("script_location")
        assert script_loc is not None
        assert Path(script_loc).is_absolute()

    def test_preserves_absolute_script_location(self, tmp_path, monkeypatch):
        """Should not modify an already-absolute script_location."""
        abs_path = str(tmp_path / "my_alembic")
        ini_file = tmp_path / "alembic.ini"
        ini_file.write_text(f"[alembic]\nscript_location = {abs_path}\n")
        monkeypatch.setattr(migrations_mod, "find_alembic_ini", lambda: ini_file)

        cfg = get_alembic_config()
        assert cfg.get_main_option("script_location") == abs_path


class TestRunMigrations:
    """Tests for run_migrations() and run_migrations_async()."""

    def test_run_migrations_calls_alembic_upgrade(self, tmp_path, monkeypatch):
        """Should call alembic command.upgrade with 'head'."""
        monkeypatch.chdir(tmp_path)

        ini_file = tmp_path / "alembic.ini"
        ini_file.write_text("[alembic]\nscript_location = alembic\n")
        alembic_dir = tmp_path / "alembic"
        alembic_dir.mkdir()

        with patch("aegra_api.core.migrations.command") as mock_command:
            from aegra_api.core.migrations import run_migrations

            run_migrations()
            mock_command.upgrade.assert_called_once()
            args = mock_command.upgrade.call_args
            assert args[0][1] == "head"  # Second positional arg is "head"

    @pytest.mark.asyncio
    async def test_run_migrations_async_dispatches_to_lockfree_path(self):
        """run_migrations_async should hand off to the lock-free helper."""
        with patch("aegra_api.core.migrations.asyncio.to_thread") as mock_to_thread:
            from aegra_api.core.migrations import (
                run_migrations_async,
                run_migrations_if_needed,
            )

            await run_migrations_async()
            mock_to_thread.assert_called_once_with(run_migrations_if_needed)


class TestRunMigrationsIfNeeded:
    """Tests for the lock-free fast-path used at FastAPI startup."""

    def _setup_alembic_ini(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "alembic.ini").write_text("[alembic]\nscript_location = alembic\n")
        (tmp_path / "alembic").mkdir()

    def test_skips_upgrade_when_database_at_head(self, tmp_path, monkeypatch):
        """When current revision matches head, no upgrade is invoked."""
        self._setup_alembic_ini(tmp_path, monkeypatch)

        with (
            patch("aegra_api.core.migrations._is_database_up_to_date", return_value=True) as mock_check,
            patch("aegra_api.core.migrations.command") as mock_command,
        ):
            from aegra_api.core.migrations import run_migrations_if_needed

            run_migrations_if_needed()

            mock_check.assert_called_once()
            mock_command.upgrade.assert_not_called()

    def test_runs_upgrade_when_database_behind_head(self, tmp_path, monkeypatch):
        """When the precheck reports drift, upgrade is invoked."""
        self._setup_alembic_ini(tmp_path, monkeypatch)

        with (
            patch("aegra_api.core.migrations._is_database_up_to_date", return_value=False),
            patch("aegra_api.core.migrations.command") as mock_command,
        ):
            from aegra_api.core.migrations import run_migrations_if_needed

            run_migrations_if_needed()

            mock_command.upgrade.assert_called_once()
            assert mock_command.upgrade.call_args[0][1] == "head"

    def test_falls_back_to_upgrade_when_precheck_fails(self, tmp_path, monkeypatch):
        """If the precheck raises (e.g. alembic_version missing on first install)
        the upgrade still runs so a fresh database can bootstrap.
        """
        self._setup_alembic_ini(tmp_path, monkeypatch)

        with (
            patch(
                "aegra_api.core.migrations._is_database_up_to_date",
                side_effect=RuntimeError("alembic_version table missing"),
            ),
            patch("aegra_api.core.migrations.command") as mock_command,
        ):
            from aegra_api.core.migrations import run_migrations_if_needed

            run_migrations_if_needed()

            mock_command.upgrade.assert_called_once()


class TestIsDatabaseUpToDate:
    """Tests for the lock-free revision precheck helper."""

    def _patch_psycopg(self, current_revision: str | None, *, raises: Exception | None = None):
        """Build a context-managed psycopg.connect mock whose cursor returns
        a single-row alembic_version result of ``current_revision`` (or None
        when the table is empty). Pass ``raises`` to simulate UndefinedTable.
        """
        cursor = MagicMock()
        if raises is not None:
            cursor.execute.side_effect = raises
        else:
            cursor.fetchone.return_value = (current_revision,) if current_revision is not None else None

        cursor_cm = MagicMock()
        cursor_cm.__enter__.return_value = cursor
        cursor_cm.__exit__.return_value = False

        connection = MagicMock()
        connection.cursor.return_value = cursor_cm

        connect_cm = MagicMock()
        connect_cm.__enter__.return_value = connection
        connect_cm.__exit__.return_value = False
        return connection, connect_cm

    def test_returns_true_when_revisions_match(self):
        from aegra_api.core.migrations import _is_database_up_to_date

        cfg = MagicMock()
        _, connect_cm = self._patch_psycopg("abc123")
        script = MagicMock()
        script.get_current_head.return_value = "abc123"

        with (
            patch("aegra_api.core.migrations.psycopg.connect", return_value=connect_cm),
            patch("aegra_api.core.migrations.ScriptDirectory.from_config", return_value=script),
        ):
            assert _is_database_up_to_date(cfg) is True

    def test_returns_false_when_revisions_differ(self):
        from aegra_api.core.migrations import _is_database_up_to_date

        cfg = MagicMock()
        _, connect_cm = self._patch_psycopg("abc123")
        script = MagicMock()
        script.get_current_head.return_value = "def456"

        with (
            patch("aegra_api.core.migrations.psycopg.connect", return_value=connect_cm),
            patch("aegra_api.core.migrations.ScriptDirectory.from_config", return_value=script),
        ):
            assert _is_database_up_to_date(cfg) is False

    def test_returns_false_when_no_current_revision(self):
        """A fresh database with no alembic_version row needs the upgrade path."""
        from aegra_api.core.migrations import _is_database_up_to_date

        cfg = MagicMock()
        _, connect_cm = self._patch_psycopg(None)
        script = MagicMock()
        script.get_current_head.return_value = "abc123"

        with (
            patch("aegra_api.core.migrations.psycopg.connect", return_value=connect_cm),
            patch("aegra_api.core.migrations.ScriptDirectory.from_config", return_value=script),
        ):
            assert _is_database_up_to_date(cfg) is False

    def test_returns_false_when_alembic_version_table_missing(self):
        """Fresh install: alembic_version doesn't exist yet. Treat as 'not up to date'
        so the upgrade path runs and bootstraps the schema.
        """
        import psycopg

        from aegra_api.core.migrations import _is_database_up_to_date

        cfg = MagicMock()
        _, connect_cm = self._patch_psycopg(None, raises=psycopg.errors.UndefinedTable("alembic_version"))
        script = MagicMock()
        script.get_current_head.return_value = "abc123"

        with (
            patch("aegra_api.core.migrations.psycopg.connect", return_value=connect_cm),
            patch("aegra_api.core.migrations.ScriptDirectory.from_config", return_value=script),
        ):
            assert _is_database_up_to_date(cfg) is False

    def test_returns_true_when_script_directory_empty(self):
        """No revisions at all means nothing to apply; skip the upgrade path."""
        from aegra_api.core.migrations import _is_database_up_to_date

        cfg = MagicMock()
        script = MagicMock()
        script.get_current_head.return_value = None

        with (
            patch("aegra_api.core.migrations.ScriptDirectory.from_config", return_value=script),
            patch("aegra_api.core.migrations.psycopg.connect") as mock_connect,
        ):
            assert _is_database_up_to_date(cfg) is True
            # Should short-circuit before opening any DB connection.
            mock_connect.assert_not_called()

    def test_uses_libpq_url_for_multihost_compatibility(self):
        """Precheck hands libpq URL straight to psycopg (multi-host PR #299)."""
        from aegra_api.core.migrations import _is_database_up_to_date

        cfg = MagicMock()
        _, connect_cm = self._patch_psycopg("abc123")
        script = MagicMock()
        script.get_current_head.return_value = "abc123"

        with (
            patch("aegra_api.core.migrations.psycopg.connect", return_value=connect_cm) as mock_connect,
            patch("aegra_api.core.migrations.ScriptDirectory.from_config", return_value=script),
        ):
            _is_database_up_to_date(cfg)

        # Single positional arg, equal to settings.db.database_url_sync, no
        # SQLAlchemy URL rewrite layered on top.
        assert mock_connect.call_count == 1
        passed_url = mock_connect.call_args.args[0]
        from aegra_api.settings import settings

        assert passed_url == settings.db.database_url_sync
