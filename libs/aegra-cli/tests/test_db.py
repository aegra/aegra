"""Tests for ``aegra db`` commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from aegra_cli.cli import cli


def test_upgrade_calls_run_migrations(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Operator upgrade must use the helper that goes through locked env.py."""
    with (
        cli_runner.isolated_filesystem(temp_dir=tmp_path),
        patch("aegra_api.core.migrations.run_migrations") as mock_run,
    ):
        result = cli_runner.invoke(cli, ["db", "upgrade"])

    assert result.exit_code == 0
    mock_run.assert_called_once_with()
