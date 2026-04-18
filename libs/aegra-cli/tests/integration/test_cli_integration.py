"""Integration tests for the CLI entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from aegra_cli.cli import cli


def _create_completed_process(returncode: int = 0) -> MagicMock:
    """Create a mocked child process that exits successfully."""
    process = MagicMock()
    process.poll.return_value = returncode
    process.wait.return_value = returncode
    process.returncode = returncode
    return process


@pytest.mark.integration
def test_dev_no_reload_skips_reload_flag(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invoking the CLI from a real project directory omits the reload flag."""
    project_dir = tmp_path / "integration-project"
    project_dir.mkdir()
    (project_dir / "aegra.json").write_text('{"graphs": {}}', encoding="utf-8")
    monkeypatch.chdir(project_dir)

    with patch("aegra_cli.cli.subprocess.Popen") as mock_popen:
        mock_popen.return_value = _create_completed_process(0)
        result = cli_runner.invoke(cli, ["dev", "--no-db-check", "--no-reload"])

    assert result.exit_code == 0, result.output
    mock_popen.assert_called_once()

    call_args = mock_popen.call_args.args[0]
    assert call_args[0] == sys.executable
    assert "-m" in call_args
    assert "uvicorn" in call_args
    assert "aegra_api.main:app" in call_args
    assert "--reload" not in call_args
