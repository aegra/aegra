"""Tests for main CLI commands (version, dev, up, down)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from aegra_cli.cli import (
    cli,
    ensure_docker_files,
    find_config_file,
    get_project_slug,
)
from aegra_cli.env import load_env_file

if TYPE_CHECKING:
    pass


def create_mock_popen(returncode: int = 0):
    """Create a mock Popen object that behaves like a completed process."""
    mock_process = MagicMock()
    mock_process.poll.return_value = returncode  # Process has finished
    mock_process.wait.return_value = returncode
    mock_process.returncode = returncode
    return mock_process


class TestVersion:
    """Tests for the version command."""

    def test_version_shows_aegra_cli_version(self, cli_runner: CliRunner) -> None:
        """Test that version command shows aegra-cli version."""
        result = cli_runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert "aegra-cli" in result.output

    def test_version_shows_aegra_api_version(self, cli_runner: CliRunner) -> None:
        """Test that version command shows aegra-api version (or not installed)."""
        result = cli_runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert "aegra-api" in result.output

    def test_version_flag(self, cli_runner: CliRunner) -> None:
        """Test that --version flag works on main CLI."""
        result = cli_runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "aegra-cli" in result.output


class TestDevCommand:
    """Tests for the dev command.

    Note: Most tests use --no-db-check to skip automatic PostgreSQL/Docker checks,
    allowing us to test the uvicorn command building in isolation.
    All tests create an aegra.json file since dev command requires a config.
    """

    def test_dev_builds_correct_command(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that dev command builds the correct uvicorn command."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with patch("aegra_cli.cli.subprocess.Popen") as mock_popen:
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(cli, ["dev", "--no-db-check"])

                # Verify subprocess.Popen was called
                mock_popen.assert_called_once()
                call_args = mock_popen.call_args[0][0]

                # Check command structure
                assert call_args[0] == sys.executable
                assert "-m" in call_args
                assert "uvicorn" in call_args
                assert "aegra_api.main:app" in call_args
                assert "--reload" in call_args

    def test_dev_default_host_and_port(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that dev command uses default host and port."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with patch("aegra_cli.cli.subprocess.Popen") as mock_popen:
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(cli, ["dev", "--no-db-check"])

                call_args = mock_popen.call_args[0][0]
                assert "--host" in call_args
                host_idx = call_args.index("--host")
                assert call_args[host_idx + 1] == "127.0.0.1"

                assert "--port" in call_args
                port_idx = call_args.index("--port")
                assert call_args[port_idx + 1] == "2026"

    def test_dev_custom_host(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that dev command accepts custom host."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with patch("aegra_cli.cli.subprocess.Popen") as mock_popen:
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(cli, ["dev", "--no-db-check", "--host", "0.0.0.0"])

                call_args = mock_popen.call_args[0][0]
                host_idx = call_args.index("--host")
                assert call_args[host_idx + 1] == "0.0.0.0"

    def test_dev_custom_port(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that dev command accepts custom port."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with patch("aegra_cli.cli.subprocess.Popen") as mock_popen:
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(cli, ["dev", "--no-db-check", "--port", "3000"])

                call_args = mock_popen.call_args[0][0]
                port_idx = call_args.index("--port")
                assert call_args[port_idx + 1] == "3000"

    def test_dev_port_from_env_var(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that dev command picks up PORT from env var when no CLI flag is passed."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')
            Path(".env").write_text("PORT=9090\n")

            with (
                patch.dict(os.environ, {}, clear=False),
                patch("aegra_cli.cli.subprocess.Popen") as mock_popen,
            ):
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(cli, ["dev", "--no-db-check"])
                assert result.exit_code == 0, result.output

                call_args = mock_popen.call_args[0][0]
                port_idx = call_args.index("--port")
                assert call_args[port_idx + 1] == "9090"

    def test_dev_cli_flag_overrides_env_var(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that explicit --port flag takes precedence over PORT env var."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')
            Path(".env").write_text("PORT=9090\n")

            with (
                patch.dict(os.environ, {}, clear=False),
                patch("aegra_cli.cli.subprocess.Popen") as mock_popen,
            ):
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(cli, ["dev", "--no-db-check", "--port", "7777"])
                assert result.exit_code == 0, result.output

                call_args = mock_popen.call_args[0][0]
                port_idx = call_args.index("--port")
                assert call_args[port_idx + 1] == "7777"

    def test_dev_host_from_env_var(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that dev command picks up HOST from env var when no CLI flag is passed."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')
            Path(".env").write_text("HOST=0.0.0.0\n")

            with (
                patch.dict(os.environ, {}, clear=False),
                patch("aegra_cli.cli.subprocess.Popen") as mock_popen,
            ):
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(cli, ["dev", "--no-db-check"])
                assert result.exit_code == 0, result.output

                call_args = mock_popen.call_args[0][0]
                host_idx = call_args.index("--host")
                assert call_args[host_idx + 1] == "0.0.0.0"

    def test_dev_custom_app(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that dev command accepts custom app path."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with patch("aegra_cli.cli.subprocess.Popen") as mock_popen:
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(cli, ["dev", "--no-db-check", "--app", "myapp.main:app"])

                call_args = mock_popen.call_args[0][0]
                assert "myapp.main:app" in call_args

    def test_dev_uvicorn_not_installed(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test error handling when uvicorn is not installed."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with patch("aegra_cli.cli.subprocess.Popen") as mock_popen:
                mock_popen.side_effect = FileNotFoundError("uvicorn not found")
                result = cli_runner.invoke(cli, ["dev", "--no-db-check"])

                assert result.exit_code == 1
                assert "uvicorn is not installed" in result.output

    def test_dev_keyboard_interrupt(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test handling of keyboard interrupt during dev server."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with patch("aegra_cli.cli.subprocess.Popen") as mock_popen:
                mock_process = create_mock_popen(0)
                mock_process.wait.side_effect = KeyboardInterrupt()
                mock_popen.return_value = mock_process
                result = cli_runner.invoke(cli, ["dev", "--no-db-check"])

                assert result.exit_code == 0
                assert "Server stopped by user" in result.output

    def test_dev_shows_server_info(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that dev command shows server info in output."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with patch("aegra_cli.cli.subprocess.Popen") as mock_popen:
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(
                    cli, ["dev", "--no-db-check", "--host", "0.0.0.0", "--port", "9000"]
                )

                assert "0.0.0.0" in result.output
                assert "9000" in result.output

    def test_dev_with_env_file(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that dev command loads .env file when specified."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')
            # Create a test .env file
            Path(".env").write_text("POSTGRES_USER=testuser\nPOSTGRES_DB=testdb\n")

            with patch("aegra_cli.cli.subprocess.Popen") as mock_popen:
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(cli, ["dev", "--no-db-check", "--env-file", ".env"])

                assert result.exit_code == 0
                assert "Loaded environment from" in result.output
                # Check that .env is mentioned (path may be wrapped by Rich)
                assert ".env" in result.output

    def test_dev_with_env_file_short_flag(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that dev command accepts -e short flag for env file."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')
            Path(".env").write_text("TEST_VAR=value\n")

            with patch("aegra_cli.cli.subprocess.Popen") as mock_popen:
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(cli, ["dev", "--no-db-check", "-e", ".env"])

                assert result.exit_code == 0
                assert "Loaded environment from" in result.output

    def test_dev_with_debug_port_wraps_debugpy(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that providing --debug-port wraps uvicorn with debugpy listen."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with (
                patch("importlib.util.find_spec", return_value=object()),
                patch("aegra_cli.cli.subprocess.Popen") as mock_popen,
            ):
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(cli, ["dev", "--no-db-check", "--debug-port", "5678"])

                assert result.exit_code == 0
                mock_popen.assert_called_once()
                call_args = mock_popen.call_args[0][0]
                # Ensure debugpy is used and listening on given port bound to loopback
                assert "debugpy" in call_args
                assert "--listen" in call_args
                listen_idx = call_args.index("--listen")
                assert call_args[listen_idx + 1] == "5678"
                # Ensure uvicorn still present
                assert "uvicorn" in call_args
                assert "--reload" in call_args

    def test_dev_with_ipv6_debug_host_formats_listen(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that providing an IPv6 debug host (::1) is accepted and bracketed."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with (
                patch("importlib.util.find_spec", return_value=object()),
                patch("aegra_cli.cli.subprocess.Popen") as mock_popen,
            ):
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(
                    cli, ["dev", "--no-db-check", "--debug-port", "5678", "--debug-host", "::1"]
                )

                assert result.exit_code == 0
                mock_popen.assert_called_once()
                call_args = mock_popen.call_args[0][0]
                # Ensure debugpy listen uses bracketed IPv6 literal
                assert "--listen" in call_args
                joined = " ".join(map(str, call_args))
                assert "[::1]:5678" in joined

    def test_dev_with_localhost_debug_host_formats_listen(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that providing a localhost debug host is accepted and formatted correctly."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with (
                patch("importlib.util.find_spec", return_value=object()),
                patch("aegra_cli.cli.subprocess.Popen") as mock_popen,
            ):
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(
                    cli,
                    ["dev", "--no-db-check", "--debug-port", "5678", "--debug-host", "localhost"],
                )

                assert result.exit_code == 0
                mock_popen.assert_called_once()
                call_args = mock_popen.call_args[0][0]
                # Ensure debugpy listen uses bracketed IPv6 literal
                assert "--listen" in call_args
                joined = " ".join(map(str, call_args))
                assert "localhost:5678" in joined

    def test_dev_with_invalid_debug_host_warns_and_falls_back_to_loopback(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that an invalid debug host prints a warning and keeps loopback binding."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with (
                patch("importlib.util.find_spec", return_value=object()),
                patch("aegra_cli.cli.subprocess.Popen") as mock_popen,
            ):
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(
                    cli,
                    [
                        "dev",
                        "--no-db-check",
                        "--debug-port",
                        "5678",
                        "--debug-host",
                        "not-a-valid-host",
                    ],
                )

                assert result.exit_code == 0
                assert "Invalid debug host specified" in result.output
                mock_popen.assert_called_once()
                call_args = mock_popen.call_args[0][0]
                listen_idx = call_args.index("--listen")
                assert call_args[listen_idx + 1] == "5678"
                assert "not-a-valid-host:5678" not in " ".join(map(str, call_args))

    def test_dev_debug_host_non_loopback_warns(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that using a non-loopback --debug-host prints a warning panel."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with (
                patch("importlib.util.find_spec", return_value=object()),
                patch("aegra_cli.cli.subprocess.Popen") as mock_popen,
            ):
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(
                    cli,
                    [
                        "dev",
                        "--no-db-check",
                        "--debug-port",
                        "5678",
                        "--debug-host",
                        "0.0.0.0",
                    ],
                )

                assert result.exit_code == 0
                assert "Debug Exposure Warning" in result.output

    def test_dev_with_wait_for_client_adds_flag(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that --wait-for-client adds the debugpy wait flag."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with (
                patch("importlib.util.find_spec", return_value=object()),
                patch("aegra_cli.cli.subprocess.Popen") as mock_popen,
            ):
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(
                    cli, ["dev", "--no-db-check", "--debug-port", "5678", "--wait-for-client"]
                )

                assert result.exit_code == 0
                call_args = mock_popen.call_args[0][0]
                assert "--wait-for-client" in call_args

    def test_dev_debugpy_not_installed_shows_error(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that missing debugpy yields an instructive error message."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with patch("importlib.util.find_spec", return_value=None):
                result = cli_runner.invoke(cli, ["dev", "--no-db-check", "--debug-port", "5678"])

                assert result.exit_code == 1
                assert "debugpy is not installed" in result.output

    def test_dev_wait_for_client_without_debug_port_errors(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that using --wait-for-client without --debug-port errors fast with UsageError."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            result = cli_runner.invoke(cli, ["dev", "--no-db-check", "--wait-for-client"])

            assert result.exit_code != 0
            assert "--wait-for-client requires --debug-port" in result.output

    def test_dev_debug_host_without_debug_port_errors(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that using --debug-host without --debug-port errors fast with UsageError."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            result = cli_runner.invoke(cli, ["dev", "--no-db-check", "--debug-host", "0.0.0.0"])

            assert result.exit_code != 0
            assert "--debug-host requires --debug-port" in result.output


class TestUpCommand:
    """Tests for the up command."""

    def test_up_builds_correct_command(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that up command builds the correct docker compose command."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                result = cli_runner.invoke(cli, ["up"])

                mock_run.assert_called_once()
                call_args = mock_run.call_args[0][0]

                assert call_args[0] == "docker"
                assert call_args[1] == "compose"
                assert "up" in call_args
                assert "-d" in call_args

    def test_up_includes_build_by_default(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that up command includes --build by default."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                result = cli_runner.invoke(cli, ["up"])

                call_args = mock_run.call_args[0][0]
                assert "--build" in call_args

    def test_up_with_specific_services(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that up command passes specific services."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                result = cli_runner.invoke(cli, ["up", "postgres", "redis"])

                call_args = mock_run.call_args[0][0]
                assert "postgres" in call_args
                assert "redis" in call_args

    def test_up_with_compose_file(self, cli_runner: CliRunner, mock_compose_file: Path) -> None:
        """Test that up command accepts custom compose file."""
        with patch("aegra_cli.cli.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = cli_runner.invoke(cli, ["up", "-f", str(mock_compose_file)])

            call_args = mock_run.call_args[0][0]
            assert "-f" in call_args
            file_idx = call_args.index("-f")
            assert call_args[file_idx + 1] == str(mock_compose_file)

    def test_up_success_message(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that up command shows success message."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                result = cli_runner.invoke(cli, ["up"])

                assert "Services started successfully" in result.output

    def test_up_failure_shows_error(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that up command shows error on failure."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 1
                result = cli_runner.invoke(cli, ["up"])

                assert result.exit_code == 1
                assert "Error" in result.output

    def test_up_docker_not_installed(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test error handling when docker is not installed."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("docker not found")
                result = cli_runner.invoke(cli, ["up"])

                assert result.exit_code == 1
                assert "docker is not installed" in result.output

    def test_up_shows_running_command(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that up command shows the command being run."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                result = cli_runner.invoke(cli, ["up"])

                assert "Running:" in result.output
                assert "docker compose" in result.output


class TestDownCommand:
    """Tests for the down command."""

    def test_down_with_compose_file(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that down command uses docker-compose.yml."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("docker-compose.yml").write_text("services: {}")

            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                result = cli_runner.invoke(cli, ["down"])

                mock_run.assert_called_once()
                call_args = mock_run.call_args[0][0]

                assert call_args[0] == "docker"
                assert call_args[1] == "compose"
                assert "down" in call_args
                assert "docker-compose.yml" in " ".join(call_args)

    def test_down_with_volumes_flag(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that down command includes -v when --volumes is specified."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("docker-compose.yml").write_text("services: {}")

            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                result = cli_runner.invoke(cli, ["down", "--volumes"])

                call_args = mock_run.call_args[0][0]
                assert "-v" in call_args

    def test_down_with_v_short_flag(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that down command accepts -v short flag."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("docker-compose.yml").write_text("services: {}")

            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                result = cli_runner.invoke(cli, ["down", "-v"])

                call_args = mock_run.call_args[0][0]
                assert "-v" in call_args

    def test_down_volumes_shows_warning(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that down --volumes shows a warning about data loss."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("docker-compose.yml").write_text("services: {}")

            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                result = cli_runner.invoke(cli, ["down", "-v"])

                assert "Warning" in result.output
                assert "data will be lost" in result.output

    def test_down_with_custom_compose_file(
        self, cli_runner: CliRunner, mock_compose_file: Path
    ) -> None:
        """Test that down command accepts custom compose file."""
        with patch("aegra_cli.cli.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = cli_runner.invoke(cli, ["down", "-f", str(mock_compose_file)])

            call_args = mock_run.call_args[0][0]
            assert "-f" in call_args
            file_idx = call_args.index("-f")
            assert call_args[file_idx + 1] == str(mock_compose_file)

    def test_down_success_message(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that down command shows success message."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("docker-compose.yml").write_text("services: {}")

            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                result = cli_runner.invoke(cli, ["down"])

                assert "Services stopped successfully" in result.output

    def test_down_failure_shows_error(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that down command shows error on failure."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("docker-compose.yml").write_text("services: {}")

            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 1
                result = cli_runner.invoke(cli, ["down"])

                assert result.exit_code == 1
                assert "failed to stop" in result.output

    def test_down_docker_not_installed(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test error handling when docker is not installed."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("docker-compose.yml").write_text("services: {}")

            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("docker not found")
                result = cli_runner.invoke(cli, ["down"])

                assert result.exit_code == 1
                assert "docker is not installed" in result.output

    def test_down_shows_running_command(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that down command shows the command being run."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("docker-compose.yml").write_text("services: {}")

            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                result = cli_runner.invoke(cli, ["down"])

                assert "Running:" in result.output
                assert "docker compose" in result.output


class TestCLIHelp:
    """Tests for CLI help messages."""

    def test_main_help(self, cli_runner: CliRunner) -> None:
        """Test that main CLI shows help."""
        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Aegra CLI" in result.output

    def test_version_help(self, cli_runner: CliRunner) -> None:
        """Test that version command shows help."""
        result = cli_runner.invoke(cli, ["version", "--help"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_dev_help(self, cli_runner: CliRunner) -> None:
        """Test that dev command shows help."""
        result = cli_runner.invoke(cli, ["dev", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--app" in result.output
        assert "--env-file" in result.output
        assert "-e" in result.output

    def test_up_help(self, cli_runner: CliRunner) -> None:
        """Test that up command shows help."""
        result = cli_runner.invoke(cli, ["up", "--help"])
        assert result.exit_code == 0
        assert "--file" in result.output
        assert "--build" in result.output
        # --dev flag should not be present
        assert "--dev" not in result.output

    def test_down_help(self, cli_runner: CliRunner) -> None:
        """Test that down command shows help."""
        result = cli_runner.invoke(cli, ["down", "--help"])
        assert result.exit_code == 0
        assert "--file" in result.output
        assert "--volumes" in result.output
        # --all flag should not be present
        assert "--all" not in result.output


class TestCLIEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_command(self, cli_runner: CliRunner) -> None:
        """Test that unknown commands show an error."""
        result = cli_runner.invoke(cli, ["unknown-command"])
        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_dev_invalid_port(self, cli_runner: CliRunner) -> None:
        """Test that dev command rejects invalid port."""
        result = cli_runner.invoke(cli, ["dev", "--port", "not-a-number"])
        assert result.exit_code != 0

    def test_up_nonexistent_compose_file(self, cli_runner: CliRunner) -> None:
        """Test that up command handles nonexistent compose file."""
        result = cli_runner.invoke(cli, ["up", "-f", "/nonexistent/docker-compose.yml"])
        assert result.exit_code != 0

    def test_down_nonexistent_compose_file(self, cli_runner: CliRunner) -> None:
        """Test that down command handles nonexistent compose file."""
        result = cli_runner.invoke(cli, ["down", "-f", "/nonexistent/docker-compose.yml"])
        assert result.exit_code != 0


class TestConfigDiscovery:
    """Tests for config file auto-discovery."""

    def test_find_config_file_aegra_json(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that find_config_file finds aegra.json in current directory."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            result = find_config_file()
            assert result is not None
            assert result.name == "aegra.json"

    def test_find_config_file_langgraph_json(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that find_config_file finds langgraph.json as fallback."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("langgraph.json").write_text('{"graphs": {}}')

            result = find_config_file()
            assert result is not None
            assert result.name == "langgraph.json"

    def test_find_config_file_prefers_aegra(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that find_config_file prefers aegra.json over langgraph.json."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')
            Path("langgraph.json").write_text('{"graphs": {}}')

            result = find_config_file()
            assert result is not None
            assert result.name == "aegra.json"

    def test_find_config_file_not_found(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that find_config_file returns None when no config found."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = find_config_file()
            assert result is None

    def test_dev_fails_without_config(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that dev command fails when no config file is found."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = cli_runner.invoke(cli, ["dev", "--no-db-check"])

            assert result.exit_code == 1
            assert "Could not find aegra.json" in result.output

    def test_dev_uses_discovered_config(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that dev command uses auto-discovered config."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            # Create aegra.json
            Path("aegra.json").write_text('{"graphs": {}}')

            with patch("aegra_cli.cli.subprocess.Popen") as mock_popen:
                mock_popen.return_value = create_mock_popen(0)
                result = cli_runner.invoke(cli, ["dev", "--no-db-check"])

                assert result.exit_code == 0
                assert "Using config:" in result.output
                # Check for path parts (Rich may wrap long paths across lines)
                assert "aegra" in result.output and ".json" in result.output

    def test_dev_uses_specified_config(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that dev command uses config specified with -c flag."""
        # Create config file outside the isolated filesystem
        config_file = tmp_path / "custom" / "aegra.json"
        config_file.parent.mkdir(parents=True)
        config_file.write_text('{"graphs": {}}')

        with patch("aegra_cli.cli.subprocess.Popen") as mock_popen:
            mock_popen.return_value = create_mock_popen(0)
            result = cli_runner.invoke(cli, ["dev", "--no-db-check", "-c", str(config_file)])

            assert result.exit_code == 0
            assert "Using config:" in result.output
            assert "aegra.json" in result.output

    def test_dev_sets_aegra_config_env_var(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that dev command sets AEGRA_CONFIG environment variable."""
        import os

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            # Clear any existing AEGRA_CONFIG
            original = os.environ.pop("AEGRA_CONFIG", None)
            try:
                with patch("aegra_cli.cli.subprocess.Popen") as mock_popen:
                    mock_popen.return_value = create_mock_popen(0)
                    result = cli_runner.invoke(cli, ["dev", "--no-db-check"])

                    # The env var should have been set before Popen was called
                    assert "AEGRA_CONFIG" in os.environ
                    assert os.environ["AEGRA_CONFIG"].endswith("aegra.json")
            finally:
                # Restore original
                if original:
                    os.environ["AEGRA_CONFIG"] = original
                else:
                    os.environ.pop("AEGRA_CONFIG", None)

    def test_dev_help_shows_config_option(self, cli_runner: CliRunner) -> None:
        """Test that dev --help shows the config option."""
        result = cli_runner.invoke(cli, ["dev", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "-c" in result.output


class TestGetProjectSlug:
    """Tests for the get_project_slug helper function."""

    def test_get_project_slug_from_config(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that get_project_slug reads name from config file."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            config_path = Path("aegra.json")
            config_path.write_text('{"name": "My Test Project"}')
            slug = get_project_slug(config_path)
            assert slug == "my_test_project"

    def test_get_project_slug_fallback_to_dirname(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that get_project_slug falls back to directory name."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            slug = get_project_slug(None)
            # Falls back to current directory name
            assert slug is not None
            assert len(slug) > 0

    def test_get_project_slug_invalid_json(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that get_project_slug handles invalid JSON gracefully."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            config_path = Path("aegra.json")
            config_path.write_text("not valid json")
            slug = get_project_slug(config_path)
            # Should fall back to directory name
            assert slug is not None

    def test_get_project_slug_no_name_in_config(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that get_project_slug handles config without name field."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            config_path = Path("aegra.json")
            config_path.write_text('{"graphs": {}}')
            slug = get_project_slug(config_path)
            # Should fall back to directory name
            assert slug is not None


class TestEnsureDockerFiles:
    """Tests for the ensure_docker_files function."""

    def test_ensure_docker_files_creates_compose_and_dockerfile(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that ensure_docker_files creates both files."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            project_path = Path.cwd()
            result = ensure_docker_files(project_path, "myapp")
            assert result.exists()
            assert result.name == "docker-compose.yml"

            content = result.read_text()
            assert "postgres" in content
            assert "myapp-postgres" in content
            assert "myapp:" in content
            assert "build:" in content

            dockerfile = project_path / "Dockerfile"
            assert dockerfile.exists()
            assert "aegra" in dockerfile.read_text()

    def test_ensure_docker_files_skips_existing(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that ensure_docker_files doesn't overwrite existing files."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            project_path = Path.cwd()
            compose_path = project_path / "docker-compose.yml"
            compose_path.write_text("existing compose")
            dockerfile_path = project_path / "Dockerfile"
            dockerfile_path.write_text("existing dockerfile")

            ensure_docker_files(project_path, "myapp")

            assert compose_path.read_text() == "existing compose"
            assert dockerfile_path.read_text() == "existing dockerfile"

    def test_ensure_docker_files_creates_missing_dockerfile(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that ensure_docker_files creates Dockerfile even if compose exists."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            project_path = Path.cwd()
            compose_path = project_path / "docker-compose.yml"
            compose_path.write_text("existing compose")

            ensure_docker_files(project_path, "myapp")

            assert compose_path.read_text() == "existing compose"
            dockerfile = project_path / "Dockerfile"
            assert dockerfile.exists()
            assert "aegra" in dockerfile.read_text()


class TestUpCommandExtended:
    """Extended tests for the up command."""

    def test_up_with_no_build_flag(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that up --no-build skips building."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                result = cli_runner.invoke(cli, ["up", "--no-build"])

                assert result.exit_code == 0
                cmd = mock_run.call_args[0][0]
                assert "--build" not in cmd

    def test_up_auto_generates_docker_files(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that up auto-generates Docker files."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                result = cli_runner.invoke(cli, ["up"])

                assert result.exit_code == 0
                # Files should be created
                assert Path("docker-compose.yml").exists()
                assert Path("Dockerfile").exists()

    def test_up_nonexistent_custom_file(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that up fails with nonexistent custom compose file."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = cli_runner.invoke(cli, ["up", "-f", "nonexistent.yml"])
            assert result.exit_code == 1
            assert "not found" in result.output


class TestDownCommandExtended:
    """Extended tests for the down command."""

    def test_down_uses_docker_compose_yml(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that down uses docker-compose.yml."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("docker-compose.yml").write_text("services: {}")

            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                result = cli_runner.invoke(cli, ["down"])

                assert result.exit_code == 0
                assert mock_run.call_count == 1
                cmd = mock_run.call_args[0][0]
                assert "docker-compose.yml" in " ".join(cmd)

    def test_down_no_compose_files(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that down handles missing compose files gracefully."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = cli_runner.invoke(cli, ["down"])
            assert result.exit_code == 0
            assert "No docker-compose.yml found" in result.output


class TestServeCommand:
    """Tests for the serve command."""

    def test_serve_help(self, cli_runner: CliRunner) -> None:
        """Test that serve --help shows all options."""
        result = cli_runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--config" in result.output

    def test_serve_fails_without_config(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that serve fails when no config file is found."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = cli_runner.invoke(cli, ["serve"])
            assert result.exit_code == 1
            assert "Could not find aegra.json" in result.output

    def test_serve_runs_uvicorn(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that serve runs uvicorn with correct arguments."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                result = cli_runner.invoke(cli, ["serve"])

                assert result.exit_code == 0
                mock_run.assert_called_once()
                cmd = mock_run.call_args[0][0]
                assert "uvicorn" in cmd
                assert "--host" in cmd
                assert "0.0.0.0" in cmd
                assert "--port" in cmd
                assert "2026" in cmd
                # No --reload flag (production mode)
                assert "--reload" not in cmd

    def test_serve_port_from_env_var(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that serve command picks up PORT from env var when no CLI flag is passed."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')
            Path(".env").write_text("PORT=9090\n")

            with (
                patch.dict(os.environ, {}, clear=False),
                patch("aegra_cli.cli.subprocess.run") as mock_run,
            ):
                mock_run.return_value = MagicMock(returncode=0)
                result = cli_runner.invoke(cli, ["serve"])
                assert result.exit_code == 0, result.output

                cmd = mock_run.call_args[0][0]
                port_idx = cmd.index("--port")
                assert cmd[port_idx + 1] == "9090"

    def test_serve_cli_flag_overrides_env_var(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that explicit --port flag takes precedence over PORT env var."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')
            Path(".env").write_text("PORT=9090\n")

            with (
                patch.dict(os.environ, {}, clear=False),
                patch("aegra_cli.cli.subprocess.run") as mock_run,
            ):
                mock_run.return_value = MagicMock(returncode=0)
                result = cli_runner.invoke(cli, ["serve", "--port", "7777"])
                assert result.exit_code == 0, result.output

                cmd = mock_run.call_args[0][0]
                port_idx = cmd.index("--port")
                assert cmd[port_idx + 1] == "7777"

    def test_serve_host_from_env_var(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test that serve command picks up HOST from env var when no CLI flag is passed."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')
            Path(".env").write_text("HOST=192.168.1.5\n")

            with (
                patch.dict(os.environ, {}, clear=False),
                patch("aegra_cli.cli.subprocess.run") as mock_run,
            ):
                mock_run.return_value = MagicMock(returncode=0)
                result = cli_runner.invoke(cli, ["serve"])
                assert result.exit_code == 0, result.output

                cmd = mock_run.call_args[0][0]
                host_idx = cmd.index("--host")
                assert cmd[host_idx + 1] == "192.168.1.5"

    def test_serve_uvicorn_not_installed(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test error handling when uvicorn is not installed."""
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with patch("aegra_cli.cli.subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("uvicorn not found")
                result = cli_runner.invoke(cli, ["serve"])

                assert result.exit_code == 1
                assert "uvicorn is not installed" in result.output


class TestLoadEnvFile:
    """Tests for the .env file parser."""

    def test_loads_simple_key_value(self, tmp_path: Path) -> None:
        """Test basic KEY=value parsing."""
        env_file = tmp_path / ".env"
        env_file.write_text("MY_VAR=hello\n")
        with patch.dict("os.environ", {}, clear=True):
            load_env_file(env_file)

            assert os.environ["MY_VAR"] == "hello"

    def test_strips_inline_comments_from_unquoted_values(self, tmp_path: Path) -> None:
        """Test that inline comments are stripped from unquoted values."""
        env_file = tmp_path / ".env"
        env_file.write_text("ENV_MODE=LOCAL # this is a comment\n")
        with patch.dict("os.environ", {}, clear=True):
            load_env_file(env_file)

            assert os.environ["ENV_MODE"] == "LOCAL"

    def test_preserves_hash_in_quoted_values(self, tmp_path: Path) -> None:
        """Test that # inside quoted values is preserved."""
        env_file = tmp_path / ".env"
        env_file.write_text('PASSWORD="my#secret"\n')
        with patch.dict("os.environ", {}, clear=True):
            load_env_file(env_file)

            assert os.environ["PASSWORD"] == "my#secret"

    def test_preserves_hash_in_single_quoted_values(self, tmp_path: Path) -> None:
        """Test that # inside single-quoted values is preserved."""
        env_file = tmp_path / ".env"
        env_file.write_text("PASSWORD='my#secret'\n")
        with patch.dict("os.environ", {}, clear=True):
            load_env_file(env_file)

            assert os.environ["PASSWORD"] == "my#secret"

    def test_skips_comment_lines(self, tmp_path: Path) -> None:
        """Test that full-line comments are skipped."""
        env_file = tmp_path / ".env"
        env_file.write_text("# this is a comment\nKEY=value\n")
        with patch.dict("os.environ", {}, clear=True):
            load_env_file(env_file)

            assert os.environ.get("KEY") == "value"
            assert "# this is a comment" not in os.environ

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        """Test that empty lines are skipped."""
        env_file = tmp_path / ".env"
        env_file.write_text("\n\nKEY=value\n\n")
        with patch.dict("os.environ", {}, clear=True):
            load_env_file(env_file)

            assert os.environ["KEY"] == "value"

    def test_does_not_override_existing_env_vars(self, tmp_path: Path) -> None:
        """Test that existing env vars are not overwritten."""
        env_file = tmp_path / ".env"
        env_file.write_text("MY_VAR=from_file\n")
        with patch.dict("os.environ", {"MY_VAR": "from_env"}, clear=True):
            load_env_file(env_file)

            assert os.environ["MY_VAR"] == "from_env"

    def test_returns_none_for_nonexistent_file(self, tmp_path: Path) -> None:
        """Test that None is returned when .env file doesn't exist."""
        result = load_env_file(tmp_path / "nonexistent.env")
        assert result is None

    def test_returns_path_on_success(self, tmp_path: Path) -> None:
        """Test that the file path is returned on success."""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=value\n")
        with patch.dict("os.environ", {}, clear=True):
            result = load_env_file(env_file)
            assert result == env_file

    def test_inline_comment_with_multiple_hashes(self, tmp_path: Path) -> None:
        """Test stripping inline comment when value itself has no quotes."""
        env_file = tmp_path / ".env"
        env_file.write_text("AUTH_TYPE=noop  # noop, custom\n")
        with patch.dict("os.environ", {}, clear=True):
            load_env_file(env_file)

            assert os.environ["AUTH_TYPE"] == "noop"
