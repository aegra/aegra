"""Tests for Docker utility functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aegra_cli.utils.docker import (
    find_compose_file,
    get_compose_command,
    get_container_command,
    get_docker_start_instructions,
    is_container_runtime_installed,
    is_docker_installed,
    is_docker_running,
    is_postgres_container_running,
)


class TestGetComposeCommand:
    """Tests for get_compose_command function."""

    def test_returns_docker_compose_when_plugin_and_daemon_available(self) -> None:
        """Test that docker compose v2 plugin is preferred when daemon is running."""
        with patch("aegra_cli.utils.docker.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = get_compose_command()
            assert result == ["docker", "compose"]
            assert mock_run.call_count == 2
            mock_run.assert_any_call(
                ["docker", "compose", "version"], capture_output=True, timeout=10
            )
            mock_run.assert_any_call(["docker", "info"], capture_output=True, timeout=10)

    def test_falls_back_when_docker_daemon_not_running(self) -> None:
        """Test fallback to podman-compose when Docker CLI exists but daemon is down."""

        with (
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
            patch("aegra_cli.utils.docker.shutil.which") as mock_which,
        ):

            def run_side_effect(cmd, **kwargs):
                mock_result = MagicMock()
                if cmd == ["docker", "compose", "version"]:
                    mock_result.returncode = 0
                elif cmd == ["docker", "info"]:
                    mock_result.returncode = 1
                elif cmd == ["podman", "info"]:
                    mock_result.returncode = 0
                else:
                    mock_result.returncode = 1
                return mock_result

            mock_run.side_effect = run_side_effect
            mock_which.side_effect = lambda cmd: (
                "/usr/bin/podman-compose" if cmd == "podman-compose" else None
            )
            result = get_compose_command()
            assert result == ["podman-compose"]

    def test_returns_podman_compose_when_docker_plugin_unavailable(self) -> None:
        """Test fallback to podman-compose when docker compose plugin fails."""
        with (
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
            patch("aegra_cli.utils.docker.shutil.which") as mock_which,
        ):

            def run_side_effect(cmd, **kwargs):
                mock_result = MagicMock()
                if cmd == ["podman", "info"]:
                    mock_result.returncode = 0
                else:
                    mock_result.returncode = 1
                return mock_result

            mock_run.side_effect = run_side_effect
            mock_which.side_effect = lambda cmd: (
                "/usr/bin/podman-compose" if cmd == "podman-compose" else None
            )
            result = get_compose_command()
            assert result == ["podman-compose"]

    def test_skips_podman_compose_when_podman_runtime_unavailable(self) -> None:
        """Test that podman-compose is skipped when Podman runtime isn't available."""
        with (
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
            patch("aegra_cli.utils.docker.shutil.which") as mock_which,
        ):

            def run_side_effect(cmd, **kwargs):
                mock_result = MagicMock()
                mock_result.returncode = 1
                return mock_result

            mock_run.side_effect = run_side_effect
            mock_which.side_effect = lambda cmd: {
                "podman-compose": "/usr/bin/podman-compose",
                "docker-compose": "/usr/bin/docker-compose",
            }.get(cmd)
            result = get_compose_command()
            assert result == ["docker-compose"]

    def test_returns_docker_compose_v1_as_last_fallback(self) -> None:
        """Test fallback to standalone docker-compose v1."""
        with (
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
            patch("aegra_cli.utils.docker.shutil.which") as mock_which,
        ):
            mock_run.return_value.returncode = 1
            mock_which.side_effect = lambda cmd: (
                "/usr/bin/docker-compose" if cmd == "docker-compose" else None
            )
            result = get_compose_command()
            assert result == ["docker-compose"]

    def test_raises_when_no_compose_tool_found(self) -> None:
        """Test that FileNotFoundError is raised when no compose tool exists."""
        with (
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
            patch("aegra_cli.utils.docker.shutil.which") as mock_which,
        ):
            mock_run.side_effect = FileNotFoundError
            mock_which.return_value = None
            with pytest.raises(FileNotFoundError, match="No container compose tool found"):
                get_compose_command()

    def test_handles_docker_compose_timeout(self) -> None:
        """Test fallback when docker compose version times out."""
        import subprocess as sp

        with (
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
            patch("aegra_cli.utils.docker.shutil.which") as mock_which,
        ):

            def run_side_effect(cmd, **kwargs):
                if cmd == ["podman", "info"]:
                    mock_result = MagicMock()
                    mock_result.returncode = 0
                    return mock_result
                raise sp.TimeoutExpired("docker", 10)

            mock_run.side_effect = run_side_effect
            mock_which.side_effect = lambda cmd: (
                "/usr/bin/podman-compose" if cmd == "podman-compose" else None
            )
            result = get_compose_command()
            assert result == ["podman-compose"]


class TestGetContainerCommand:
    """Tests for get_container_command function."""

    def test_returns_docker_when_available(self) -> None:
        """Test that docker is preferred when both are available."""
        with patch("aegra_cli.utils.docker.shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: (
                f"/usr/bin/{cmd}" if cmd in ("docker", "podman") else None
            )
            assert get_container_command() == "docker"

    def test_returns_podman_when_docker_unavailable(self) -> None:
        """Test fallback to podman when docker is not installed."""
        with patch("aegra_cli.utils.docker.shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: "/usr/bin/podman" if cmd == "podman" else None
            assert get_container_command() == "podman"

    def test_raises_when_neither_available(self) -> None:
        """Test that FileNotFoundError is raised when no runtime exists."""
        with patch("aegra_cli.utils.docker.shutil.which") as mock_which:
            mock_which.return_value = None
            with pytest.raises(FileNotFoundError, match="No container runtime found"):
                get_container_command()


class TestIsContainerRuntimeInstalled:
    """Tests for is_container_runtime_installed function."""

    def test_returns_true_when_docker_available(self) -> None:
        """Test returns True when docker is in PATH."""
        with patch("aegra_cli.utils.docker.shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: "/usr/bin/docker" if cmd == "docker" else None
            assert is_container_runtime_installed() is True

    def test_returns_true_when_podman_available(self) -> None:
        """Test returns True when only podman is in PATH."""
        with patch("aegra_cli.utils.docker.shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: "/usr/bin/podman" if cmd == "podman" else None
            assert is_container_runtime_installed() is True

    def test_returns_false_when_neither_available(self) -> None:
        """Test returns False when no container runtime is installed."""
        with patch("aegra_cli.utils.docker.shutil.which") as mock_which:
            mock_which.return_value = None
            assert is_container_runtime_installed() is False


class TestIsDockerInstalled:
    """Tests for is_docker_installed function."""

    def test_returns_true_when_docker_found(self) -> None:
        """Test that function returns True when docker is in PATH."""
        with patch("aegra_cli.utils.docker.shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/docker"
            assert is_docker_installed() is True
            mock_which.assert_called_once_with("docker")

    def test_returns_false_when_docker_not_found(self) -> None:
        """Test that function returns False when docker is not in PATH."""
        with patch("aegra_cli.utils.docker.shutil.which") as mock_which:
            mock_which.return_value = None
            assert is_docker_installed() is False


class TestIsDockerRunning:
    """Tests for is_docker_running function."""

    def test_returns_true_when_docker_daemon_responds(self) -> None:
        """Test that function returns True when docker info succeeds."""
        with (
            patch("aegra_cli.utils.docker.shutil.which") as mock_which,
            patch("aegra_cli.utils.docker.is_docker_installed") as mock_installed,
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
        ):
            mock_which.return_value = None  # no podman
            mock_installed.return_value = True
            mock_run.return_value.returncode = 0

            assert is_docker_running() is True
            mock_run.assert_called_once_with(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )

    def test_returns_true_when_podman_available(self) -> None:
        """Test that function returns True when podman info succeeds."""
        with (
            patch("aegra_cli.utils.docker.shutil.which") as mock_which,
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
        ):
            mock_which.return_value = "/usr/bin/podman"
            mock_run.return_value.returncode = 0
            assert is_docker_running() is True
            mock_run.assert_called_once_with(
                ["podman", "info"],
                capture_output=True,
                timeout=10,
            )

    def test_returns_false_when_podman_not_ready(self) -> None:
        """Test that function returns False when podman info fails (e.g., VM not running)."""
        with (
            patch("aegra_cli.utils.docker.shutil.which") as mock_which,
            patch("aegra_cli.utils.docker.is_docker_installed") as mock_installed,
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
        ):
            mock_which.return_value = "/usr/bin/podman"
            mock_installed.return_value = False
            mock_run.return_value.returncode = 1
            assert is_docker_running() is False

    def test_returns_false_when_docker_not_installed(self) -> None:
        """Test that function returns False when neither docker nor podman is installed."""
        with (
            patch("aegra_cli.utils.docker.shutil.which") as mock_which,
            patch("aegra_cli.utils.docker.is_docker_installed") as mock_installed,
        ):
            mock_which.return_value = None  # no podman
            mock_installed.return_value = False
            assert is_docker_running() is False

    def test_returns_false_when_docker_daemon_not_running(self) -> None:
        """Test that function returns False when docker info fails."""
        with (
            patch("aegra_cli.utils.docker.shutil.which") as mock_which,
            patch("aegra_cli.utils.docker.is_docker_installed") as mock_installed,
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
        ):
            mock_which.return_value = None  # no podman
            mock_installed.return_value = True
            mock_run.return_value.returncode = 1

            assert is_docker_running() is False

    def test_returns_false_on_timeout(self) -> None:
        """Test that function returns False when docker info times out."""
        import subprocess

        with (
            patch("aegra_cli.utils.docker.shutil.which") as mock_which,
            patch("aegra_cli.utils.docker.is_docker_installed") as mock_installed,
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
        ):
            mock_which.return_value = None  # no podman
            mock_installed.return_value = True
            mock_run.side_effect = subprocess.TimeoutExpired("docker", 10)

            assert is_docker_running() is False


class TestGetDockerStartInstructions:
    """Tests for get_docker_start_instructions function."""

    def test_returns_macos_instructions_on_darwin(self) -> None:
        """Test that function returns macOS instructions on Darwin."""
        with patch("aegra_cli.utils.docker.platform.system") as mock_system:
            mock_system.return_value = "Darwin"
            instructions = get_docker_start_instructions()

            assert "macOS" in instructions
            assert "open -a Docker" in instructions

    def test_returns_linux_instructions_on_linux(self) -> None:
        """Test that function returns Linux instructions on Linux."""
        with patch("aegra_cli.utils.docker.platform.system") as mock_system:
            mock_system.return_value = "Linux"
            instructions = get_docker_start_instructions()

            assert "Linux" in instructions
            assert "systemctl start docker" in instructions

    def test_returns_windows_instructions_on_windows(self) -> None:
        """Test that function returns Windows instructions on Windows."""
        with patch("aegra_cli.utils.docker.platform.system") as mock_system:
            mock_system.return_value = "Windows"
            instructions = get_docker_start_instructions()

            assert "Windows" in instructions
            assert "Docker Desktop" in instructions


class TestIsPostgresContainerRunning:
    """Tests for is_postgres_container_running function."""

    def test_returns_true_when_postgres_in_running_services(self) -> None:
        """Test that function returns True when postgres is running."""
        with (
            patch("aegra_cli.utils.docker.get_compose_command") as mock_compose,
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
        ):
            mock_compose.return_value = ["docker", "compose"]
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "postgres\nredis\n"

            assert is_postgres_container_running() is True

    def test_returns_false_when_postgres_not_running(self) -> None:
        """Test that function returns False when postgres is not in running services."""
        with (
            patch("aegra_cli.utils.docker.get_compose_command") as mock_compose,
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
        ):
            mock_compose.return_value = ["docker", "compose"]
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "redis\n"

            assert is_postgres_container_running() is False

    def test_returns_false_when_no_services_running(self) -> None:
        """Test that function returns False when no services are running."""
        with (
            patch("aegra_cli.utils.docker.get_compose_command") as mock_compose,
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
        ):
            mock_compose.return_value = ["docker", "compose"]
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ""

            assert is_postgres_container_running() is False

    def test_returns_false_when_no_compose_tool_found(self) -> None:
        """Test that function returns False when no compose tool is available."""
        with patch("aegra_cli.utils.docker.get_compose_command") as mock_compose:
            mock_compose.side_effect = FileNotFoundError
            assert is_postgres_container_running() is False

    def test_returns_false_on_compose_ps_failure(self) -> None:
        """Test that function returns False when compose ps command fails."""
        with (
            patch("aegra_cli.utils.docker.get_compose_command") as mock_compose,
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
        ):
            mock_compose.return_value = ["docker", "compose"]
            mock_run.return_value.returncode = 1

            assert is_postgres_container_running() is False

    def test_uses_compose_file_when_provided(self) -> None:
        """Test that function uses compose file when provided."""
        with (
            patch("aegra_cli.utils.docker.get_compose_command") as mock_compose,
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
        ):
            mock_compose.return_value = ["docker", "compose"]
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "postgres\n"

            compose_file = Path("/path/to/docker-compose.yml")
            result = is_postgres_container_running(compose_file)

            call_args = mock_run.call_args[0][0]
            assert "-f" in call_args
            assert str(compose_file) in call_args

    def test_returns_true_with_podman_compose_when_postgres_running(self) -> None:
        """Test that podman-compose derives runtime from compose_cmd and uses labels."""
        with (
            patch("aegra_cli.utils.docker.get_compose_command") as mock_compose,
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
            patch("aegra_cli.utils.docker.Path.cwd") as mock_cwd,
        ):
            mock_compose.return_value = ["podman-compose"]
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "abc123\n"
            mock_cwd.return_value = Path("/home/user/myproject")

            assert is_postgres_container_running() is True

            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "podman"
            cmd_str = " ".join(call_args)
            assert "label=com.docker.compose.service=postgres" in cmd_str
            assert "label=com.docker.compose.project=myproject" in cmd_str

    def test_returns_false_with_podman_compose_when_no_postgres(self) -> None:
        """Test that podman-compose path returns False when no matching container."""
        with (
            patch("aegra_cli.utils.docker.get_compose_command") as mock_compose,
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
            patch("aegra_cli.utils.docker.Path.cwd") as mock_cwd,
        ):
            mock_compose.return_value = ["podman-compose"]
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ""
            mock_cwd.return_value = Path("/home/user/myproject")

            assert is_postgres_container_running() is False

    def test_runtime_derived_from_compose_cmd_not_container_command(self) -> None:
        """Regression: when both runtimes are installed but Docker daemon is down,
        runtime must come from compose_cmd, not get_container_command()."""
        with (
            patch("aegra_cli.utils.docker.get_compose_command") as mock_compose,
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
            patch("aegra_cli.utils.docker.Path.cwd") as mock_cwd,
        ):
            mock_compose.return_value = ["podman-compose"]
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "abc123\n"
            mock_cwd.return_value = Path("/home/user/aegra")

            assert is_postgres_container_running() is True

            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "podman"

    def test_docker_compose_v1_derives_docker_runtime(self) -> None:
        """Test that docker-compose v1 derives 'docker' as the runtime."""
        with (
            patch("aegra_cli.utils.docker.get_compose_command") as mock_compose,
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
            patch("aegra_cli.utils.docker.Path.cwd") as mock_cwd,
        ):
            mock_compose.return_value = ["docker-compose"]
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "abc123\n"
            mock_cwd.return_value = Path("/home/user/aegra")

            assert is_postgres_container_running() is True

            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "docker"

    def test_project_scoped_from_compose_file_parent(self) -> None:
        """Test that project label uses compose file's parent directory name."""
        with (
            patch("aegra_cli.utils.docker.get_compose_command") as mock_compose,
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
        ):
            mock_compose.return_value = ["podman-compose"]
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "abc123\n"

            compose_file = Path("/opt/apps/MyApp/docker-compose.yml")
            assert is_postgres_container_running(compose_file) is True

            cmd_str = " ".join(mock_run.call_args[0][0])
            assert "label=com.docker.compose.project=myapp" in cmd_str

    def test_project_from_compose_project_name_env(self) -> None:
        """Test that COMPOSE_PROJECT_NAME env var takes precedence over directory name."""
        with (
            patch("aegra_cli.utils.docker.get_compose_command") as mock_compose,
            patch("aegra_cli.utils.docker.subprocess.run") as mock_run,
            patch("aegra_cli.utils.docker.Path.cwd") as mock_cwd,
            patch.dict("os.environ", {"COMPOSE_PROJECT_NAME": "custom-project"}),
        ):
            mock_compose.return_value = ["podman-compose"]
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "abc123\n"
            mock_cwd.return_value = Path("/home/user/myproject")

            assert is_postgres_container_running() is True

            cmd_str = " ".join(mock_run.call_args[0][0])
            assert "label=com.docker.compose.project=custom-project" in cmd_str
            assert "myproject" not in cmd_str


class TestFindComposeFile:
    """Tests for find_compose_file function."""

    def test_finds_compose_yml_in_current_directory(self, tmp_path: Path) -> None:
        """Test that function finds docker-compose.yml in current directory."""
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("version: '3'\n")

        with patch("aegra_cli.utils.docker.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path
            result = find_compose_file()

            assert result == compose_file

    def test_finds_compose_yaml_in_current_directory(self, tmp_path: Path) -> None:
        """Test that function finds docker-compose.yaml in current directory."""
        compose_file = tmp_path / "docker-compose.yaml"
        compose_file.write_text("version: '3'\n")

        with patch("aegra_cli.utils.docker.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path
            result = find_compose_file()

            assert result == compose_file

    def test_finds_compose_file_in_parent_directory(self, tmp_path: Path) -> None:
        """Test that function finds docker-compose.yml in parent directory."""
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("version: '3'\n")

        subdir = tmp_path / "subdir"
        subdir.mkdir()

        with patch("aegra_cli.utils.docker.Path.cwd") as mock_cwd:
            mock_cwd.return_value = subdir
            result = find_compose_file()

            assert result == compose_file

    def test_returns_none_when_no_compose_file_found(self, tmp_path: Path) -> None:
        """Test that function returns None when no compose file exists."""
        with patch("aegra_cli.utils.docker.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path
            result = find_compose_file()

            assert result is None

    def test_prefers_yml_over_yaml(self, tmp_path: Path) -> None:
        """Test that function prefers .yml over .yaml when both exist."""
        yml_file = tmp_path / "docker-compose.yml"
        yml_file.write_text("version: '3'\n")

        yaml_file = tmp_path / "docker-compose.yaml"
        yaml_file.write_text("version: '3'\n")

        with patch("aegra_cli.utils.docker.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path
            result = find_compose_file()

            # Should find .yml first since it's checked first
            assert result == yml_file


class TestDevCommandWithDockerCheck:
    """Tests for dev command Docker check functionality."""

    def test_dev_help_shows_no_db_check_option(self, cli_runner) -> None:
        """Test that dev --help shows --no-db-check option."""
        from aegra_cli.cli import cli

        result = cli_runner.invoke(cli, ["dev", "--help"])
        assert result.exit_code == 0
        assert "--no-db-check" in result.output

    def test_dev_help_shows_file_option(self, cli_runner) -> None:
        """Test that dev --help shows --file/-f option for compose file."""
        from aegra_cli.cli import cli

        result = cli_runner.invoke(cli, ["dev", "--help"])
        assert result.exit_code == 0
        assert "--file" in result.output or "-f" in result.output

    def test_dev_fails_when_no_runtime_installed(self, cli_runner, tmp_path) -> None:
        """Test that dev fails gracefully when no container runtime is installed."""
        from pathlib import Path

        from aegra_cli.cli import cli

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with patch("aegra_cli.utils.docker.is_container_runtime_installed") as mock_installed:
                mock_installed.return_value = False
                result = cli_runner.invoke(cli, ["dev"])

                assert result.exit_code == 1
                assert "No container runtime found" in result.output

    def test_dev_fails_when_docker_not_running(self, cli_runner, tmp_path) -> None:
        """Test that dev fails gracefully when Docker is not running."""
        from pathlib import Path

        from aegra_cli.cli import cli

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with (
                patch("aegra_cli.utils.docker.is_container_runtime_installed") as mock_installed,
                patch("aegra_cli.utils.docker.is_docker_running") as mock_running,
                patch("aegra_cli.utils.docker.try_start_docker") as mock_try_start,
                patch("aegra_cli.utils.docker.shutil.which") as mock_which,
            ):
                mock_installed.return_value = True
                mock_running.return_value = False
                mock_try_start.return_value = False
                mock_which.side_effect = lambda cmd: "/usr/bin/docker" if cmd == "docker" else None

                result = cli_runner.invoke(cli, ["dev"])

                assert result.exit_code == 1
                assert "Docker is not running" in result.output

    def test_dev_fails_with_podman_vm_guidance_on_macos(self, cli_runner, tmp_path) -> None:
        """Test that dev shows 'podman machine start' on macOS when Podman VM is down."""
        from pathlib import Path

        from aegra_cli.cli import cli

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with (
                patch("aegra_cli.utils.docker.is_container_runtime_installed") as mock_installed,
                patch("aegra_cli.utils.docker.is_docker_running") as mock_running,
                patch("aegra_cli.utils.docker.shutil.which") as mock_which,
                patch("aegra_cli.utils.docker.platform.system") as mock_system,
            ):
                mock_installed.return_value = True
                mock_running.return_value = False
                mock_which.side_effect = lambda cmd: "/usr/bin/podman" if cmd == "podman" else None
                mock_system.return_value = "Darwin"

                result = cli_runner.invoke(cli, ["dev"])

                assert result.exit_code == 1
                assert "Podman is not ready" in result.output
                assert "podman machine start" in result.output

    def test_dev_fails_with_podman_diagnostic_guidance_on_linux(self, cli_runner, tmp_path) -> None:
        """Test that dev shows diagnostic guidance on Linux when Podman fails."""
        from pathlib import Path

        from aegra_cli.cli import cli

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with (
                patch("aegra_cli.utils.docker.is_container_runtime_installed") as mock_installed,
                patch("aegra_cli.utils.docker.is_docker_running") as mock_running,
                patch("aegra_cli.utils.docker.shutil.which") as mock_which,
                patch("aegra_cli.utils.docker.platform.system") as mock_system,
            ):
                mock_installed.return_value = True
                mock_running.return_value = False
                mock_which.side_effect = lambda cmd: "/usr/bin/podman" if cmd == "podman" else None
                mock_system.return_value = "Linux"

                result = cli_runner.invoke(cli, ["dev"])

                assert result.exit_code == 1
                assert "Podman is not working" in result.output
                assert "podman info" in result.output


@pytest.fixture
def cli_runner():
    """Provide a CliRunner for tests."""
    from click.testing import CliRunner

    return CliRunner()
