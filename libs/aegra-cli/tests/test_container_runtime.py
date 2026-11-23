"""Tests for the Docker and Podman runtime helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aegra_cli.utils.container_runtime import (
    ContainerRuntime,
    ProjectOwnership,
    RuntimeSelectionError,
    _wait_for_podman_ready,
    compose_argv,
    console,
    ensure_compose_plugin,
    ensure_postgres_running,
    ensure_runtime_ready,
    find_compose_file,
    get_docker_start_instructions,
    get_podman_start_instructions,
    get_runtime_start_instructions,
    is_docker_installed,
    is_docker_running,
    is_podman_installed,
    is_podman_running,
    is_postgres_container_running,
    is_runtime_running,
    missing_binary_message,
    podman_compose_available,
    project_container_ids,
    resolve_container_runtime,
    runtime_label,
    runtimes_owning_project,
    start_postgres_container,
    try_start_podman,
    try_start_runtime,
)


class TestIsDockerInstalled:
    """Tests for is_docker_installed function."""

    def test_returns_true_when_docker_found(self) -> None:
        """Test that function returns True when docker is in PATH."""
        with patch("aegra_cli.utils.container_runtime.shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/docker"
            assert is_docker_installed() is True
            mock_which.assert_called_once_with("docker")

    def test_returns_false_when_docker_not_found(self) -> None:
        """Test that function returns False when docker is not in PATH."""
        with patch("aegra_cli.utils.container_runtime.shutil.which") as mock_which:
            mock_which.return_value = None
            assert is_docker_installed() is False


class TestIsDockerRunning:
    """Tests for is_docker_running function."""

    def test_returns_true_when_docker_daemon_responds(self) -> None:
        """Test that function returns True when docker info succeeds."""
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed") as mock_installed,
            patch("aegra_cli.utils.container_runtime.subprocess.run") as mock_run,
        ):
            mock_installed.return_value = True
            mock_run.return_value.returncode = 0

            assert is_docker_running() is True
            mock_run.assert_called_once_with(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )

    def test_returns_false_when_docker_not_installed(self) -> None:
        """Test that function returns False when docker is not installed."""
        with patch("aegra_cli.utils.container_runtime.is_docker_installed") as mock_installed:
            mock_installed.return_value = False
            assert is_docker_running() is False

    def test_returns_false_when_docker_daemon_not_running(self) -> None:
        """Test that function returns False when docker info fails."""
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed") as mock_installed,
            patch("aegra_cli.utils.container_runtime.subprocess.run") as mock_run,
        ):
            mock_installed.return_value = True
            mock_run.return_value.returncode = 1

            assert is_docker_running() is False

    def test_returns_false_on_timeout(self) -> None:
        """Test that function returns False when docker info times out."""
        import subprocess

        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed") as mock_installed,
            patch("aegra_cli.utils.container_runtime.subprocess.run") as mock_run,
        ):
            mock_installed.return_value = True
            mock_run.side_effect = subprocess.TimeoutExpired("docker", 10)

            assert is_docker_running() is False

    def test_reports_why_when_docker_info_times_out(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["docker", "info"], 10),
            ),
            console.capture() as captured,
        ):
            assert is_docker_running() is False
        output = " ".join(captured.get().split())
        assert "Could not check Docker" in output
        assert "timed out" in output

    def test_reports_why_when_docker_binary_disappears(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=FileNotFoundError(2, "No such file or directory", "docker"),
            ),
            console.capture() as captured,
        ):
            assert is_docker_running() is False
        output = " ".join(captured.get().split())
        assert "Could not check Docker" in output
        assert "[Errno 2] No such file or directory" in output


class TestGetDockerStartInstructions:
    """Tests for get_docker_start_instructions function."""

    def test_returns_macos_instructions_on_darwin(self) -> None:
        """Test that function returns macOS instructions on Darwin."""
        with patch("aegra_cli.utils.container_runtime.platform.system") as mock_system:
            mock_system.return_value = "Darwin"
            instructions = get_docker_start_instructions()

            assert "macOS" in instructions
            assert "open -a Docker" in instructions

    def test_returns_linux_instructions_on_linux(self) -> None:
        """Test that function returns Linux instructions on Linux."""
        with patch("aegra_cli.utils.container_runtime.platform.system") as mock_system:
            mock_system.return_value = "Linux"
            instructions = get_docker_start_instructions()

            assert "Linux" in instructions
            assert "systemctl start docker" in instructions

    def test_returns_windows_instructions_on_windows(self) -> None:
        """Test that function returns Windows instructions on Windows."""
        with patch("aegra_cli.utils.container_runtime.platform.system") as mock_system:
            mock_system.return_value = "Windows"
            instructions = get_docker_start_instructions()

            assert "Windows" in instructions
            assert "Docker Desktop" in instructions


class TestIsPostgresContainerRunning:
    """Tests for is_postgres_container_running function."""

    def test_returns_true_when_postgres_in_running_services(self) -> None:
        """Test that function returns True when postgres is running."""
        with patch("aegra_cli.utils.container_runtime.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "postgres\nredis\n"

            assert is_postgres_container_running(runtime=ContainerRuntime.DOCKER) is True

    def test_returns_false_when_postgres_not_running(self) -> None:
        """Test that function returns False when postgres is not in running services."""
        with patch("aegra_cli.utils.container_runtime.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "redis\n"

            assert is_postgres_container_running(runtime=ContainerRuntime.DOCKER) is False

    def test_returns_false_when_no_services_running(self) -> None:
        """Test that function returns False when no services are running."""
        with patch("aegra_cli.utils.container_runtime.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ""

            assert is_postgres_container_running(runtime=ContainerRuntime.DOCKER) is False

    def test_returns_false_on_docker_compose_failure(self) -> None:
        """Test that function returns False when docker compose fails."""
        with patch("aegra_cli.utils.container_runtime.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1

            assert is_postgres_container_running(runtime=ContainerRuntime.DOCKER) is False

    def test_uses_compose_file_when_provided(self) -> None:
        """Test that function uses compose file when provided."""
        with patch("aegra_cli.utils.container_runtime.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "postgres\n"

            compose_file = Path("/path/to/docker-compose.yml")
            assert (
                is_postgres_container_running(compose_file, runtime=ContainerRuntime.DOCKER) is True
            )

            call_args = mock_run.call_args[0][0]
            assert "-f" in call_args
            assert str(compose_file) in call_args

    def test_reports_timeout_when_compose_ps_times_out(self) -> None:
        with (
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["podman", "compose", "ps"], 10),
            ),
            console.capture() as captured,
        ):
            assert is_postgres_container_running(runtime=ContainerRuntime.PODMAN) is False
        output = " ".join(captured.get().split())
        assert "Could not check PostgreSQL with podman" in output
        assert "timed out" in output

    def test_reports_missing_binary_when_compose_ps_cannot_run(self) -> None:
        with (
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=FileNotFoundError(2, "No such file or directory", "podman"),
            ),
            console.capture() as captured,
        ):
            assert is_postgres_container_running(runtime=ContainerRuntime.PODMAN) is False
        output = " ".join(captured.get().split())
        assert "Could not check PostgreSQL with podman" in output
        assert "[Errno 2] No such file or directory" in output


class TestFindComposeFile:
    """Tests for find_compose_file function."""

    def test_finds_compose_yml_in_current_directory(self, tmp_path: Path) -> None:
        """Test that function finds docker-compose.yml in current directory."""
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("version: '3'\n")

        with patch("aegra_cli.utils.container_runtime.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path
            result = find_compose_file()

            assert result == compose_file

    def test_finds_compose_yaml_in_current_directory(self, tmp_path: Path) -> None:
        """Test that function finds docker-compose.yaml in current directory."""
        compose_file = tmp_path / "docker-compose.yaml"
        compose_file.write_text("version: '3'\n")

        with patch("aegra_cli.utils.container_runtime.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path
            result = find_compose_file()

            assert result == compose_file

    def test_finds_compose_file_in_parent_directory(self, tmp_path: Path) -> None:
        """Test that function finds docker-compose.yml in parent directory."""
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("version: '3'\n")

        subdir = tmp_path / "subdir"
        subdir.mkdir()

        with patch("aegra_cli.utils.container_runtime.Path.cwd") as mock_cwd:
            mock_cwd.return_value = subdir
            result = find_compose_file()

            assert result == compose_file

    def test_returns_none_when_no_compose_file_found(self, tmp_path: Path) -> None:
        """Test that function returns None when no compose file exists."""
        with patch("aegra_cli.utils.container_runtime.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path
            result = find_compose_file()

            assert result is None

    def test_prefers_yml_over_yaml(self, tmp_path: Path) -> None:
        """Test that function prefers .yml over .yaml when both exist."""
        yml_file = tmp_path / "docker-compose.yml"
        yml_file.write_text("version: '3'\n")

        yaml_file = tmp_path / "docker-compose.yaml"
        yaml_file.write_text("version: '3'\n")

        with patch("aegra_cli.utils.container_runtime.Path.cwd") as mock_cwd:
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

    def test_dev_fails_when_docker_not_installed(self, cli_runner, tmp_path) -> None:
        """Test that dev fails gracefully when Docker is not installed."""
        from pathlib import Path

        from aegra_cli.cli import cli

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with (
                patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=False),
                patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=False),
            ):
                result = cli_runner.invoke(cli, ["dev"])

                assert result.exit_code == 1
                assert "Neither Docker nor Podman" in result.output

    def test_dev_fails_when_docker_not_running(self, cli_runner, tmp_path) -> None:
        """Test that dev fails gracefully when Docker is not running."""
        from pathlib import Path

        from aegra_cli.cli import cli

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            Path("aegra.json").write_text('{"graphs": {}}')

            with (
                patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=True),
                patch("aegra_cli.utils.container_runtime.is_docker_running", return_value=False),
                patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=False),
                patch("aegra_cli.utils.container_runtime.try_start_docker", return_value=False),
            ):
                result = cli_runner.invoke(cli, ["dev"])

                assert result.exit_code == 1
                assert "Docker is not running" in result.output


def _completed(returncode: int, stdout: str = "") -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    return result


@pytest.fixture(autouse=True)
def clear_container_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # setenv records the original value so a later os.environ write is undone.
    monkeypatch.setenv("AEGRA_CONTAINER_RUNTIME", "")
    monkeypatch.delenv("AEGRA_CONTAINER_RUNTIME")


@pytest.fixture
def cli_runner():
    """Provide a CliRunner for tests."""
    from click.testing import CliRunner

    return CliRunner()


class TestPodmanDetection:
    """Podman install and health checks."""

    def test_installed_when_binary_is_on_path(self) -> None:
        with patch(
            "aegra_cli.utils.container_runtime.shutil.which", return_value="/usr/bin/podman"
        ):
            assert is_podman_installed() is True

    def test_not_installed_when_binary_is_missing(self) -> None:
        with patch("aegra_cli.utils.container_runtime.shutil.which", return_value=None):
            assert is_podman_installed() is False

    def test_running_when_info_succeeds(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime.subprocess.run", return_value=_completed(0)),
        ):
            assert is_podman_running() is True

    def test_not_running_when_not_installed(self) -> None:
        with patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=False):
            assert is_podman_running() is False

    def test_not_running_when_info_fails(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime.subprocess.run", return_value=_completed(1)),
        ):
            assert is_podman_running() is False

    def test_not_running_reports_timeout(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["podman", "info"], 10),
            ),
            console.capture() as captured,
        ):
            assert is_podman_running() is False
        output = " ".join(captured.get().split())
        assert "Could not check Podman" in output
        assert "['podman', 'info']" in output
        assert "timed out" in output

    def test_not_running_reports_missing_binary(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=FileNotFoundError(2, "No such file or directory", "podman"),
            ),
            console.capture() as captured,
        ):
            assert is_podman_running() is False
        output = " ".join(captured.get().split())
        assert "Could not check Podman" in output
        assert "[Errno 2] No such file or directory" in output

    def test_runtime_running_follows_each_cli(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_running", return_value=True),
            patch("aegra_cli.utils.container_runtime.is_podman_running", return_value=False),
        ):
            assert is_runtime_running(ContainerRuntime.DOCKER) is True
            assert is_runtime_running(ContainerRuntime.PODMAN) is False


class TestResolveContainerRuntime:
    """Selection via AEGRA_CONTAINER_RUNTIME and auto-detect."""

    def test_explicit_docker_ignores_case(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AEGRA_CONTAINER_RUNTIME", " Docker ")
        with patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=True):
            assert resolve_container_runtime() is ContainerRuntime.DOCKER

    def test_explicit_podman(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AEGRA_CONTAINER_RUNTIME", "podman")
        with patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=True):
            assert resolve_container_runtime() is ContainerRuntime.PODMAN

    def test_explicit_docker_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AEGRA_CONTAINER_RUNTIME", "docker")
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=False),
            pytest.raises(RuntimeSelectionError, match="Docker is not installed"),
        ):
            resolve_container_runtime()

    def test_explicit_podman_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AEGRA_CONTAINER_RUNTIME", "podman")
        with (
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=False),
            pytest.raises(RuntimeSelectionError, match="Podman is not installed"),
        ):
            resolve_container_runtime()

    def test_invalid_value_does_not_auto_detect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AEGRA_CONTAINER_RUNTIME", "containerd")
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=True),
            pytest.raises(RuntimeSelectionError, match="Invalid AEGRA_CONTAINER_RUNTIME"),
        ):
            resolve_container_runtime()

    def test_auto_prefers_running_docker_over_podman(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime.is_docker_running", return_value=True),
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=True),
        ):
            assert resolve_container_runtime() is ContainerRuntime.DOCKER

    def test_auto_uses_podman_when_docker_is_stopped(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime.is_docker_running", return_value=False),
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=True),
        ):
            assert resolve_container_runtime() is ContainerRuntime.PODMAN

    def test_auto_uses_podman_when_only_podman_is_installed(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=False),
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=True),
        ):
            assert resolve_container_runtime() is ContainerRuntime.PODMAN

    def test_auto_uses_stopped_docker_when_podman_is_absent(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime.is_docker_running", return_value=False),
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=False),
        ):
            assert resolve_container_runtime() is ContainerRuntime.DOCKER

    def test_auto_errors_when_neither_runtime_exists(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=False),
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=False),
            pytest.raises(RuntimeSelectionError, match="Neither Docker nor Podman"),
        ):
            resolve_container_runtime()

    def test_blank_env_uses_auto_detect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AEGRA_CONTAINER_RUNTIME", "   ")
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=False),
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=True),
        ):
            assert resolve_container_runtime() is ContainerRuntime.PODMAN


class TestComposeCommand:
    """Compose argv and the Podman compose plugin check."""

    def test_docker_argv_includes_file_and_args(self) -> None:
        command = compose_argv(
            ContainerRuntime.DOCKER,
            Path("docker-compose.yml"),
            "up",
            "-d",
        )
        assert command == ["docker", "compose", "-f", "docker-compose.yml", "up", "-d"]

    def test_podman_argv_without_file(self) -> None:
        assert compose_argv(ContainerRuntime.PODMAN, None, "down") == [
            "podman",
            "compose",
            "down",
        ]

    def test_podman_compose_version_success(self) -> None:
        with patch(
            "aegra_cli.utils.container_runtime.subprocess.run",
            return_value=_completed(0),
        ) as mock_run:
            assert podman_compose_available() is True
            assert mock_run.call_args[0][0] == ["podman", "compose", "version"]

    def test_podman_compose_version_failure(self) -> None:
        with patch("aegra_cli.utils.container_runtime.subprocess.run", return_value=_completed(1)):
            assert podman_compose_available() is False

    def test_podman_compose_version_reports_timeout(self) -> None:
        with (
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["podman", "compose", "version"], 10),
            ),
            console.capture() as captured,
        ):
            assert podman_compose_available() is False
        output = " ".join(captured.get().split())
        assert "Could not check Podman Compose" in output
        assert "timed out" in output

    def test_podman_compose_version_reports_missing_binary(self) -> None:
        with (
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=FileNotFoundError(2, "No such file or directory", "podman"),
            ),
            console.capture() as captured,
        ):
            assert podman_compose_available() is False
        output = " ".join(captured.get().split())
        assert "Could not check Podman Compose" in output
        assert "[Errno 2] No such file or directory" in output

    def test_docker_skips_podman_plugin_check(self) -> None:
        with patch("aegra_cli.utils.container_runtime.podman_compose_available") as mock_check:
            assert ensure_compose_plugin(ContainerRuntime.DOCKER) is True
            mock_check.assert_not_called()

    def test_podman_plugin_present(self) -> None:
        with patch("aegra_cli.utils.container_runtime.podman_compose_available", return_value=True):
            assert ensure_compose_plugin(ContainerRuntime.PODMAN) is True

    def test_podman_plugin_missing_names_podman_compose(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.podman_compose_available", return_value=False),
            console.capture() as captured,
        ):
            assert ensure_compose_plugin(ContainerRuntime.PODMAN) is False
        message = captured.get()
        assert "podman compose" in message
        assert "podman-compose" in message

    def test_missing_binary_message_names_the_runtime(self) -> None:
        assert "docker is not installed" in missing_binary_message(ContainerRuntime.DOCKER)
        assert "podman is not installed" in missing_binary_message(ContainerRuntime.PODMAN)
        assert runtime_label(ContainerRuntime.DOCKER) == "Docker"
        assert runtime_label(ContainerRuntime.PODMAN) == "Podman"


class TestPodmanStart:
    """Start attempts and hints for Podman."""

    def test_instructions_for_each_platform(self) -> None:
        """Each platform names its own start command, with a generic hint for unknown systems."""
        expectations = {
            "Darwin": "podman machine start",
            "Linux": "systemctl --user start podman.socket",
            "Windows": "podman machine start",
            "FreeBSD": "start Podman manually",
        }
        for system, snippet in expectations.items():
            with patch("aegra_cli.utils.container_runtime.platform.system", return_value=system):
                instructions = get_podman_start_instructions()
                assert snippet in instructions

    def test_runtime_instructions_delegate(self) -> None:
        with patch("aegra_cli.utils.container_runtime.platform.system", return_value="Linux"):
            assert "systemctl start docker" in get_runtime_start_instructions(
                ContainerRuntime.DOCKER
            )
            assert "podman.socket" in get_runtime_start_instructions(ContainerRuntime.PODMAN)

    def test_darwin_machine_start_success(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.platform.system", return_value="Darwin"),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                return_value=_completed(0),
            ) as mock_run,
            patch("aegra_cli.utils.container_runtime._wait_for_podman_ready", return_value=True),
        ):
            assert try_start_podman() is True
            assert mock_run.call_args_list[0].args[0] == ["podman", "machine", "start"]

    def test_darwin_falls_back_to_podman_desktop(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.platform.system", return_value="Darwin"),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=[_completed(1), _completed(0)],
            ) as mock_run,
            patch(
                "aegra_cli.utils.container_runtime._wait_for_podman_ready", return_value=True
            ) as mock_wait,
        ):
            assert try_start_podman() is True
            assert mock_run.call_args_list[1].args[0] == ["open", "-a", "Podman Desktop"]
            mock_wait.assert_called_once_with()

    def test_darwin_returns_false_when_desktop_open_fails(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.platform.system", return_value="Darwin"),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=[_completed(1), _completed(1)],
            ),
            patch(
                "aegra_cli.utils.container_runtime._wait_for_podman_ready", return_value=True
            ) as mock_wait,
        ):
            assert try_start_podman() is False
            mock_wait.assert_not_called()

    def test_linux_starts_user_socket(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.platform.system", return_value="Linux"),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                return_value=_completed(0),
            ) as mock_run,
            patch("aegra_cli.utils.container_runtime._wait_for_podman_ready", return_value=True),
        ):
            assert try_start_podman() is True
            assert mock_run.call_args.args[0] == [
                "systemctl",
                "--user",
                "start",
                "podman.socket",
            ]

    def test_linux_returns_false_when_systemctl_fails(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.platform.system", return_value="Linux"),
            patch("aegra_cli.utils.container_runtime.subprocess.run", return_value=_completed(1)),
            patch("aegra_cli.utils.container_runtime._wait_for_podman_ready") as mock_wait,
        ):
            assert try_start_podman() is False
            mock_wait.assert_not_called()

    def test_windows_starts_podman_machine(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.platform.system", return_value="Windows"),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                return_value=_completed(0),
            ) as mock_run,
            patch("aegra_cli.utils.container_runtime._wait_for_podman_ready", return_value=True),
        ):
            assert try_start_podman() is True
            assert mock_run.call_args.args[0] == ["podman", "machine", "start"]

    def test_windows_returns_false_when_machine_start_fails(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.platform.system", return_value="Windows"),
            patch("aegra_cli.utils.container_runtime.subprocess.run", return_value=_completed(1)),
        ):
            assert try_start_podman() is False

    def test_unknown_platform_returns_false(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.platform.system", return_value="FreeBSD"),
            patch("aegra_cli.utils.container_runtime.subprocess.run") as mock_run,
        ):
            assert try_start_podman() is False
            mock_run.assert_not_called()

    def test_reports_timeout_when_start_command_times_out(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.platform.system", return_value="Darwin"),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["podman", "machine", "start"], 60),
            ),
            console.capture() as captured,
        ):
            assert try_start_podman() is False
        output = " ".join(captured.get().split())
        assert "Could not start Podman" in output
        assert "timed out" in output

    def test_reports_denial_when_start_command_is_not_permitted(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.platform.system", return_value="Linux"),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=PermissionError(13, "Permission denied", "systemctl"),
            ),
            console.capture() as captured,
        ):
            assert try_start_podman() is False
        output = " ".join(captured.get().split())
        assert "Could not start Podman" in output
        assert "[Errno 13] Permission denied" in output

    def test_try_start_runtime_delegates(self) -> None:
        with (
            patch(
                "aegra_cli.utils.container_runtime.try_start_docker", return_value=True
            ) as mock_docker,
            patch(
                "aegra_cli.utils.container_runtime.try_start_podman", return_value=False
            ) as mock_podman,
        ):
            assert try_start_runtime(ContainerRuntime.DOCKER) is True
            assert try_start_runtime(ContainerRuntime.PODMAN) is False
            mock_docker.assert_called_once_with()
            mock_podman.assert_called_once_with()

    def test_wait_returns_true_when_podman_is_ready(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_podman_running", return_value=True),
            patch("aegra_cli.utils.container_runtime.time.sleep"),
            console.capture() as captured,
        ):
            assert _wait_for_podman_ready(timeout_seconds=1) is True
        assert "Podman is ready" in captured.get()

    def test_wait_times_out(self) -> None:
        ticks = iter((0, 0, 0, 30))

        def fake_time() -> int:
            return next(ticks)

        with (
            patch("aegra_cli.utils.container_runtime.is_podman_running", return_value=False),
            patch("aegra_cli.utils.container_runtime.time.time", side_effect=fake_time),
            patch("aegra_cli.utils.container_runtime.time.sleep"),
            console.capture() as captured,
        ):
            assert _wait_for_podman_ready(timeout_seconds=1) is False
        assert "Timed out" in captured.get()


class TestEnsureRuntimeReady:
    """Runtime health gate used by dev, up, and down."""

    def test_returns_true_when_already_running(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_runtime_running", return_value=True),
            patch("aegra_cli.utils.container_runtime.try_start_runtime") as mock_start,
        ):
            assert ensure_runtime_ready(ContainerRuntime.DOCKER) is True
            mock_start.assert_not_called()

    def test_returns_true_after_start(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_runtime_running", return_value=False),
            patch("aegra_cli.utils.container_runtime.try_start_runtime", return_value=True),
            console.capture() as captured,
        ):
            assert ensure_runtime_ready(ContainerRuntime.PODMAN) is True
        assert "Podman started successfully" in captured.get()

    def test_returns_false_when_start_fails(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_runtime_running", return_value=False),
            patch("aegra_cli.utils.container_runtime.try_start_runtime", return_value=False),
            patch("aegra_cli.utils.container_runtime.platform.system", return_value="Linux"),
            console.capture() as captured,
        ):
            assert ensure_runtime_ready(ContainerRuntime.PODMAN) is False
        message = captured.get()
        assert "Podman is not running" in message
        assert "podman.socket" in message


class TestPostgresCompose:
    """Postgres lifecycle commands follow the selected runtime."""

    def test_podman_ps_command(self) -> None:
        with patch(
            "aegra_cli.utils.container_runtime.subprocess.run",
            return_value=_completed(0, "postgres\n"),
        ) as mock_run:
            assert is_postgres_container_running(runtime=ContainerRuntime.PODMAN) is True
            assert mock_run.call_args.args[0][:2] == ["podman", "compose"]

    def test_ps_timeout_returns_false(self) -> None:
        with patch(
            "aegra_cli.utils.container_runtime.subprocess.run",
            side_effect=subprocess.TimeoutExpired("docker", 10),
        ):
            assert is_postgres_container_running(runtime=ContainerRuntime.DOCKER) is False

    def test_ps_missing_binary_returns_false(self) -> None:
        with patch(
            "aegra_cli.utils.container_runtime.subprocess.run",
            side_effect=FileNotFoundError("docker"),
        ):
            assert is_postgres_container_running(runtime=ContainerRuntime.DOCKER) is False

    def test_ps_returns_false_when_no_runtime_is_installed(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=False),
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=False),
            console.capture() as captured,
        ):
            assert is_postgres_container_running() is False
        assert "Neither Docker nor Podman" in captured.get()

    def test_start_uses_podman_compose(self, tmp_path: Path) -> None:
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("services: {}\n")
        with (
            patch("aegra_cli.utils.container_runtime.ensure_compose_plugin", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                return_value=_completed(0),
            ) as mock_run,
        ):
            assert start_postgres_container(compose_file, runtime=ContainerRuntime.PODMAN) is True
        command = mock_run.call_args.args[0]
        assert command[:2] == ["podman", "compose"]
        assert str(compose_file) in command
        assert command[-4:] == ["up", "-d", "--wait", "postgres"]

    def test_start_resolves_explicit_podman(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AEGRA_CONTAINER_RUNTIME", "podman")
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("services: {}\n")
        with (
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime.ensure_compose_plugin", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                return_value=_completed(0),
            ) as mock_run,
        ):
            assert start_postgres_container(compose_file) is True
        assert mock_run.call_args.args[0][0] == "podman"

    def test_start_returns_false_on_nonzero_exit(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.ensure_compose_plugin", return_value=True),
            patch("aegra_cli.utils.container_runtime.subprocess.run", return_value=_completed(2)),
        ):
            assert start_postgres_container(runtime=ContainerRuntime.DOCKER) is False

    def test_start_returns_false_on_timeout(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.ensure_compose_plugin", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=subprocess.TimeoutExpired("docker", 120),
            ),
        ):
            assert start_postgres_container(runtime=ContainerRuntime.DOCKER) is False

    def test_start_reports_missing_podman_binary(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.ensure_compose_plugin", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=FileNotFoundError("podman"),
            ),
            console.capture() as captured,
        ):
            assert start_postgres_container(runtime=ContainerRuntime.PODMAN) is False
        assert "podman command not found" in captured.get()

    def test_start_stops_when_plugin_is_missing(self) -> None:
        with patch("aegra_cli.utils.container_runtime.ensure_compose_plugin", return_value=False):
            assert start_postgres_container(runtime=ContainerRuntime.PODMAN) is False

    def test_start_returns_false_when_no_runtime_is_installed(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=False),
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=False),
            console.capture() as captured,
        ):
            assert start_postgres_container() is False
        assert "Neither Docker nor Podman" in captured.get()


class TestEnsurePostgresRunning:
    """dev's database check uses the resolved runtime."""

    def test_reports_when_neither_runtime_is_installed(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=False),
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=False),
            console.capture() as captured,
        ):
            assert ensure_postgres_running() is False
        assert "Neither Docker nor Podman" in captured.get()

    def test_returns_false_when_runtime_does_not_start(self, tmp_path: Path) -> None:
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("services: {}\n")
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime.is_docker_running", return_value=False),
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=False),
            patch("aegra_cli.utils.container_runtime.try_start_docker", return_value=False),
        ):
            assert ensure_postgres_running(compose_file) is False

    def test_returns_false_when_podman_compose_plugin_is_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AEGRA_CONTAINER_RUNTIME", "podman")
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("services: {}\n")
        with (
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime.is_podman_running", return_value=True),
            patch("aegra_cli.utils.container_runtime.podman_compose_available", return_value=False),
            console.capture() as captured,
        ):
            assert ensure_postgres_running(compose_file) is False
        assert "podman-compose" in captured.get()

    def test_returns_false_when_compose_file_is_missing(self) -> None:
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime.is_docker_running", return_value=True),
            patch("aegra_cli.utils.container_runtime.find_compose_file", return_value=None),
            console.capture() as captured,
        ):
            assert ensure_postgres_running() is False
        assert "No docker-compose.yml found" in captured.get()

    def test_returns_true_when_postgres_is_already_running(self, tmp_path: Path) -> None:
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("services: {}\n")
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime.is_docker_running", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.is_postgres_container_running", return_value=True
            ),
            patch("aegra_cli.utils.container_runtime.start_postgres_container") as mock_start,
        ):
            assert ensure_postgres_running(compose_file) is True
            mock_start.assert_not_called()

    def test_starts_postgres_with_the_selected_runtime(self, tmp_path: Path) -> None:
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("services: {}\n")
        with (
            patch("aegra_cli.utils.container_runtime.is_docker_installed", return_value=False),
            patch("aegra_cli.utils.container_runtime.is_podman_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime.is_podman_running", return_value=True),
            patch("aegra_cli.utils.container_runtime.podman_compose_available", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.is_postgres_container_running",
                return_value=False,
            ),
            patch(
                "aegra_cli.utils.container_runtime.start_postgres_container",
                return_value=True,
            ) as mock_start,
        ):
            assert ensure_postgres_running(compose_file) is True
        assert mock_start.call_args.kwargs["runtime"] is ContainerRuntime.PODMAN


class TestProjectOwnership:
    """down follows the runtime that still has this Compose project."""

    def test_lists_stopped_containers(self, tmp_path: Path) -> None:
        compose_file = tmp_path / "docker-compose.yml"
        with (
            patch("aegra_cli.utils.container_runtime._runtime_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime._probe_runtime", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                return_value=_completed(0, "abc\ndef\n"),
            ) as mock_run,
        ):
            assert project_container_ids(ContainerRuntime.PODMAN, compose_file) == ["abc", "def"]
        command = mock_run.call_args.args[0]
        assert command[:2] == ["podman", "compose"]
        assert command[-3:] == ["ps", "--all", "-q"]

    def test_falls_back_when_all_flag_is_rejected(self, tmp_path: Path) -> None:
        compose_file = tmp_path / "docker-compose.yml"
        with (
            patch("aegra_cli.utils.container_runtime._runtime_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime._probe_runtime", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=[_completed(1, ""), _completed(0, "abc\n")],
            ) as mock_run,
        ):
            assert project_container_ids(ContainerRuntime.DOCKER, compose_file) == ["abc"]
        assert mock_run.call_args_list[0].args[0][-3:] == ["ps", "--all", "-q"]
        assert mock_run.call_args_list[1].args[0][-2:] == ["ps", "-q"]

    def test_empty_when_all_listing_finds_nothing(self, tmp_path: Path) -> None:
        with (
            patch("aegra_cli.utils.container_runtime._runtime_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime._probe_runtime", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run", return_value=_completed(0, "")
            ),
        ):
            assert project_container_ids(ContainerRuntime.DOCKER, tmp_path / "c.yml") == []

    def test_none_when_fallback_finds_no_running_containers(self, tmp_path: Path) -> None:
        with (
            patch("aegra_cli.utils.container_runtime._runtime_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime._probe_runtime", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=[_completed(1, ""), _completed(0, "")],
            ),
            console.capture() as captured,
        ):
            assert project_container_ids(ContainerRuntime.DOCKER, tmp_path / "c.yml") is None
        assert "Could not list stopped docker Compose containers" in captured.get()

    def test_empty_fallback_flags_listing_failure_and_keeps_other_owners(
        self, tmp_path: Path
    ) -> None:
        with (
            patch("aegra_cli.utils.container_runtime._runtime_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime._probe_runtime", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=[_completed(1, ""), _completed(0, ""), _completed(0, "abc\n")],
            ),
        ):
            owners = runtimes_owning_project(tmp_path / "c.yml")

        assert owners == ProjectOwnership([ContainerRuntime.PODMAN], True)

    def test_empty_when_runtime_is_not_installed(self, tmp_path: Path) -> None:
        with patch("aegra_cli.utils.container_runtime._runtime_installed", return_value=False):
            assert project_container_ids(ContainerRuntime.DOCKER, tmp_path / "c.yml") == []

    def test_empty_when_runtime_reports_stopped(self, tmp_path: Path) -> None:
        with (
            patch("aegra_cli.utils.container_runtime._runtime_installed", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run", return_value=_completed(1)
            ) as mock_run,
        ):
            assert project_container_ids(ContainerRuntime.DOCKER, tmp_path / "c.yml") == []
        mock_run.assert_called_once()
        assert mock_run.call_args.args[0] == ["docker", "info"]

    def test_none_when_runtime_check_times_out(self, tmp_path: Path) -> None:
        with (
            patch("aegra_cli.utils.container_runtime._runtime_installed", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["podman", "info"], 10),
            ) as mock_run,
            console.capture() as captured,
        ):
            assert project_container_ids(ContainerRuntime.PODMAN, tmp_path / "c.yml") is None
        mock_run.assert_called_once()
        assert "Could not check Podman" in captured.get()

    def test_none_when_runtime_check_cannot_run(self, tmp_path: Path) -> None:
        with (
            patch("aegra_cli.utils.container_runtime._runtime_installed", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=FileNotFoundError(2, "No such file or directory", "docker"),
            ),
            console.capture() as captured,
        ):
            assert project_container_ids(ContainerRuntime.DOCKER, tmp_path / "c.yml") is None
        assert "Could not check Docker" in captured.get()

    def test_hung_runtime_flags_listing_failure_and_keeps_other_owners(
        self, tmp_path: Path
    ) -> None:
        with (
            patch("aegra_cli.utils.container_runtime._runtime_installed", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=[
                    _completed(0),
                    _completed(0, "abc\n"),
                    subprocess.TimeoutExpired(["podman", "info"], 10),
                ],
            ),
        ):
            owners = runtimes_owning_project(tmp_path / "c.yml")

        assert owners == ProjectOwnership([ContainerRuntime.DOCKER], True)

    def test_none_when_fallback_listing_times_out(self, tmp_path: Path) -> None:
        with (
            patch("aegra_cli.utils.container_runtime._runtime_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime._probe_runtime", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=[_completed(1, ""), subprocess.TimeoutExpired("docker", 10)],
            ),
            console.capture() as captured,
        ):
            assert project_container_ids(ContainerRuntime.DOCKER, tmp_path / "c.yml") is None
        assert "Could not list docker Compose containers" in captured.get()

    def test_none_when_ps_fails(self, tmp_path: Path) -> None:
        with (
            patch("aegra_cli.utils.container_runtime._runtime_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime._probe_runtime", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run", return_value=_completed(1, "")
            ),
            console.capture() as captured,
        ):
            assert project_container_ids(ContainerRuntime.DOCKER, tmp_path / "c.yml") is None
        assert "Could not list docker Compose containers" in captured.get()

    def test_none_when_ps_times_out(self, tmp_path: Path) -> None:
        with (
            patch("aegra_cli.utils.container_runtime._runtime_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime._probe_runtime", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=subprocess.TimeoutExpired("docker", 10),
            ),
            console.capture() as captured,
        ):
            assert project_container_ids(ContainerRuntime.DOCKER, tmp_path / "c.yml") is None
        assert "Could not list docker Compose containers" in captured.get()

    def test_none_when_binary_disappears(self, tmp_path: Path) -> None:
        with (
            patch("aegra_cli.utils.container_runtime._runtime_installed", return_value=True),
            patch("aegra_cli.utils.container_runtime._probe_runtime", return_value=True),
            patch(
                "aegra_cli.utils.container_runtime.subprocess.run",
                side_effect=FileNotFoundError("docker"),
            ),
            console.capture() as captured,
        ):
            assert project_container_ids(ContainerRuntime.DOCKER, tmp_path / "c.yml") is None
        assert "Could not list docker Compose containers" in captured.get()

    def test_owners_are_runtimes_with_containers(self, tmp_path: Path) -> None:
        """A runtime with an empty listing is not an owner."""
        compose_file = tmp_path / "docker-compose.yml"

        def containers(runtime: ContainerRuntime, _compose_file: Path) -> list[str] | None:
            """Podman has a container and Docker has none."""
            if runtime is ContainerRuntime.PODMAN:
                return ["abc"]
            return []

        with patch(
            "aegra_cli.utils.container_runtime.project_container_ids", side_effect=containers
        ):
            assert runtimes_owning_project(compose_file) == ProjectOwnership(
                [ContainerRuntime.PODMAN],
                False,
            )

    def test_owners_include_both_runtimes(self, tmp_path: Path) -> None:
        compose_file = tmp_path / "docker-compose.yml"
        with patch(
            "aegra_cli.utils.container_runtime.project_container_ids",
            return_value=["abc"],
        ):
            assert runtimes_owning_project(compose_file) == ProjectOwnership(
                [ContainerRuntime.DOCKER, ContainerRuntime.PODMAN],
                False,
            )

    def test_owners_none_when_a_listing_fails(self, tmp_path: Path) -> None:
        """A failed listing is flagged without dropping the runtime that was listed."""
        compose_file = tmp_path / "docker-compose.yml"

        def containers(runtime: ContainerRuntime, _compose_file: Path) -> list[str] | None:
            """Docker's listing fails and Podman has a container."""
            if runtime is ContainerRuntime.DOCKER:
                return None
            return ["abc"]

        with patch(
            "aegra_cli.utils.container_runtime.project_container_ids", side_effect=containers
        ):
            assert runtimes_owning_project(compose_file) == ProjectOwnership(
                [ContainerRuntime.PODMAN],
                True,
            )
