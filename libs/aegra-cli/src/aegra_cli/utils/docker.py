"""Docker utility functions for aegra-cli.

This module provides helpers to detect Docker installation, check if Docker is running,
and manage PostgreSQL containers for local development.
"""

import platform
import shutil
import subprocess
from pathlib import Path

from rich.console import Console

console = Console()


def get_compose_command() -> list[str]:
    """Detect the available compose command.

    Checks in order: docker compose (v2 plugin), podman-compose, docker-compose (v1).
    Returns the command as a list suitable for subprocess calls.

    Raises:
        FileNotFoundError: If no compose tool is found.
    """
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            daemon = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            if daemon.returncode == 0:
                return ["docker", "compose"]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    if shutil.which("podman-compose"):
        return ["podman-compose"]

    if shutil.which("docker-compose"):
        return ["docker-compose"]

    raise FileNotFoundError(
        "No container compose tool found. Install one of: "
        "Docker Desktop, podman-compose, or docker-compose."
    )


def get_container_command() -> str:
    """Detect the available container runtime command.

    Returns 'docker' or 'podman', preferring docker if both are available.

    Raises:
        FileNotFoundError: If neither docker nor podman is found.
    """
    if shutil.which("docker"):
        return "docker"
    if shutil.which("podman"):
        return "podman"
    raise FileNotFoundError("No container runtime found. Install Docker or Podman.")


def is_container_runtime_installed() -> bool:
    """Check if a container runtime (Docker or Podman) is installed."""
    return shutil.which("docker") is not None or shutil.which("podman") is not None


def is_docker_installed() -> bool:
    """Check if Docker CLI is installed and available in PATH."""
    return shutil.which("docker") is not None


def is_docker_running() -> bool:
    """Check if a container runtime is available.

    For Docker, checks if the daemon is responding. Podman is daemonless
    and is always considered running if installed.
    """
    if shutil.which("podman"):
        try:
            result = subprocess.run(
                ["podman", "info"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    if not is_docker_installed():
        return False

    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_docker_start_instructions() -> str:
    """Get platform-specific instructions for starting Docker."""
    system = platform.system().lower()

    if system == "darwin":
        return (
            "[bold cyan]macOS:[/bold cyan] Start Docker Desktop from Applications,\n"
            "or run: [cyan]open -a Docker[/cyan]"
        )
    elif system == "linux":
        return (
            "[bold cyan]Linux:[/bold cyan] Start the Docker daemon:\n"
            "  [cyan]sudo systemctl start docker[/cyan]\n"
            "Or if using Docker Desktop:\n"
            "  [cyan]systemctl --user start docker-desktop[/cyan]"
        )
    elif system == "windows":
        return (
            "[bold cyan]Windows:[/bold cyan] Start Docker Desktop from the Start menu,\n"
            "or run from PowerShell: [cyan]Start-Process 'Docker Desktop'[/cyan]"
        )
    else:
        return "[dim]Please start Docker manually for your platform.[/dim]"


def try_start_docker() -> bool:
    """Attempt to start Docker daemon based on platform.

    Returns True if Docker was successfully started.
    """
    system = platform.system().lower()

    console.print("[yellow]Attempting to start Docker...[/yellow]")

    try:
        if system == "darwin":
            # macOS - try to open Docker Desktop
            result = subprocess.run(
                ["open", "-a", "Docker"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                return _wait_for_docker_ready()

        elif system == "linux":
            # Linux - try systemctl for daemon or Docker Desktop
            # First try the system daemon
            result = subprocess.run(
                ["systemctl", "start", "docker"],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                return _wait_for_docker_ready()

            # If that fails, try Docker Desktop (user service)
            result = subprocess.run(
                ["systemctl", "--user", "start", "docker-desktop"],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                return _wait_for_docker_ready()

        elif system == "windows":
            # Windows - try multiple methods to start Docker Desktop
            # Method 1: Try common Docker Desktop paths
            import os

            docker_paths = [
                os.path.expandvars(r"%ProgramFiles%\Docker\Docker\Docker Desktop.exe"),
                os.path.expandvars(r"%LOCALAPPDATA%\Docker\Docker Desktop.exe"),
                r"C:\Program Files\Docker\Docker\Docker Desktop.exe",
            ]

            for docker_path in docker_paths:
                if os.path.exists(docker_path):
                    try:
                        # Start Docker Desktop without waiting
                        subprocess.Popen(
                            [docker_path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        return _wait_for_docker_ready()
                    except (OSError, subprocess.SubprocessError):
                        continue

            # Method 2: Try using PowerShell Start-Process
            try:
                result = subprocess.run(
                    ["powershell", "-Command", "Start-Process 'Docker Desktop'"],
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    return _wait_for_docker_ready()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        pass

    return False


def _wait_for_docker_ready(timeout_seconds: int = 60) -> bool:
    """Wait for Docker daemon to be ready.

    Args:
        timeout_seconds: Maximum time to wait for Docker to be ready.

    Returns:
        True if Docker became ready within the timeout.
    """
    import time

    from rich.progress import Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[dim]Waiting for Docker to be ready..."),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("waiting", total=None)

        start_time = time.time()
        check_count = 0
        while time.time() - start_time < timeout_seconds:
            if is_docker_running():
                progress.stop()
                console.print("[green]Docker is ready![/green]")
                return True

            check_count += 1
            elapsed = int(time.time() - start_time)
            progress.update(
                task, description=f"[dim]Waiting for Docker to be ready... ({elapsed}s)"
            )
            time.sleep(2)

    console.print(f"[yellow]Timed out after {timeout_seconds}s waiting for Docker.[/yellow]")
    return False


def is_postgres_container_running(compose_file: Path | None = None) -> bool:
    """Check if PostgreSQL container is running.

    Args:
        compose_file: Optional path to docker-compose.yml file.

    Returns:
        True if postgres service is running.
    """
    try:
        compose_cmd = get_compose_command()
    except FileNotFoundError:
        return False

    cmd = list(compose_cmd)

    if compose_file:
        cmd.extend(["-f", str(compose_file)])

    cmd.extend(["ps", "--services", "--filter", "status=running"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            running_services = result.stdout.strip().split("\n")
            return "postgres" in running_services
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return False


def _supports_wait_flag(compose_cmd: list[str]) -> bool:
    """Check if the compose command supports the --wait flag.

    Docker Compose v2 (plugin) supports --wait. podman-compose and
    docker-compose v1 do not.
    """
    return compose_cmd == ["docker", "compose"]


def start_postgres_container(compose_file: Path | None = None) -> bool:
    """Start the PostgreSQL container using the detected compose tool.

    Uses --wait for Docker Compose v2. For podman-compose and docker-compose v1,
    starts the container and polls for readiness.

    Args:
        compose_file: Optional path to docker-compose.yml file.

    Returns:
        True if postgres was started successfully.
    """
    try:
        compose_cmd = get_compose_command()
    except FileNotFoundError:
        console.print("[red]No compose tool found[/red]")
        return False

    cmd = list(compose_cmd)

    if compose_file:
        cmd.extend(["-f", str(compose_file)])

    if _supports_wait_flag(compose_cmd):
        cmd.extend(["up", "-d", "--wait", "postgres"])
    else:
        cmd.extend(["up", "-d", "postgres"])

    console.print("[cyan]Starting PostgreSQL container...[/cyan]")
    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")

    try:
        result = subprocess.run(cmd, timeout=120)
        if result.returncode != 0:
            console.print(
                f"[red]Failed to start PostgreSQL container (exit code {result.returncode})[/red]"
            )
            return False
    except subprocess.TimeoutExpired:
        console.print("[red]Timeout while starting PostgreSQL container[/red]")
        return False
    except FileNotFoundError:
        console.print("[red]Compose command not found[/red]")
        return False

    if not _supports_wait_flag(compose_cmd):
        return _wait_for_postgres_ready(compose_file)

    console.print("[green]PostgreSQL container started successfully![/green]")
    return True


def _wait_for_postgres_ready(
    compose_file: Path | None = None,
    timeout_seconds: int = 30,
) -> bool:
    """Poll PostgreSQL container until it accepts connections.

    Uses pg_isready via compose exec for a true readiness check,
    rather than just checking container status.
    Used when the compose tool doesn't support --wait.
    """
    import time

    try:
        compose_cmd = get_compose_command()
    except FileNotFoundError:
        return False

    check_cmd = list(compose_cmd)
    if compose_file:
        check_cmd.extend(["-f", str(compose_file)])
    check_cmd.extend(["exec", "-T", "postgres", "pg_isready", "-U", "postgres"])

    console.print("[dim]Waiting for PostgreSQL to be ready...[/dim]")
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        try:
            result = subprocess.run(check_cmd, capture_output=True, timeout=10)
            if result.returncode == 0:
                console.print("[green]PostgreSQL container started successfully![/green]")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        time.sleep(1)

    console.print(f"[red]PostgreSQL did not become ready within {timeout_seconds}s[/red]")
    return False


def find_compose_file() -> Path | None:
    """Find docker-compose.yml in current directory or parent directories.

    Returns:
        Path to docker-compose.yml if found, None otherwise.
    """
    current = Path.cwd()

    # Check current directory and up to 3 parent directories
    for _ in range(4):
        compose_file = current / "docker-compose.yml"
        if compose_file.exists():
            return compose_file

        compose_file = current / "docker-compose.yaml"
        if compose_file.exists():
            return compose_file

        parent = current.parent
        if parent == current:
            break
        current = parent

    return None


def ensure_postgres_running(compose_file: Path | None = None) -> bool:
    """Ensure PostgreSQL is running, starting it if necessary.

    This function checks Docker and PostgreSQL status, attempting to start
    them if needed. It provides helpful error messages if something fails.

    Args:
        compose_file: Optional path to docker-compose.yml file.

    Returns:
        True if PostgreSQL is running (either was already running or was started).
    """
    # Step 1: Check if a container runtime is installed
    if not is_container_runtime_installed():
        console.print(
            "\n[bold red]No container runtime found![/bold red]\n\n"
            "Aegra requires Docker or Podman to run PostgreSQL for state persistence.\n\n"
            "[bold]Installation instructions:[/bold]\n"
            "  • Docker: [cyan]https://docs.docker.com/get-docker/[/cyan]\n"
            "  • Podman: [cyan]https://podman.io/docs/installation[/cyan]\n"
        )
        return False

    # Step 2: Check if container runtime is ready
    if not is_docker_running():
        if shutil.which("podman") and not shutil.which("docker"):
            console.print(
                "\n[bold red]Podman is not ready.[/bold red]\n\n"
                "Start your Podman machine:\n"
                "  [cyan]podman machine start[/cyan]\n\n"
                "[dim]Then run 'aegra dev' again.[/dim]"
            )
            return False

        console.print("\n[yellow]Docker is not running.[/yellow]")

        # Try to start Docker automatically
        if try_start_docker():
            console.print("[green]Docker started successfully![/green]")
        else:
            console.print(
                "\n[bold red]Could not start Docker automatically.[/bold red]\n\n"
                f"{get_docker_start_instructions()}\n\n"
                "[dim]Then run 'aegra dev' again.[/dim]"
            )
            return False

    # Step 3: Find docker-compose.yml if not provided
    if compose_file is None:
        compose_file = find_compose_file()
        if compose_file is None:
            console.print(
                "\n[bold red]No docker-compose.yml found![/bold red]\n\n"
                "Create one with: [cyan]aegra init[/cyan]\n"
                "Or create it manually with a postgres service."
            )
            return False

    console.print(f"[dim]Using compose file: {compose_file}[/dim]")

    # Step 4: Check if PostgreSQL container is already running
    if is_postgres_container_running(compose_file):
        console.print("[green]PostgreSQL is already running.[/green]")
        return True

    # Step 5: Start PostgreSQL container
    console.print("\n[yellow]PostgreSQL is not running.[/yellow]")
    return start_postgres_container(compose_file)
