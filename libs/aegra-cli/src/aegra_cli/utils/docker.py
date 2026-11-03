"""Docker and Podman helpers for local Compose workflows."""

import os
import platform
import shutil
import subprocess
import time
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

ENV_CONTAINER_RUNTIME = "AEGRA_CONTAINER_RUNTIME"


class ContainerRuntime(StrEnum):
    """Local container CLI used to run Compose."""

    DOCKER = "docker"
    PODMAN = "podman"


class RuntimeSelectionError(Exception):
    """Raised when no usable Docker or Podman runtime can be selected."""


class ProjectOwnership(NamedTuple):
    runtimes: list[ContainerRuntime]
    listing_failed: bool


def _probe_runtime(runtime: ContainerRuntime) -> bool | None:
    """Whether `<runtime> info` succeeds. None means the check itself could not finish."""
    try:
        result = subprocess.run(
            [runtime.value, "info"],
            capture_output=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        console.print(f"[yellow]Could not check {runtime_label(runtime)}: {exc}[/yellow]")
        return None
    return result.returncode == 0


def is_docker_installed() -> bool:
    """Check if Docker CLI is installed and available in PATH."""
    return shutil.which("docker") is not None


def is_docker_running() -> bool:
    """Check if Docker daemon is running.

    Returns True if we can communicate with Docker daemon.
    """
    if not is_docker_installed():
        return False

    return _probe_runtime(ContainerRuntime.DOCKER) is True


def is_podman_installed() -> bool:
    return shutil.which("podman") is not None


def is_podman_running() -> bool:
    """True when `podman info` succeeds. A failed or hung check counts as not running."""
    if not is_podman_installed():
        return False

    return _probe_runtime(ContainerRuntime.PODMAN) is True


def is_runtime_running(runtime: ContainerRuntime) -> bool:
    """Dispatch to the Docker or Podman check. _probe_runtime tells stopped from unknown."""
    if runtime is ContainerRuntime.DOCKER:
        return is_docker_running()
    return is_podman_running()


def runtime_label(runtime: ContainerRuntime) -> str:
    """Human-readable name for messages. Commands use the enum value instead."""
    if runtime is ContainerRuntime.DOCKER:
        return "Docker"
    return "Podman"


def _runtime_installed(runtime: ContainerRuntime) -> bool:
    """Whether the CLI is on PATH. Says nothing about whether its daemon or machine is up."""
    if runtime is ContainerRuntime.DOCKER:
        return is_docker_installed()
    return is_podman_installed()


def resolve_container_runtime() -> ContainerRuntime:
    """Honor AEGRA_CONTAINER_RUNTIME, else a running Docker daemon, else Podman."""
    raw = os.environ.get(ENV_CONTAINER_RUNTIME, "").strip().lower()
    if raw:
        try:
            runtime = ContainerRuntime(raw)
        except ValueError:
            raise RuntimeSelectionError(
                f"Invalid {ENV_CONTAINER_RUNTIME}={raw!r}. Expected 'docker' or 'podman'."
            ) from None
        if not _runtime_installed(runtime):
            url = (
                "https://docs.docker.com/get-docker/"
                if runtime is ContainerRuntime.DOCKER
                else "https://podman.io/docs/installation"
            )
            raise RuntimeSelectionError(
                f"{runtime_label(runtime)} is not installed. Install it from {url}"
            )
        return runtime

    if is_docker_installed() and is_docker_running():
        return ContainerRuntime.DOCKER
    if is_podman_installed():
        return ContainerRuntime.PODMAN
    if is_docker_installed():
        return ContainerRuntime.DOCKER
    raise RuntimeSelectionError(
        "Neither Docker nor Podman is available. "
        "Install Docker (https://docs.docker.com/get-docker/) "
        "or Podman (https://podman.io/docs/installation)."
    )


def compose_argv(
    runtime: ContainerRuntime,
    compose_file: Path | None,
    *args: str,
) -> list[str]:
    """Build a Compose command. Without compose_file, the CLI searches the working directory."""
    command = [runtime.value, "compose"]
    if compose_file is not None:
        command.extend(["-f", str(compose_file)])
    command.extend(args)
    return command


def podman_compose_available() -> bool:
    """Whether `podman compose` runs. It relies on a provider installed separately from Podman."""
    try:
        result = subprocess.run(
            ["podman", "compose", "version"],
            capture_output=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        console.print(f"[yellow]Could not check Podman Compose: {exc}[/yellow]")
        return False
    return result.returncode == 0


def ensure_compose_plugin(runtime: ContainerRuntime) -> bool:
    """Podman must provide `podman compose`. Aegra does not call podman-compose."""
    if runtime is not ContainerRuntime.PODMAN:
        return True
    if podman_compose_available():
        return True
    console.print(
        "\n[bold red]Podman Compose is not available.[/bold red]\n\n"
        "Aegra runs [cyan]podman compose[/cyan] and does not invoke [cyan]podman-compose[/cyan].\n"
        "Install the Podman Compose plugin, then run the command again."
    )
    return False


def missing_binary_message(runtime: ContainerRuntime) -> str:
    """Rich-markup error for a CLI that vanished after the runtime was selected."""
    if runtime is ContainerRuntime.DOCKER:
        return (
            "[bold red]Error:[/bold red] docker is not installed or not in PATH.\n"
            "Please install Docker Desktop or Docker Engine."
        )
    return (
        "[bold red]Error:[/bold red] podman is not installed or not in PATH.\n"
        "Please install Podman."
    )


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


def get_podman_start_instructions() -> str:
    """Rich-markup steps for starting Podman by hand on this platform."""
    system = platform.system().lower()
    if system == "darwin":
        return (
            "[bold cyan]macOS:[/bold cyan] Start Podman Machine or Podman Desktop:\n"
            "  [cyan]podman machine start[/cyan]\n"
            "  [cyan]open -a 'Podman Desktop'[/cyan]"
        )
    if system == "linux":
        return (
            "[bold cyan]Linux:[/bold cyan] Start the Podman socket:\n"
            "  [cyan]systemctl --user start podman.socket[/cyan]"
        )
    if system == "windows":
        return (
            "[bold cyan]Windows:[/bold cyan] Start Podman Machine:\n"
            "  [cyan]podman machine start[/cyan]"
        )
    return "[dim]Please start Podman manually for your platform.[/dim]"


def get_runtime_start_instructions(runtime: ContainerRuntime) -> str:
    """Rich-markup steps for starting the given runtime by hand on this platform."""
    if runtime is ContainerRuntime.DOCKER:
        return get_docker_start_instructions()
    return get_podman_start_instructions()


def _wait_for_podman_ready(timeout_seconds: int = 60) -> bool:
    """Wait until `podman info` succeeds."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[dim]Waiting for Podman to be ready..."),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("waiting", total=None)
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if is_podman_running():
                progress.stop()
                console.print("[green]Podman is ready![/green]")
                return True
            elapsed = int(time.time() - start_time)
            progress.update(
                task, description=f"[dim]Waiting for Podman to be ready... ({elapsed}s)"
            )
            time.sleep(2)

    console.print(f"[yellow]Timed out after {timeout_seconds}s waiting for Podman.[/yellow]")
    return False


def try_start_podman() -> bool:
    """Attempt to start Podman. Returns True when `podman info` succeeds."""
    system = platform.system().lower()
    console.print("[yellow]Attempting to start Podman...[/yellow]")
    try:
        if system == "darwin":
            started = subprocess.run(
                ["podman", "machine", "start"],
                capture_output=True,
                timeout=60,
            )
            if started.returncode == 0 and _wait_for_podman_ready():
                return True
            opened = subprocess.run(
                ["open", "-a", "Podman Desktop"],
                capture_output=True,
                timeout=10,
            )
            if opened.returncode == 0:
                return _wait_for_podman_ready()
        elif system == "linux":
            started = subprocess.run(
                ["systemctl", "--user", "start", "podman.socket"],
                capture_output=True,
                timeout=30,
            )
            if started.returncode == 0:
                return _wait_for_podman_ready()
        elif system == "windows":
            started = subprocess.run(
                ["podman", "machine", "start"],
                capture_output=True,
                timeout=60,
            )
            if started.returncode == 0:
                return _wait_for_podman_ready()
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError) as exc:
        console.print(f"[yellow]Could not start Podman: {exc}[/yellow]")
        return False
    return False


def try_start_runtime(runtime: ContainerRuntime) -> bool:
    """Try to start the given runtime. True only once it responds."""
    if runtime is ContainerRuntime.DOCKER:
        return try_start_docker()
    return try_start_podman()


def ensure_runtime_ready(runtime: ContainerRuntime) -> bool:
    """Return True when the runtime is running, starting it when needed."""
    if is_runtime_running(runtime):
        return True
    label = runtime_label(runtime)
    console.print(f"\n[yellow]{label} is not running.[/yellow]")
    if try_start_runtime(runtime):
        console.print(f"[green]{label} started successfully![/green]")
        return True
    console.print(
        f"\n[bold red]Could not start {label} automatically.[/bold red]\n\n"
        f"{get_runtime_start_instructions(runtime)}\n\n"
        "[dim]Then run the command again.[/dim]"
    )
    return False


def _selected_runtime(runtime: ContainerRuntime | None) -> ContainerRuntime | None:
    """Use the given runtime or resolve one. None means none is usable, after printing why."""
    if runtime is not None:
        return runtime
    try:
        return resolve_container_runtime()
    except RuntimeSelectionError as exc:
        console.print(f"\n[bold red]{exc}[/bold red]")
        return None


def is_postgres_container_running(
    compose_file: Path | None = None,
    *,
    runtime: ContainerRuntime | None = None,
) -> bool:
    """Whether Compose lists postgres as running. Any failed check counts as not running."""
    selected = _selected_runtime(runtime)
    if selected is None:
        return False

    cmd = compose_argv(
        selected,
        compose_file,
        "ps",
        "--services",
        "--filter",
        "status=running",
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        console.print(f"[yellow]Could not check PostgreSQL with {selected.value}: {exc}[/yellow]")
        return False
    if result.returncode != 0:
        return False
    running_services = result.stdout.strip().split("\n")
    return "postgres" in running_services


def start_postgres_container(
    compose_file: Path | None = None,
    *,
    runtime: ContainerRuntime | None = None,
) -> bool:
    """Start postgres with `up --wait`. False means it did not start, after printing why."""
    selected = _selected_runtime(runtime)
    if selected is None:
        return False
    if not ensure_compose_plugin(selected):
        return False

    cmd = compose_argv(selected, compose_file, "up", "-d", "--wait", "postgres")

    console.print("[cyan]Starting PostgreSQL container...[/cyan]")
    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")

    try:
        result = subprocess.run(cmd, timeout=120)
    except subprocess.TimeoutExpired:
        console.print("[red]Timeout while starting PostgreSQL container[/red]")
        return False
    except FileNotFoundError:
        console.print(f"[red]{selected.value} command not found[/red]")
        return False

    if result.returncode == 0:
        console.print("[green]PostgreSQL container started successfully![/green]")
        return True
    console.print(
        f"[red]Failed to start PostgreSQL container (exit code {result.returncode})[/red]"
    )
    return False


def _compose_ps_quiet(
    runtime: ContainerRuntime,
    compose_file: Path,
    *,
    include_stopped: bool,
) -> subprocess.CompletedProcess[str] | None:
    """Run `compose ps -q`. None means it never finished; a non-zero exit is left to the caller."""
    command = compose_argv(runtime, compose_file, "ps", "-q")
    if include_stopped:
        command = compose_argv(runtime, compose_file, "ps", "--all", "-q")
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        console.print(f"[yellow]Could not list {runtime.value} Compose containers: {exc}[/yellow]")
        return None


def project_container_ids(runtime: ContainerRuntime, compose_file: Path) -> list[str] | None:
    """Container ids for this file, including stopped ones. None means the listing failed."""
    if not _runtime_installed(runtime):
        return []
    running = _probe_runtime(runtime)
    if running is None:
        return None
    if not running:
        return []
    listed = _compose_ps_quiet(runtime, compose_file, include_stopped=True)
    if listed is None:
        return None
    saw_stopped = listed.returncode == 0
    if not saw_stopped:
        # Some Compose builds reject --all. Running containers are still visible.
        listed = _compose_ps_quiet(runtime, compose_file, include_stopped=False)
        if listed is None:
            return None
        if listed.returncode != 0:
            console.print(
                f"[yellow]Could not list {runtime.value} Compose containers "
                f"(exit {listed.returncode}).[/yellow]"
            )
            return None
    found = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    if not found and not saw_stopped:
        # Stopped containers are invisible here, so empty does not mean none exist.
        console.print(
            f"[yellow]Could not list stopped {runtime.value} Compose containers.[/yellow]"
        )
        return None
    return found


def runtimes_owning_project(compose_file: Path) -> ProjectOwnership:
    """Runtimes that still have containers. A failed listing does not drop the others."""
    owners: list[ContainerRuntime] = []
    listing_failed = False
    for runtime in (ContainerRuntime.DOCKER, ContainerRuntime.PODMAN):
        found = project_container_ids(runtime, compose_file)
        if found is None:
            listing_failed = True
            continue
        if found:
            owners.append(runtime)
    return ProjectOwnership(owners, listing_failed)


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
    """Ensure PostgreSQL is running, starting Docker or Podman if needed.

    Args:
        compose_file: Optional path to docker-compose.yml file.

    Returns:
        True if PostgreSQL is running (either was already running or was started).
    """
    try:
        runtime = resolve_container_runtime()
    except RuntimeSelectionError as exc:
        console.print(f"\n[bold red]{exc}[/bold red]")
        return False

    if not ensure_runtime_ready(runtime):
        return False
    if not ensure_compose_plugin(runtime):
        return False

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

    if is_postgres_container_running(compose_file, runtime=runtime):
        console.print("[green]PostgreSQL is already running.[/green]")
        return True

    console.print("\n[yellow]PostgreSQL is not running.[/yellow]")
    return start_postgres_container(compose_file, runtime=runtime)
