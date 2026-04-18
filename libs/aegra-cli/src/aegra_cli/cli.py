"""Aegra CLI - Command-line interface for managing self-hosted agent deployments."""

import importlib.util
import ipaddress
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aegra_cli import __version__
from aegra_cli.commands import init
from aegra_cli.env import load_env_file
from aegra_cli.templates import (
    get_docker_compose,
    get_dockerfile,
    slugify,
)
from aegra_cli.utils.docker import ensure_postgres_running

console = Console()

# Default values for server options (single source of truth)
_DEFAULT_DEV_HOST = "127.0.0.1"
_DEFAULT_SERVE_HOST = "0.0.0.0"  # noqa: S104  # nosec B104 - intentional for Docker
_DEFAULT_PORT = 2026


def _resolve_server_option(
    ctx: click.Context,
    param_name: str,
    cli_value: str | int,
    *,
    env_var: str,
    default: str | int,
) -> str | int:
    """Resolve a server option with precedence: CLI flag > env var > default.

    Args:
        ctx: Click context to check parameter source.
        param_name: Name of the Click parameter.
        cli_value: Value from Click (may be the default).
        env_var: Environment variable name to check.
        default: The hardcoded default value.

    Returns:
        The resolved value with correct precedence.
    """
    source = ctx.get_parameter_source(param_name)
    if source == click.core.ParameterSource.COMMANDLINE:
        return cli_value
    env_val = os.environ.get(env_var)
    if env_val:
        try:
            return type(default)(env_val)
        except (ValueError, TypeError):
            msg = f"Invalid value for {env_var}: {env_val!r} (expected {type(default).__name__})"
            raise click.ClickException(msg) from None
    return default


# Attempt to get aegra-api version
try:
    from aegra_api import __version__ as api_version
except ImportError:
    api_version = "not installed"


@click.group()
@click.version_option(version=__version__, prog_name="aegra-cli")
def cli():
    """Aegra CLI - Manage your self-hosted agent deployments.

    Aegra is an open-source, self-hosted alternative to LangSmith Deployments.
    Use this CLI to run development servers, manage Docker services, and more.
    """
    pass


@cli.command()
def version():
    """Show version information for aegra-cli and aegra-api."""
    table = Table(title="Aegra Version Information", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="bold")
    table.add_column("Version", style="green")

    table.add_row("aegra-cli", __version__)
    table.add_row("aegra-api", api_version)

    console.print()
    console.print(table)
    console.print()


def find_config_file() -> Path | None:
    """Find aegra.json or langgraph.json in current directory.

    Returns:
        Path to config file if found, None otherwise
    """
    # Check for aegra.json first
    aegra_config = Path.cwd() / "aegra.json"
    if aegra_config.exists():
        return aegra_config

    # Fallback to langgraph.json
    langgraph_config = Path.cwd() / "langgraph.json"
    if langgraph_config.exists():
        return langgraph_config

    return None


def get_project_slug(config_path: Path | None) -> str:
    """Get project slug from config file or directory name.

    Args:
        config_path: Path to aegra.json config file

    Returns:
        Slugified project name
    """
    if config_path and config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
                if "name" in config:
                    return slugify(config["name"])
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback to directory name
    return slugify(Path.cwd().name)


def ensure_docker_files(project_path: Path, slug: str) -> Path:
    """Ensure docker-compose.yml and Dockerfile exist.

    Args:
        project_path: Project directory path
        slug: Project slug for naming

    Returns:
        Path to docker-compose.yml
    """
    compose_path = project_path / "docker-compose.yml"
    if not compose_path.exists():
        console.print(f"[cyan]Creating[/cyan] {compose_path}")
        compose_path.write_text(get_docker_compose(slug), encoding="utf-8")

    dockerfile_path = project_path / "Dockerfile"
    if not dockerfile_path.exists():
        console.print(f"[cyan]Creating[/cyan] {dockerfile_path}")
        dockerfile_path.write_text(get_dockerfile(), encoding="utf-8")

    return compose_path


@cli.command()
@click.option(
    "--host",
    default=_DEFAULT_DEV_HOST,
    help="Host to bind the server to.",
    show_default=True,
)
@click.option(
    "--port",
    default=_DEFAULT_PORT,
    type=int,
    help="Port to bind the server to.",
    show_default=True,
)
@click.option(
    "--app",
    default="aegra_api.main:app",
    help="Application import path.",
    show_default=True,
)
@click.option(
    "--config",
    "-c",
    "config_file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to aegra.json config file (auto-discovered if not specified).",
)
@click.option(
    "--env-file",
    "-e",
    "env_file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to .env file (default: .env in project directory).",
)
@click.option(
    "--no-db-check",
    is_flag=True,
    default=False,
    help="Skip automatic PostgreSQL/Docker check.",
)
@click.option(
    "--file",
    "-f",
    "compose_file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to docker-compose.yml file for PostgreSQL.",
)
@click.option(
    "--debug-port",
    default=None,
    type=int,
    help="Port for debugger to listen on (no debugger if not specified).",
    show_default=True,
)
@click.option(
    "--debug-host",
    default=None,
    help="Host for debugger to bind to (127.0.0.1 if not specified).",
    show_default=True,
)
@click.option(
    "--wait-for-client",
    is_flag=True,
    default=False,
    help="Break and wait for a debugger client to attach before starting.",
)
@click.pass_context
def dev(
    ctx: click.Context,
    host: str,
    port: int,
    app: str,
    config_file: Path | None,
    env_file: Path | None,
    no_db_check: bool,
    compose_file: Path | None,
    debug_port: int | None,
    debug_host: str | None,
    wait_for_client: bool,
) -> None:
    """Run the development server with hot reload.

    Starts uvicorn with --reload flag for development.
    The server will automatically restart when code changes are detected.

    Aegra auto-discovers aegra.json in the current directory, so you
    should run 'aegra dev' from your project root.

    By default, Aegra will check if Docker is running and start PostgreSQL
    automatically if needed. Use --no-db-check to skip this behavior.

    Examples:

        aegra dev                        # Auto-discover config, start server

        aegra dev -c /path/to/aegra.json # Use specific config file

        aegra dev -e /path/to/.env       # Use specific .env file

        aegra dev --no-db-check          # Start without database check
    """
    # Validate debug options: --wait-for-client only makes sense with --debug-port
    if wait_for_client and debug_port is None:
        raise click.UsageError("--wait-for-client requires --debug-port to be set")
    if debug_host is not None and debug_port is None:
        raise click.UsageError("--debug-host requires --debug-port to be set")

    # Discover or validate config file
    if config_file is not None:
        # User specified a config file explicitly
        resolved_config = config_file.resolve()
    else:
        # Auto-discover config file in current directory
        resolved_config = find_config_file()

    if resolved_config is None:
        console.print(
            "[bold red]Error:[/bold red] Could not find aegra.json or langgraph.json.\n"
            "Run [cyan]aegra init[/cyan] to create a new project, or specify "
            "[cyan]--config[/cyan] to point to your config file."
        )
        sys.exit(1)

    console.print(f"[dim]Using config: {resolved_config}[/dim]")

    # Set AEGRA_CONFIG env var so aegra-api resolves paths relative to config location
    os.environ["AEGRA_CONFIG"] = str(resolved_config)

    # Load environment variables from .env file
    # Default: look in config file's directory first, then cwd
    if env_file is None:
        # Try config directory first
        config_dir_env = resolved_config.parent / ".env"
        if config_dir_env.exists():
            env_file = config_dir_env

    # Auto-copy .env.example to .env if .env doesn't exist
    if env_file is None:
        dot_env = resolved_config.parent / ".env"
        dot_env_example = resolved_config.parent / ".env.example"
        if not dot_env.exists() and dot_env_example.exists():
            shutil.copy2(dot_env_example, dot_env)
            console.print(f"[cyan]Created[/cyan] {dot_env} [dim](copied from .env.example)[/dim]")

    loaded_env = load_env_file(env_file)
    if loaded_env:
        console.print(f"[dim]Loaded environment from: {loaded_env}[/dim]")
    elif env_file is not None:
        # User specified a file but it doesn't exist (shouldn't happen due to click validation)
        console.print(f"[yellow]Warning: .env file not found: {env_file}[/yellow]")

    # Resolve host/port with precedence: CLI flag > env var > default
    host = _resolve_server_option(ctx, "host", host, env_var="HOST", default=_DEFAULT_DEV_HOST)
    port = _resolve_server_option(ctx, "port", port, env_var="PORT", default=_DEFAULT_PORT)

    # Check and start PostgreSQL unless disabled
    if not no_db_check:
        console.print()

        # Auto-generate docker-compose.yml if not specified and doesn't exist
        if compose_file is None:
            project_path = resolved_config.parent
            default_compose = project_path / "docker-compose.yml"
            if default_compose.exists():
                compose_file = default_compose
            else:
                slug = get_project_slug(resolved_config)
                compose_file = ensure_docker_files(project_path, slug)

        if not ensure_postgres_running(compose_file):
            console.print(
                "\n[bold red]Cannot start server without PostgreSQL.[/bold red]\n"
                "[dim]Use --no-db-check to skip this check.[/dim]"
            )
            sys.exit(1)
        console.print()

    # Build info panel content
    info_lines = [
        "[bold green]Starting Aegra development server[/bold green]\n",
        f"[cyan]Host:[/cyan] {host}",
        f"[cyan]Port:[/cyan] {port}",
        f"[cyan]App:[/cyan] {app}",
        f"[cyan]Config:[/cyan] {resolved_config}",
    ]
    if loaded_env:
        info_lines.append(f"[cyan]Env:[/cyan] {loaded_env}")
    if debug_port is not None:
        info_lines.append(f"[cyan]Debug port:[/cyan] {debug_port}")
        if debug_host is not None:
            info_lines.append(f"[cyan]Debug host:[/cyan] {debug_host}")
        if wait_for_client:
            info_lines.append("[cyan]Wait for client to attach:[/cyan] True")
    info_lines.append("\n[dim]Press Ctrl+C to stop the server[/dim]")

    console.print(
        Panel(
            "\n".join(info_lines),
            title="[bold]Aegra Dev Server[/bold]",
            border_style="green",
        )
    )

    # Build command. If debug_port is provided, wrap uvicorn with debugpy.
    cmd_uvicorn = ["-m", "uvicorn", app, "--host", host, "--port", str(port), "--reload"]

    if debug_port is not None:
        if importlib.util.find_spec("debugpy") is None:
            console.print(
                "[bold red]Error:[/bold red] debugpy is not installed.\n"
                "Install it with: [cyan]pip install 'aegra-cli[debug]'[/cyan]"
            )
            sys.exit(1)

        console.print(
            "[yellow]Note:[/yellow] debugpy is active. Hot-reload will disconnect the debugger "
            "on each file change — reattach after every reload."
        )

        # Always bind debugpy to the debug host (default loopback) regardless of --host
        listen = str(debug_port)
        is_loopback = True

        # If debug_host is specified, warn if it's not loopback. Also handle ip v4 and v6 addresses
        if debug_host is not None:
            try:
                address = ipaddress.ip_address(debug_host)
                if address.version == 4:
                    listen = f"{debug_host}:{debug_port}"
                else:
                    listen = f"[{debug_host}]:{debug_port}"
                is_loopback = address.is_loopback
            except ValueError:
                if debug_host in ("localhost"):
                    listen = f"{debug_host}:{debug_port}"
                else:
                    console.print(
                        "[yellow]Warning:[/yellow] Invalid debug host specified. "
                        "Falling back to loopback binding for debugpy."
                    )

        if not is_loopback:
            console.print(
                Panel(
                    "[bold red]Warning:[/bold red] Debug host is non-loopback and exposes an "
                    "unauthenticated debug server!\n"
                    "[dim]Binding debugpy to a non-loopback address can allow remote code "
                    "execution. Use only if you understand the risk.[/dim]",
                    title="[bold red]Debug Exposure Warning[/bold red]",
                    border_style="red",
                )
            )
        cmd = [sys.executable, "-m", "debugpy", "--listen", listen]
        if wait_for_client:
            cmd.append("--wait-for-client")
        cmd.extend(cmd_uvicorn)
    else:
        cmd = [sys.executable] + cmd_uvicorn

    process = None
    try:
        # Use Popen for better signal handling across platforms
        process = subprocess.Popen(cmd)

        # Set up signal handler to forward signals to child process
        def signal_handler(signum, frame):
            if process and process.poll() is None:  # Process still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            console.print("\n[yellow]Server stopped by user.[/yellow]")
            sys.exit(0)

        # Register signal handlers (SIGTERM not available on Windows)
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, signal_handler)

        # Wait for the process to complete
        returncode = process.wait()
        sys.exit(returncode)

    except FileNotFoundError:
        console.print(
            "[bold red]Error:[/bold red] uvicorn is not installed.\n"
            "Install it with: [cyan]pip install uvicorn[/cyan]"
        )
        sys.exit(1)
    except KeyboardInterrupt:
        # Fallback handler if signal handler didn't catch it
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        console.print("\n[yellow]Server stopped by user.[/yellow]")
        sys.exit(0)


@cli.command()
@click.option(
    "--host",
    default=_DEFAULT_SERVE_HOST,
    help="Host to bind the server to.",
    show_default=True,
)
@click.option(
    "--port",
    default=_DEFAULT_PORT,
    type=int,
    help="Port to bind the server to.",
    show_default=True,
)
@click.option(
    "--app",
    default="aegra_api.main:app",
    help="Application import path.",
    show_default=True,
)
@click.option(
    "--config",
    "-c",
    "config_file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to aegra.json config file (auto-discovered if not specified).",
)
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, app: str, config_file: Path | None) -> None:
    """Run the production server.

    Starts uvicorn without --reload for production use.
    This command is typically used inside Docker containers.

    Examples:

        aegra serve                         # Start production server

        aegra serve --host 0.0.0.0 --port 8080

    """
    # Discover or validate config file
    if config_file is not None:
        resolved_config = config_file.resolve()
    else:
        resolved_config = find_config_file()

    if resolved_config is None:
        console.print(
            "[bold red]Error:[/bold red] Could not find aegra.json or langgraph.json.\n"
            "Run [cyan]aegra init[/cyan] to create a new project, or specify "
            "[cyan]--config[/cyan] to point to your config file."
        )
        sys.exit(1)

    # Set AEGRA_CONFIG env var
    os.environ["AEGRA_CONFIG"] = str(resolved_config)

    # Load .env file from config directory (same logic as dev command)
    config_dir_env = resolved_config.parent / ".env"
    loaded_env = load_env_file(config_dir_env if config_dir_env.exists() else None)

    # Resolve host/port with precedence: CLI flag > env var > default
    host = _resolve_server_option(ctx, "host", host, env_var="HOST", default=_DEFAULT_SERVE_HOST)
    port = _resolve_server_option(ctx, "port", port, env_var="PORT", default=_DEFAULT_PORT)

    info_lines = [
        "[bold green]Starting Aegra production server[/bold green]\n",
        f"[cyan]Host:[/cyan] {host}",
        f"[cyan]Port:[/cyan] {port}",
        f"[cyan]Config:[/cyan] {resolved_config}",
    ]
    if loaded_env:
        info_lines.append(f"[cyan]Env:[/cyan] {loaded_env}")

    console.print(
        Panel(
            "\n".join(info_lines),
            title="[bold]Aegra Server[/bold]",
            border_style="green",
        )
    )

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        app,
        "--host",
        host,
        "--port",
        str(port),
    ]

    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except FileNotFoundError:
        console.print(
            "[bold red]Error:[/bold red] uvicorn is not installed.\n"
            "Install it with: [cyan]pip install uvicorn[/cyan]"
        )
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
        sys.exit(0)


@cli.command()
@click.option(
    "--file",
    "-f",
    "compose_file",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to docker-compose file.",
)
@click.option(
    "--build/--no-build",
    default=True,
    help="Build images before starting containers.",
)
@click.argument("services", nargs=-1)
def up(compose_file: Path | None, build: bool, services: tuple[str, ...]):
    """Start services with Docker Compose.

    Uses docker-compose.yml which contains both postgres and the API service.
    Auto-generates Docker files if they don't exist.

    Examples:

        aegra up                    # Build and start all services

        aegra up --no-build         # Start without rebuilding

        aegra up postgres           # Start only postgres

        aegra up -f ./custom.yml    # Use custom compose file
    """
    # Determine which compose file to use
    project_path = Path.cwd()
    config_file = find_config_file()
    slug = get_project_slug(config_file)

    if compose_file is None:
        compose_file = ensure_docker_files(project_path, slug)
    elif not compose_file.exists():
        console.print(f"[bold red]Error:[/bold red] Compose file not found: {compose_file}")
        sys.exit(1)

    console.print(
        Panel(
            "[bold green]Starting Aegra services[/bold green]\n\n"
            f"[cyan]Compose file:[/cyan] {compose_file}",
            title="[bold]Aegra Up[/bold]",
            border_style="green",
        )
    )

    cmd = ["docker", "compose", "-f", str(compose_file)]

    cmd.append("up")
    cmd.append("-d")

    # Build unless --no-build is specified
    if build:
        cmd.append("--build")

    if services:
        cmd.extend(services)
        console.print(f"[cyan]Services:[/cyan] {', '.join(services)}")
    else:
        console.print("[cyan]Services:[/cyan] all")

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]\n")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            console.print("\n[bold green]Services started successfully![/bold green]")
            console.print()
            console.print(
                "[dim]View logs:    docker compose -f " + str(compose_file) + " logs -f[/dim]"
            )
            console.print("[dim]Stop:         aegra down[/dim]")
        else:
            console.print(
                f"\n[bold red]Error:[/bold red] Docker Compose exited with code {result.returncode}"
            )
        sys.exit(result.returncode)
    except FileNotFoundError:
        console.print(
            "[bold red]Error:[/bold red] docker is not installed or not in PATH.\n"
            "Please install Docker Desktop or Docker Engine."
        )
        sys.exit(1)


@cli.command()
@click.option(
    "--file",
    "-f",
    "compose_file",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to docker-compose file.",
)
@click.option(
    "--volumes",
    "-v",
    is_flag=True,
    default=False,
    help="Remove named volumes declared in the compose file.",
)
def down(compose_file: Path | None, volumes: bool):
    """Stop services with Docker Compose.

    Runs 'docker compose down' to stop and remove containers.

    Examples:

        aegra down                  # Stop services

        aegra down -v               # Stop and remove volumes

        aegra down -f ./custom.yml  # Stop specific compose file
    """
    console.print(
        Panel(
            "[bold yellow]Stopping Aegra services[/bold yellow]",
            title="[bold]Aegra Down[/bold]",
            border_style="yellow",
        )
    )

    if volumes:
        console.print("[yellow]Warning:[/yellow] Removing volumes - data will be lost!")

    project_path = Path.cwd()

    if compose_file:
        if not compose_file.exists():
            console.print(f"[bold red]Error:[/bold red] Compose file not found: {compose_file}")
            sys.exit(1)
        target_compose = compose_file
    else:
        target_compose = project_path / "docker-compose.yml"
        if not target_compose.exists():
            console.print("[yellow]No docker-compose.yml found. Nothing to stop.[/yellow]")
            sys.exit(0)

    console.print(f"\n[cyan]Stopping:[/cyan] {target_compose}")

    cmd = ["docker", "compose", "-f", str(target_compose), "down"]

    if volumes:
        cmd.append("-v")

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            console.print("\n[bold green]Services stopped successfully![/bold green]")
            sys.exit(0)
        else:
            console.print("\n[bold red]Some services failed to stop.[/bold red]")
            sys.exit(1)
    except FileNotFoundError:
        console.print(
            "[bold red]Error:[/bold red] docker is not installed or not in PATH.\n"
            "Please install Docker Desktop or Docker Engine."
        )
        sys.exit(1)


# Register command groups and commands from the commands package
cli.add_command(init)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
