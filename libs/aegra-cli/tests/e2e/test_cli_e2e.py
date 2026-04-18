"""End-to-end tests for the packaged aegra CLI."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from aegra_api.test_utils import check_and_skip_if_geo_blocked, elog


def _find_cli_executable() -> Path:
    """Return the installed aegra console script path."""
    venv_script = Path(sys.executable).resolve().with_name("aegra")
    if venv_script.exists():
        return venv_script

    script_on_path = shutil.which("aegra")
    if script_on_path is None:
        pytest.skip("Installed aegra CLI executable was not found")

    return Path(script_on_path)


async def _stream_process_output(
    process: asyncio.subprocess.Process,
    output_queue: asyncio.Queue[str | None],
    output_lines: list[str],
) -> None:
    """Stream process output into a queue while preserving the full log."""
    stdout = process.stdout
    if stdout is None:
        await output_queue.put(None)
        return

    while True:
        line = await stdout.readline()
        if not line:
            break

        decoded_line = line.decode("utf-8", errors="replace")
        output_lines.append(decoded_line)
        await output_queue.put(decoded_line)

    await output_queue.put(None)


@asynccontextmanager
async def _run_cli_process(
    *,
    cli_executable: Path,
    cwd: Path,
    env: dict[str, str],
) -> AsyncIterator[tuple[asyncio.subprocess.Process, asyncio.Queue[str | None], list[str]]]:
    """Start the packaged CLI and ensure it is shut down cleanly."""
    process = await asyncio.create_subprocess_exec(
        str(cli_executable),
        "dev",
        "--no-db-check",
        "--no-reload",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--app",
        "test_app:app",
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    output_lines: list[str] = []
    output_queue: asyncio.Queue[str | None] = asyncio.Queue()
    output_task = asyncio.create_task(_stream_process_output(process, output_queue, output_lines))

    try:
        yield process, output_queue, output_lines
    finally:
        if process.returncode is None:
            process.send_signal(signal.SIGINT)
        try:
            await asyncio.wait_for(process.wait(), timeout=10.0)
        except TimeoutError:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=5.0)

        await asyncio.wait_for(output_task, timeout=5.0)


async def _wait_for_health(
    process: asyncio.subprocess.Process,
    *,
    output_queue: asyncio.Queue[str | None],
    timeout: float,
) -> int:
    """Wait until the spawned server responds on /health and return its bound port."""
    deadline = asyncio.get_running_loop().time() + timeout
    last_error: Exception | None = None
    port: int | None = None
    port_pattern = re.compile(r"Uvicorn running on http://127\.0\.0\.1:(\d+)")

    async with httpx.AsyncClient(timeout=0.5) as client:
        while asyncio.get_running_loop().time() < deadline:
            if process.returncode is not None:
                raise AssertionError(
                    f"CLI exited before server became healthy with code {process.returncode}."
                )

            try:
                line = await asyncio.wait_for(output_queue.get(), timeout=0.1)
            except TimeoutError:
                line = None

            if line:
                match = port_pattern.search(line)
                if match is not None:
                    port = int(match.group(1))

            if port is None:
                continue

            try:
                response = await client.get(f"http://127.0.0.1:{port}/health")
            except httpx.HTTPError as exc:
                last_error = exc
                await asyncio.sleep(0.1)
                continue

            if response.status_code == 200:
                elog("CLI health check", {"port": port, "payload": response.json()})
                return port

    raise AssertionError(
        f"Timed out waiting for CLI server health check on port {port}. Last error: {last_error}"
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_dev_no_reload_skips_reload_flag(tmp_path: Path) -> None:
    """The packaged CLI starts a healthy server without uvicorn auto-reload."""
    check_and_skip_if_geo_blocked({})
    elog("CLI E2E setup", {"tmp_path": str(tmp_path)})

    project_dir = tmp_path / "e2e-project"
    project_dir.mkdir()

    (project_dir / "aegra.json").write_text('{"graphs": {}}', encoding="utf-8")
    (project_dir / "test_app.py").write_text(
        "from fastapi import FastAPI\n"
        "\n"
        "app = FastAPI()\n"
        "\n"
        "@app.get('/health')\n"
        "def health() -> dict[str, str]:\n"
        "    return {'status': 'ok'}\n",
        encoding="utf-8",
    )

    cli_executable = _find_cli_executable()

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(project_dir)
        if not existing_pythonpath
        else f"{project_dir}{os.pathsep}{existing_pythonpath}"
    )

    async with _run_cli_process(
        cli_executable=cli_executable,
        cwd=project_dir,
        env=environment,
    ) as (process, output_queue, output_lines):
        port = await _wait_for_health(process, output_queue=output_queue, timeout=15.0)
        assert process.returncode is None
        elog("CLI process healthy", {"port": port})

    output = "".join(output_lines)
    elog("CLI process output", {"returncode": process.returncode, "output": output})

    assert process.returncode == 0, output
    assert "Started reloader process" not in output
    assert "Uvicorn running on http://127.0.0.1:" in output
    assert "Server stopped by user." in output
