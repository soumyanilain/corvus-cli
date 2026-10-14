from __future__ import annotations

import asyncio
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
}


def _resolve_inside_workspace(
    workspace: Path,
    requested: str | Path,
) -> Path:
    workspace = workspace.resolve()
    requested_path = Path(requested)

    if not requested_path.is_absolute():
        requested_path = workspace / requested_path

    requested_path = requested_path.resolve()

    if (
        requested_path != workspace
        and workspace not in requested_path.parents
    ):
        raise ValueError(
            f"Path must stay inside workspace: {workspace}"
        )

    return requested_path


def _kill_process_tree(pid: int) -> None:
    """
    Kill the shell AND everything it started. Killing only the
    shell leaves children (e.g. a dev server) holding the output
    pipe open, so the timeout never actually ends the wait.
    """

    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
            )
        else:
            os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


async def run_command(
    command: str,
    *,
    workspace: str | Path,
    cwd: str | Path = ".",
    timeout: int = 120,
    max_output_chars: int = 20_000,
) -> dict[str, Any]:
    """
    Run a shell command inside the project workspace.
    """

    workspace_path = Path(workspace).resolve()

    cwd_path = _resolve_inside_workspace(
        workspace_path,
        cwd,
    )

    process = await asyncio.create_subprocess_shell(
        command,
        cwd=str(cwd_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # Own process group on Mac/Linux so we can kill the whole tree.
        start_new_session=(sys.platform != "win32"),
    )

    try:
        stdout_bytes, stderr_bytes = (
            await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        )

        timed_out = False

    except asyncio.TimeoutError:
        _kill_process_tree(process.pid)

        try:
            stdout_bytes, stderr_bytes = (
                await asyncio.wait_for(
                    process.communicate(),
                    timeout=5,
                )
            )
        except asyncio.TimeoutError:
            stdout_bytes, stderr_bytes = b"", b""

        timed_out = True

    stdout = stdout_bytes.decode(
        "utf-8",
        errors="replace",
    )

    stderr = stderr_bytes.decode(
        "utf-8",
        errors="replace",
    )

    if len(stdout) > max_output_chars:
        stdout = (
            stdout[:max_output_chars]
            + "\n... [stdout truncated]"
        )

    if len(stderr) > max_output_chars:
        stderr = (
            stderr[:max_output_chars]
            + "\n... [stderr truncated]"
        )

    return {
        "command": command,
        "cwd": str(cwd_path),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
    }


async def grep_code(
    pattern: str,
    *,
    workspace: str | Path,
    path: str | Path = ".",
    regex: bool = False,
    case_sensitive: bool = False,
    max_matches: int = 100,
) -> dict[str, Any]:
    """
    Search text files inside the workspace.
    """

    workspace_path = Path(workspace).resolve()

    search_root = _resolve_inside_workspace(
        workspace_path,
        path,
    )

    flags = (
        0
        if case_sensitive
        else re.IGNORECASE
    )

    matcher = re.compile(
        pattern if regex else re.escape(pattern),
        flags,
    )

    matches: list[dict[str, Any]] = []

    files_scanned = 0

    # os.walk() yields nothing for a single file, so handle that case
    # explicitly (otherwise grep on one file reports files_scanned: 0).
    if search_root.is_file():
        walker = [(str(search_root.parent), [], [search_root.name])]
    else:
        walker = os.walk(search_root)

    for current_root, dir_names, file_names in walker:

        dir_names[:] = [
            name
            for name in dir_names
            if name not in DEFAULT_IGNORED_DIRS
        ]

        for file_name in file_names:
            file_path = Path(current_root) / file_name

            try:
                if file_path.stat().st_size > 2_000_000:
                    continue
            except OSError:
                continue

            try:
                text = file_path.read_text(
                    encoding="utf-8"
                )
            except (UnicodeDecodeError, OSError):
                continue

            files_scanned += 1

            for line_number, line in enumerate(
                text.splitlines(),
                start=1,
            ):
                if matcher.search(line):

                    matches.append(
                        {
                            "path": str(
                                file_path.relative_to(
                                    workspace_path
                                )
                            ),
                            "line": line_number,
                            "text": line.strip(),
                        }
                    )

                    if len(matches) >= max_matches:
                        return {
                            "pattern": pattern,
                            "files_scanned": files_scanned,
                            "matches": matches,
                            "truncated": True,
                        }

    return {
        "pattern": pattern,
        "files_scanned": files_scanned,
        "matches": matches,
        "truncated": False,
    }