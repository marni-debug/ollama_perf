from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ALLOWED_NAMES = {"python", "python3", "pytest"}
DENIED_NAMES = {"pip", "pip3", "curl", "wget", "ssh", "sudo", "apt", "npm", "git"}
DEFAULT_TIMEOUT = 30


class SandboxError(Exception):
    pass


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    rejected: bool = False


def resolve_in_workspace(workspace: Path, raw: str) -> Path:
    root = workspace.resolve()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SandboxError(f"path escapes workspace: {raw}") from exc
    return resolved


def sandbox_env(workspace: Path, venv_bin: Path) -> dict[str, str]:
    path_parts = [str(venv_bin.resolve()), "/usr/bin", "/bin"]
    return {
        "PATH": os.pathsep.join(path_parts),
        "HOME": str(workspace.resolve()),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "VIRTUAL_ENV": str(venv_bin.resolve().parent),
    }


_NETNS_PREFIX: list[str] | None = None


def _netns_prefix() -> list[str]:
    global _NETNS_PREFIX
    if _NETNS_PREFIX is not None:
        return _NETNS_PREFIX
    unshare = shutil.which("unshare")
    if unshare:
        probe = subprocess.run(
            [unshare, "-n", "true"],
            capture_output=True,
            timeout=3,
            check=False,
        )
        if probe.returncode == 0:
            _NETNS_PREFIX = [unshare, "-n"]
            return _NETNS_PREFIX
    _NETNS_PREFIX = []
    return _NETNS_PREFIX


def prepare_argv(workspace: Path, venv_bin: Path, command: str) -> list[str]:
    try:
        argv = shlex.split(command, posix=True)
    except ValueError as exc:
        raise SandboxError(f"could not parse command: {exc}") from exc
    if not argv:
        raise SandboxError("empty command")

    name = Path(argv[0]).name
    if name in DENIED_NAMES or name not in ALLOWED_NAMES:
        raise SandboxError(f"command not allowed: {argv[0]}")

    venv_python = venv_bin / "python"
    venv_pytest = venv_bin / "pytest"
    if name in {"python", "python3"}:
        if "-c" in argv[1:]:
            raise SandboxError("python -c is not allowed")
        if len(argv) >= 3 and argv[1] == "-m" and argv[2] != "pytest":
            raise SandboxError(f"python -m {argv[2]} is not allowed")
        argv[0] = str(venv_python)
    elif name == "pytest":
        argv[0] = str(venv_pytest)

    root = workspace.resolve()
    for arg in argv[1:]:
        if arg.startswith("-"):
            continue
        looks_like_path = "/" in arg or arg.endswith(".py") or arg in {".", ".."}
        if not looks_like_path:
            continue
        candidate = Path(arg)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except (ValueError, OSError) as exc:
            raise SandboxError(f"argument path outside workspace: {arg}") from exc
    return argv


def run_command(
    workspace: Path,
    venv_bin: Path,
    command: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> CommandResult:
    try:
        argv = prepare_argv(workspace, venv_bin, command)
    except SandboxError as exc:
        return CommandResult(argv=[], returncode=127, stdout="", stderr=str(exc), rejected=True)

    prefix = _netns_prefix()
    full = prefix + argv if prefix else argv
    env = sandbox_env(workspace, venv_bin)
    try:
        proc = subprocess.run(
            full,
            cwd=str(workspace.resolve()),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(argv=argv, returncode=124, stdout="", stderr=f"timeout after {timeout}s")
    except FileNotFoundError as exc:
        return CommandResult(argv=argv, returncode=127, stdout="", stderr=str(exc))
    return CommandResult(
        argv=argv,
        returncode=proc.returncode,
        stdout=proc.stdout[-8000:],
        stderr=proc.stderr[-8000:],
    )
