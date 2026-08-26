from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from llm_fitness.paths import FIXTURES_DIR, RUNS_DIR, VENV_CACHE_DIR

IGNORE_SNAPSHOT = {".venv", "__pycache__", ".pytest_cache", ".git"}


@dataclass
class Workspace:
    root: Path
    venv_bin: Path
    fixture_name: str

    def snapshot(self) -> dict[str, str]:
        files: dict[str, str] = {}
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root)
            if any(part in IGNORE_SNAPSHOT for part in rel.parts):
                continue
            files[rel.as_posix()] = path.read_text(encoding="utf-8", errors="replace")
        return files


def _hash_requirements(requirements: Path) -> str:
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()[:12]
    return digest


def cached_venv(requirements: Path) -> Path:
    VENV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    venv_dir = VENV_CACHE_DIR / f"venv-{_hash_requirements(requirements)}"
    marker = venv_dir / ".installed"
    python = venv_dir / "bin" / "python"
    if marker.exists() and python.exists():
        return venv_dir
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    subprocess.run(["python3", "-m", "venv", str(venv_dir)], check=True)
    pip = venv_dir / "bin" / "pip"
    subprocess.run(
        [str(pip), "install", "--disable-pip-version-check", "-r", str(requirements)],
        check=True,
        cwd=str(requirements.parent),
    )
    marker.write_text("ok\n", encoding="utf-8")
    return venv_dir


def _link_venv(workspace: Path, venv_dir: Path) -> None:
    target = workspace / ".venv"
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(venv_dir, target_is_directory=True)
    except OSError:
        shutil.copytree(venv_dir, target)


def create_workspace(
    fixture_name: str = "flask_mini",
    overlay: Path | None = None,
    keep: bool = False,
) -> Workspace:
    source = FIXTURES_DIR / fixture_name
    if not source.is_dir():
        raise FileNotFoundError(f"missing fixture: {source}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"llm-fitness-{fixture_name}-"
    if keep:
        root = Path(tempfile.mkdtemp(prefix=prefix, dir=str(RUNS_DIR)))
    else:
        # Prefer /tmp so the run lives fully outside the git repo.
        root = Path(tempfile.mkdtemp(prefix=prefix, dir="/tmp"))
    shutil.copytree(source, root, dirs_exist_ok=True)
    if overlay and overlay.exists():
        shutil.copytree(overlay, root, dirs_exist_ok=True)
    requirements = root / "requirements.txt"
    venv_dir = cached_venv(requirements)
    _link_venv(root, venv_dir)
    os.chmod(root, stat.S_IRWXU)
    return Workspace(root=root, venv_bin=venv_dir / "bin", fixture_name=fixture_name)


def cleanup_workspace(workspace: Workspace) -> None:
    shutil.rmtree(workspace.root, ignore_errors=True)
