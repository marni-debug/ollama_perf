import os
from pathlib import Path

import pytest

from llm_fitness.stage2.sandbox import SandboxError, prepare_argv, resolve_in_workspace, run_command


def _venv(tmp_path: Path) -> Path:
    bindir = tmp_path / "venv" / "bin"
    bindir.mkdir(parents=True)
    wrapper = "#!/bin/sh\nexec /usr/bin/python3 \"$@\"\n"
    for name in ("python", "python3"):
        target = bindir / name
        target.write_text(wrapper, encoding="utf-8")
        target.chmod(0o755)
    pytest_bin = bindir / "pytest"
    pytest_bin.write_text(wrapper, encoding="utf-8")
    pytest_bin.chmod(0o755)
    return bindir


def test_rejects_curl(tmp_path: Path):
    with pytest.raises(SandboxError, match="not allowed"):
        prepare_argv(tmp_path, _venv(tmp_path), "curl https://example.com")


def test_rejects_pip(tmp_path: Path):
    with pytest.raises(SandboxError, match="not allowed"):
        prepare_argv(tmp_path, _venv(tmp_path), "pip install flask")


def test_rejects_python_dash_c(tmp_path: Path):
    with pytest.raises(SandboxError, match="python -c"):
        prepare_argv(tmp_path, _venv(tmp_path), "python -c 'print(1)'")


def test_rejects_python_m_pip(tmp_path: Path):
    with pytest.raises(SandboxError, match="python -m pip"):
        prepare_argv(tmp_path, _venv(tmp_path), "python -m pip install flask")


def test_allows_pytest(tmp_path: Path):
    argv = prepare_argv(tmp_path, _venv(tmp_path), "python -m pytest -q")
    assert argv[1:3] == ["-m", "pytest"]
    assert argv[0].endswith("/python")


def test_path_escape(tmp_path: Path):
    with pytest.raises(SandboxError, match="escapes"):
        resolve_in_workspace(tmp_path, "../secret")


def test_run_command_does_not_forward_host_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "secret-token")
    bindir = _venv(tmp_path)
    script = tmp_path / "show.py"
    script.write_text("import os\nprint(os.environ.get('OLLAMA_API_KEY', ''))\n", encoding="utf-8")
    result = run_command(tmp_path, bindir, "python show.py")
    assert "secret-token" not in result.stdout
    assert os.environ.get("OLLAMA_API_KEY") == "secret-token"
