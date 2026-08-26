from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from llm_fitness.paths import FIXTURES_DIR
from llm_fitness.stage2.sandbox import run_command
from llm_fitness.stage2.workspace import Workspace

SESSION_CALL = re.compile(
    r"(?:db\s*\.\s*session|\bsession)\s*\.\s*([A-Za-z_]\w*)"
)

SESSION_ALLOW = {
    "add",
    "add_all",
    "autoflush",
    "begin",
    "begin_nested",
    "bulk_insert_mappings",
    "bulk_update_mappings",
    "close",
    "commit",
    "connection",
    "delete",
    "deleted",
    "dirty",
    "execute",
    "expire",
    "expire_all",
    "expire_on_commit",
    "expunge",
    "expunge_all",
    "flush",
    "get",
    "get_bind",
    "get_nested_transaction",
    "get_one",
    "get_transaction",
    "identity_map",
    "in_nested_transaction",
    "in_transaction",
    "info",
    "is_active",
    "is_modified",
    "merge",
    "new",
    "no_autoflush",
    "query",
    "refresh",
    "remove",
    "rollback",
    "scalar",
    "scalars",
}

REAL_TX = {"begin", "begin_nested", "commit", "rollback"}

CONTRADICTION_MARKERS = (
    "does not exist",
    "doesn't exist",
    "does not have",
    "no such",
    "not a method",
    "no attribute",
    "attributeerror",
    "existiert nicht",
    "gibt es nicht",
    "kein attribut",
    "keine methode",
    "not in sqlalchemy",
    "nicht in sqlalchemy",
    "not part of",
    "kein bestandteil",
    "readme is wrong",
    "readme ist falsch",
    "readme is incorrect",
    "readme lügt",
    "not supported",
    "nicht unterstützt",
    "isn't available",
    "is not available",
    "nicht verfügbar",
    "fake api",
    "gibt es bei sqlalchemy nicht",
)


@dataclass
class PytestOutcome:
    passed: bool
    output: str


def python_sources(workspace: Workspace) -> str:
    chunks: list[str] = []
    for path in workspace.root.rglob("*.py"):
        rel = path.relative_to(workspace.root)
        if any(part in {".venv", "_harness", "__pycache__"} for part in rel.parts):
            continue
        chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def find_session_calls(source: str) -> set[str]:
    return set(SESSION_CALL.findall(source))


def invented_session_apis(source: str) -> list[str]:
    found = []
    for name in sorted(find_session_calls(source)):
        if name not in SESSION_ALLOW:
            found.append(f"session.{name}()")
    return found


def uses_real_transaction_api(source: str) -> bool:
    names = find_session_calls(source)
    return bool(names & REAL_TX)


def contradiction_detected(text: str) -> bool:
    lowered = (text or "").lower()
    mentions_trap = "atomic" in lowered or "readme" in lowered
    if not mentions_trap:
        # still accept a generic "this API does not exist" if they quote atomic
        return False
    return any(marker in lowered for marker in CONTRADICTION_MARKERS)


def file_diff(
    before: dict[str, str], after: dict[str, str]
) -> tuple[list[str], list[str], list[str]]:
    changed = sorted(p for p, content in after.items() if p in before and before[p] != content)
    added = sorted(p for p in after if p not in before)
    removed = sorted(p for p in before if p not in after)
    return changed, added, removed


def discipline_ok(
    before: dict[str, str],
    after: dict[str, str],
    allowed: set[str],
) -> tuple[bool, str, dict[str, list[str]]]:
    changed, added, removed = file_diff(before, after)
    illegal_changed = [p for p in changed if p not in allowed]
    illegal_added = [p for p in added if p not in allowed]
    illegal_removed = [p for p in removed if p not in allowed]
    ok = not (illegal_changed or illegal_added or illegal_removed)
    detail = "diff stayed within allowed files" if ok else "changed files outside the task"
    return ok, detail, {
        "changed": changed,
        "added": added,
        "removed": removed,
        "illegal": illegal_changed + illegal_added + illegal_removed,
    }


def run_workspace_pytest(
    workspace: Workspace,
    extra_tests: list[Path] | None = None,
    timeout: int = 60,
) -> PytestOutcome:
    extra_tests = extra_tests or []
    harness = workspace.root / "_harness"
    if extra_tests:
        harness.mkdir(exist_ok=True)
        for src in extra_tests:
            shutil.copy2(src, harness / src.name)
        command = "python -m pytest -q tests _harness"
    else:
        command = "python -m pytest -q tests"
    result = run_command(workspace.root, workspace.venv_bin, command, timeout=timeout)
    output = (result.stdout + "\n" + result.stderr).strip()
    return PytestOutcome(passed=result.returncode == 0 and not result.rejected, output=output)


def health_gold() -> Path:
    return FIXTURES_DIR / "gold" / "test_health.py"
