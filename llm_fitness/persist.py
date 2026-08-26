from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from llm_fitness.paths import RESULTS_DIR
from llm_fitness.results import RunResult, run_result_from_dict


def _slug(model: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in model).strip("-")


def save_run(run: RunResult, directory: Path | None = None) -> Path:
    target_dir = directory or RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = target_dir / f"{_slug(run.model)}-{stamp}.json"
    payload = run.to_dict()
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_run(path: Path) -> RunResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    return run_result_from_dict(data)
