"""A/B holdout runner: hash prompt files, pair per-task outcomes, assemble JSON.

Does not compile prompts or call fitness graders / agent_loop.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compiler.evaluate import (
    DEFAULT_TEMPERATURE,
    EVALUATOR_NAME,
    EvaluationConfig,
    Evaluator,
    NUM_CTX,
    NUM_PREDICT,
    TaskOutcome,
)

RESULT_VERSION = 1

NOTES = (
    "Ollama chat has no seed; temperature=0 does not freeze sampling.",
    "Live evaluator is non_empty_response: passed=bool(content.strip()); not a coding-quality grader.",
    "Default evaluator is chat-only (no tools); workspace_root isolates arms only.",
)


class ExperimentError(Exception):
    pass


@dataclass(frozen=True)
class HoldoutTask:
    id: str
    input: str


@dataclass(frozen=True)
class Holdout:
    name: str
    tasks: tuple[HoldoutTask, ...]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prompt_file(path: Path) -> tuple[str, str]:
    """Return (sha256 hex of exact file bytes, decoded UTF-8 text). Does not strip."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ExperimentError(f"cannot read prompt file: {exc}") from exc
    if len(raw) == 0:
        raise ExperimentError(f"empty prompt file: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExperimentError(f"cannot read prompt file: {exc}") from exc
    return digest, text


def load_holdout(path: Path) -> Holdout:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ExperimentError(f"cannot read benchmark file: {exc}") from exc
    except OSError as exc:
        raise ExperimentError(f"cannot read benchmark file: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ExperimentError("benchmark JSON must be an object")
    if "tasks" not in data:
        raise ExperimentError("missing required field: tasks")
    if not isinstance(data["tasks"], list):
        raise ExperimentError("tasks must be an array")
    if "name" not in data:
        raise ExperimentError("missing required field: name")
    if not isinstance(data["name"], str):
        raise ExperimentError("name must be a string")
    tasks: list[HoldoutTask] = []
    seen: set[str] = set()
    for index, item in enumerate(data["tasks"]):
        if not isinstance(item, dict):
            raise ExperimentError(f"task {index} must be an object")
        if "id" not in item:
            raise ExperimentError(f"task {index} missing required field: id")
        if "input" not in item:
            raise ExperimentError(f"task {index} missing required field: input")
        if not isinstance(item["id"], str):
            raise ExperimentError(f"task {index} id must be a string")
        if not isinstance(item["input"], str):
            raise ExperimentError(f"task {index} input must be a string")
        if item["id"] in seen:
            raise ExperimentError(f"duplicate task id: {item['id']}")
        seen.add(item["id"])
        tasks.append(HoldoutTask(id=item["id"], input=item["input"]))
    return Holdout(name=data["name"], tasks=tuple(tasks))


def compare_passed(baseline_passed: bool, compiled_passed: bool) -> str:
    if (not baseline_passed) and compiled_passed:
        return "improved"
    if baseline_passed and (not compiled_passed):
        return "regressed"
    return "unchanged"


def _arm_stats(outcomes: list[TaskOutcome]) -> dict[str, Any]:
    passed = sum(1 for outcome in outcomes if outcome.passed)
    return {
        "failed": len(outcomes) - passed,
        "passed": passed,
        "score": float(sum(outcome.score for outcome in outcomes)),
    }


def _knobs() -> dict[str, float | int]:
    return {
        "num_ctx": NUM_CTX,
        "num_predict": NUM_PREDICT,
        "temperature": DEFAULT_TEMPERATURE,
    }


def _arm_workspace(parent: Path, arm: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"{arm}-", dir=parent))


def run_experiment(
    *,
    model: str,
    benchmark_path: str | Path,
    baseline_prompt_path: str | Path,
    compiled_prompt_path: str | Path,
    evaluator: Evaluator,
) -> dict[str, Any]:
    baseline_sha, baseline_prompt = load_prompt_file(Path(baseline_prompt_path))
    compiled_sha, compiled_prompt = load_prompt_file(Path(compiled_prompt_path))
    holdout = load_holdout(Path(benchmark_path))

    baseline_outcomes: list[TaskOutcome] = []
    compiled_outcomes: list[TaskOutcome] = []
    tasks_out: list[dict[str, Any]] = []
    parent = Path(tempfile.mkdtemp(prefix="pc-ab-"))
    try:
        for task in holdout.tasks:
            baseline_config = EvaluationConfig(
                model=model,
                prompt=baseline_prompt,
                task_id=task.id,
                task_input=task.input,
                workspace_root=_arm_workspace(parent, "a"),
            )
            compiled_config = EvaluationConfig(
                model=model,
                prompt=compiled_prompt,
                task_id=task.id,
                task_input=task.input,
                workspace_root=_arm_workspace(parent, "b"),
            )
            baseline_raw = evaluator.evaluate(baseline_config)
            compiled_raw = evaluator.evaluate(compiled_config)
            baseline_outcome = TaskOutcome(
                task_id=task.id,
                passed=baseline_raw.passed,
                score=float(baseline_raw.score),
            )
            compiled_outcome = TaskOutcome(
                task_id=task.id,
                passed=compiled_raw.passed,
                score=float(compiled_raw.score),
            )
            baseline_outcomes.append(baseline_outcome)
            compiled_outcomes.append(compiled_outcome)
            tasks_out.append(
                {
                    "baseline": {
                        "passed": baseline_outcome.passed,
                        "score": float(baseline_outcome.score),
                    },
                    "comparison": compare_passed(baseline_outcome.passed, compiled_outcome.passed),
                    "compiled": {
                        "passed": compiled_outcome.passed,
                        "score": float(compiled_outcome.score),
                    },
                    "task_id": task.id,
                }
            )
    finally:
        shutil.rmtree(parent, ignore_errors=True)

    baseline_stats = _arm_stats(baseline_outcomes)
    compiled_stats = _arm_stats(compiled_outcomes)
    baseline_stats["prompt_sha256"] = baseline_sha
    compiled_stats["prompt_sha256"] = compiled_sha
    return {
        "baseline": baseline_stats,
        "benchmark": holdout.name,
        "compiled": compiled_stats,
        "delta": {
            "passed": compiled_stats["passed"] - baseline_stats["passed"],
            "score": compiled_stats["score"] - baseline_stats["score"],
        },
        "evaluator": EVALUATOR_NAME,
        "knobs": _knobs(),
        "model": model,
        "notes": list(NOTES),
        "tasks": tasks_out,
        "version": RESULT_VERSION,
    }
