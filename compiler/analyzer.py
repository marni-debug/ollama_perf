"""JSON fitness result → CapabilityProfile."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CATEGORIES = (
    "agent_discipline",
    "code_reasoning",
    "hallucination_resistance",
    "instruction_following",
    "self_correction",
)

CLASS_STRENGTH = "strength"
CLASS_WEAKNESS = "weakness"
CLASS_NEUTRAL = "neutral"


class CompilerError(Exception):
    pass


@dataclass(frozen=True)
class Thresholds:
    strong: float
    weak: float


@dataclass(frozen=True)
class CapabilityProfile:
    model: str
    scores: dict[str, float]
    classifications: dict[str, str]
    strengths: list[str]
    weaknesses: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "scores": dict(self.scores),
            "classifications": dict(self.classifications),
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
        }


def load_fitness(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CompilerError(f"cannot read fitness file: {exc}") from exc
    except OSError as exc:
        raise CompilerError(f"cannot read fitness file: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CompilerError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CompilerError("fitness JSON must be an object")
    if "model" not in data:
        raise CompilerError("missing required field: model")
    if not isinstance(data["model"], str):
        raise CompilerError("model must be a string")
    if not isinstance(data.get("dimensions"), dict):
        raise CompilerError("missing required field: dimensions")
    return data


def classify(score: float, thresholds: Thresholds) -> str:
    if score >= thresholds.strong:
        return CLASS_STRENGTH
    if score < thresholds.weak:
        return CLASS_WEAKNESS
    return CLASS_NEUTRAL


def analyze(data: dict[str, Any], thresholds: Thresholds) -> CapabilityProfile:
    model = data["model"]
    if not isinstance(model, str):
        raise CompilerError("model must be a string")
    dimensions = data["dimensions"]
    scores: dict[str, float] = {}
    classifications: dict[str, str] = {}
    for name in CATEGORIES:
        if name not in dimensions:
            continue
        raw = dimensions[name]
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise CompilerError(f"invalid score for {name}")
        score = float(raw) / 10.0
        scores[name] = score
        classifications[name] = classify(score, thresholds)
    strengths = [name for name in sorted(classifications) if classifications[name] == CLASS_STRENGTH]
    weaknesses = [name for name in sorted(classifications) if classifications[name] == CLASS_WEAKNESS]
    return CapabilityProfile(
        model=model,
        scores=scores,
        classifications=classifications,
        strengths=strengths,
        weaknesses=weaknesses,
    )
