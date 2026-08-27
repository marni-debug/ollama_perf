"""Load YAML rules and select weakness/failure instructions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compiler.analyzer import CATEGORIES, CapabilityProfile, CompilerError, Thresholds
from compiler.yaml_util import YamlError, load_yaml_file

TEST_ID_TO_CATEGORY = {
    "1": "hallucination_resistance",
    "2": "code_reasoning",
    "3": "code_reasoning",
    "4": "code_reasoning",
    "5": "instruction_following",
    "6": "agent_discipline",
    "7": "self_correction",
    "8": "hallucination_resistance",
}

# Fallback when category YAML is missing; keep aligned with shipped YAML[0].
FAILURE_INSTRUCTIONS = {
    "1": "Use only APIs, types, and modules that exist in the repository or its declared dependencies.",
    "2": "Read the relevant existing code before editing and match its types, control flow, and conventions.",
    "3": "Read the relevant existing code before editing and match its types, control flow, and conventions.",
    "4": "Read the relevant existing code before editing and match its types, control flow, and conventions.",
    "5": "Treat every explicit requirement in the task as mandatory; do not skip, reinterpret, or replace it.",
    "6": "Change only the files needed to complete the task; do not refactor, restyle, or edit unrelated code.",
    "7": "After changing code, run the existing tests that cover the change.",
    "8": "Use only APIs, types, and modules that exist in the repository or its declared dependencies.",
}

INVENTED_API_INSTRUCTION = "Do not use APIs not present in installed dependencies."


@dataclass(frozen=True)
class CategoryRule:
    description: str
    instructions: tuple[str, ...]


@dataclass(frozen=True)
class RuleSet:
    thresholds: Thresholds
    categories: dict[str, CategoryRule]


@dataclass(frozen=True)
class SelectedRules:
    by_category: dict[str, tuple[str, ...]]
    operating: tuple[str, ...]
    extras: tuple[str, ...]


def default_rules_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "rules"


def load_rules(rules_dir: Path | None = None) -> RuleSet:
    directory = rules_dir or default_rules_dir()
    try:
        raw_thresholds = load_yaml_file(directory / "thresholds.yaml")
    except (OSError, YamlError) as exc:
        raise CompilerError(f"cannot load thresholds: {exc}") from exc
    inner = raw_thresholds.get("thresholds", raw_thresholds) if isinstance(raw_thresholds, dict) else None
    if not isinstance(inner, dict) or "strong" not in inner or "weak" not in inner:
        raise CompilerError("thresholds.yaml must define thresholds.strong and thresholds.weak")
    try:
        thresholds = Thresholds(strong=float(inner["strong"]), weak=float(inner["weak"]))
    except (TypeError, ValueError) as exc:
        raise CompilerError(f"invalid thresholds: {exc}") from exc

    categories: dict[str, CategoryRule] = {}
    for name in CATEGORIES:
        path = directory / f"{name}.yaml"
        if not path.is_file():
            categories[name] = CategoryRule(description="", instructions=())
            continue
        try:
            data = load_yaml_file(path)
        except (OSError, YamlError) as exc:
            raise CompilerError(f"cannot load {path.name}: {exc}") from exc
        categories[name] = _category_rule(data, path.name)
    return RuleSet(thresholds=thresholds, categories=categories)


def select_rules(profile: CapabilityProfile, fitness: dict[str, Any], rules: RuleSet) -> SelectedRules:
    weakness_set = set(profile.weaknesses)
    yaml_by_category: dict[str, list[str]] = {name: [] for name in CATEGORIES}
    fail_by_category: dict[str, list[str]] = {name: [] for name in CATEGORIES}
    for name in CATEGORIES:
        if name in weakness_set:
            yaml_by_category[name].extend(rules.categories[name].instructions)

    for test in _tests(fitness):
        if test.get("scored", True) is False:
            continue
        if test.get("passed") is not False:
            continue
        test_id = str(test.get("id", ""))
        category = TEST_ID_TO_CATEGORY.get(test_id)
        if category is None:
            continue
        extra = _failure_extra(test_id, rules)
        if extra:
            fail_by_category[category].append(extra)

    invented = INVENTED_API_INSTRUCTION if _invented_api_flag(fitness) else None

    operating: list[str] = []
    by_category: dict[str, list[str]] = {name: [] for name in CATEGORIES}
    for name in CATEGORIES:
        if name in weakness_set:
            by_category[name].extend(yaml_by_category[name])
            by_category[name].extend(fail_by_category[name])
            if name == "hallucination_resistance" and invented:
                by_category[name].append(invented)
            by_category[name] = _dedupe(by_category[name])
        operating.extend(yaml_by_category[name])
        operating.extend(fail_by_category[name])
        if name == "hallucination_resistance" and invented:
            operating.append(invented)
    operating_tuple = tuple(_dedupe(operating))
    frozen = {name: tuple(by_category[name]) for name in CATEGORIES}
    section_rules = {item for items in frozen.values() for item in items}
    extras = tuple(item for item in operating_tuple if item not in section_rules)
    return SelectedRules(by_category=frozen, operating=operating_tuple, extras=extras)


def _failure_extra(test_id: str, rules: RuleSet) -> str | None:
    category = TEST_ID_TO_CATEGORY.get(test_id)
    if category is None:
        return None
    yaml_instructions = rules.categories[category].instructions
    if yaml_instructions:
        return yaml_instructions[0]
    return FAILURE_INSTRUCTIONS.get(test_id)


def _category_rule(data: Any, filename: str) -> CategoryRule:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise CompilerError(f"{filename} must be a mapping")
    description = data.get("description") or ""
    if not isinstance(description, str):
        description = str(description)
    raw_instructions = data.get("instructions") or []
    if not isinstance(raw_instructions, list):
        raise CompilerError(f"{filename} instructions must be a list")
    instructions = tuple(str(item) for item in raw_instructions if item is not None and str(item) != "")
    return CategoryRule(description=description, instructions=instructions)


def _tests(fitness: dict[str, Any]) -> list[dict[str, Any]]:
    tests = fitness.get("tests") or []
    if not isinstance(tests, list):
        return []
    return [item for item in tests if isinstance(item, dict)]


def _invented_api_flag(fitness: dict[str, Any]) -> bool:
    if fitness.get("critical_hallucination"):
        return True
    invented = fitness.get("invented_apis") or []
    if invented:
        return True
    for test in _tests(fitness):
        if test.get("critical_hallucination"):
            return True
        if test.get("invented_apis"):
            return True
    return False


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
