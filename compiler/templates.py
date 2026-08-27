"""Deterministic prompt and explain rendering."""

from __future__ import annotations

import json

from compiler.analyzer import CLASS_WEAKNESS, CapabilityProfile
from compiler.rules import SelectedRules

ROLE_LINE = "You are an autonomous coding agent."
NO_EXTRA_INSTRUCTIONS = "No additional operating instructions."
NEUTRAL_REQUIREMENTS = "Follow the task requirements as written."
NEUTRAL_IMPLEMENTATION = "Keep changes minimal and consistent with the existing codebase."
NEUTRAL_VERIFICATION = "Verify the completed work against the task before finishing."
COMPLETION_BASE = "Completion requires satisfying the task."
COMPLETION_TESTS = "Do not declare the task complete until relevant tests have been run."

HEADERS = (
    "ROLE",
    "MODEL-SPECIFIC OPERATING PROFILE",
    "TASK",
    "REQUIREMENT HANDLING",
    "IMPLEMENTATION GUIDANCE",
    "VERIFICATION",
    "COMPLETION CRITERIA",
)


def render_prompt(profile: CapabilityProfile, task: str, selected: SelectedRules) -> str:
    # First occurrence wins so identical instruction strings are not repeated
    # across Operating instructions and later sections.
    seen: set[str] = set()
    head = "\n\n".join(
        [
            f"ROLE\n\n{ROLE_LINE}",
            f"MODEL-SPECIFIC OPERATING PROFILE\n\n{_profile_body(profile, selected, seen)}",
        ]
    )
    task_block = f"TASK\n\n{task}"
    if not task.endswith("\n"):
        task_block += "\n"
    tail = "\n\n".join(
        [
            f"REQUIREMENT HANDLING\n\n{_requirement_body(selected, seen)}",
            f"IMPLEMENTATION GUIDANCE\n\n{_implementation_body(selected, seen)}",
            f"VERIFICATION\n\n{_verification_body(selected, seen)}",
            f"COMPLETION CRITERIA\n\n{_completion_body(profile)}",
        ]
    )
    return f"{head}\n\n{task_block}\n{tail}\n"


def render_explain(profile: CapabilityProfile, selected: SelectedRules) -> str:
    blocks: list[str] = []
    for name in sorted(profile.classifications):
        lines = [
            f"Category: {name}",
            f"Score: {json.dumps(profile.scores[name])}",
            f"Classification: {profile.classifications[name]}",
        ]
        rules = selected.by_category.get(name, ())
        if rules:
            lines.append("Applied rules:")
            lines.extend(f"{i}. {rule}" for i, rule in enumerate(rules, 1))
        else:
            lines.append("Applied rules: none")
        blocks.append("\n".join(lines))
    if selected.extras:
        extra_lines = ["Extra operating instructions:"]
        extra_lines.extend(f"{i}. {rule}" for i, rule in enumerate(selected.extras, 1))
        blocks.append("\n".join(extra_lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _profile_body(profile: CapabilityProfile, selected: SelectedRules, seen: set[str]) -> str:
    # Section bodies own weakness YAML; operating only lists extras not used later.
    unique = _take(selected.extras, seen)
    if unique:
        instructions = _numbered(unique)
    else:
        instructions = NO_EXTRA_INSTRUCTIONS
    return "\n".join(
        [
            f"Model: {profile.model}",
            "",
            "Strengths:",
            _name_block(profile.strengths),
            "",
            "Weaknesses:",
            _name_block(profile.weaknesses),
            "",
            "Operating instructions:",
            instructions,
        ]
    )


def _requirement_body(selected: SelectedRules, seen: set[str]) -> str:
    rules = _take(selected.by_category.get("instruction_following", ()), seen)
    if rules:
        return _numbered(rules)
    return NEUTRAL_REQUIREMENTS


def _implementation_body(selected: SelectedRules, seen: set[str]) -> str:
    rules = _take(
        _dedupe_join(
            selected.by_category.get("agent_discipline", ()),
            selected.by_category.get("code_reasoning", ()),
        ),
        seen,
    )
    if rules:
        return _numbered(rules)
    return NEUTRAL_IMPLEMENTATION


def _verification_body(selected: SelectedRules, seen: set[str]) -> str:
    rules = _take(
        _dedupe_join(
            selected.by_category.get("hallucination_resistance", ()),
            selected.by_category.get("self_correction", ()),
        ),
        seen,
    )
    if rules:
        return _numbered(rules)
    return NEUTRAL_VERIFICATION


def _completion_body(profile: CapabilityProfile) -> str:
    lines = [COMPLETION_BASE]
    if profile.classifications.get("self_correction") == CLASS_WEAKNESS:
        lines.append(COMPLETION_TESTS)
    return "\n".join(lines)


def _name_block(names: list[str]) -> str:
    if not names:
        return "None"
    return "\n".join(f"- {name}" for name in names)


def _numbered(items: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))


def _dedupe_join(*groups: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for item in group:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
    return out


def _take(items: tuple[str, ...] | list[str], seen: set[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
