"""Orchestrate fitness analysis, rule selection, and prompt rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from compiler.analyzer import CapabilityProfile, CompilerError, analyze, load_fitness
from compiler.rules import RuleSet, SelectedRules, default_rules_dir, load_rules, select_rules
from compiler.templates import render_explain, render_prompt


@dataclass(frozen=True)
class CompileResult:
    prompt: str
    profile: CapabilityProfile
    selected: SelectedRules
    rules: RuleSet


def compile_prompt(
    fitness_path: str | Path,
    task_path: str | Path,
    *,
    rules_dir: str | Path | None = None,
) -> CompileResult:
    fitness = load_fitness(Path(fitness_path))
    rules = load_rules(Path(rules_dir) if rules_dir is not None else default_rules_dir())
    profile = analyze(fitness, rules.thresholds)
    selected = select_rules(profile, fitness, rules)
    try:
        task = Path(task_path).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CompilerError(f"cannot read task file: {exc}") from exc
    except OSError as exc:
        raise CompilerError(f"cannot read task file: {exc}") from exc
    prompt = render_prompt(profile, task, selected)
    return CompileResult(prompt=prompt, profile=profile, selected=selected, rules=rules)


def explain_text(result: CompileResult) -> str:
    return render_explain(result.profile, result.selected)
