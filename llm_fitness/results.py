from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PerfSample:
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    total_duration_ns: int | None = None

    def generation_duration_ns(self) -> int | None:
        prompt = self.prompt_eval_duration_ns or 0
        eval_ = self.eval_duration_ns or 0
        if prompt or eval_:
            return prompt + eval_
        if self.total_duration_ns is None:
            return None
        load = self.load_duration_ns or 0
        return max(0, self.total_duration_ns - load)


@dataclass
class TestResult:
    __test__ = False
    id: str
    name: str
    stage: int
    passed: bool | None
    points: float
    max_points: float
    detail: str = ""
    checks: dict[str, Any] = field(default_factory=dict)
    critical_hallucination: bool = False
    invented_apis: list[str] = field(default_factory=list)
    response: str | None = None
    perf: list[PerfSample] = field(default_factory=list)
    scored: bool = True


@dataclass
class RunResult:
    model: str
    ollama_version: str
    context: str
    mode: str
    tests: list[TestResult]
    total: float
    total_max: float
    penalty: float
    dimensions: dict[str, float | None]
    verdict: str | None
    avg_response_s: float | None
    tokens_per_sec: float | None
    critical_hallucination: bool = False
    invented_apis: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    context_effective: int = 8192
    context_model_max: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        effective = data.pop("context_effective", 8192)
        model_max = data.pop("context_model_max", None)
        data["context"] = {"effective": effective, "model_max": model_max}
        return data


def _context_fields(data: dict[str, Any]) -> tuple[str, int, int | None]:
    raw = data.get("context", "")
    if isinstance(raw, dict):
        effective = int(raw.get("effective") or 8192)
        model_max = raw.get("model_max")
        if model_max is not None:
            model_max = int(model_max)
        display = f"{effective // 1024}K" if effective >= 1024 and effective % 1024 == 0 else str(effective)
        return display, effective, model_max
    extra_eff = data.get("context_effective")
    extra_max = data.get("context_model_max")
    return (
        str(raw or ""),
        int(extra_eff) if extra_eff is not None else 8192,
        int(extra_max) if extra_max is not None else None,
    )


def run_result_from_dict(data: dict[str, Any]) -> RunResult:
    tests = []
    for raw in data.get("tests", []):
        perf = [PerfSample(**p) for p in raw.pop("perf", [])]
        tests.append(TestResult(**raw, perf=perf))
    context, context_effective, context_model_max = _context_fields(data)
    return RunResult(
        model=data["model"],
        ollama_version=data.get("ollama_version", ""),
        context=context,
        context_effective=context_effective,
        context_model_max=context_model_max,
        mode=data.get("mode", "full"),
        tests=tests,
        total=data.get("total", 0),
        total_max=data.get("total_max", 100),
        penalty=data.get("penalty", 0),
        dimensions=data.get("dimensions", {}),
        verdict=data.get("verdict"),
        avg_response_s=data.get("avg_response_s"),
        tokens_per_sec=data.get("tokens_per_sec"),
        critical_hallucination=data.get("critical_hallucination", False),
        invented_apis=data.get("invented_apis", []),
        extra=data.get("extra", {}),
    )
