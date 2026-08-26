from __future__ import annotations

from llm_fitness.results import RunResult, TestResult

WIDTH = 54


def _dots(label: str, status: str, width: int = 42) -> str:
    fill = width - len(label) - 1
    if fill < 2:
        fill = 2
    return f"{label} {'.' * fill} {status}"


def _status(test: TestResult) -> str:
    if test.passed is None:
        return "SKIP"
    return "PASS" if test.passed else "FAIL"


def _fmt_dim(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}/10"


def _fmt_num(value: float | None, suffix: str, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def progress_line(index: int, total: int, name: str, passed: bool | None) -> str:
    status = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
    return f"[{index}/{total}] {_dots(name, status)}"


def render_header(model: str, context: str) -> str:
    return "\n".join(
        [
            "╔" + "═" * WIDTH + "╗",
            "║" + "LLM CODING AGENT FITNESS TEST".center(WIDTH) + "║",
            "╚" + "═" * WIDTH + "╝",
            "",
            f"Model: {model}",
            "Backend: Ollama",
            f"Context: {context}",
            "",
        ]
    )


def render_footer(run: RunResult) -> str:
    lines = [
        "",
        "─" * 50,
        f"Score:       {run.total:.0f}/{run.total_max:.0f}",
        f"Hallucination resistance:  {_fmt_dim(run.dimensions.get('hallucination_resistance'))}",
        f"Code reasoning:             {_fmt_dim(run.dimensions.get('code_reasoning'))}",
        f"Instruction following:      {_fmt_dim(run.dimensions.get('instruction_following'))}",
        f"Agent discipline:           {_fmt_dim(run.dimensions.get('agent_discipline'))}",
        f"Self-correction:            {_fmt_dim(run.dimensions.get('self_correction'))}",
        "",
        f"Average response time:     {_fmt_num(run.avg_response_s, ' s')}",
        f"Tokens/sec:                {_fmt_num(run.tokens_per_sec, '', 1)}",
        "─" * 50,
    ]

    if run.critical_hallucination:
        apis = ", ".join(run.invented_apis) or "(unknown)"
        lines += [
            "",
            "Critical Hallucination",
            "Agent used API not in installed dependencies:",
            f"    {apis}",
            f"Penalty: -{run.penalty:.0f}",
        ]

    details = [t for t in run.tests if t.detail and t.passed is False]
    if details:
        lines += ["", "Notes:"]
        for test in details:
            lines.append(f"- {test.name}: {test.detail}")

    if run.verdict:
        lines += ["", "Verdict:", run.verdict]
    return "\n".join(lines) + "\n"


def render(run: RunResult) -> str:
    scored = [t for t in run.tests if t.scored]
    extra = [t for t in run.tests if not t.scored]
    n = len(scored)
    body = [f"[{i}/{n}] {_dots(test.name, _status(test))}" for i, test in enumerate(scored, start=1)]
    body += [f"[extra] {_dots(test.name, _status(test))}" for test in extra]
    return render_header(run.model, run.context) + "\n".join(body) + "\n" + render_footer(run)


def render_compare(runs: list[RunResult]) -> str:
    if not runs:
        return "No results to compare.\n"

    def slug(run: RunResult) -> str:
        return run.model.replace(":", "").replace(".", "").upper()[:12]

    headers = [""] + [slug(r) for r in runs]
    rows: list[tuple[str, list[str]]] = []

    dim_labels = [
        ("API / Hallucination", "hallucination_resistance"),
        ("Code reasoning", "code_reasoning"),
        ("Instruction", "instruction_following"),
        ("Agent discipline", "agent_discipline"),
        ("Self-correction", "self_correction"),
    ]
    for label, key in dim_labels:
        cells = []
        for run in runs:
            value = run.dimensions.get(key)
            cells.append(f"{value * 10:.0f}" if value is not None else "-")
        rows.append((label, cells))

    rows.append(("", [""] * len(runs)))
    rows.append((
        "TOTAL",
        [f"{r.total:.0f}/{r.total_max:.0f}" for r in runs],
    ))
    rows.append((
        "Latency",
        [_fmt_num(r.avg_response_s, "s") for r in runs],
    ))
    rows.append((
        "Tokens/s",
        [_fmt_num(r.tokens_per_sec, "", 0) for r in runs],
    ))

    col0 = max(len(h) for h, _ in rows) + 2
    col0 = max(col0, 18)
    other = 12
    lines = [" " * col0 + "".join(h.rjust(other) for h in headers[1:])]
    lines.append("-" * (col0 + other * len(runs)))
    for label, cells in rows:
        if not label:
            lines.append("")
            continue
        lines.append(label.ljust(col0) + "".join(c.rjust(other) for c in cells))
    return "\n".join(lines) + "\n"
