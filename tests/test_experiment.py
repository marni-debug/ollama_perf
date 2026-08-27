import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from compiler.cli import main
from compiler.evaluate import (
    DEFAULT_TEMPERATURE,
    EvaluationConfig,
    NUM_CTX,
    NUM_PREDICT,
    OllamaChatEvaluator,
    TaskOutcome,
)
from compiler.experiment import ExperimentError, file_sha256, load_holdout, run_experiment


class FakeEvaluator:
    def __init__(self, outcomes: dict[tuple[str, str], tuple[bool, float]] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.configs: list[EvaluationConfig] = []

    def evaluate(self, config: EvaluationConfig) -> TaskOutcome:
        self.configs.append(config)
        passed, score = self.outcomes[(config.prompt, config.task_id)]
        return TaskOutcome(task_id=config.task_id, passed=passed, score=score)


class ConstEvaluator:
    def __init__(self, passed: bool = True, score: float = 1.0) -> None:
        self.passed = passed
        self.score = score
        self.configs: list[EvaluationConfig] = []

    def evaluate(self, config: EvaluationConfig) -> TaskOutcome:
        self.configs.append(config)
        return TaskOutcome(task_id=config.task_id, passed=self.passed, score=self.score)


def write_prompt(path: Path, text: str | bytes) -> Path:
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def write_holdout(path: Path, tasks: list[dict], name: str = "holdout") -> Path:
    path.write_text(json.dumps({"name": name, "tasks": tasks}), encoding="utf-8")
    return path


def make_tasks(*ids: str, input_prefix: str = "do ") -> list[dict]:
    return [{"id": task_id, "input": f"{input_prefix}{task_id}"} for task_id in ids]


def run(
    tmp_path: Path,
    *,
    baseline: str | bytes = "BASELINE\n",
    compiled: str | bytes = "COMPILED\n",
    tasks: list[dict] | None = None,
    evaluator: object | None = None,
    model: str = "test-model",
    name: str = "holdout",
) -> tuple[dict, Path, Path, Path]:
    bench = write_holdout(tmp_path / "holdout.json", tasks if tasks is not None else [], name=name)
    base = write_prompt(tmp_path / "baseline.md", baseline)
    comp = write_prompt(tmp_path / "compiled.md", compiled)
    result = run_experiment(
        model=model,
        benchmark_path=bench,
        baseline_prompt_path=base,
        compiled_prompt_path=comp,
        evaluator=evaluator if evaluator is not None else ConstEvaluator(),
    )
    return result, bench, base, comp


def config_without_prompt(config: EvaluationConfig) -> dict:
    data = asdict(config)
    del data["prompt"]
    return data


def test_same_inputs_same_hashes_and_task_order(tmp_path: Path):
    tasks = make_tasks("agent_017", "agent_018")
    evaluator = ConstEvaluator()
    first, _, base, comp = run(tmp_path, tasks=tasks, evaluator=evaluator)
    second = run_experiment(
        model="test-model",
        benchmark_path=tmp_path / "holdout.json",
        baseline_prompt_path=base,
        compiled_prompt_path=comp,
        evaluator=ConstEvaluator(),
    )
    assert first["baseline"]["prompt_sha256"] == second["baseline"]["prompt_sha256"]
    assert first["compiled"]["prompt_sha256"] == second["compiled"]["prompt_sha256"]
    assert [item["task_id"] for item in first["tasks"]] == ["agent_017", "agent_018"]
    assert [item["task_id"] for item in second["tasks"]] == ["agent_017", "agent_018"]
    assert first == second
    dumped = json.dumps(first, indent=2, sort_keys=True) + "\n"
    assert json.dumps(second, indent=2, sort_keys=True) + "\n" == dumped


def test_changing_only_compiled_bytes_changes_compiled_hash(tmp_path: Path):
    tasks = make_tasks("t1")
    result_a, _, base, comp = run(
        tmp_path,
        baseline=b"same-baseline\n",
        compiled=b"compiled-v1\n",
        tasks=tasks,
        evaluator=ConstEvaluator(),
    )
    baseline_hash = result_a["baseline"]["prompt_sha256"]
    compiled_hash = result_a["compiled"]["prompt_sha256"]
    comp.write_bytes(b"compiled-v2\n")
    result_b = run_experiment(
        model="test-model",
        benchmark_path=tmp_path / "holdout.json",
        baseline_prompt_path=base,
        compiled_prompt_path=comp,
        evaluator=ConstEvaluator(),
    )
    assert result_b["baseline"]["prompt_sha256"] == baseline_hash
    assert result_b["compiled"]["prompt_sha256"] != compiled_hash
    assert result_b["baseline"]["prompt_sha256"] == file_sha256(base)
    assert result_b["compiled"]["prompt_sha256"] == file_sha256(comp)


@pytest.mark.parametrize("n", [0, 1, 3])
def test_n_tasks_pair_one_to_one_no_missing_or_dup_ids(tmp_path: Path, n: int):
    tasks = make_tasks(*[f"t{i}" for i in range(n)])
    evaluator = ConstEvaluator()
    result, _, _, _ = run(tmp_path, tasks=tasks, evaluator=evaluator)
    ids = [item["task_id"] for item in result["tasks"]]
    assert ids == [task["id"] for task in tasks]
    assert len(ids) == n
    assert len(set(ids)) == n
    assert len(evaluator.configs) == 2 * n
    baseline_ids = [cfg.task_id for cfg in evaluator.configs[0::2]]
    compiled_ids = [cfg.task_id for cfg in evaluator.configs[1::2]]
    assert baseline_ids == ids
    assert compiled_ids == ids
    assert result["baseline"]["passed"] + result["baseline"]["failed"] == n
    assert result["compiled"]["passed"] + result["compiled"]["failed"] == n


def test_comparison_matrix(tmp_path: Path):
    baseline = "BASE\n"
    compiled = "COMP\n"
    tasks = [
        {"id": "false_true", "input": "ft"},
        {"id": "true_false", "input": "tf"},
        {"id": "true_true", "input": "tt"},
        {"id": "false_false", "input": "ff"},
    ]
    outcomes = {
        (baseline, "false_true"): (False, 0.0),
        (compiled, "false_true"): (True, 1.0),
        (baseline, "true_false"): (True, 1.0),
        (compiled, "true_false"): (False, 0.0),
        (baseline, "true_true"): (True, 1.0),
        (compiled, "true_true"): (True, 1.0),
        (baseline, "false_false"): (False, 0.0),
        (compiled, "false_false"): (False, 0.0),
    }
    result, _, _, _ = run(
        tmp_path,
        baseline=baseline,
        compiled=compiled,
        tasks=tasks,
        evaluator=FakeEvaluator(outcomes),
    )
    by_id = {item["task_id"]: item for item in result["tasks"]}
    assert by_id["false_true"]["comparison"] == "improved"
    assert by_id["true_false"]["comparison"] == "regressed"
    assert by_id["true_true"]["comparison"] == "unchanged"
    assert by_id["false_false"]["comparison"] == "unchanged"
    assert [item["comparison"] for item in result["tasks"]] == [
        "improved",
        "regressed",
        "unchanged",
        "unchanged",
    ]


@pytest.mark.parametrize(
    ("baseline_scores", "compiled_scores", "expected_delta"),
    [
        ((0.0, 0.0), (1.0, 1.0), 2.0),
        ((1.0, 1.0), (0.0, 0.0), -2.0),
        ((1.0, 0.0), (0.5, 0.5), 0.0),
    ],
)
def test_score_delta(tmp_path: Path, baseline_scores, compiled_scores, expected_delta):
    baseline = "BASE\n"
    compiled = "COMP\n"
    tasks = make_tasks("a", "b")
    outcomes = {
        (baseline, "a"): (True, baseline_scores[0]),
        (baseline, "b"): (True, baseline_scores[1]),
        (compiled, "a"): (True, compiled_scores[0]),
        (compiled, "b"): (True, compiled_scores[1]),
    }
    result, _, _, _ = run(
        tmp_path,
        baseline=baseline,
        compiled=compiled,
        tasks=tasks,
        evaluator=FakeEvaluator(outcomes),
    )
    assert result["delta"]["score"] == pytest.approx(expected_delta)
    assert result["compiled"]["score"] - result["baseline"]["score"] == pytest.approx(expected_delta)


def test_sha256_identical_bytes_same_one_byte_change_different(tmp_path: Path):
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_bytes(b"hello\n")
    b.write_bytes(b"hello\n")
    assert file_sha256(a) == file_sha256(b)
    assert file_sha256(a) == hashlib.sha256(b"hello\n").hexdigest()
    b.write_bytes(b"hello\n!")
    assert file_sha256(a) != file_sha256(b)
    spaced = tmp_path / "spaced.md"
    spaced.write_bytes(b"hello\n ")
    assert file_sha256(spaced) != file_sha256(a)


def test_evaluation_config_equal_except_prompt(tmp_path: Path):
    baseline = "BASELINE PROMPT\n"
    compiled = "COMPILED PROMPT\n"
    tasks = [
        {"id": "agent_017", "input": "  keep whitespace\nand lines\n"},
        {"id": "agent_018", "input": "second"},
    ]
    evaluator = ConstEvaluator()
    result, _, _, _ = run(
        tmp_path,
        baseline=baseline,
        compiled=compiled,
        tasks=tasks,
        evaluator=evaluator,
        model="gemma4:12b",
    )
    assert len(evaluator.configs) == 4
    for index, task in enumerate(tasks):
        a = evaluator.configs[2 * index]
        b = evaluator.configs[2 * index + 1]
        assert config_without_prompt(a) == config_without_prompt(b)
        assert a.prompt == baseline
        assert b.prompt == compiled
        assert a.prompt != b.prompt
        assert a.model == b.model == "gemma4:12b"
        assert a.task_id == b.task_id == task["id"]
        assert a.task_input == b.task_input == task["input"]
        assert a.temperature == b.temperature == DEFAULT_TEMPERATURE
        assert a.num_ctx == b.num_ctx == NUM_CTX
        assert a.num_predict == b.num_predict == NUM_PREDICT
        assert result["tasks"][index]["task_id"] == task["id"]
        assert result["tasks"][index]["baseline"]["passed"] is True
        assert result["tasks"][index]["compiled"]["passed"] is True
    assert result["knobs"] == {
        "num_ctx": NUM_CTX,
        "num_predict": NUM_PREDICT,
        "temperature": DEFAULT_TEMPERATURE,
    }


def test_identical_prompts_unchanged_zero_delta(tmp_path: Path):
    prompt = "SAME PROMPT\n"
    tasks = make_tasks("t1", "t2")
    evaluator = ConstEvaluator(passed=True, score=1.0)
    result, _, base, comp = run(
        tmp_path,
        baseline=prompt,
        compiled=prompt,
        tasks=tasks,
        evaluator=evaluator,
    )
    assert file_sha256(base) == file_sha256(comp)
    assert result["baseline"]["prompt_sha256"] == result["compiled"]["prompt_sha256"]
    assert all(item["comparison"] == "unchanged" for item in result["tasks"])
    assert result["delta"]["score"] == 0.0
    assert result["delta"]["passed"] == 0
    for a, b in zip(evaluator.configs[0::2], evaluator.configs[1::2], strict=True):
        assert a.prompt == b.prompt == prompt


def test_delta_passed_counts(tmp_path: Path):
    baseline = "A\n"
    compiled = "B\n"
    tasks = make_tasks("x", "y", "z")
    outcomes = {
        (baseline, "x"): (False, 0.0),
        (compiled, "x"): (True, 1.0),
        (baseline, "y"): (True, 1.0),
        (compiled, "y"): (True, 1.0),
        (baseline, "z"): (False, 0.0),
        (compiled, "z"): (False, 0.0),
    }
    result, _, _, _ = run(
        tmp_path,
        baseline=baseline,
        compiled=compiled,
        tasks=tasks,
        evaluator=FakeEvaluator(outcomes),
    )
    assert result["baseline"]["passed"] == 1
    assert result["baseline"]["failed"] == 2
    assert result["compiled"]["passed"] == 2
    assert result["compiled"]["failed"] == 1
    assert result["delta"]["passed"] == 1
    assert result["delta"]["score"] == pytest.approx(1.0)


def test_empty_prompt_file_error(tmp_path: Path):
    write_holdout(tmp_path / "holdout.json", [])
    write_prompt(tmp_path / "baseline.md", "ok\n")
    write_prompt(tmp_path / "compiled.md", b"")
    with pytest.raises(ExperimentError, match="empty prompt file"):
        run_experiment(
            model="m",
            benchmark_path=tmp_path / "holdout.json",
            baseline_prompt_path=tmp_path / "baseline.md",
            compiled_prompt_path=tmp_path / "compiled.md",
            evaluator=ConstEvaluator(),
        )


def test_missing_prompt_file_error(tmp_path: Path):
    write_holdout(tmp_path / "holdout.json", [])
    write_prompt(tmp_path / "baseline.md", "ok\n")
    with pytest.raises(ExperimentError, match="cannot read prompt file"):
        run_experiment(
            model="m",
            benchmark_path=tmp_path / "holdout.json",
            baseline_prompt_path=tmp_path / "baseline.md",
            compiled_prompt_path=tmp_path / "missing.md",
            evaluator=ConstEvaluator(),
        )


def test_invalid_benchmark_json_error(tmp_path: Path):
    bad = tmp_path / "holdout.json"
    bad.write_text("{not json", encoding="utf-8")
    write_prompt(tmp_path / "baseline.md", "ok\n")
    write_prompt(tmp_path / "compiled.md", "ok\n")
    with pytest.raises(ExperimentError, match="invalid JSON"):
        run_experiment(
            model="m",
            benchmark_path=bad,
            baseline_prompt_path=tmp_path / "baseline.md",
            compiled_prompt_path=tmp_path / "compiled.md",
            evaluator=ConstEvaluator(),
        )


def test_missing_tasks_array_error(tmp_path: Path):
    (tmp_path / "holdout.json").write_text(json.dumps({"name": "holdout"}), encoding="utf-8")
    write_prompt(tmp_path / "baseline.md", "ok\n")
    write_prompt(tmp_path / "compiled.md", "ok\n")
    with pytest.raises(ExperimentError, match="tasks"):
        run_experiment(
            model="m",
            benchmark_path=tmp_path / "holdout.json",
            baseline_prompt_path=tmp_path / "baseline.md",
            compiled_prompt_path=tmp_path / "compiled.md",
            evaluator=ConstEvaluator(),
        )


def test_load_holdout_empty_tasks(tmp_path: Path):
    path = write_holdout(tmp_path / "holdout.json", [])
    holdout = load_holdout(path)
    assert holdout.name == "holdout"
    assert holdout.tasks == ()


def test_result_has_no_datetime_or_hostname(tmp_path: Path):
    result, _, _, _ = run(tmp_path, tasks=make_tasks("t1"), evaluator=ConstEvaluator())
    blob = json.dumps(result)
    assert "datetime" not in blob.lower()
    assert "hostname" not in blob.lower()
    assert "T" not in "".join(result["notes"])
    assert result["version"] == 1
    assert result["model"] == "test-model"
    assert result["benchmark"] == "holdout"
    assert any("seed" in note.lower() for note in result["notes"])


def test_cli_experiment_writes_sorted_json(tmp_path: Path, capsys):
    write_holdout(tmp_path / "holdout.json", make_tasks("t1"))
    write_prompt(tmp_path / "baseline.md", "BASE\n")
    write_prompt(tmp_path / "compiled.md", "COMP\n")
    out_path = tmp_path / "nested" / "experiment.json"
    code = main(
        [
            "experiment",
            "--model",
            "test-model",
            "--benchmark",
            str(tmp_path / "holdout.json"),
            "--baseline-prompt",
            str(tmp_path / "baseline.md"),
            "--compiled-prompt",
            str(tmp_path / "compiled.md"),
            "--output",
            str(out_path),
        ],
        evaluator=ConstEvaluator(),
    )
    assert code == 0
    assert capsys.readouterr().out == ""
    text = out_path.read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert text == json.dumps(parsed, indent=2, sort_keys=True) + "\n"
    assert parsed["version"] == 1
    assert parsed["tasks"][0]["task_id"] == "t1"


def test_cli_invalid_benchmark_does_not_write(tmp_path: Path, capsys):
    bad = tmp_path / "holdout.json"
    bad.write_text("{", encoding="utf-8")
    write_prompt(tmp_path / "baseline.md", "BASE\n")
    write_prompt(tmp_path / "compiled.md", "COMP\n")
    out_path = tmp_path / "out" / "experiment.json"
    code = main(
        [
            "experiment",
            "--model",
            "m",
            "--benchmark",
            str(bad),
            "--baseline-prompt",
            str(tmp_path / "baseline.md"),
            "--compiled-prompt",
            str(tmp_path / "compiled.md"),
            "--output",
            str(out_path),
        ],
        evaluator=ConstEvaluator(),
    )
    assert code == 1
    assert "invalid JSON" in capsys.readouterr().err
    assert not out_path.exists()


def test_cli_missing_prompt_does_not_write(tmp_path: Path, capsys):
    write_holdout(tmp_path / "holdout.json", [])
    write_prompt(tmp_path / "baseline.md", "BASE\n")
    out_path = tmp_path / "experiment.json"
    code = main(
        [
            "experiment",
            "--model",
            "m",
            "--benchmark",
            str(tmp_path / "holdout.json"),
            "--baseline-prompt",
            str(tmp_path / "baseline.md"),
            "--compiled-prompt",
            str(tmp_path / "missing.md"),
            "--output",
            str(out_path),
        ],
        evaluator=ConstEvaluator(),
    )
    assert code == 1
    assert "cannot read prompt file" in capsys.readouterr().err
    assert not out_path.exists()


def test_cli_empty_prompt_does_not_write(tmp_path: Path, capsys):
    write_holdout(tmp_path / "holdout.json", [])
    write_prompt(tmp_path / "baseline.md", "BASE\n")
    write_prompt(tmp_path / "compiled.md", b"")
    out_path = tmp_path / "experiment.json"
    code = main(
        [
            "experiment",
            "--model",
            "m",
            "--benchmark",
            str(tmp_path / "holdout.json"),
            "--baseline-prompt",
            str(tmp_path / "baseline.md"),
            "--compiled-prompt",
            str(tmp_path / "compiled.md"),
            "--output",
            str(out_path),
        ],
        evaluator=ConstEvaluator(),
    )
    assert code == 1
    assert "empty prompt file" in capsys.readouterr().err
    assert not out_path.exists()


def test_compile_cli_still_dispatches(tmp_path: Path, capsys):
    fitness = {
        "model": "test-model",
        "dimensions": {
            "hallucination_resistance": 8.0,
            "code_reasoning": 8.0,
            "instruction_following": 8.0,
            "agent_discipline": 8.0,
            "self_correction": 8.0,
        },
    }
    fitness_path = tmp_path / "f.json"
    fitness_path.write_text(json.dumps(fitness), encoding="utf-8")
    task = tmp_path / "task.md"
    task.write_text("Add a health endpoint.\n", encoding="utf-8")
    code = main(["--fitness", str(fitness_path), "--task", str(task)])
    assert code == 0
    assert capsys.readouterr().out.startswith("ROLE\n")


class _StubChat:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[tuple[str, list[dict[str, str]], float]] = []

    def chat(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0):
        self.calls.append((model, messages, temperature))
        return self


def test_ollama_chat_evaluator_nonempty_content_no_client_instance():
    stub = _StubChat("  hello  ")
    evaluator = OllamaChatEvaluator(client=stub)
    config = EvaluationConfig(
        model="m",
        prompt="SYS",
        task_id="t1",
        task_input="user-task",
    )
    outcome = evaluator.evaluate(config)
    assert outcome == TaskOutcome(task_id="t1", passed=True, score=1.0)
    assert stub.calls == [
        (
            "m",
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "user-task"},
            ],
            DEFAULT_TEMPERATURE,
        )
    ]


def test_ollama_chat_evaluator_blank_content_fails():
    stub = _StubChat("   \n")
    evaluator = OllamaChatEvaluator(client=stub)
    outcome = evaluator.evaluate(
        EvaluationConfig(model="m", prompt="SYS", task_id="t1", task_input="x")
    )
    assert outcome.passed is False
    assert outcome.score == 0.0
