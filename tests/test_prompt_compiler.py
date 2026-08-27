import json
import subprocess
import sys
from pathlib import Path

import pytest

from compiler.analyzer import CompilerError, analyze, load_fitness
from compiler.cli import main
from compiler.compiler import compile_prompt
from compiler.rules import FAILURE_INSTRUCTIONS, INVENTED_API_INSTRUCTION, load_rules
from compiler.templates import (
    COMPLETION_BASE,
    COMPLETION_TESTS,
    HEADERS,
    NEUTRAL_IMPLEMENTATION,
    NEUTRAL_REQUIREMENTS,
    NEUTRAL_VERIFICATION,
    NO_EXTRA_INSTRUCTIONS,
    render_explain,
)
from compiler.yaml_util import load_yaml, load_yaml_file

ROOT = Path(__file__).resolve().parents[1]


def make_fitness(**overrides):
    data = {
        "model": "test-model",
        "ollama_version": "0.0.0",
        "context": {"effective": 8192, "model_max": None},
        "mode": "full",
        "tests": [],
        "total": 0,
        "total_max": 100,
        "penalty": 0,
        "dimensions": {
            "hallucination_resistance": 8.0,
            "code_reasoning": 8.0,
            "instruction_following": 8.0,
            "agent_discipline": 8.0,
            "self_correction": 8.0,
        },
        "verdict": None,
        "avg_response_s": None,
        "tokens_per_sec": None,
        "critical_hallucination": False,
        "invented_apis": [],
        "extra": {},
        "schema_version": 2,
    }
    if "dimensions" in overrides:
        dims = dict(data["dimensions"])
        extra_dims = overrides.pop("dimensions")
        if extra_dims is None:
            del data["dimensions"]
        else:
            dims.update(extra_dims)
            data["dimensions"] = dims
    data.update(overrides)
    return data


def failed_test(test_id, *, scored=True, detail="", invented_apis=None, critical=False):
    return {
        "id": str(test_id),
        "name": f"test {test_id}",
        "stage": 1,
        "passed": False,
        "points": 0.0,
        "max_points": 12.0,
        "detail": detail,
        "checks": {},
        "critical_hallucination": critical,
        "invented_apis": invented_apis or [],
        "scored": scored,
    }


def write_fitness(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_task(path: Path, text: str = "Add a health endpoint.\n") -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def write_rules_dir(tmp_path: Path, *, skip=(), extra_files=None) -> Path:
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "thresholds.yaml").write_text(
        "thresholds:\n  strong: 0.75\n  weak: 0.60\n",
        encoding="utf-8",
    )
    catalog = {
        "agent_discipline": "AGENT_DISC_RULE",
        "code_reasoning": "CODE_REASON_RULE",
        "hallucination_resistance": "HALLUC_RULE",
        "instruction_following": "INSTR_FOLLOW_RULE",
        "self_correction": "SELF_CORR_RULE",
    }
    for name, instr in catalog.items():
        if name in skip:
            continue
        (rules / f"{name}.yaml").write_text(
            f'description: "{name}"\ninstructions:\n  - "{instr}"\n',
            encoding="utf-8",
        )
    for fname, content in (extra_files or {}).items():
        (rules / fname).write_text(content, encoding="utf-8")
    return rules


def compile_tmp(tmp_path: Path, data: dict, task: str = "Add a health endpoint.\n", rules_dir=None):
    fitness = write_fitness(tmp_path / "fitness.json", data)
    task_path = write_task(tmp_path / "task.md", task)
    return compile_prompt(fitness, task_path, rules_dir=rules_dir)


def test_yaml_subset_mappings_and_lists():
    text = (
        "thresholds:\n"
        "  strong: 0.75\n"
        '  weak: 0.60\n'
        "instructions:\n"
        '  - "alpha"\n'
        "  - beta\n"
        'description: "quoted"\n'
    )
    data = load_yaml(text)
    assert data["thresholds"]["strong"] == 0.75
    assert data["thresholds"]["weak"] == pytest.approx(0.6)
    assert data["instructions"] == ["alpha", "beta"]
    assert data["description"] == "quoted"


def test_shipped_rules_load():
    rules = load_rules()
    assert rules.thresholds.strong == 0.75
    assert rules.thresholds.weak == pytest.approx(0.6)
    assert rules.categories["self_correction"].instructions
    assert load_yaml_file(ROOT / "rules" / "instruction_following.yaml")["instructions"]


def test_load_valid_json(tmp_path: Path):
    path = write_fitness(tmp_path / "ok.json", make_fitness())
    data = load_fitness(path)
    assert data["model"] == "test-model"
    assert isinstance(data["dimensions"], dict)


def test_load_invalid_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CompilerError, match="invalid JSON"):
        load_fitness(path)


def test_load_missing_model(tmp_path: Path):
    data = make_fitness()
    del data["model"]
    path = write_fitness(tmp_path / "no_model.json", data)
    with pytest.raises(CompilerError, match="model"):
        load_fitness(path)


def test_load_missing_dimensions(tmp_path: Path):
    data = make_fitness()
    del data["dimensions"]
    path = write_fitness(tmp_path / "no_dims.json", data)
    with pytest.raises(CompilerError, match="dimensions"):
        load_fitness(path)


def test_load_dimensions_not_dict(tmp_path: Path):
    data = make_fitness()
    data["dimensions"] = ["nope"]
    path = write_fitness(tmp_path / "dims.json", data)
    with pytest.raises(CompilerError, match="dimensions"):
        load_fitness(path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (10.0, "strength"),
        (7.5, "strength"),
        (7.49, "neutral"),
        (6.0, "neutral"),
        (6.25, "neutral"),
        (5.99, "weakness"),
        (0.0, "weakness"),
    ],
)
def test_classification_boundaries(raw, expected):
    rules = load_rules()
    profile = analyze(make_fitness(dimensions={"instruction_following": raw}), rules.thresholds)
    assert profile.scores["instruction_following"] == pytest.approx(raw / 10.0)
    assert profile.classifications["instruction_following"] == expected
    if expected == "strength":
        assert "instruction_following" in profile.strengths
        assert "instruction_following" not in profile.weaknesses
    elif expected == "weakness":
        assert "instruction_following" in profile.weaknesses
        assert "instruction_following" not in profile.strengths
    else:
        assert "instruction_following" not in profile.strengths
        assert "instruction_following" not in profile.weaknesses


def test_null_dimension_omitted():
    rules = load_rules()
    data = make_fitness()
    data["dimensions"] = {
        "code_reasoning": None,
        "agent_discipline": 10.0,
        "hallucination_resistance": None,
        "instruction_following": None,
        "self_correction": None,
    }
    profile = analyze(data, rules.thresholds)
    assert "code_reasoning" not in profile.scores
    assert "code_reasoning" not in profile.classifications
    assert profile.strengths == ["agent_discipline"]
    assert profile.weaknesses == []


def test_unknown_dimension_ignored():
    rules = load_rules()
    profile = analyze(make_fitness(dimensions={"planning": 0.0, "autonomy": 10.0}), rules.thresholds)
    assert "planning" not in profile.classifications
    assert "autonomy" not in profile.classifications
    assert set(profile.classifications) <= {
        "agent_discipline",
        "code_reasoning",
        "hallucination_resistance",
        "instruction_following",
        "self_correction",
    }


def test_strengths_and_weaknesses_sorted():
    rules = load_rules()
    profile = analyze(
        make_fitness(
            dimensions={
                "self_correction": 0.0,
                "agent_discipline": 0.0,
                "hallucination_resistance": 10.0,
                "code_reasoning": 10.0,
                "instruction_following": 7.0,
            }
        ),
        rules.thresholds,
    )
    assert profile.strengths == ["code_reasoning", "hallucination_resistance"]
    assert profile.weaknesses == ["agent_discipline", "self_correction"]


def test_unknown_capability_yaml_ignored(tmp_path: Path):
    rules_dir = write_rules_dir(
        tmp_path,
        extra_files={"planning.yaml": 'instructions:\n  - "SHOULD_NOT_APPEAR"\n'},
    )
    result = compile_tmp(
        tmp_path,
        make_fitness(dimensions={"instruction_following": 0.0}),
        rules_dir=rules_dir,
    )
    assert "SHOULD_NOT_APPEAR" not in result.prompt
    assert "INSTR_FOLLOW_RULE" in result.prompt


def test_missing_category_yaml_still_classifies(tmp_path: Path):
    rules_dir = write_rules_dir(tmp_path, skip=("code_reasoning",))
    result = compile_tmp(
        tmp_path,
        make_fitness(dimensions={"code_reasoning": 0.0}),
        rules_dir=rules_dir,
    )
    assert result.profile.classifications["code_reasoning"] == "weakness"
    assert "CODE_REASON_RULE" not in result.prompt
    assert "code_reasoning" in result.profile.weaknesses


def test_empty_failures_no_extra_instructions(tmp_path: Path):
    rules_dir = write_rules_dir(tmp_path)
    result = compile_tmp(
        tmp_path,
        make_fitness(dimensions={"self_correction": 0.0}, tests=[]),
        rules_dir=rules_dir,
    )
    assert FAILURE_INSTRUCTIONS["7"] not in result.prompt
    assert INVENTED_API_INSTRUCTION not in result.prompt
    assert "SELF_CORR_RULE" in result.prompt


def test_rule_selection_only_weaknesses(tmp_path: Path):
    rules_dir = write_rules_dir(tmp_path)
    result = compile_tmp(
        tmp_path,
        make_fitness(
            dimensions={
                "instruction_following": 0.0,
                "agent_discipline": 10.0,
                "code_reasoning": 8.0,
                "hallucination_resistance": 8.0,
                "self_correction": 8.0,
            }
        ),
        rules_dir=rules_dir,
    )
    assert "INSTR_FOLLOW_RULE" in result.prompt
    assert "AGENT_DISC_RULE" not in result.prompt
    assert "CODE_REASON_RULE" not in result.prompt
    assert "HALLUC_RULE" not in result.prompt
    assert "SELF_CORR_RULE" not in result.prompt
    assert NEUTRAL_REQUIREMENTS not in result.prompt
    assert NEUTRAL_IMPLEMENTATION in result.prompt


def test_failure_handling_known_table(tmp_path: Path):
    rules_dir = write_rules_dir(tmp_path)
    result = compile_tmp(
        tmp_path,
        make_fitness(
            dimensions={"self_correction": 0.0},
            tests=[failed_test("7", detail="UNIQUE_DETAIL_TOKEN_XYZ")],
        ),
        rules_dir=rules_dir,
    )
    assert "SELF_CORR_RULE" in result.prompt
    assert result.prompt.count("SELF_CORR_RULE") == 2  # operating + VERIFICATION
    assert "UNIQUE_DETAIL_TOKEN_XYZ" not in result.prompt
    assert COMPLETION_TESTS in result.prompt


def test_unknown_test_id_ignored(tmp_path: Path):
    rules_dir = write_rules_dir(tmp_path)
    result = compile_tmp(
        tmp_path,
        make_fitness(
            dimensions={"self_correction": 0.0},
            tests=[failed_test("99", detail="nope")],
        ),
        rules_dir=rules_dir,
    )
    assert "nope" not in result.prompt
    assert FAILURE_INSTRUCTIONS["7"] not in result.prompt


def test_unscored_failure_ignored(tmp_path: Path):
    rules_dir = write_rules_dir(tmp_path)
    result = compile_tmp(
        tmp_path,
        make_fitness(
            dimensions={"self_correction": 0.0},
            tests=[failed_test("7", scored=False)],
        ),
        rules_dir=rules_dir,
    )
    assert FAILURE_INSTRUCTIONS["7"] not in result.prompt


def test_scored_fail_adds_table_extra_even_if_not_weakness(tmp_path: Path):
    rules_dir = write_rules_dir(tmp_path)
    result = compile_tmp(
        tmp_path,
        make_fitness(
            dimensions={"self_correction": 10.0},
            tests=[failed_test("7")],
        ),
        rules_dir=rules_dir,
    )
    assert "SELF_CORR_RULE" in result.selected.operating
    assert result.selected.by_category["self_correction"] == ()
    assert "SELF_CORR_RULE" in result.selected.extras
    ver = result.prompt.split("VERIFICATION\n\n", 1)[1].split("\n\nCOMPLETION", 1)[0]
    assert NEUTRAL_VERIFICATION in ver
    assert "SELF_CORR_RULE" not in ver
    assert COMPLETION_TESTS not in result.prompt


def test_invented_apis_adds_known_instruction(tmp_path: Path):
    rules_dir = write_rules_dir(tmp_path)
    result = compile_tmp(
        tmp_path,
        make_fitness(
            dimensions={"hallucination_resistance": 10.0},
            invented_apis=["session.atomic()"],
            critical_hallucination=True,
        ),
        rules_dir=rules_dir,
    )
    assert INVENTED_API_INSTRUCTION in result.prompt
    assert INVENTED_API_INSTRUCTION in result.selected.operating
    assert INVENTED_API_INSTRUCTION in result.selected.extras
    assert result.selected.by_category["hallucination_resistance"] == ()
    assert "HALLUC_RULE" not in result.prompt
    assert "session.atomic()" not in result.prompt
    ver = result.prompt.split("VERIFICATION\n\n", 1)[1].split("\n\nCOMPLETION", 1)[0]
    assert NEUTRAL_VERIFICATION in ver
    assert INVENTED_API_INSTRUCTION not in ver


def test_duplicate_failure_instructions_deduped(tmp_path: Path):
    rules_dir = write_rules_dir(tmp_path)
    result = compile_tmp(
        tmp_path,
        make_fitness(
            dimensions={"code_reasoning": 0.0},
            tests=[failed_test("2"), failed_test("3"), failed_test("4")],
        ),
        rules_dir=rules_dir,
    )
    assert result.selected.operating.count("CODE_REASON_RULE") == 1
    impl = result.prompt.split("IMPLEMENTATION GUIDANCE\n\n", 1)[1].split("\n\nVERIFICATION", 1)[0]
    assert impl.count("CODE_REASON_RULE") == 1


def test_compilation_headers_and_verbatim_task(tmp_path: Path):
    rules_dir = write_rules_dir(tmp_path)
    task = "Use the exact phrase TASK_PAYLOAD_42.\nDo not rewrite me."
    result = compile_tmp(tmp_path, make_fitness(), task=task, rules_dir=rules_dir)
    for header in HEADERS:
        assert f"\n{header}\n" in f"\n{result.prompt}"
    assert result.prompt.startswith("ROLE\n")
    assert "Use the exact phrase TASK_PAYLOAD_42.\nDo not rewrite me." in result.prompt
    assert "You are an autonomous coding agent." in result.prompt
    assert COMPLETION_BASE in result.prompt
    assert NEUTRAL_REQUIREMENTS in result.prompt
    assert NEUTRAL_IMPLEMENTATION in result.prompt
    assert NEUTRAL_VERIFICATION in result.prompt
    assert NO_EXTRA_INSTRUCTIONS in result.prompt
    assert COMPLETION_TESTS not in result.prompt


def test_determinism_compile_twice(tmp_path: Path):
    rules_dir = write_rules_dir(tmp_path)
    data = make_fitness(
        dimensions={"self_correction": 0.0, "agent_discipline": 5.0},
        tests=[failed_test("6"), failed_test("7")],
    )
    a = compile_tmp(tmp_path, data, rules_dir=rules_dir)
    b = compile_tmp(tmp_path, data, rules_dir=rules_dir)
    assert a.prompt == b.prompt
    assert a.profile.to_dict() == b.profile.to_dict()


def test_section_rules_follow_weaknesses(tmp_path: Path):
    rules_dir = write_rules_dir(tmp_path)
    result = compile_tmp(
        tmp_path,
        make_fitness(
            dimensions={
                "instruction_following": 0.0,
                "agent_discipline": 0.0,
                "code_reasoning": 0.0,
                "self_correction": 0.0,
                "hallucination_resistance": 10.0,
            }
        ),
        rules_dir=rules_dir,
    )
    req = result.prompt.split("REQUIREMENT HANDLING\n\n", 1)[1].split("\n\nIMPLEMENTATION", 1)[0]
    impl = result.prompt.split("IMPLEMENTATION GUIDANCE\n\n", 1)[1].split("\n\nVERIFICATION", 1)[0]
    ver = result.prompt.split("VERIFICATION\n\n", 1)[1].split("\n\nCOMPLETION", 1)[0]
    assert "INSTR_FOLLOW_RULE" in req
    assert "AGENT_DISC_RULE" in impl
    assert "CODE_REASON_RULE" in impl
    assert impl.index("AGENT_DISC_RULE") < impl.index("CODE_REASON_RULE")
    assert "SELF_CORR_RULE" in ver
    assert "HALLUC_RULE" not in ver


def test_cli_stdout(tmp_path: Path, capsys):
    fitness = write_fitness(tmp_path / "f.json", make_fitness())
    task = write_task(tmp_path / "task.md")
    code = main(["--fitness", str(fitness), "--task", str(task)])
    assert code == 0
    out = capsys.readouterr().out
    assert out.startswith("ROLE\n")
    assert "Add a health endpoint." in out


def test_cli_output_file(tmp_path: Path, capsys):
    fitness = write_fitness(tmp_path / "f.json", make_fitness())
    task = write_task(tmp_path / "task.md")
    out_path = tmp_path / "nested" / "prompt.md"
    code = main(["--fitness", str(fitness), "--task", str(task), "--output", str(out_path)])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    text = out_path.read_text(encoding="utf-8")
    assert text.startswith("ROLE\n")
    assert compile_prompt(fitness, task).prompt == text


def test_cli_invalid_json_does_not_write(tmp_path: Path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    task = write_task(tmp_path / "task.md")
    out_path = tmp_path / "out" / "prompt.md"
    code = main(["--fitness", str(bad), "--task", str(task), "--output", str(out_path)])
    assert code == 1
    err = capsys.readouterr().err
    assert "invalid JSON" in err
    assert not out_path.exists()


def test_cli_missing_fields(tmp_path: Path, capsys):
    data = make_fitness()
    del data["model"]
    fitness = write_fitness(tmp_path / "f.json", data)
    task = write_task(tmp_path / "task.md")
    out_path = tmp_path / "prompt.md"
    code = main(["--fitness", str(fitness), "--task", str(task), "--output", str(out_path)])
    assert code == 1
    assert "model" in capsys.readouterr().err
    assert not out_path.exists()


def test_cli_format_error(tmp_path: Path, capsys):
    fitness = write_fitness(tmp_path / "f.json", make_fitness())
    task = write_task(tmp_path / "task.md")
    code = main(["--fitness", str(fitness), "--task", str(task), "--format", "html"])
    assert code == 1
    assert "markdown" in capsys.readouterr().err.lower()


def test_cli_profile_and_explain_do_not_change_prompt(tmp_path: Path, capsys):
    data = make_fitness(dimensions={"self_correction": 0.0})
    fitness = write_fitness(tmp_path / "f.json", data)
    task = write_task(tmp_path / "task.md")
    expected = compile_prompt(fitness, task).prompt
    out_path = tmp_path / "prompt.md"
    code = main(
        [
            "--fitness",
            str(fitness),
            "--task",
            str(task),
            "--output",
            str(out_path),
            "--profile",
            "--explain",
        ]
    )
    assert code == 0
    assert out_path.read_text(encoding="utf-8") == expected
    stdout = capsys.readouterr().out
    profile_text, explain_text = stdout.split("\nCategory:", 1)
    profile = json.loads(profile_text)
    assert profile["model"] == "test-model"
    assert profile["classifications"]["self_correction"] == "weakness"
    assert list(profile.keys()) == sorted(profile.keys())
    assert "Category: self_correction" in "Category:" + explain_text
    assert "Classification: weakness" in explain_text
    assert "Score:" in explain_text
    assert "Applied rules:" in explain_text


def test_cli_profile_sort_keys(tmp_path: Path, capsys):
    fitness = write_fitness(tmp_path / "f.json", make_fitness())
    task = write_task(tmp_path / "task.md")
    out_path = tmp_path / "prompt.md"
    assert main(["--fitness", str(fitness), "--task", str(task), "--output", str(out_path), "--profile"]) == 0
    dumped = capsys.readouterr().out
    parsed = json.loads(dumped)
    assert dumped == json.dumps(parsed, indent=2, sort_keys=True) + "\n"


def test_partial_credit_fail_still_adds_table_extra(tmp_path: Path):
    rules_dir = write_rules_dir(tmp_path)
    result = compile_tmp(
        tmp_path,
        make_fitness(
            dimensions={"hallucination_resistance": 7.5},
            tests=[failed_test("8")],
        ),
        rules_dir=rules_dir,
    )
    assert result.profile.classifications["hallucination_resistance"] == "strength"
    assert "HALLUC_RULE" in result.selected.operating
    assert result.selected.by_category["hallucination_resistance"] == ()
    ver = result.prompt.split("VERIFICATION\n\n", 1)[1].split("\n\nCOMPLETION", 1)[0]
    assert NEUTRAL_VERIFICATION in ver
    assert "HALLUC_RULE" not in ver


def test_invented_api_with_omitted_hall_dimension(tmp_path: Path):
    rules_dir = write_rules_dir(tmp_path)
    data = make_fitness(invented_apis=["session.atomic()"])
    data["dimensions"]["hallucination_resistance"] = None
    result = compile_tmp(tmp_path, data, rules_dir=rules_dir)
    assert "hallucination_resistance" not in result.profile.classifications
    assert INVENTED_API_INSTRUCTION in result.selected.operating
    assert INVENTED_API_INSTRUCTION in result.selected.extras
    ver = result.prompt.split("VERIFICATION\n\n", 1)[1].split("\n\nCOMPLETION", 1)[0]
    assert NEUTRAL_VERIFICATION in ver
    explain = render_explain(result.profile, result.selected)
    assert "Extra operating instructions:" in explain
    assert INVENTED_API_INSTRUCTION in explain
    assert "Category: hallucination_resistance" not in explain


def test_load_non_utf8_fitness(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_bytes(b"\xff\xfe{not utf-8}")
    with pytest.raises(CompilerError, match="cannot read fitness file"):
        load_fitness(path)


def test_load_model_not_string(tmp_path: Path):
    data = make_fitness()
    data["model"] = 12
    path = write_fitness(tmp_path / "model.json", data)
    with pytest.raises(CompilerError, match="model must be a string"):
        load_fitness(path)


def test_cli_non_utf8_does_not_write(tmp_path: Path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe")
    task = write_task(tmp_path / "task.md")
    out_path = tmp_path / "out" / "prompt.md"
    code = main(["--fitness", str(bad), "--task", str(task), "--output", str(out_path)])
    assert code == 1
    assert "cannot read fitness file" in capsys.readouterr().err
    assert not out_path.exists()


def test_cli_non_utf8_task_does_not_write(tmp_path: Path, capsys):
    fitness = write_fitness(tmp_path / "f.json", make_fitness())
    task = tmp_path / "task.md"
    task.write_bytes(b"\xff\xfe")
    out_path = tmp_path / "prompt.md"
    code = main(["--fitness", str(fitness), "--task", str(task), "--output", str(out_path)])
    assert code == 1
    assert "cannot read task file" in capsys.readouterr().err
    assert not out_path.exists()


def test_cli_output_directory_error(tmp_path: Path, capsys):
    fitness = write_fitness(tmp_path / "f.json", make_fitness())
    task = write_task(tmp_path / "task.md")
    code = main(["--fitness", str(fitness), "--task", str(task), "--output", str(tmp_path)])
    assert code == 1
    assert "cannot write output file" in capsys.readouterr().err


def test_cli_output_empty_name(tmp_path: Path, capsys):
    fitness = write_fitness(tmp_path / "f.json", make_fitness())
    task = write_task(tmp_path / "task.md")
    code = main(["--fitness", str(fitness), "--task", str(task), "--output", "."])
    assert code == 1
    assert "cannot write output file" in capsys.readouterr().err


def test_cli_subprocess_entry(tmp_path: Path):
    fitness = write_fitness(tmp_path / "f.json", make_fitness())
    task = write_task(tmp_path / "task.md", "SUBPROCESS_TASK\n")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "prompt_compiler.py"), "--fitness", str(fitness), "--task", str(task)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "ROLE" in proc.stdout
    assert "SUBPROCESS_TASK" in proc.stdout
    assert proc.stderr == ""
