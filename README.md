# ollama-perf

Personal tools around local Ollama models.

## LLM Coding-Agent Fitness Test

Checks how well a local model behaves as an autonomous coding agent (Cline-style), not as a generic HumanEval clone.

```bash
python check_llm.py -m ornith:35b
python check_llm.py -m qwen3.8:27b --mode knowledge
python check_llm.py --compare results/*.json
```

`--mode knowledge` is the fast prompt-only path. `--mode full` (default) also runs isolated agent workspaces.

Ollama must be running. Results are written to `results/`. Temperature defaults to `0` to reduce scatter; it does not guarantee identical tokens.

`ollama_perf_logger.py` is unchanged and independent of this test.

## Prompt Compiler

Builds a deterministic coding-agent prompt from a fitness-test JSON result and a task file. No LLM calls.

Input: `--fitness` result JSON (schema_version 2) and `--task` markdown/text. Output: a markdown prompt with fixed section headers. Weaknesses get compensation instructions from `rules/`; strengths are listed but do not add extra instructions. Null fitness dimensions are omitted.

```bash
python prompt_compiler.py --fitness results/gemma4-12b-20260826T224852Z.json --task task.md --output prompt.md
python prompt_compiler.py --fitness results/gemma4-12b-20260826T224852Z.json --task task.md --profile --explain
```

The same inputs always produce the same prompt. `--profile` prints the capability profile JSON to stdout; `--explain` prints per-category score, classification, and applied rules.

## A/B holdout experiment

Three separate layers — do not treat their numbers as the same measurement:

| Layer | Role |
|-------|------|
| **Fitness** (`check_llm.py`) | Real agent scoring (workspace tests, diffs, graders). |
| **Compiler** | Deterministic prompt bytes from fitness JSON + task. No LLM. |
| **A/B experiment** | Same model and holdout, two already-built prompt files, controlled knobs. |

The compiler is byte-deterministic. An LLM run is not. The experiment *definition* is deterministic (file hashes, task pairing, knobs); sampling is not.

Live A/B default is a smoke check (`evaluator: "non_empty_response"`): `passed` if the chat response is non-empty. No tools. `improved` does not mean the task was solved. An A/B delta is not a Fitness delta.

Each task arm gets its own temp workspace (A and B never share a path). The default chat evaluator ignores it. Future work (not implemented): a deterministic task evaluator (pytest, diffs, invariants).

```bash
python prompt_compiler.py experiment --model MODEL --benchmark holdout.json --baseline-prompt baseline.md --compiled-prompt compiled.md --output experiment.json
```
