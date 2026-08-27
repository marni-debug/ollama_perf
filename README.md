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
