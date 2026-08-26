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
