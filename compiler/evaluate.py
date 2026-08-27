"""Evaluator protocol and A/B evaluation config.

Import-only use of llm_fitness.ollama_client constants. Holdout grading is
injected; the live chat helper only checks non-empty content. No tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from llm_fitness.ollama_client import NUM_CTX, NUM_PREDICT

DEFAULT_TEMPERATURE = 0.0
EVALUATOR_NAME = "non_empty_response"


@dataclass(frozen=True)
class EvaluationConfig:
    model: str
    prompt: str
    task_id: str
    task_input: str
    temperature: float = DEFAULT_TEMPERATURE
    num_ctx: int = NUM_CTX
    num_predict: int = NUM_PREDICT
    workspace_root: Path | None = None


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    passed: bool
    score: float


class Evaluator(Protocol):
    def evaluate(self, config: EvaluationConfig) -> TaskOutcome:
        ...


class OllamaChatEvaluator:
    """Default live path: one chat call, no tools.

    Scores passed=bool(content.strip()). Ignores workspace_root.
    """

    name = EVALUATOR_NAME

    def __init__(self, client: object | None = None) -> None:
        if client is None:
            from llm_fitness.ollama_client import OllamaClient

            client = OllamaClient()
        self._client = client

    def evaluate(self, config: EvaluationConfig) -> TaskOutcome:
        response = self._client.chat(
            config.model,
            [
                {"role": "system", "content": config.prompt},
                {"role": "user", "content": config.task_input},
            ],
            temperature=config.temperature,
        )
        passed = bool(response.content.strip())
        return TaskOutcome(
            task_id=config.task_id,
            passed=passed,
            score=1.0 if passed else 0.0,
        )


def default_evaluator() -> OllamaChatEvaluator:
    return OllamaChatEvaluator()
