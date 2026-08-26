from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from llm_fitness.results import PerfSample

# Fixed context for every model. 8192 is 2× Ollama's 4096 default, enough for
# this fixture's system prompt + file reads + tool results, without using
# each model's maximum (32K–256K). Same value for all tags.
NUM_CTX = 8192


class OllamaError(Exception):
    pass


class ChatResponse:
    def __init__(
        self,
        content: str,
        thinking: str,
        perf: PerfSample,
        raw: dict[str, Any],
    ):
        self.content = content
        self.thinking = thinking
        self.perf = perf
        self.raw = raw


def chat_payload(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": NUM_CTX},
    }


def chat_response_from_body(body: dict[str, Any]) -> ChatResponse:
    message = body.get("message") or {}
    return ChatResponse(
        content=(message.get("content") or "").strip(),
        thinking=(message.get("thinking") or "").strip(),
        perf=_perf_from_body(body),
        raw=body,
    )


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = self.base_url + path
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise OllamaError(f"Ollama HTTP {exc.code} on {path}: {body}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Ollama is not reachable at {self.base_url}: {exc.reason}"
            ) from exc

    def version(self) -> str:
        data = self._request("GET", "/api/version")
        return str(data.get("version", "unknown"))

    def show(self, model: str) -> dict[str, Any]:
        return self._request("POST", "/api/show", {"name": model})

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> ChatResponse:
        body = self._request(
            "POST",
            "/api/chat",
            chat_payload(model, messages, temperature=temperature),
        )
        return chat_response_from_body(body)


def _perf_from_body(body: dict[str, Any]) -> PerfSample:
    def ns(key: str) -> int | None:
        value = body.get(key)
        if value is None:
            return None
        return int(value)

    def count(key: str) -> int | None:
        value = body.get(key)
        if value is None:
            return None
        return int(value)

    return PerfSample(
        load_duration_ns=ns("load_duration"),
        prompt_eval_count=count("prompt_eval_count"),
        prompt_eval_duration_ns=ns("prompt_eval_duration"),
        eval_count=count("eval_count"),
        eval_duration_ns=ns("eval_duration"),
        total_duration_ns=ns("total_duration"),
    )


def model_max_context(show: dict[str, Any]) -> int | None:
    info = show.get("model_info") or {}
    if not isinstance(info, dict):
        return None
    for key, value in info.items():
        if str(key).endswith("context_length") or str(key) == "context_length":
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def format_num_ctx(length: int | None) -> str:
    if length is None:
        return "unknown"
    if length >= 1024 and length % 1024 == 0:
        return f"{length // 1024}K"
    if length >= 1024:
        return f"{length / 1024:.0f}K"
    return str(length)


def format_context(show: dict[str, Any]) -> str:
    return format_num_ctx(model_max_context(show))


def aggregate_perf(samples: list[PerfSample]) -> tuple[float | None, float | None]:
    gen_durations = []
    eval_counts = []
    eval_durations = []
    for sample in samples:
        gen = sample.generation_duration_ns()
        if gen:
            gen_durations.append(gen)
        if sample.eval_count and sample.eval_duration_ns:
            eval_counts.append(sample.eval_count)
            eval_durations.append(sample.eval_duration_ns)
    avg_s = (sum(gen_durations) / len(gen_durations) / 1e9) if gen_durations else None
    tps = None
    if eval_counts and eval_durations and sum(eval_durations) > 0:
        tps = sum(eval_counts) / (sum(eval_durations) / 1e9)
    return avg_s, tps
