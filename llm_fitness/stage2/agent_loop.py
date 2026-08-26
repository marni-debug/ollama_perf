from __future__ import annotations

from dataclasses import dataclass, field

from llm_fitness.ollama_client import OllamaClient
from llm_fitness.results import PerfSample
from llm_fitness.stage2.protocol import ToolCall, ToolProtocol
from llm_fitness.stage2.tools import ToolExecutor
from llm_fitness.stage2.xml_protocol import XmlProtocol

MAX_TURNS = 20
NUDGE = (
    "Please use the XML tools to inspect the project, make the change, "
    "and run pytest. Call attempt_completion when done."
)


@dataclass
class AgentRun:
    transcript: list[dict[str, str]]
    completion: str | None
    turns: int
    perf: list[PerfSample] = field(default_factory=list)
    tool_log: list[dict[str, str]] = field(default_factory=list)
    empty_tool_turns: int = 0


def run_agent(
    client: OllamaClient,
    model: str,
    executor: ToolExecutor,
    user_task: str,
    protocol: ToolProtocol | None = None,
    max_turns: int = MAX_TURNS,
) -> AgentRun:
    proto: ToolProtocol = protocol or XmlProtocol()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": proto.system_prompt(str(executor.workspace))},
        {"role": "user", "content": user_task},
    ]
    perf: list[PerfSample] = []
    completion: str | None = None
    empty = 0
    turn = 0

    for turn in range(1, max_turns + 1):
        reply = client.chat(model, messages)
        perf.append(reply.perf)
        messages.append({"role": "assistant", "content": reply.content})
        calls = proto.parse(reply.content)
        if not calls:
            empty += 1
            messages.append({"role": "user", "content": NUDGE})
            continue
        remaining: list[ToolCall] = []
        finish: ToolCall | None = None
        for call in calls:
            if call.name == "attempt_completion":
                finish = call
            else:
                remaining.append(call)
        chunks: list[str] = []
        for call in remaining:
            result = executor.execute(call)
            chunks.append(proto.format_result(result))
        if finish is not None:
            completion = finish.arguments.get("result") or reply.content
            if remaining:
                messages.append({"role": "user", "content": "\n\n".join(chunks)})
            break
        messages.append({"role": "user", "content": "\n\n".join(chunks)})

    return AgentRun(
        transcript=messages,
        completion=completion,
        turns=turn,
        perf=perf,
        tool_log=list(executor.log),
        empty_tool_turns=empty,
    )
