from pathlib import Path

from llm_fitness.ollama_client import ChatResponse
from llm_fitness.results import PerfSample
from llm_fitness.stage2.agent_loop import run_agent
from llm_fitness.stage2.tools import ToolExecutor
from llm_fitness.stage2.xml_protocol import XmlProtocol


class FakeClient:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)

    def chat(self, model, messages, temperature=0.0):
        return ChatResponse(
            content=self.replies.pop(0),
            thinking="",
            perf=PerfSample(),
            raw={},
        )


def test_agent_writes_file_and_completes(tmp_path: Path):
    (tmp_path / "app.py").write_text("print('old')\n", encoding="utf-8")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python").write_text("#!/bin/sh\nexit 0\n")
    (bindir / "python").chmod(0o755)
    (bindir / "pytest").write_text("#!/bin/sh\nexit 0\n")
    (bindir / "pytest").chmod(0o755)

    client = FakeClient(
        [
            """
<write_to_file>
<path>app.py</path>
<content>
print("new")
</content>
</write_to_file>
""",
            """
<attempt_completion>
<result>updated app.py</result>
</attempt_completion>
""",
        ]
    )
    executor = ToolExecutor(tmp_path, bindir)
    run = run_agent(client, "dummy", executor, "please edit", protocol=XmlProtocol(), max_turns=5)
    assert run.completion == "updated app.py"
    assert (tmp_path / "app.py").read_text(encoding="utf-8").strip() == 'print("new")'


def test_nudge_when_no_tools(tmp_path: Path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python").write_text("#!/bin/sh\nexit 0\n")
    (bindir / "python").chmod(0o755)
    client = FakeClient(["just thinking", "<attempt_completion><result>ok</result></attempt_completion>"])
    executor = ToolExecutor(tmp_path, bindir)
    run = run_agent(client, "dummy", executor, "task", max_turns=5)
    assert run.empty_tool_turns == 1
    assert run.completion == "ok"
