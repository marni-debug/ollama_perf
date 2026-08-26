from llm_fitness.stage2.xml_protocol import XmlProtocol


def test_parse_multiple_tools():
    proto = XmlProtocol()
    text = """
I'll inspect the repo first.

<list_files>
<path>.</path>
</list_files>

<read_file>
<path>app.py</path>
</read_file>
"""
    calls = proto.parse(text)
    assert [c.name for c in calls] == ["list_files", "read_file"]
    assert calls[1].arguments["path"] == "app.py"


def test_parse_write_and_completion():
    proto = XmlProtocol()
    text = """
<write_to_file>
<path>app.py</path>
<content>
print("hi")
</content>
</write_to_file>
<attempt_completion>
<result>done</result>
</attempt_completion>
"""
    calls = proto.parse(text)
    assert calls[0].arguments["content"].strip() == 'print("hi")'
    assert calls[1].name == "attempt_completion"
    assert calls[1].arguments["result"] == "done"


def test_parse_execute_command_bare_body():
    proto = XmlProtocol()
    calls = proto.parse("<execute_command>pytest</execute_command>")
    assert calls[0].arguments["command"] == "pytest"
