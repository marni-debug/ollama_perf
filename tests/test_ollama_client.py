from llm_fitness.ollama_client import (
    NUM_CTX,
    NUM_PREDICT,
    chat_payload,
    chat_response_from_body,
)
from llm_fitness.stage2.xml_protocol import XmlProtocol


def test_thinking_stays_separate_from_content():
    reply = chat_response_from_body(
        {
            "message": {
                "content": "OK",
                "thinking": "lange interne Reasoning-Kette",
            }
        }
    )
    assert reply.content == "OK"
    assert reply.thinking == "lange interne Reasoning-Kette"


def test_empty_content_is_not_replaced_by_thinking():
    thinking = "<read_file>\n<path>app.py</path>\n</read_file>"
    reply = chat_response_from_body({"message": {"content": "", "thinking": thinking}})
    assert reply.content == ""
    assert reply.thinking == thinking
    assert XmlProtocol().parse(reply.content) == []


def test_xml_parsed_only_from_content_not_thinking():
    content = "text <read_file>\n<path>README.md</path>\n</read_file>"
    thinking = "<read_file>\n<path>fake.md</path>\n</read_file>"
    reply = chat_response_from_body({"message": {"content": content, "thinking": thinking}})
    calls = XmlProtocol().parse(reply.content)
    assert [c.arguments.get("path") for c in calls] == ["README.md"]
    assert all(c.arguments.get("path") != "fake.md" for c in calls)


def test_chat_payload_sends_fixed_num_ctx():
    payload = chat_payload("qwen3:14b", [{"role": "user", "content": "OK"}])
    assert payload["options"]["num_ctx"] == NUM_CTX
    assert payload["options"]["temperature"] == 0.0
    assert NUM_CTX == 8192


def test_chat_payload_sends_fixed_num_predict():
    payload = chat_payload("qwen3:14b", [{"role": "user", "content": "OK"}])
    assert payload["options"]["num_predict"] == NUM_PREDICT
    assert NUM_PREDICT == 4096
