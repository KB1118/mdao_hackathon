import json

from proxy.translate_request import to_responses_request
from proxy.translate_response import StreamConverter, to_anthropic_message


def _parse_sse(chunks):
    events = []
    text = b"".join(chunks).decode("utf-8")
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_request_maps_system_tools_and_tool_results():
    body = {
        "model": "gpt-6-luna",
        "system": [{"type": "text", "text": "sys"}],
        "max_tokens": 100,
        "stream": True,
        "tools": [{"name": "Read", "description": "read", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "auto"},
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "a"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "file body"},
                    {"type": "text", "text": "thanks"},
                ],
            },
        ],
    }
    request = to_responses_request(body, "gpt-6-luna")

    assert request["instructions"] == "sys"
    assert request["max_output_tokens"] == 100
    assert request["stream"] is True
    assert request["store"] is False
    assert request["tool_choice"] == "auto"
    assert request["tools"][0]["type"] == "function"
    assert request["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
        {"type": "function_call", "call_id": "t1", "name": "Read", "arguments": '{"path": "a"}'},
        {"type": "function_call_output", "call_id": "t1", "output": "file body"},
        {"role": "user", "content": [{"type": "input_text", "text": "thanks"}]},
    ]


def test_non_stream_response_maps_text_tool_use_and_cached_usage():
    response = {
        "id": "resp_1",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": "Hi"}]},
            {"type": "function_call", "call_id": "call_1", "name": "Read", "arguments": '{"path": "a"}'},
        ],
        "usage": {"input_tokens": 20, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 8}},
    }
    message = to_anthropic_message(response, "gpt-6-luna")

    assert message["content"] == [
        {"type": "text", "text": "Hi"},
        {"type": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "a"}},
    ]
    assert message["stop_reason"] == "tool_use"
    assert message["usage"]["input_tokens"] == 12
    assert message["usage"]["cache_read_input_tokens"] == 8


def test_stream_emits_anthropic_event_sequence():
    events = [
        {"type": "response.created", "response": {"id": "resp_9"}},
        {"type": "response.output_item.added", "item": {"type": "message", "id": "m1"}},
        {"type": "response.output_text.delta", "delta": "Hel"},
        {"type": "response.output_text.delta", "delta": "lo"},
        {"type": "response.output_item.done", "item": {"type": "message"}},
        {"type": "response.output_item.added", "item": {"type": "function_call", "call_id": "call_1", "name": "Read"}},
        {"type": "response.function_call_arguments.delta", "delta": '{"path":'},
        {"type": "response.function_call_arguments.delta", "delta": '"a.py"}'},
        {"type": "response.output_item.done", "item": {"type": "function_call"}},
        {"type": "response.completed",
         "response": {"id": "resp_9", "usage": {"input_tokens": 20, "output_tokens": 5,
                                                "input_tokens_details": {"cached_tokens": 8}, "cost": 0.001}}},
    ]
    converter = StreamConverter("gpt-6-luna")
    chunks = []
    for event in events:
        chunks += converter.feed(event)
    parsed = _parse_sse(chunks)

    names = [name for name, _ in parsed]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert parsed[1][1]["content_block"] == {"type": "text", "text": ""}
    assert parsed[5][1]["content_block"]["name"] == "Read"
    assert parsed[9][1]["delta"]["stop_reason"] == "tool_use"
    assert converter.usage["cache_read_input_tokens"] == 8
    assert converter.cost == "0.001"
