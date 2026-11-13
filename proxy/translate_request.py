"""Anthropic Messages request -> OpenAI Responses API request."""

from __future__ import annotations

import json
from typing import Any


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "")
        for block in content or []
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _tool_result_text(block: dict[str, Any]) -> str:
    text = _text(block.get("content", ""))
    if block.get("is_error"):
        text = f"Error: {text}"
    return text


def _image(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source", {})
    if source.get("type") == "base64":
        url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
    elif source.get("type") == "url":
        url = source.get("url", "")
    else:
        return None
    return {"type": "input_image", "image_url": url}


def _document(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source", {})
    if source.get("type") != "base64":
        return None
    media = source.get("media_type", "application/pdf")
    return {
        "type": "input_file",
        "filename": "document.pdf",
        "file_data": f"data:{media};base64,{source.get('data', '')}",
    }


def _user_message(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"role": "user", "content": [{"type": "input_text", "text": content}]}]
    items: list[dict[str, Any]] = [
        {"type": "function_call_output", "call_id": block.get("tool_use_id", ""), "output": _tool_result_text(block)}
        for block in content
        if block.get("type") == "tool_result"
    ]
    parts: list[dict[str, Any]] = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            parts.append({"type": "input_text", "text": block.get("text", "")})
        elif kind == "image" and (part := _image(block)):
            parts.append(part)
        elif kind == "document" and (part := _document(block)):
            parts.append(part)
    if parts:
        items.append({"role": "user", "content": parts})
    return items


def _assistant_message(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"role": "assistant", "content": [{"type": "output_text", "text": content}]}]
    items: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            texts.append({"type": "output_text", "text": block.get("text", "")})
        elif kind == "tool_use":
            if texts:
                items.append({"role": "assistant", "content": texts})
                texts = []
            items.append(
                {
                    "type": "function_call",
                    "call_id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                }
            )
    if texts:
        items.append({"role": "assistant", "content": texts})
    return items


def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if message.get("role") == "user":
            items += _user_message(content)
        else:
            items += _assistant_message(content)
    return items


def _tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["input_schema"],
            "strict": False,
        }
        for tool in tools or []
        if "input_schema" in tool
    ]


def _tool_choice(choice: dict[str, Any]) -> Any:
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool":
        return {"type": "function", "name": choice.get("name", "")}
    return None


def _reasoning(thinking: dict[str, Any] | None) -> dict[str, str] | None:
    if not thinking or thinking.get("type") != "enabled":
        return None
    budget = thinking.get("budget_tokens", 0)
    effort = "low" if budget < 4000 else "medium" if budget < 16000 else "high"
    return {"effort": effort}


def to_responses_request(body: dict[str, Any], model: str) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "input": _messages(body.get("messages", [])),
        "store": False,
        "stream": bool(body.get("stream")),
    }
    system = _text(body.get("system"))
    if system:
        request["instructions"] = system
    if body.get("max_tokens"):
        request["max_output_tokens"] = body["max_tokens"]
    tools = _tools(body.get("tools"))
    if tools:
        request["tools"] = tools
        choice = body.get("tool_choice") or {}
        mapped = _tool_choice(choice)
        if mapped is not None:
            request["tool_choice"] = mapped
        if choice.get("disable_parallel_tool_use"):
            request["parallel_tool_calls"] = False
    reasoning = _reasoning(body.get("thinking"))
    if reasoning:
        request["reasoning"] = reasoning
    return request
