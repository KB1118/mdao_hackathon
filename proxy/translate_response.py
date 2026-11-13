"""OpenAI Responses API output -> Anthropic Messages output, including SSE streaming."""

from __future__ import annotations

import json
from typing import Any


def usage_from_responses(usage: dict[str, Any] | None) -> dict[str, Any]:
    usage = usage or {}
    details = usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    cache_write = details.get("cache_write_tokens") or 0
    total_input = usage.get("input_tokens") or 0
    return {
        "input_tokens": max(total_input - cached - cache_write, 0),
        "output_tokens": usage.get("output_tokens") or 0,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": cache_write,
    }


def reported_cost(usage: dict[str, Any] | None) -> str | None:
    if usage and usage.get("cost") is not None:
        return str(usage["cost"])
    return None


def _stop_reason(response: dict[str, Any], has_tool: bool) -> str:
    if has_tool:
        return "tool_use"
    reason = (response.get("incomplete_details") or {}).get("reason")
    return "max_tokens" if reason == "max_output_tokens" else "end_turn"


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def to_anthropic_message(response: dict[str, Any], requested_model: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for item in response.get("output", []):
        kind = item.get("type")
        if kind == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    content.append({"type": "text", "text": part.get("text", "")})
        elif kind == "function_call":
            content.append(
                {
                    "type": "tool_use",
                    "id": item.get("call_id") or item.get("id", ""),
                    "name": item.get("name", ""),
                    "input": _parse_arguments(item.get("arguments")),
                }
            )
    has_tool = any(block["type"] == "tool_use" for block in content)
    return {
        "id": "msg_" + str(response.get("id", "proxy")),
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": _stop_reason(response, has_tool),
        "stop_sequence": None,
        "usage": usage_from_responses(response.get("usage")),
    }


class StreamConverter:
    """Turns Responses API stream events into Anthropic Messages stream events."""

    def __init__(self, requested_model: str) -> None:
        self.model = requested_model
        self.message_id = "msg_proxy"
        self.started = False
        self.next_index = 0
        self.open_index: int | None = None
        self.open_kind: str | None = None
        self.has_tool = False
        self.usage: dict[str, Any] = {}
        self.cost: str | None = None
        self.stop_reason: str | None = None

    def feed(self, event: dict[str, Any]) -> list[bytes]:
        kind = event.get("type", "")
        out: list[bytes] = []
        if kind == "response.created":
            self.message_id = "msg_" + str(event.get("response", {}).get("id", "proxy"))
            out += self._start()
        elif kind == "response.output_item.added":
            item = event.get("item", {})
            if item.get("type") == "function_call":
                out += self._start()
                out += self._close()
                self.has_tool = True
                out += self._open(
                    "tool",
                    {
                        "type": "tool_use",
                        "id": item.get("call_id") or item.get("id", ""),
                        "name": item.get("name", ""),
                        "input": {},
                    },
                )
        elif kind == "response.output_text.delta":
            out += self._start()
            if self.open_kind != "text":
                out += self._close()
                out += self._open("text", {"type": "text", "text": ""})
            out += self._emit(
                "content_block_delta",
                {"type": "content_block_delta", "index": self.open_index,
                 "delta": {"type": "text_delta", "text": event.get("delta", "")}},
            )
        elif kind == "response.function_call_arguments.delta":
            if self.open_kind == "tool":
                out += self._emit(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": self.open_index,
                     "delta": {"type": "input_json_delta", "partial_json": event.get("delta", "")}},
                )
        elif kind == "response.output_item.done":
            if event.get("item", {}).get("type") in ("message", "function_call"):
                out += self._close()
        elif kind in ("response.completed", "response.incomplete"):
            response = event.get("response", {})
            out += self._start()
            out += self._close()
            self.usage = usage_from_responses(response.get("usage"))
            self.cost = reported_cost(response.get("usage"))
            self.stop_reason = _stop_reason(response, self.has_tool)
            out += self._finish(self.stop_reason)
        elif kind in ("response.failed", "error"):
            out += self._close()
            message = event.get("message") or event.get("response", {}).get("error", {}).get("message", "Upstream error")
            out += self._emit("error", {"type": "error", "error": {"type": "api_error", "message": message}})
        return out

    def _emit(self, name: str, data: dict[str, Any]) -> list[bytes]:
        return [f"event: {name}\ndata: {json.dumps(data)}\n\n".encode("utf-8")]

    def _start(self) -> list[bytes]:
        if self.started:
            return []
        self.started = True
        return self._emit(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

    def _open(self, kind: str, block: dict[str, Any]) -> list[bytes]:
        self.open_index = self.next_index
        self.next_index += 1
        self.open_kind = kind
        return self._emit(
            "content_block_start",
            {"type": "content_block_start", "index": self.open_index, "content_block": block},
        )

    def _close(self) -> list[bytes]:
        if self.open_index is None:
            return []
        out = self._emit("content_block_stop", {"type": "content_block_stop", "index": self.open_index})
        self.open_index = None
        self.open_kind = None
        return out

    def _finish(self, stop_reason: str) -> list[bytes]:
        return self._emit(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": self.usage,
            },
        ) + self._emit("message_stop", {"type": "message_stop"})
