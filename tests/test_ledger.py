import hashlib
import json
from datetime import datetime

import pytest

from proxy import ledger


def _body():
    return {
        "model": "claude-haiku-5-5",
        "system": "system prompt",
        "tools": [{"name": "Read", "description": "read", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": "hello there"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "file body"},
                {"type": "text", "text": "thanks"},
            ]},
        ],
    }


def _write_request(tmp_path, session_key="claude:abc-123:f1", run_label="run-A", body=None):
    return ledger.request_event(
        tmp_path,
        session_key=session_key,
        session_source="claude_header",
        request_id="req-1",
        run_label=run_label,
        requested_model="claude-haiku-5-5[1m]",
        model="claude-haiku-5-5",
        route="anthropic",
        stream=False,
        body=body or _body(),
    )


def test_request_event_shape_and_file_content(tmp_path):
    entry = _write_request(tmp_path)
    assert entry["type"] == "request"
    assert entry["requested_model"] == "claude-haiku-5-5[1m]"
    assert entry["model"] == "claude-haiku-5-5"
    assert entry["route"] == "anthropic"
    assert entry["stream"] is False
    assert entry["run_label"] == "run-A"
    assert entry["session_key"] == "claude:abc-123:f1"
    assert entry["session_source"] == "claude_header"
    assert entry["request_id"] == "req-1"
    assert entry["message_count"] == 3

    profiles = entry["messages"]
    assert [m["role"] for m in profiles] == ["user", "assistant", "user"]
    # plain-string content counts as one text block
    assert profiles[0]["block_counts"] == {"text": 1}
    # block types are counted per message
    assert profiles[1]["block_counts"] == {"text": 1, "tool_use": 1}
    assert profiles[2]["block_counts"] == {"tool_result": 1, "text": 1}

    assert entry["system"]["chars"] == len(json.dumps("system prompt", sort_keys=True,
                                                     separators=(",", ":"), ensure_ascii=False))
    assert entry["tools"]["count"] == 1
    assert entry["tools"]["names"] == ["Read"]

    path = ledger.session_path(tmp_path, "claude:abc-123:f1")
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == entry

    # ts is a parseable UTC timestamp
    parsed = datetime.fromisoformat(entry["ts"])
    assert parsed.utcoffset().total_seconds() == 0


def test_hashes_are_stable_and_payloads_never_stored(tmp_path):
    first = _write_request(tmp_path)
    second = _write_request(tmp_path, run_label="run-B")
    assert first["messages"][0]["hash"] == second["messages"][0]["hash"]
    assert first["system"]["hash"] == second["system"]["hash"]

    # the hash is the sha256 of the canonical serialization
    content = "hello there"
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert first["messages"][0]["hash"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # a different body yields a different hash
    changed = _write_request(tmp_path, session_key="claude:other:f2",
                            body={"messages": [{"role": "user", "content": "hello there!"}]})
    assert changed["messages"][0]["hash"] != first["messages"][0]["hash"]

    # payload text never appears anywhere in the ledger file
    raw = ledger.session_path(tmp_path, "claude:abc-123:f1").read_text(encoding="utf-8")
    assert "hello there" not in raw
    assert "system prompt" not in raw


def test_response_and_stage_and_error_shapes(tmp_path):
    stage = ledger.stage_event(
        tmp_path, session_key="claude:s1:f", session_source="claude_header",
        request_id="req-9", run_label="ablate", feature="compression",
        provenance={"tokens_before": 100, "tokens_after": 60, "tokens_saved": 40,
                    "fallback": None, "duration_ms": 321},
    )
    assert stage["feature"] == "compression"
    assert stage["provenance"]["tokens_saved"] == 40
    assert stage["provenance"]["duration_ms"] == 321

    response = ledger.response_event(
        tmp_path, session_key="claude:s1:f", session_source="claude_header",
        request_id="req-9", run_label="ablate", status=200, duration_ms=1234,
        usage={"input_tokens": 12, "output_tokens": 5,
               "cache_read_input_tokens": 7, "cache_creation_input_tokens": 2},
        stop_reason="tool_use",
        content=[{"type": "text", "text": "answer"},
                 {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "a"}}],
    )
    assert response["type"] == "response"
    assert response["status"] == 200
    assert response["duration_ms"] == 1234
    assert response["usage"] == {"input": 12, "output": 5, "cache_read": 7, "cache_write": 2}
    assert response["stop_reason"] == "tool_use"
    assert response["content"]["block_counts"] == {"text": 1, "tool_use": 1}
    assert response["content"]["chars"] > 0
    assert "answer" not in ledger.session_path(tmp_path, "claude:s1:f").read_text(encoding="utf-8")

    # streaming responses have no content fingerprint: the field is omitted, usage still split
    streamed = ledger.response_event(
        tmp_path, session_key="claude:s1:f", session_source="claude_header",
        request_id="req-9", run_label="ablate", status=200, duration_ms=1000,
        usage={"input_tokens": 1, "output_tokens": 2}, stop_reason=None,
    )
    assert streamed["content"] is None
    assert streamed["usage"] == {"input": 1, "output": 2, "cache_read": 0, "cache_write": 0}
    assert streamed["stop_reason"] is None

    error = ledger.error_event(
        tmp_path, session_key="claude:s1:f", session_source="claude_header",
        request_id="req-9", run_label="ablate", stage="compression", message="boom",
    )
    assert error == {
        "type": "error",
        "ts": error["ts"],
        "run_label": "ablate",
        "session_key": "claude:s1:f",
        "session_source": "claude_header",
        "request_id": "req-9",
        "stage": "compression",
        "message": "boom",
    }


def test_file_naming_sanitizes_and_never_collides(tmp_path):
    dirty = "claude:not@all/safe ?:*here"
    path = ledger.session_path(tmp_path, dirty)
    name = path.name
    assert name.endswith(".jsonl")
    assert all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in name)
    # distinct keys that sanitize to the same prefix get different hash suffixes
    assert ledger.session_path(tmp_path, "a b") != ledger.session_path(tmp_path, "a/b")
    assert ledger.session_path(tmp_path, "a/b").stem.split("_")[0] == "a"
    # the suffix is the sha256 of the raw key, so the same key always maps to the same file
    expected = hashlib.sha256(b"a b").hexdigest()[:8]
    assert ledger.session_path(tmp_path, "a b").stem.endswith(expected)


def test_missing_session_writes_to_pseudo_session(tmp_path, capsys):
    entry = _write_request(tmp_path, session_key=None, run_label="")
    assert entry["run_label"] == ""
    assert ledger.session_path(tmp_path, None).name.startswith(ledger.NO_SESSION_KEY)
    ledger.request_event(tmp_path, session_key=None, session_source=None, request_id="r",
                         run_label="", requested_model="m", model="m", route="anthropic",
                         stream=True, body={"messages": []})
    assert len(list(tmp_path.glob("*.jsonl"))) == 1


def test_ledger_writes_fail_open(tmp_path, capsys):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("occupied", encoding="utf-8")
    # none of these raise
    ledger.request_event(blocker, session_key="s", session_source=None, request_id="r",
                         run_label="run", requested_model="m", model="m", route="anthropic",
                         stream=False, body={"messages": []})
    ledger.stage_event(blocker, session_key="s", session_source=None, request_id="r",
                       run_label="run", feature="compression", provenance={"tokens_saved": 1})
    ledger.response_event(blocker, session_key="s", session_source=None, request_id="r",
                          run_label="run", status=200, duration_ms=1)
    ledger.error_event(blocker, session_key="s", session_source=None, request_id="r",
                       run_label="run", stage="compression", message="x")
    err = capsys.readouterr().err
    assert err.count("ledger write failed") == 4


def test_list_and_read_sessions(tmp_path):
    _write_request(tmp_path, run_label="ablate-1")
    ledger.response_event(tmp_path, session_key="claude:abc-123:f1", session_source="claude_header",
                          request_id="req-1", run_label="ablate-1", status=200, duration_ms=10)
    _write_request(tmp_path, session_key="codex:other", run_label="ablate-2")

    sessions = ledger.list_sessions(tmp_path)
    assert len(sessions) == 2
    by_key = {s["key"]: s for s in sessions}
    main = by_key[ledger.session_path(tmp_path, "claude:abc-123:f1").stem]
    assert main["session_key"] == "claude:abc-123:f1"
    assert main["session_source"] == "claude_header"
    assert main["run_label"] == "ablate-1"
    assert main["requests"] == 1
    assert main["entries"] == 2
    assert main["size_bytes"] > 0
    assert main["first_ts"] <= main["last_ts"]

    detail = ledger.read_session(tmp_path, main["key"])
    assert [e["type"] for e in detail["entries"]] == ["request", "response"]
    assert detail["session_key"] == "claude:abc-123:f1"
    assert detail["run_label"] == "ablate-1"

    # unreadable keys: traversal, empty, dots
    assert ledger.read_session(tmp_path, "..") is None
    assert ledger.read_session(tmp_path, "") is None
    assert ledger.read_session(tmp_path, "no-such-session") is None
    assert ledger.read_session(tmp_path, "bad/key") is None

    # empty dir and missing dir both list cleanly
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ledger.list_sessions(empty) == []
    assert ledger.list_sessions(tmp_path / "missing") == []
