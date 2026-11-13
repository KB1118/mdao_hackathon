import json
import time

import httpx
from fastapi.testclient import TestClient

from proxy import headroom_settings as hs
from proxy.app import create_app
from proxy.config import Settings, save_settings
from proxy import usage as usage_log

UPSTREAM_OK = {
    "id": "msg_up",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 1, "cache_read_input_tokens": 3,
              "cache_creation_input_tokens": 0},
}

CLAUDE_HEADERS = {"x-claude-code-session-id": "sess-1", "x-api-key": "k"}


def _request_body():
    return {
        "model": "claude-haiku-5-5",
        "max_tokens": 10,
        "system": "sys",
        "messages": [{"role": "user", "content": "original long text"}],
    }


def _app(tmp_path, headroom_values, sidecar_handler, upstream_seen):
    settings_path = tmp_path / "config.json"
    save_settings(
        Settings(upstream_url="http://gateway.test", exposed_models=["claude-haiku-5-5"],
                 headroom=hs.normalize(headroom_values)),
        settings_path,
    )

    def upstream(request: httpx.Request) -> httpx.Response:
        upstream_seen.append(json.loads(request.content))
        return httpx.Response(200, json=UPSTREAM_OK)

    return create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude.json",
        upstream_transport=httpx.MockTransport(upstream),
        sidecar_transport=httpx.MockTransport(sidecar_handler),
        manage_sidecar=False,
    )


def _compressed_reply(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/compress":
        return httpx.Response(200, json={
            "body": {"model": "claude-haiku-5-5", "system": "sys",
                     "messages": [{"role": "user", "content": "short"}]},
            "turn_id": "turn-1",
            "obligations": ["relay_usage"],
            "tokens_before": 50, "tokens_after": 10, "tokens_saved": 40,
            "transforms_applied": ["router"],
            "session": {"cached_prefix_replayed": False},
        })
    return httpx.Response(200, json={"action": "done"})


def test_compressed_body_is_what_upstream_receives(tmp_path):
    upstream_seen = []
    app = _app(tmp_path, {"retries": 0, "backoff_ms": 0}, _compressed_reply, upstream_seen)
    client = TestClient(app)
    resp = client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
    assert resp.status_code == 200
    assert upstream_seen[0]["messages"][0]["content"] == "short"
    assert upstream_seen[0]["system"] == "sys"


def test_usage_is_logged_and_relayed(tmp_path):
    db = tmp_path / "usage.db"
    upstream_seen = []
    app = _app(tmp_path, {"retries": 0, "backoff_ms": 0}, _compressed_reply, upstream_seen)
    client = TestClient(app)
    client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
    row = None
    for _ in range(50):
        row = usage_log.recent(db, 1)[0]
        if row["relayed"]:
            break
        time.sleep(0.05)
    assert row["compressed"] == 1
    assert row["tokens_saved"] == 40
    assert row["session_source"] == "claude_header"
    assert row["turn_id"] == "turn-1"
    assert row["relayed"] == 1


def test_failure_forwards_original_by_default(tmp_path):
    db = tmp_path / "usage.db"
    upstream_seen = []

    def failing(request):
        return httpx.Response(503, json={"error": {"type": "compression_timeout"}})

    app = _app(tmp_path, {"retries": 1, "backoff_ms": 0}, failing, upstream_seen)
    client = TestClient(app)
    resp = client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
    assert resp.status_code == 200
    assert upstream_seen[0]["messages"][0]["content"] == "original long text"
    row = usage_log.recent(db, 1)[0]
    assert row["fallback"].startswith("forwarded_original")
    assert row["compressed"] == 0


def test_failure_can_be_set_to_fail_the_request(tmp_path):
    upstream_seen = []

    def failing(request):
        return httpx.Response(503, json={})

    app = _app(tmp_path, {"retries": 0, "backoff_ms": 0, "failure_policy": "fail_request"},
               failing, upstream_seen)
    client = TestClient(app)
    resp = client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
    assert resp.status_code == 503
    assert upstream_seen == []


def test_requests_without_session_signal_are_not_compressed(tmp_path):
    db = tmp_path / "usage.db"
    upstream_seen = []
    calls = []

    def sidecar(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={})

    app = _app(tmp_path, {}, sidecar, upstream_seen)
    client = TestClient(app)
    client.post("/v1/messages", json=_request_body(), headers={"x-api-key": "k"})
    assert calls == []
    assert usage_log.recent(db, 1)[0]["fallback"] == "no_session_key"


def test_disabled_sidecar_forwards_untouched(tmp_path):
    db = tmp_path / "usage.db"
    upstream_seen = []
    calls = []

    def sidecar(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={})

    app = _app(tmp_path, {"sidecar_enabled": False}, sidecar, upstream_seen)
    client = TestClient(app)
    client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
    assert calls == []
    assert upstream_seen[0]["messages"][0]["content"] == "original long text"
    assert usage_log.recent(db, 1)[0]["fallback"] == "sidecar_disabled"
