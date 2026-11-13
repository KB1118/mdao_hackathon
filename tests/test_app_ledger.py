"""End-to-end request-path tests: flags on and off, ledger events, sessions API."""

import json

import httpx
from fastapi.testclient import TestClient

from proxy import features as fs
from proxy import ledger
from proxy import usage as usage_log
from proxy.app import create_app
from proxy.config import Settings, save_settings
from proxy.headroom_settings import normalize as headroom_normalize

CLAUDE_HEADERS = {"x-claude-code-session-id": "smoke-sess-1", "x-api-key": "k"}

UPSTREAM_OK = {
    "id": "msg_up", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "ok from gateway"}], "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 1, "cache_read_input_tokens": 3,
              "cache_creation_input_tokens": 2},
}

UPSTREAM_SSE = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":8}}}\n\n'
    'event: content_block_start\n'
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"chunked"}}\n\n'
    'event: content_block_stop\n'
    'data: {"type":"content_block_stop","index":0}\n\n'
    'event: message_delta\n'
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},'
    '"usage":{"output_tokens":4,"cache_read_input_tokens":6}}\n\n'
    'event: message_stop\n'
    'data: {"type":"message_stop"}\n\n'
)

COMPRESSED_REPLY = {
    "turn_id": "turn-42",
    "obligations": ["relay_usage"],
    "tokens_before": 80, "tokens_after": 20, "tokens_saved": 60,
    "transforms_applied": ["router"],
    "session": {"cached_prefix_replayed": False},
}


def _sidecar(request):
    if request.url.path != "/v1/compress":
        return httpx.Response(200, json={"action": "done"})
    # mirrors the real headroom gateway contract: the provider body keeps the
    # request's own fields (stream, max_tokens, ...) minus the proxy-only ones
    payload = json.loads(request.content)
    body = {
        key: value for key, value in payload.items()
        if key not in ("token_budget", "config", "gateway", "messages", "system")
    }
    body["model"] = payload.get("model", "claude-haiku-5-5")
    body["system"] = "sys"
    body["messages"] = [{"role": "user", "content": "compressed"}]
    return httpx.Response(200, json={"body": body, **COMPRESSED_REPLY})


def _request_body(stream=False):
    return {
        "model": "claude-haiku-5-5", "max_tokens": 10, "system": "sys", "stream": stream,
        "messages": [{"role": "user", "content": "the original request text"}],
    }


def _app(tmp_path, *, feature_values=None, upstream=None, sidecar=None, upstream_seen=None):
    settings_path = tmp_path / "config.json"
    settings = Settings(upstream_url="http://gateway.test", exposed_models=["claude-haiku-5-5"])
    settings.headroom = headroom_normalize({"retries": 0, "backoff_ms": 0})
    settings.features = fs.normalize(feature_values or {"tracing": True, "compression": True,
                                                        "run_label": "ablate-X"})
    save_settings(settings, settings_path)

    def default_upstream(request):
        if upstream_seen is not None:
            upstream_seen.append(json.loads(request.content))
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(200, content=UPSTREAM_SSE.encode(),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=UPSTREAM_OK)

    def default_sidecar(request):
        return _sidecar(request)

    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude.json",
        ledger_dir=tmp_path / "sessions",
        upstream_transport=httpx.MockTransport(upstream or default_upstream),
        sidecar_transport=httpx.MockTransport(sidecar or default_sidecar),
        manage_sidecar=False,
    )
    return app


def _entries_for(tmp_path, session_key):
    with ledger.session_path(tmp_path / "sessions", session_key).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_full_round_trip_writes_request_stage_and_response(tmp_path):
    app = _app(tmp_path)
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
    assert resp.status_code == 200

    session_key = "claude:smoke-sess-1:" + _fingerprint()
    entries = _entries_for(tmp_path, session_key)
    types = [e["type"] for e in entries]
    assert types == ["request", "stage", "response"]

    request, stage, response = entries
    # every entry correlates on request_id and carries the run label from config
    assert len({e["request_id"] for e in entries}) == 1
    assert {e["run_label"] for e in entries} == {"ablate-X"}
    assert {e["session_key"] for e in entries} == {session_key}

    # request event: original (pre-compression) body fingerprint, never the payload
    assert request["model"] == "claude-haiku-5-5"
    assert request["route"] == "anthropic"
    assert request["system"]["chars"] > 0
    assert request["messages"][0]["role"] == "user"
    raw = (tmp_path / "sessions").glob("*.jsonl").__next__().read_text(encoding="utf-8")
    assert "the original request text" not in raw
    assert "sys" not in request["tools"]["names"]

    # stage provenance for compression
    assert stage["feature"] == "compression"
    assert stage["provenance"]["tokens_saved"] == 60
    assert stage["provenance"]["tokens_before"] == 80
    assert stage["provenance"]["fallback"] is None
    assert stage["provenance"]["duration_ms"] >= 0

    # response: usage split, stop reason, assistant fingerprint
    assert response["status"] == 200
    assert response["usage"] == {"input": 5, "output": 1, "cache_read": 3, "cache_write": 2}
    assert response["stop_reason"] == "end_turn"
    assert response["content"]["block_counts"] == {"text": 1}

    # the usage row still works exactly as before
    row = usage_log.recent(tmp_path / "usage.db", 1)[0]
    assert row["compressed"] == 1
    assert row["tokens_saved"] == 60
    assert row["turn_id"] == "turn-42"


def _fingerprint():
    from proxy.session_key import fingerprint
    return fingerprint({"system": "sys", "messages": [{"role": "user", "content": "the original request text"}]})


def test_streaming_round_trip_logs_final_usage_at_stream_end(tmp_path):
    app = _app(tmp_path)
    with TestClient(app) as client:
        with client.stream("POST", "/v1/messages", json=_request_body(stream=True),
                           headers=CLAUDE_HEADERS) as resp:
            body = "".join(resp.iter_text())
    assert "chunked" in body

    entries = _entries_for(tmp_path, "claude:smoke-sess-1:" + _fingerprint())
    response = [e for e in entries if e["type"] == "response"][0]
    assert response["status"] == 200
    # streamed assistant content is not fingerprinted
    assert response["content"] is None
    # usage merged from message_start + message_delta, stop reason from the delta
    assert response["usage"] == {"input": 8, "output": 4, "cache_read": 6, "cache_write": 0}
    assert response["stop_reason"] == "end_turn"


def test_tracing_off_writes_nothing(tmp_path):
    app = _app(tmp_path, feature_values={"tracing": False})
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
    assert resp.status_code == 200
    assert not (tmp_path / "sessions").exists()


def test_compression_failure_fail_open_logs_stage_and_error(tmp_path):
    def failing(request):
        return httpx.Response(503, json={"error": {}})

    app = _app(tmp_path, sidecar=failing)
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
    assert resp.status_code == 200  # fail-open still succeeds
    entries = _entries_for(tmp_path, "claude:smoke-sess-1:" + _fingerprint())
    stages = [e for e in entries if e["type"] == "stage"]
    errors = [e for e in entries if e["type"] == "error"]
    assert len(stages) == 1 and len(errors) == 1
    assert stages[0]["provenance"]["fallback"].startswith("forwarded_original")
    assert errors[0]["stage"] == "compression"
    assert "HTTP 503" in errors[0]["message"]


def test_upstream_error_is_observed_without_breaking_the_stream(tmp_path):
    def erroring(request):
        return httpx.Response(500, json={"error": {"message": "gateway exploded"}})

    app = _app(tmp_path, upstream=erroring)
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
    assert resp.status_code == 500
    entries = _entries_for(tmp_path, "claude:smoke-sess-1:" + _fingerprint())
    response = [e for e in entries if e["type"] == "response"][0]
    assert response["status"] == 500
    assert response["usage"] == {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}


def test_sessions_api_lists_and_reads(tmp_path):
    app = _app(tmp_path)
    with TestClient(app) as client:
        client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
        client.post("/v1/messages", json=_request_body(),
                    headers={"x-claude-code-session-id": "smoke-sess-2", "x-api-key": "k"})

        listing = client.get("/api/sessions").json()["sessions"]
        assert len(listing) == 2
        first = [s for s in listing if s["session_key"].startswith("claude:smoke-sess-1:")][0]
        assert first["requests"] == 1
        assert first["entries"] == 3
        assert first["run_label"] == "ablate-X"
        assert first["size_bytes"] > 0
        assert first["first_ts"] and first["last_ts"]

        detail = client.get(f"/api/sessions/{first['key']}").json()
        assert [e["type"] for e in detail["entries"]] == ["request", "stage", "response"]
        assert detail["run_label"] == "ablate-X"

        assert client.get("/api/sessions/no-such-session").status_code == 404


def test_sessions_api_rejects_unsafe_keys(tmp_path):
    app = _app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/sessions/..").status_code == 404
        response = client.get("/api/sessions/bad%2Fkey")
        assert response.status_code == 404
