import json

import httpx
from fastapi.testclient import TestClient

from proxy import features as fs
from proxy import usage as usage_log
from proxy.app import create_app
from proxy.config import Settings, load_settings, save_settings
from proxy.headroom_settings import normalize as headroom_normalize


def test_defaults_match_the_work_order():
    assert fs.defaults() == {
        "compression": True,
        "cost_control": False,
        "shunt": False,
        "green_metrics": False,
        "tracing": False,
        "run_label": "",
    }


def test_normalize_coerces_and_rejects():
    values = fs.normalize({"compression": "off", "run_label": "  ablate-A  "})
    assert values["compression"] is False
    assert values["run_label"] == "ablate-A"
    assert values["tracing"] is False  # missing key takes default
    try:
        fs.normalize({"not_a_feature": True})
        assert False, "unknown feature must be rejected"
    except ValueError:
        pass
    try:
        fs.normalize({"compression": "maybe"})
        assert False, "impossible bool must be rejected"
    except ValueError:
        pass


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "config.json"
    settings = Settings(upstream_url="http://gateway.test", exposed_models=["claude-haiku-5-5"])
    settings.features = fs.normalize({"compression": False, "tracing": True, "run_label": "ablate-B"})
    save_settings(settings, path)
    loaded = load_settings(path)
    assert loaded.features == settings.features
    assert json.loads(path.read_text(encoding="utf-8"))["features"]["run_label"] == "ablate-B"


def test_load_features_reads_config_at_request_time_and_fails_open(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"features": {"compression": False, "run_label": "run-42"}}), encoding="utf-8")
    assert fs.load_features(path) == {**fs.defaults(), "compression": False, "run_label": "run-42"}
    # a config rewritten underneath us (current run_label preserved, flag flipped) is picked up
    path.write_text(json.dumps({"features": {"compression": True, "run_label": "run-42"}}), encoding="utf-8")
    assert fs.load_features(path)["compression"] is True
    # missing file, corrupt file, and unknown/corrupt values all fail open to defaults
    assert fs.load_features(tmp_path / "missing.json") == fs.defaults()
    path.write_text("{not json", encoding="utf-8")
    assert fs.load_features(path) == fs.defaults()
    path.write_text(json.dumps({"features": {"compression": {}}}), encoding="utf-8")
    assert fs.load_features(path) == fs.defaults()


CLAUDE_HEADERS = {"x-claude-code-session-id": "sess-1", "x-api-key": "k"}
UPSTREAM_OK = {
    "id": "msg_up", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 1, "cache_read_input_tokens": 3,
              "cache_creation_input_tokens": 0},
}
REQUEST_BODY = {
    "model": "claude-haiku-5-5", "max_tokens": 10, "system": "sys",
    "messages": [{"role": "user", "content": "original long text"}],
}


def _app(tmp_path, feature_values, sidecar_handler, upstream_seen, ledger_dir):
    settings_path = tmp_path / "config.json"
    settings = Settings(upstream_url="http://gateway.test", exposed_models=["claude-haiku-5-5"])
    settings.headroom = headroom_normalize({"retries": 0, "backoff_ms": 0})
    settings.features = fs.normalize(feature_values)
    save_settings(settings, settings_path)

    def upstream(request):
        upstream_seen.append(json.loads(request.content))
        return httpx.Response(200, json=UPSTREAM_OK)

    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude.json",
        ledger_dir=ledger_dir,
        upstream_transport=httpx.MockTransport(upstream),
        sidecar_transport=httpx.MockTransport(sidecar_handler),
        manage_sidecar=False,
    )
    return TestClient(app)


def _compressed_reply(request):
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


def test_compression_off_forwards_uncompressed_and_skips_sidecar(tmp_path):
    upstream_seen = []
    sidecar_calls = []
    ledger_dir = tmp_path / "sessions"

    def sidecar(request):
        sidecar_calls.append(request.url.path)
        return httpx.Response(200, json={})

    client = _app(tmp_path, {"compression": False}, sidecar, upstream_seen, ledger_dir)
    resp = client.post("/v1/messages", json=REQUEST_BODY, headers=CLAUDE_HEADERS)
    assert resp.status_code == 200
    assert sidecar_calls == []  # the flag gates the call site: no sidecar round trip at all
    assert upstream_seen[0]["messages"][0]["content"] == "original long text"
    row = usage_log.recent(tmp_path / "usage.db", 1)[0]
    assert row["fallback"] == "compression_disabled"
    assert row["compressed"] == 0


def test_flags_are_read_per_request_and_flipped_without_restart(tmp_path):
    upstream_seen = []
    ledger_dir = tmp_path / "sessions"
    client = _app(tmp_path, {}, _compressed_reply, upstream_seen, ledger_dir)

    # default on: the compressed body reaches the gateway
    client.post("/v1/messages", json=REQUEST_BODY, headers=CLAUDE_HEADERS)
    assert upstream_seen[0]["messages"][0]["content"] == "short"

    # flip via the dashboard API: no app rebuild, no restart
    resp = client.put("/api/features", json={"values": {"compression": False}})
    assert resp.status_code == 200
    client.post("/v1/messages", json=REQUEST_BODY, headers=CLAUDE_HEADERS)
    assert upstream_seen[1]["messages"][0]["content"] == "original long text"

    # a config.json rewritten on disk (the ablation harness's path) is honored too
    settings_path = tmp_path / "config.json"
    config = json.loads(settings_path.read_text(encoding="utf-8"))
    config["features"]["compression"] = True
    settings_path.write_text(json.dumps(config), encoding="utf-8")
    client.post("/v1/messages", json=REQUEST_BODY, headers=CLAUDE_HEADERS)
    assert upstream_seen[2]["messages"][0]["content"] == "short"


def test_features_api_rejects_unknown_keys(tmp_path):
    upstream_seen = []
    client = _app(tmp_path, {}, _compressed_reply, upstream_seen, tmp_path / "sessions")
    resp = client.put("/api/features", json={"values": {"not_a_feature": True}})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
    data = client.get("/api/features").json()
    assert data["values"] == fs.defaults()
    assert {e["key"] for e in data["schema"]} == set(fs.defaults())
