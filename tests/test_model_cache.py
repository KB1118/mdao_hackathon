"""Kompress model-cache pre-heat: check, download, fail-open, and app wiring."""

import threading

import httpx
from fastapi.testclient import TestClient

from proxy import model_cache
from proxy.app import create_app
from proxy.config import Settings, save_settings
from proxy.headroom_settings import normalize as headroom_normalize


def _reset_state(monkeypatch):
    monkeypatch.setattr(model_cache, "_state", {
        "status": "unknown", "detail": "", "missing": [], "changed_at": None})
    monkeypatch.setattr(model_cache, "_worker", None)
    monkeypatch.setattr(model_cache, "_lock", threading.Lock())


def test_audit_on_a_cold_cache_lists_missing_without_network(monkeypatch, tmp_path):
    _reset_state(monkeypatch)
    # point every cache location at empty dirs: the audit must report missing
    # files and the tiktoken vocabulary without ever reaching the network.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hf" / "hub"))
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "tiktoken"))
    info = model_cache.audit()
    assert info["cached"] is False
    assert info["probe_error"] == ""
    expected = {f"{repo}/{f}@{spec['revision'][:12]}" for repo, spec in model_cache.REPOS.items()
                for f in spec["files"]}
    expected.add("tiktoken:o200k_base")
    assert set(info["missing"]) == expected


def test_ensure_downloads_when_missing_and_marks_cached(monkeypatch):
    _reset_state(monkeypatch)
    calls = {"download": 0, "ready": 0}
    monkeypatch.setattr(model_cache, "audit",
                        lambda: {"cached": False, "missing": ["x"], "probe_error": ""} if calls["download"] == 0
                        else {"cached": True, "missing": [], "probe_error": ""})
    monkeypatch.setattr(model_cache, "_download", lambda: calls.__setitem__("download", calls["download"] + 1))
    result = model_cache.ensure(background=False, on_ready=lambda: calls.__setitem__("ready", calls["ready"] + 1))
    assert result["status"] == "cached"
    assert calls == {"download": 1, "ready": 1}


def test_ensure_skips_download_when_cached(monkeypatch):
    _reset_state(monkeypatch)
    calls = {"download": 0, "ready": 0}
    monkeypatch.setattr(model_cache, "audit", lambda: {"cached": True, "missing": [], "probe_error": ""})
    monkeypatch.setattr(model_cache, "_download", lambda: calls.__setitem__("download", calls["download"] + 1))
    result = model_cache.ensure(background=False, on_ready=lambda: calls.__setitem__("ready", calls["ready"] + 1))
    assert result["status"] == "cached"
    assert calls == {"download": 0, "ready": 0}


def test_download_failure_is_recorded_not_raised(monkeypatch):
    _reset_state(monkeypatch)

    def boom():
        raise RuntimeError("no network")

    monkeypatch.setattr(model_cache, "audit", lambda: {"cached": False, "missing": ["x"], "probe_error": ""})
    monkeypatch.setattr(model_cache, "_download", boom)
    result = model_cache.ensure(background=False)
    assert result["status"] == "error"
    assert "no network" in result["detail"]


def test_background_download_runs_in_a_worker_and_calls_ready(monkeypatch):
    _reset_state(monkeypatch)
    calls = {"download": 0, "ready": 0}
    monkeypatch.setattr(model_cache, "audit",
                        lambda: {"cached": False, "missing": ["x"], "probe_error": ""} if calls["download"] == 0
                        else {"cached": True, "missing": [], "probe_error": ""})
    monkeypatch.setattr(model_cache, "_download", lambda: calls.__setitem__("download", calls["download"] + 1))
    result = model_cache.ensure(background=True, on_ready=lambda: calls.__setitem__("ready", calls["ready"] + 1))
    assert result["status"] == "downloading"
    worker = model_cache._worker
    assert worker is not None and worker.is_alive()
    worker.join(timeout=10)
    assert calls == {"download": 1, "ready": 1}
    assert model_cache.status()["status"] == "cached"


def test_tiktoken_cache_check_is_offline_and_fail_open(monkeypatch, tmp_path):
    ok_dir = tmp_path / "has"
    ok_dir.mkdir()
    import hashlib
    name = hashlib.sha1(model_cache.TIKTOKEN_BLOB_URL.encode()).hexdigest()
    (ok_dir / name).write_text("bpe", encoding="utf-8")
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(ok_dir))
    assert model_cache.tiktoken_cached() is True
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "nope"))
    assert model_cache.tiktoken_cached() is False
    # a probe crash reads as "not cached", never as an exception
    monkeypatch.setattr(model_cache, "tiktoken_cached", None)
    monkeypatch.setattr(model_cache, "tiktoken_cached",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert model_cache.audit()["cached"] is False


_ENSURE_CALLS: list[dict] = []


def _app(tmp_path, *, manage_sidecar, headroom_values=None, monkeypatch=None):
    settings_path = tmp_path / "config.json"
    settings = Settings(upstream_url="http://gateway.test", exposed_models=["claude-haiku-5-5"])
    settings.headroom = headroom_normalize(headroom_values or {})
    save_settings(settings, settings_path)
    if monkeypatch is not None:
        # never start a real sidecar or download in tests; record ensure() calls
        monkeypatch.setattr("proxy.app.Sidecar", _FakeSidecar, raising=True)
        monkeypatch.setattr(
            "proxy.model_cache.ensure",
            lambda **kwargs: _ENSURE_CALLS.append(kwargs) or model_cache.status())
        monkeypatch.setattr("proxy.model_cache.current_status",
                            lambda: {"status": "cached", "detail": "", "missing": [], "changed_at": "now"})
    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude.json",
        ledger_dir=tmp_path / "sessions",
        upstream_transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        manage_sidecar=manage_sidecar,
    )
    return app


class _FakeSidecar:
    def __init__(self, log_path):  # noqa: ANN001
        self.running = False

    def start(self, env, port):  # noqa: ANN001
        self.running = True

    def stop(self):
        self.running = False

    def restart(self, env, port):  # noqa: ANN001
        self.running = True

    def status(self):
        return {"running": self.running, "pid": None, "port": None, "uptime_seconds": None,
                "log_path": ""}


def test_model_endpoints_and_lifespan_prefetch(tmp_path, monkeypatch):
    _ENSURE_CALLS.clear()
    app = _app(tmp_path, manage_sidecar=True, monkeypatch=monkeypatch)
    with TestClient(app) as client:  # lifespan runs: sidecar starts, cache prefetch fires
        view = client.get("/api/headroom/model").json()
        assert view["status"] == "cached"
        post = client.post("/api/headroom/model").json()
        assert post["status"] == "cached"
    assert any(c.get("background") for c in _ENSURE_CALLS), "lifespan must prefetch in the background"


def test_prefetch_skipped_when_sidecar_unmanaged_or_kompress_disabled(tmp_path, monkeypatch):
    _ENSURE_CALLS.clear()
    # manage_sidecar=False: no prefetch at all
    app = _app(tmp_path, manage_sidecar=False, monkeypatch=monkeypatch)
    with TestClient(app) as client:
        client.get("/api/headroom/model")
    assert _ENSURE_CALLS == []

    # Kompress disabled: the feature is off, so no pre-heat either
    app = _app(tmp_path, manage_sidecar=True, headroom_values={"HEADROOM_DISABLE_KOMPRESS": True},
               monkeypatch=monkeypatch)
    with TestClient(app) as client:
        client.get("/api/headroom/model")
    assert _ENSURE_CALLS == []
