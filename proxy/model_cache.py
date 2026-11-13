"""Pre-heat the HuggingFace cache the Headroom sidecar needs for Kompress.

Offline mode is the sidecar's privacy posture (no telemetry, no outbound
connections), but on a fresh machine it also means the ML compressor can never
fetch its model and silently degrades to structural-only compression. This
module is the fix: the *proxy process* — not the offline-guarded sidecar —
checks the local HF cache at sidecar start, and downloads the pinned artefacts
if they are missing.

Pins match ``headroom-ai[proxy]==0.44.0`` (``headroom/onnx_runtime.py``
``_PINNED_REVISIONS`` and ``transforms/kompress_compressor.py``). A headroom
version bump requires re-checking these; the audit stays correct but downloads
the old revision otherwise. Downloads go through `huggingface_hub
snapshot_download`, so the files land in exactly the cache layout the sidecar
probes with ``local_files_only=True``.

Everything here fails open: a missing cache, a failed probe, or a failed
download is reported in the status and never raised at a caller.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPOS: dict[str, dict[str, Any]] = {
    "chopratejas/kompress-v2-base": {
        "revision": "b1563631b35bfdcee37587ad530147497d820d4c",
        "files": [
            "onnx/kompress-int8-wo.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
        ],
    },
    "answerdotai/ModernBERT-base": {
        "revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
        "files": [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
        ],
    },
}

TIKTOKEN_ENCODING = "o200k_base"
TIKTOKEN_BLOB_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"

_state: dict[str, Any] = {
    "status": "unknown",  # unknown | missing | downloading | cached | error
    "detail": "",
    "missing": [],
    "changed_at": None,
}
_lock = threading.Lock()
_worker: threading.Thread | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set(**updates: Any) -> None:
    _state.update(updates)
    _state["changed_at"] = _now()


def status() -> dict[str, Any]:
    return {**_state}


def current_status() -> dict[str, Any]:
    """Live status: re-audit locally when idle; report the download while it runs."""
    with _lock:
        if _state["status"] in ("unknown", "cached", "error") and not (_worker and _worker.is_alive()):
            info = audit()
            _set(
                status="cached" if info["cached"] else "missing",
                detail=info.get("probe_error") or "",
                missing=info["missing"],
            )
        return status()


# --- Offline probes (no network) ---------------------------------------------

def tiktoken_cached() -> bool:
    """True when the tiktoken vocabulary file is in tiktoken's own cache.

    Mirrors tiktoken/load.py: TIKTOKEN_CACHE_DIR or DATA_GYM_CACHE_DIR, else
    <tempdir>/data-gym-cache, with the sha1 of the blob URL as file name.
    """
    try:
        if "TIKTOKEN_CACHE_DIR" in os.environ:
            cache_dir = os.environ["TIKTOKEN_CACHE_DIR"]
        elif "DATA_GYM_CACHE_DIR" in os.environ:
            cache_dir = os.environ["DATA_GYM_CACHE_DIR"]
        else:
            import tempfile
            cache_dir = os.path.join(tempfile.gettempdir(), "data-gym-cache")
        if cache_dir == "":
            return False
        name = hashlib.sha1(TIKTOKEN_BLOB_URL.encode()).hexdigest()
        return (Path(cache_dir) / name).is_file()
    except Exception:  # noqa: BLE001 - a broken probe must never break the request path
        return False


def audit() -> dict[str, Any]:
    """Local-only probe of the HF cache + tiktoken vocabulary. Never any network."""
    missing: list[str] = []
    probe_error = ""
    try:
        from huggingface_hub import try_to_load_from_cache
        try:  # sentinel moved between huggingface_hub versions
            from huggingface_hub.constants import _CACHED_NO_EXIST  # type: ignore[attr-defined]
        except ImportError:  # pragma: no cover - depends on installed version
            from huggingface_hub.file_download import _CACHED_NO_EXIST
        for repo, spec in REPOS.items():
            for filename in spec["files"]:
                found = try_to_load_from_cache(repo, filename, revision=spec["revision"])
                if found is _CACHED_NO_EXIST or not found or not Path(str(found)).is_file():
                    missing.append(f"{repo}/{filename}@{spec['revision'][:12]}")
    except Exception as exc:  # noqa: BLE001
        probe_error = str(exc)
    try:
        if not tiktoken_cached():
            missing.append(f"tiktoken:{TIKTOKEN_ENCODING}")
    except Exception:  # noqa: BLE001 - a broken probe reads as "not cached"
        missing.append(f"tiktoken:{TIKTOKEN_ENCODING}")
    return {"cached": not missing and not probe_error, "missing": missing, "probe_error": probe_error}


# --- Download (the only place with network) -----------------------------------

def _download() -> None:
    """Fetch the pinned artefacts into the HF cache and warm the tiktoken vocabulary.

    Runs in the proxy process, which is not offline-guarded; the sidecar never
    gains network access. HF offline flags are cleared for the call because the
    dashboard may have set them for the child sidecar environment.
    """
    from huggingface_hub import snapshot_download

    saved = {key: os.environ.pop(key) for key in ("HF_HUB_OFFLINE", "HUGGINGFACE_HUB_OFFLINE")
             if key in os.environ}
    try:
        for repo, spec in REPOS.items():
            snapshot_download(repo_id=repo, revision=spec["revision"],
                             allow_patterns=list(spec["files"]))
        if not tiktoken_cached():
            import tiktoken
            tiktoken.get_encoding(TIKTOKEN_ENCODING)
    finally:
        os.environ.update(saved)


def _run(on_ready: Callable[[], None] | None) -> None:
    try:
        _download()
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _set(status="error", detail=str(exc)[:500])
        print(f"Kompress model cache download failed: {exc}", file=sys.stderr)
        return
    info = audit()
    with _lock:
        if info["cached"]:
            _set(status="cached", detail="", missing=[])
        else:
            _set(status="error", detail="download finished but artefacts still missing",
                 missing=info["missing"])
        ready = _state["status"] == "cached"
    if ready and on_ready is not None:
        on_ready()


def ensure(*, background: bool = False, on_ready: Callable[[], None] | None = None) -> dict[str, Any]:
    """Check the cache and, if anything is missing, download it.

    Failures are recorded in the status, never raised. ``on_ready`` runs after a
    download that ended with a warm cache (e.g. to restart the degraded sidecar).
    """
    global _worker
    with _lock:
        if _worker is not None and _worker.is_alive():
            return status()
        info = audit()
        if info["cached"]:
            _set(status="cached", detail="", missing=[])
            return status()
        _set(status="missing", detail=info.get("probe_error") or "", missing=info["missing"])
        _set(status="downloading")
        if background:
            _worker = threading.Thread(
                target=_run, args=(on_ready,), daemon=True, name="kompress-cache"
            )
            _worker.start()
            return status()
    _run(on_ready)
    with _lock:
        return status()
