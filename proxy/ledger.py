"""Append-only, privacy-first session ledger.

One JSONL file per session key under ``data/sessions/``. File names are the
sanitized key plus a short SHA-256 of the raw key, so no two distinct keys ever
collide and the name is safe on Windows and in URLs.

Entries carry metadata and SHA-256 content hashes only — conversation content,
system prompts, and tool schemas are never stored. The ledger is an observer:
every write failure goes to stderr and never breaks the request.

The ablation harness stamps each entry with the ``run_label`` from the features
block in config.json; ``request_id`` correlates all entries of one round trip.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NO_SESSION_KEY = "_no-session"
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_KEY_OK = re.compile(r"^[A-Za-z0-9._-]+$")


def session_path(ledger_dir: Path | str, session_key: str | None) -> Path:
    """Sanitized filename + short hash of the raw key, so distinct keys get distinct files."""
    key = session_key or NO_SESSION_KEY
    safe = _UNSAFE_CHARS.sub("_", key)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return Path(ledger_dir) / f"{safe}-{digest}.jsonl"


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(data: Any) -> str:
    return hashlib.sha256(_canonical(data).encode("utf-8")).hexdigest()


def _chars(data: Any) -> int:
    """Character size of the canonical serialization."""
    return len(_canonical(data))


def _block_counts(content: Any) -> dict[str, int]:
    if isinstance(content, str):
        return {"text": 1}
    return dict(Counter(block.get("type", "unknown") for block in content or [] if isinstance(block, dict)))


def _content_profile(content: Any) -> dict[str, Any] | None:
    if content is None or content == [] or content == "":
        return None
    return {"chars": _chars(content), "hash": _hash(content), "block_counts": _block_counts(content)}


def _message_profile(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content", "")
    return {
        "role": message.get("role", ""),
        "block_counts": _block_counts(content),
        "chars": _chars(content),
        "hash": _hash(content),
    }


def _envelope(
    kind: str,
    *,
    run_label: str,
    session_key: str | None,
    session_source: str | None,
    request_id: str,
) -> dict[str, Any]:
    return {
        "type": kind,
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label or "",
        "session_key": session_key,
        "session_source": session_source,
        "request_id": request_id,
    }


def _append(ledger_dir: Path | str, session_key: str | None, entry: dict[str, Any]) -> None:
    """Append one entry; a ledger failure logs to stderr and never raises."""
    try:
        path = session_path(ledger_dir, session_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"ledger write failed for session {session_key!r}: {exc}", file=sys.stderr)


def _system_profile(body: dict[str, Any]) -> dict[str, Any] | None:
    system = body.get("system")
    if system is None:
        return None
    return {"chars": _chars(system), "hash": _hash(system)}


def request_event(
    ledger_dir: Path | str,
    *,
    session_key: str | None,
    session_source: str | None,
    request_id: str,
    run_label: str,
    requested_model: str,
    model: str,
    route: str,
    stream: bool,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Log the request half: models, route, message/system/tool fingerprints. No payloads."""
    messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]
    tools = body.get("tools") or []
    entry = _envelope(
        "request", run_label=run_label, session_key=session_key,
        session_source=session_source, request_id=request_id,
    )
    entry.update({
        "requested_model": requested_model or "",
        "model": model or "",
        "route": route,
        "stream": bool(stream),
        "message_count": len(messages),
        "messages": [_message_profile(m) for m in messages],
        "system": _system_profile(body),
        "tools": {
            "names": [t.get("name", "") for t in tools if isinstance(t, dict)],
            "count": len(tools),
            "hash": _hash(tools),
        },
    })
    _append(ledger_dir, session_key, entry)
    return entry


def stage_event(
    ledger_dir: Path | str,
    *,
    session_key: str | None,
    session_source: str | None,
    request_id: str,
    run_label: str,
    feature: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log one pipeline stage and its provenance (tokens, fallback reason, duration)."""
    entry = _envelope(
        "stage", run_label=run_label, session_key=session_key,
        session_source=session_source, request_id=request_id,
    )
    entry.update({"feature": feature, "provenance": dict(provenance or {})})
    _append(ledger_dir, session_key, entry)
    return entry


def response_event(
    ledger_dir: Path | str,
    *,
    session_key: str | None,
    session_source: str | None,
    request_id: str,
    run_label: str,
    status: int,
    duration_ms: float,
    usage: dict[str, Any] | None = None,
    stop_reason: str | None = None,
    content: Any = None,
) -> dict[str, Any]:
    """Log the response half: status, duration, usage split, assistant fingerprint."""
    usage = usage or {}
    entry = _envelope(
        "response", run_label=run_label, session_key=session_key,
        session_source=session_source, request_id=request_id,
    )
    entry.update({
        "status": int(status),
        "duration_ms": int(duration_ms or 0),
        "usage": {
            "input": usage.get("input_tokens") or 0,
            "output": usage.get("output_tokens") or 0,
            "cache_read": usage.get("cache_read_input_tokens") or 0,
            "cache_write": usage.get("cache_creation_input_tokens") or 0,
        },
        "stop_reason": stop_reason,
        "content": _content_profile(content),
    })
    _append(ledger_dir, session_key, entry)
    return entry


def error_event(
    ledger_dir: Path | str,
    *,
    session_key: str | None,
    session_source: str | None,
    request_id: str,
    run_label: str,
    stage: str,
    message: Any,
) -> dict[str, Any]:
    """Log a stage failure: the stage that failed and why."""
    entry = _envelope(
        "error", run_label=run_label, session_key=session_key,
        session_source=session_source, request_id=request_id,
    )
    entry.update({"stage": stage, "message": str(message)[:1000]})
    _append(ledger_dir, session_key, entry)
    return entry


def list_sessions(ledger_dir: Path | str) -> list[dict[str, Any]]:
    """Summaries of every session file: names, run label, counts, size, first/last ts."""
    out: list[dict[str, Any]] = []
    root = Path(ledger_dir)
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.jsonl")):
        try:
            summary = _scan_file(path)
        except OSError:
            continue
        if summary is not None:
            out.append({"key": path.stem, **summary})
    return out


def _scan_file(path: Path) -> dict[str, Any] | None:
    entries = requests = 0
    first_ts = last_ts = None
    session_key = session_source = None
    run_label = ""
    with path.open("rb") as handle:
        for raw_line in handle:
            try:
                entry = json.loads(raw_line)
            except ValueError:
                continue
            if not isinstance(entry, dict):
                continue
            entries += 1
            ts = entry.get("ts")
            if first_ts is None:
                first_ts = ts
                session_key = entry.get("session_key")
                session_source = entry.get("session_source")
                run_label = entry.get("run_label") or ""
            if ts:
                last_ts = ts
            if entry.get("type") == "request":
                requests += 1
    if entries == 0:
        return None
    size_bytes = path.stat().st_size
    return {
        "session_key": session_key,
        "session_source": session_source,
        "run_label": run_label,
        "requests": requests,
        "entries": entries,
        "size_bytes": size_bytes,
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def read_session(ledger_dir: Path | str, key: str) -> dict[str, Any] | None:
    """One session's entries, keyed by the file stem used by /api/sessions."""
    if not _KEY_OK.match(key or "") or key.strip(".-") == "":
        return None
    path = Path(ledger_dir) / f"{key}.jsonl"
    if not path.is_file():
        return None
    entries: list[dict[str, Any]] = []
    session_key = session_source = None
    run_label = ""
    try:
        with path.open("rb") as handle:
            for raw_line in handle:
                try:
                    entry = json.loads(raw_line)
                except ValueError:
                    continue
                if not isinstance(entry, dict):
                    continue
                if session_key is None and entry.get("session_key"):
                    session_key = entry.get("session_key")
                    session_source = entry.get("session_source")
                    run_label = entry.get("run_label") or ""
                entries.append(entry)
    except OSError:
        return None
    return {"key": key, "session_key": session_key, "session_source": session_source,
            "run_label": run_label, "entries": entries}
