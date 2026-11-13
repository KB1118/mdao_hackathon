"""Feature flags: the ablation study's control surface.

Same declarative pattern as ``headroom_settings.py``: a schema rendered by the
dashboard, persisted under the ``features`` key in config.json. The five product
features are independently toggleable booleans; ``run_label`` is set by the
benchmark harness and stamped on every ledger entry.

Flags (and run_label) are read from config.json at request time via
:func:`load_features` — the ablation harness flips settings between runs without
restarting the proxy. A missing or corrupt config file fails open to defaults.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FEATURES: list[dict[str, Any]] = [
    {"key": "compression", "group": "Features", "kind": "flag", "type": "bool", "default": True,
     "label": "Compression",
     "help": "Compress requests through the Headroom sidecar. Off forwards every request untouched (the A/B control arm)."},
    {"key": "cost_control", "group": "Features", "kind": "flag", "type": "bool", "default": False,
     "label": "Cost control",
     "help": "Cross-session cost control: micro-skills, tool pruning, loop guard. Not implemented yet."},
    {"key": "shunt", "group": "Features", "kind": "flag", "type": "bool", "default": False,
     "label": "Shunt (delegation)",
     "help": "Delegate grunt work (bulk reads, boilerplate) to a cheaper worker model. Not implemented yet."},
    {"key": "green_metrics", "group": "Features", "kind": "flag", "type": "bool", "default": False,
     "label": "Green metrics",
     "help": "Energy and emissions estimates per request. Not implemented yet."},
    {"key": "tracing", "group": "Features", "kind": "flag", "type": "bool", "default": False,
     "label": "Session tracing",
     "help": "Append-only per-session ledger (metadata and content hashes only, never payloads) under data/sessions/."},
    {"key": "run_label", "group": "Benchmark", "kind": "string", "type": "str", "default": "",
     "label": "Run label",
     "help": "Set by the benchmark harness. Stamped on every ledger entry so ablation arms can be grouped."},
]

_BY_KEY = {entry["key"]: entry for entry in FEATURES}


def defaults() -> dict[str, Any]:
    return {entry["key"]: entry["default"] for entry in FEATURES}


def entry(key: str) -> dict[str, Any]:
    return _BY_KEY[key]


def _coerce(entry_: dict[str, Any], raw: Any) -> Any:
    key = entry_["key"]
    if entry_["type"] == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in ("1", "true", "yes", "on"):
                return True
            if lowered in ("0", "false", "no", "off"):
                return False
        raise ValueError(f"{key}: expected true or false")
    if raw is None:
        return ""
    return str(raw).strip()


def normalize(values: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce feature values; unknown keys raise, missing keys take defaults."""
    unknown = sorted(set(values) - set(_BY_KEY))
    if unknown:
        raise ValueError(f"unknown feature setting: {unknown[0]}")
    return {
        entry_["key"]: _coerce(entry_, values.get(entry_["key"], entry_["default"]))
        for entry_ in FEATURES
    }


def load_features(config_path: Path) -> dict[str, Any]:
    """Read the features block from config.json — the request-time source of truth.

    Called per request, so config edits take effect without a restart. Any read or
    validation failure fails open to defaults.
    """
    try:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return defaults()
    stored = raw.get("features") if isinstance(raw, dict) else None
    if not isinstance(stored, dict):
        return defaults()
    known = {key: value for key, value in stored.items() if key in _BY_KEY}
    try:
        return normalize(known)
    except ValueError:
        return defaults()
