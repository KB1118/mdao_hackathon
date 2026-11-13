"""Points Claude Code at the proxy by editing ~/.claude/settings.json, with a restorable backup."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"


def backup_path(settings_path: Path) -> Path:
    return settings_path.with_name(settings_path.stem + ".token-proxy-backup.json")


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return handle.read()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(tmp, path)


def _load_object(text: str | None) -> dict[str, Any]:
    if not text or not text.strip():
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Claude Code settings must be a JSON object")
    return data


def status(settings_path: Path) -> dict[str, Any]:
    base_url = None
    error = None
    try:
        data = _load_object(_read_text(settings_path))
        base_url = (data.get("env") or {}).get("ANTHROPIC_BASE_URL")
    except (ValueError, OSError) as exc:
        error = str(exc)
    return {
        "settings_path": str(settings_path),
        "base_url": base_url,
        "backup_exists": backup_path(settings_path).exists(),
        "error": error,
    }


def apply(settings_path: Path, proxy_url: str, exposed_models: list[str]) -> dict[str, Any]:
    backup = backup_path(settings_path)
    if not backup.exists():
        original = _read_text(settings_path)
        backup.write_text(
            json.dumps({"existed": original is not None, "content": original}),
            encoding="utf-8",
        )
    data = _load_object(_read_text(settings_path))
    env = data.get("env")
    if not isinstance(env, dict):
        env = {}
        data["env"] = env
    env["ANTHROPIC_BASE_URL"] = proxy_url
    if exposed_models:
        data["availableModels"] = list(exposed_models)
    _write_text(settings_path, json.dumps(data, indent=2) + "\n")
    return status(settings_path)


def restore(settings_path: Path) -> dict[str, Any]:
    backup = backup_path(settings_path)
    if not backup.exists():
        raise ValueError("No backup found, so there is nothing to restore")
    saved = json.loads(backup.read_text(encoding="utf-8"))
    if saved.get("existed"):
        _write_text(settings_path, saved.get("content") or "")
    elif settings_path.exists():
        settings_path.unlink()
    backup.unlink()
    return status(settings_path)
