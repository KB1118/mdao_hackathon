"""Derive the conversation key sent to the sidecar as config.session_id.

Sources, from probes of Claude Code 2.1.322 and Codex CLI 0.165.0:

- Claude Code sends x-claude-code-session-id, and the same value in metadata.user_id JSON.
  Subagents reuse that value, so the key also includes a fingerprint of the system prompt
  and tool names. Without it, a subagent's turns would blend into the parent's replay state.
- Codex CLI sends session-id (header) and prompt_cache_key (body), both equal to the session.

Keys are namespaced by agent family, not by header, so the same session always maps to the
same key whichever source supplied it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,256}$")


@dataclass(frozen=True)
class SessionKey:
    key: str
    source: str


def _valid(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if SESSION_ID_RE.match(value) else None


def _claude_metadata_session(user_id: Any) -> str | None:
    if not isinstance(user_id, str):
        return None
    try:
        data = json.loads(user_id)
    except ValueError:
        return None
    if isinstance(data, dict):
        return _valid(data.get("session_id"))
    return None


def fingerprint(body: Mapping[str, Any]) -> str:
    """Stable digest of the system prompt and tool names; subagents differ here."""
    system = body.get("system", body.get("instructions"))
    tools = [tool.get("name", "") for tool in body.get("tools") or [] if isinstance(tool, dict)]
    material = json.dumps(system, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    material += "\0" + json.dumps(tools, ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def extract_session(
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    enabled: Mapping[str, bool],
) -> SessionKey | None:
    lowered = {name.lower(): value for name, value in headers.items()}
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    candidates: list[tuple[str, str, str | None]] = []
    if enabled.get("claude_header", True):
        candidates.append(("claude_header", "claude", _valid(lowered.get("x-claude-code-session-id"))))
    if enabled.get("claude_metadata", True):
        candidates.append(("claude_metadata", "claude", _claude_metadata_session(metadata.get("user_id"))))
    if enabled.get("codex_header", True):
        candidates.append(("codex_header", "codex", _valid(lowered.get("session-id"))))
    if enabled.get("codex_prompt_cache_key", True):
        candidates.append(("codex_prompt_cache_key", "codex", _valid(body.get("prompt_cache_key"))))
    for source, family, raw in candidates:
        if raw:
            return SessionKey(key=f"{family}:{raw}:{fingerprint(body)}", source=source)
    return None
