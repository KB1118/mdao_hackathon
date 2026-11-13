"""Dashboard schema for the Headroom sidecar and our client policy.

Every entry is rendered as a control on the dashboard. ``kind`` decides where the value goes:

- ``env``: an environment variable read by the Headroom sidecar at startup.
- ``request``: a field in the ``config`` object of each /v1/compress call.
- ``client``: our proxy's own policy (retries, failure handling, session sources).
- ``sidecar``: the sidecar's process settings (port).

Names were checked against the installed headroom-ai package; see tests/test_headroom_settings.py.
"""

from __future__ import annotations

from typing import Any

PROXY_TOKEN_HEADER = "X-Headroom-Proxy-Token"

SCHEMA: list[dict[str, Any]] = [
    # --- Privacy and runtime -------------------------------------------------
    {"key": "HEADROOM_BEACON", "group": "Privacy and runtime", "kind": "env", "type": "choice",
     "choices": ["off", "on"], "default": "off", "restart": True,
     "label": "Anonymous usage beacon",
     "help": "Sends compression counters to Headroom Labs. Keep off so nothing leaves this machine."},
    {"key": "HEADROOM_TELEMETRY", "group": "Privacy and runtime", "kind": "env", "type": "choice",
     "choices": ["on", "off"], "default": "on", "restart": True,
     "label": "Local statistics",
     "help": "Keeps counters for /stats on this machine only."},
    {"key": "HEADROOM_OFFLINE", "group": "Privacy and runtime", "kind": "env", "type": "bool",
     "default": True, "restart": True,
     "label": "Offline mode",
     "help": "Refuses every outbound connection Headroom would start. The Kompress model must be downloaded first."},
    {"key": "HF_HUB_OFFLINE", "group": "Privacy and runtime", "kind": "env", "type": "bool",
     "default": True, "restart": True,
     "label": "Hugging Face cache only",
     "help": "Loads the Kompress model from the local cache and never contacts Hugging Face."},
    {"key": "HEADROOM_REQUIRE_RUST_CORE", "group": "Privacy and runtime", "kind": "env", "type": "bool",
     "default": True, "restart": True,
     "label": "Require Rust core",
     "help": "Refuses to start if the Rust extension will not load. Turning this off starts a degraded proxy."},
    {"key": "HEADROOM_LOG_LEVEL", "group": "Privacy and runtime", "kind": "env", "type": "choice",
     "choices": ["critical", "error", "warning", "info", "debug"], "default": "warning", "restart": True,
     "label": "Log level", "help": "Verbosity of the sidecar's uvicorn log."},
    # --- Savings posture -----------------------------------------------------
    {"key": "HEADROOM_MODE", "group": "Savings posture", "kind": "env", "type": "choice",
     "choices": ["cache", "token"], "default": "cache", "restart": True,
     "label": "Mode",
     "help": "cache keeps the provider's prompt cache intact. token removes more, but rewrites earlier turns."},
    {"key": "HEADROOM_SAVINGS_PROFILE", "group": "Savings posture", "kind": "env", "type": "choice",
     "choices": ["coding", "balanced", "agent-90", "general"], "default": "coding", "restart": True,
     "label": "Savings profile",
     "help": "Preset posture: which messages are compressed and how hard. coding is the default for agent traffic."},
    {"key": "HEADROOM_KOMPRESS_BACKEND", "group": "Savings posture", "kind": "env", "type": "choice",
     "choices": ["auto", "onnx_cpu", "onnx_coreml", "pytorch", "pytorch_mps"], "default": "onnx_cpu",
     "restart": True, "label": "Kompress backend",
     "help": "Compute backend for the prose compressor. onnx_cpu runs on any machine."},
    {"key": "HEADROOM_DISABLE_KOMPRESS", "group": "Savings posture", "kind": "env", "type": "bool",
     "default": False, "restart": True, "label": "Disable Kompress",
     "help": "Turns off ML prose compression and keeps the structural compressors."},
    {"key": "HEADROOM_FORCE_KOMPRESS_ALL", "group": "Savings posture", "kind": "env", "type": "bool",
     "default": False, "restart": True, "label": "Kompress everything",
     "help": "Sends all content through Kompress, bypassing per-type routing."},
    {"key": "HEADROOM_OUTPUT_SHAPER", "group": "Savings posture", "kind": "env", "type": "bool",
     "default": False, "restart": True, "label": "Output steering",
     "help": "Appends a terse-answer note to system prompts. Off in cache mode."},
    {"key": "HEADROOM_TOOL_SEARCH", "group": "Savings posture", "kind": "env", "type": "bool",
     "default": False, "restart": True, "available": False, "label": "Tool search",
     "help": "Defers tool schemas. Needs the headroom-tool-search extension, which is not installed."},
    # --- Compression tuning --------------------------------------------------
    {"key": "HEADROOM_COMPRESS_USER_MESSAGES", "group": "Compression tuning", "kind": "env", "type": "bool",
     "default": True, "restart": True, "label": "Compress user messages",
     "help": "Compress text inside user messages. Anthropic tool results compress either way."},
    {"key": "HEADROOM_COMPRESS_SYSTEM_MESSAGES", "group": "Compression tuning", "kind": "env", "type": "bool",
     "default": False, "restart": True, "label": "Compress system messages",
     "help": "Off by default to keep the hottest prompt-cache prefix stable."},
    {"key": "HEADROOM_PROTECT_TOOL_RESULTS", "group": "Compression tuning", "kind": "env", "type": "str",
     "default": "", "restart": True, "label": "Protected tools",
     "help": "Comma-separated tool names whose results are never lossy-compressed."},
    {"key": "HEADROOM_TOOL_DESC_MAX_CHARS", "group": "Compression tuning", "kind": "env", "type": "int",
     "default": None, "min": 1, "restart": True, "label": "Max tool description length",
     "help": "Truncates tool descriptions to this many characters. Empty leaves them unchanged."},
    # --- Request compression (sent with every call) ----------------------------
    {"key": "config_mode", "group": "Request compression", "kind": "request", "type": "choice",
     "choices": ["lossless", "lossy_inline", "unset"], "default": "lossless", "label": "Compression mode",
     "help": "lossless loses nothing. lossy_inline folds losslessly, then removes content. unset removes content with no markers."},
    {"key": "target_ratio", "group": "Request compression", "kind": "request", "type": "float",
     "default": None, "min": 0.01, "max": 1.0, "label": "Target ratio",
     "help": "Keep ratio for prose compression. Lower compresses harder. Empty uses the model's choice."},
    {"key": "protect_recent", "group": "Request compression", "kind": "request", "type": "int",
     "default": 0, "min": 0, "label": "Protect recent messages",
     "help": "Leaves the last N messages uncompressed."},
    {"key": "protect_analysis_context", "group": "Request compression", "kind": "request", "type": "bool",
     "default": True, "label": "Protect analysis context",
     "help": "Keeps code intact when the request asks for analysis or review."},
    {"key": "token_budget", "group": "Request compression", "kind": "request", "type": "int",
     "default": 1000000, "min": 1000, "label": "Context limit (tokens)",
     "help": "Context window used to judge compression pressure. Set it to your model's window."},
    # --- Session and gateway -----------------------------------------------------
    {"key": "HEADROOM_GATEWAY_CONTRACT", "group": "Session and gateway", "kind": "env", "type": "bool",
     "default": True, "restart": True, "label": "Gateway contract",
     "help": "Enables /v1/compress/response, which gateway mode needs for usage relay."},
    {"key": "HEADROOM_COMPRESS_SESSION_FROM_HEADER", "group": "Session and gateway", "kind": "env", "type": "bool",
     "default": False, "restart": True, "label": "Session from header",
     "help": "Lets x-headroom-session-id stand in for the session id. Off by default; subagents share IDs."},
    {"key": "HEADROOM_COMPRESS_SESSION_STORE", "group": "Session and gateway", "kind": "env", "type": "str",
     "default": "", "restart": True, "label": "Session store URL",
     "help": "Redis URL for sharing session state between sidecars. Empty keeps state in this process."},
    {"key": "HEADROOM_COMPRESSION_CACHE_TTL_SECONDS", "group": "Session and gateway", "kind": "env", "type": "int",
     "default": 3900, "min": 600, "restart": True, "label": "Session cache TTL (s)",
     "help": "How long an idle session keeps its replay state. Minimum 600."},
    {"key": "HEADROOM_GATEWAY_TURN_TTL_SECONDS", "group": "Session and gateway", "kind": "env", "type": "int",
     "default": 120, "min": 1, "restart": True, "label": "Pending turn TTL (s)",
     "help": "How long a turn waits for its response half."},
    {"key": "HEADROOM_GATEWAY_MAX_PENDING_TURNS", "group": "Session and gateway", "kind": "env", "type": "int",
     "default": 10000, "min": 1, "restart": True, "label": "Max pending turns",
     "help": "Upper bound on turns waiting for a response half. Oldest are evicted first."},
    {"key": "HEADROOM_GATEWAY_MAX_REDRIVES", "group": "Session and gateway", "kind": "env", "type": "int",
     "default": 8, "min": 0, "restart": True, "label": "Max re-drives",
     "help": "Re-drives allowed per turn. Only used when re-drive is enabled."},
    {"key": "can_redrive", "group": "Session and gateway", "kind": "request", "type": "bool",
     "default": False, "forced": False, "available": False, "label": "Allow re-drive",
     "help": "Not implemented in this proxy. Needed only for CCR markers and turn hooks."},
    {"key": "session_affinity", "group": "Session and gateway", "kind": "request", "type": "bool",
     "default": True, "label": "Session affinity",
     "help": "Pins a session to one sidecar. Keep on while there is one sidecar."},
    {"key": "can_relay_response", "group": "Session and gateway", "kind": "request", "type": "bool",
     "default": True, "label": "Relay usage",
     "help": "Sends provider usage back so cache hits are counted correctly."},
    # --- Sidecar process ---------------------------------------------------------
    {"key": "sidecar_enabled", "group": "Sidecar process", "kind": "client", "type": "bool",
     "default": True, "label": "Use the sidecar",
     "help": "Off forwards requests without compression, for comparison."},
    {"key": "sidecar_port", "group": "Sidecar process", "kind": "sidecar", "type": "int",
     "default": 8787, "min": 1024, "max": 65535, "restart": True, "label": "Sidecar port",
     "help": "Loopback port the sidecar listens on."},
    {"key": "HEADROOM_REQUEST_TIMEOUT", "group": "Sidecar process", "kind": "env", "type": "int",
     "default": 300, "min": 1, "restart": True, "label": "Request timeout (s)",
     "help": "Upstream request timeout used inside the sidecar."},
    {"key": "HEADROOM_COMPRESSION_MAX_WORKERS", "group": "Sidecar process", "kind": "env", "type": "int",
     "default": None, "min": 1, "restart": True, "label": "Compression threads",
     "help": "Threads for compression work. Empty uses the CPU count."},
    {"key": "HEADROOM_PROXY_TOKEN", "group": "Sidecar process", "kind": "env", "type": "str",
     "default": "", "restart": True, "label": "Sidecar token",
     "help": f"Optional shared secret. When set, the proxy sends it as {PROXY_TOKEN_HEADER}."},
    # --- Client policy ---------------------------------------------------------------
    {"key": "request_timeout_ms", "group": "Failure handling", "kind": "client", "type": "int",
     "default": 10000, "min": 100, "label": "Compression timeout (ms)",
     "help": "How long one compression call may take before it counts as failed."},
    {"key": "retries", "group": "Failure handling", "kind": "client", "type": "int",
     "default": 2, "min": 0, "max": 10, "label": "Retries",
     "help": "Extra attempts after a failed compression call."},
    {"key": "backoff_ms", "group": "Failure handling", "kind": "client", "type": "int",
     "default": 200, "min": 0, "label": "Backoff (ms)",
     "help": "Wait before the first retry. Doubles on each further retry."},
    {"key": "failure_policy", "group": "Failure handling", "kind": "client", "type": "choice",
     "choices": ["forward_original", "fail_request"], "default": "forward_original",
     "label": "When every attempt fails",
     "help": "forward_original sends the uncompressed request (costs a cache miss). fail_request returns an error."},
    {"key": "relay_usage", "group": "Failure handling", "kind": "client", "type": "bool",
     "default": True, "label": "Post usage after each turn",
     "help": "Required for cache-hit accounting in the sidecar's /stats."},
    # --- Session sources -----------------------------------------------------------------
    {"key": "session_claude_header", "group": "Session sources", "kind": "client", "type": "bool",
     "default": True, "label": "Claude Code header",
     "help": "Uses x-claude-code-session-id as the session id."},
    {"key": "session_claude_metadata", "group": "Session sources", "kind": "client", "type": "bool",
     "default": True, "label": "Claude Code metadata",
     "help": "Uses session_id inside metadata.user_id when the header is missing."},
    {"key": "session_codex_header", "group": "Session sources", "kind": "client", "type": "bool",
     "default": True, "label": "Codex header",
     "help": "Uses the session-id header sent by Codex CLI."},
    {"key": "session_codex_prompt_cache_key", "group": "Session sources", "kind": "client", "type": "bool",
     "default": True, "label": "Codex prompt cache key",
     "help": "Uses prompt_cache_key from the body when no header is present."},
]

_BY_KEY = {entry["key"]: entry for entry in SCHEMA}


def defaults() -> dict[str, Any]:
    values = {entry["key"]: entry["default"] for entry in SCHEMA}
    values["can_redrive"] = False
    return values


def _coerce(entry: dict[str, Any], raw: Any) -> Any:
    key = entry["key"]
    kind = entry["type"]
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        if kind in ("int", "float") or entry.get("default") is None:
            return None
        if kind == "str":
            return ""
    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.strip().lower() in ("1", "true", "yes", "on"):
            return True
        if isinstance(raw, str) and raw.strip().lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{key}: expected true or false")
    if kind == "int":
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}: expected a whole number") from exc
    elif kind == "float":
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}: expected a number") from exc
    elif kind == "choice":
        value = str(raw)
        if value not in entry["choices"]:
            raise ValueError(f"{key}: must be one of {', '.join(entry['choices'])}")
        return value
    else:
        return str(raw).strip()
    if "min" in entry and value < entry["min"]:
        raise ValueError(f"{key}: must be at least {entry['min']}")
    if "max" in entry and value > entry["max"]:
        raise ValueError(f"{key}: must be at most {entry['max']}")
    return value


def normalize(values: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce dashboard values; unknown keys raise, missing keys take defaults."""
    unknown = sorted(set(values) - set(_BY_KEY))
    if unknown:
        raise ValueError(f"unknown setting: {unknown[0]}")
    out: dict[str, Any] = {}
    for entry in SCHEMA:
        key = entry["key"]
        raw = values.get(key, entry["default"])
        if entry.get("forced") is not None and not entry.get("available", True):
            out[key] = entry["forced"]
            continue
        out[key] = _coerce(entry, raw)
    return out


def sidecar_env(values: dict[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {}
    for entry in SCHEMA:
        if entry["kind"] != "env":
            continue
        value = values.get(entry["key"])
        if value is None or value == "":
            continue
        if entry["type"] == "bool":
            env[entry["key"]] = "1" if value else "0"
        else:
            env[entry["key"]] = str(value)
    env.setdefault("HF_HUB_OFFLINE", "1" if values.get("HF_HUB_OFFLINE", True) else "0")
    return env


def request_config(values: dict[str, Any], session_id: str) -> dict[str, Any]:
    config: dict[str, Any] = {"session_id": session_id}
    mode = values.get("config_mode")
    if mode and mode != "unset":
        config["mode"] = mode
    if values.get("target_ratio") is not None:
        config["target_ratio"] = values["target_ratio"]
    config["protect_recent"] = values.get("protect_recent", 0)
    config["protect_analysis_context"] = values.get("protect_analysis_context", True)
    return config


def gateway_block(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "can_redrive": False,
        "can_relay_response": bool(values.get("can_relay_response", True)),
        "session_affinity": bool(values.get("session_affinity", True)),
        "plugin_version": "token-proxy/0.1",
    }


def entry(key: str) -> dict[str, Any]:
    return _BY_KEY[key]
