# Session identifiers: probe results

Captured with `tools/capture_probe.py` (a logging forwarder) against Claude Code 2.1.322 and Codex CLI 0.165.0. Raw captures are in `probe_data/capture.jsonl`.

| Agent | Header(s) | Body field(s) | Notes |
|---|---|---|---|
| Claude Code | `x-claude-code-session-id` | `metadata.user_id` (JSON string with `session_id`) | Same value in both. Subagent requests reuse the parent's session id, but have different system prompts (3,129 vs 9,006 chars) and tool lists. |
| Codex CLI | `session-id`, `thread-id`, `x-client-request-id` | `prompt_cache_key` | All equal the session id in a single run. `x-codex-turn-metadata` carries `agent_name`, which separates subagents. |
| OpenCode | deferred | deferred | Not probed. |
| Cursor, Aider, Continue | unknown | unknown | Not probed. |

## Design consequence

Claude Code subagents share the parent's session id, so the conversation key is `family:session_id:fingerprint`. The fingerprint is a hash of the system prompt and tool names. This keeps subagent turns in their own replay state.

Implemented in `proxy/session_key.py`, with tests in `tests/test_session_key.py`.
