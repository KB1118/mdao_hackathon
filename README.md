# Token Proxy

A company-internal gateway that sits between AI coding agents (currently Claude Code) and model APIs. Two things at once: an **in-path HTTP proxy** that intercepts and observes every model request, and a **managed Headroom sidecar** running out-of-path compression. Concretely it is three services:

- **Proxy** on `127.0.0.1:18090` — FastAPI app (`proxy/app.py`). Handles `/v1/messages` (Anthropic Messages API), `/v1/messages/count_tokens`, `/v1/models`, the dashboard, and its `/api/*` endpoints. Routes `gpt-*` models by translating Anthropic Messages ⇄ OpenAI Responses (both directions, including SSE streaming); everything else is Anthropic passthrough.
- **Headroom sidecar** on `127.0.0.1:8787` — headroom-ai[proxy] (pinned `0.44.0`) as a managed child process, using its gateway contract (`/v1/compress` + `/v1/compress/response` usage relay). The proxy starts, stops, and restarts it.
- **Gateway** — the upstream OpenAI/Anthropic-compatible endpoint (default `http://127.0.0.1:18080`, configured on the dashboard). It also works against any Anthropic-API-compatible upstream; OpenRouter (`https://openrouter.ai/api`) was verified end-to-end.

```
Claude Code -> proxy -> [identity] -> [ledger request event]
                    -> [compression stage, feature-gated] -> sidecar (127.0.0.1:8787)
                    -> [route: anthropic passthrough | translate to Responses]
                    -> gateway -> proxy observes the response (usage, cost, stop reason)
                    -> [ledger response event] -> relay usage back to the sidecar
```

Each `/v1/messages` round trip is a `RequestContext` (request id, session key/source, feature flags, body, normalized model, route) threaded through the stages in exactly that order. Stages are plain functions in `proxy/app.py` — no plugin framework, a monolith by design.

## Run

```
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m proxy
```

Open `http://127.0.0.1:18090/`, set the gateway URL, expose the models you want, and click **Apply to Claude Code** (edits `~/.claude/settings.json` with a restorable backup — **Restore original settings** reverts byte-for-byte). Restart Claude Code after applying or restoring.

The proxy expects the client to authenticate or falls back to its own key: whichever of `authorization` / `x-api-key` the request carries is forwarded verbatim; when a request carries neither, the proxy injects `Authorization: Bearer <key>` from the dashboard-stored key or the `TOKEN_PROXY_UPSTREAM_KEY` environment variable (env wins; keeping the env route means the key never lands in `config.json`).

## Kompress model cache (offline mode, solved)

The sidecar runs with `HEADROOM_OFFLINE=1` and `HF_HUB_OFFLINE=1` by default — it refuses every outbound connection, including HuggingFace downloads. On a fresh machine that used to mean the ML compressor silently degraded to structural-only compression. Now the **proxy process** (which is not offline-guarded) pre-heats the cache:

- At sidecar start (and from the dashboard's **Check / download Kompress model** button) the proxy audits the local HF cache — `chopratejas/kompress-v2-base` and `answerdotai/ModernBERT-base` at the exact revisions pinned by the installed headroom (`proxy/model_cache.py`), plus the `o200k_base` tiktoken vocabulary.
- Anything missing is downloaded in a background thread; the proxy and its requests are never blocked.
- When the cache is warm, a running managed sidecar restarts itself automatically, and ML compression turns on. Failures (no network, bad proxy, etc.) are surfaced on the dashboard and in the status, and the sidecar keeps running in degraded mode — fail-open, as always.
- `HEADROOM_DISABLE_KOMPRESS=on` skips the pre-heat entirely.

Status endpoints: `GET /api/headroom/model` (probe result), `POST /api/headroom/model` (check now / download in background). The pins live in `proxy/model_cache.py` and must be re-checked when the headroom-ai version changes.

## Feature flags and the ablation harness

`proxy/features.py` is the feature-flag schema, in the same declarative pattern as `proxy/headroom_settings.py`. Five booleans — `compression` (default **on**, preserves the original behavior), `cost_control`, `shunt`, `green_metrics`, `tracing` (all default off) — plus a string `run_label`. Persisted under the `features` key in `data/config.json`, rendered as the **Features and run label** settings group on the dashboard, and manageable via `GET/PUT /api/features`.

Two properties matter for the ablation study:

1. **Flags are read from `config.json` at request time**, not cached at startup. The benchmark harness can flip settings between runs (edit the file, or `PUT /api/features`) without restarting the proxy. Dashboard saves also re-read the features block from disk before persisting, so a harness edit is never clobbered.
2. **The `compression` flag gates the existing call site.** When off, the sidecar is not contacted at all (`fallback: compression_disabled` in the usage row), which is exactly the A/B control arm.

`run_label` is stamped on every ledger entry so runs can be grouped into arms.

## Session ledger (tracing)

Turning on the `tracing` flag writes an append-only, per-session JSON event log under `data/sessions/` — one JSONL file per session key (`proxy/ledger.py`). File names are the sanitized key plus an 8-hex SHA-256 of the raw key, so distinct keys never collide. When a request carries no session signal, entries go to a `_no-session` file.

Privacy first: entries are **metadata and SHA-256 content hashes only** — system prompts, messages, tool schemas, and responses are never stored. Every entry is stamped with a UTC timestamp, the `run_label`, the session key and source, and a `request_id` (uuid minted at the identity stage) that correlates everything in one round trip:

- `request` — requested + normalized model, route, stream flag, message count, per-message (role, block-type counts, char size, content hash), system prompt (chars + hash), tool catalog (names, count, hash).
- `stage` — feature name + provenance (compression: tokens before/after, fallback reason, attempts, duration).
- `response` — status, duration, usage split input/output/cache_read/cache_write, stop reason, assistant message fingerprint (chars + hash + block counts; omitted for streams). Written at stream end, reusing the existing final-usage assembly.
- `error` — stage and message.

Ledger writes fail open: a write error goes to stderr and the request is unaffected. Browsable from the dashboard's **Sessions** section (list + click-through timeline) or via `GET /api/sessions` and `GET /api/sessions/{key}`.

Session identity comes from `proxy/session_key.py` — Claude Code's `x-claude-code-session-id` header (or `session_id` inside `metadata.user_id`), namespaced `claude:<session-id>:<fingerprint>` where the fingerprint (system prompt + tool names hash) keeps subagents in their own replay state. Codex sources (`session-id` header, `prompt_cache_key`) are supported the same way. Changing the system prompt changes the fingerprint, hence the session key — by design.

## Dashboard (`/`)

- **Gateway** — upstream URL, API key, model exposure + per-model routes (`anthropic` passthrough vs `responses` translation; `gpt-*` auto-routes to responses).
- **Claude Code** — apply/restore the client config.
- **Features and run label** — the five flags + `run_label`, effective on the next request.
- **Headroom sidecar** — start/stop/restart, status, all Headroom settings (env/request/client/sidecar kinds, restart on env change), **Kompress model cache status + download**, and sidecar `/stats`.
- **Sessions** — per-session ledger files with a readable timeline.
- **Usage** — SQLite-backed `data/usage.db`: per-request tokens (fresh vs cache read/write), compression stats, fallback reasons, relay state, gateway-reported cost, duration.

## Failure handling

Fail-open everywhere: a failing compression attempt (after configured retries) forwards the original request and records the reason (`forward_original: ...`), unless the failure policy is set to `fail_request`, which returns 503. A ledger write failure logs to stderr and never breaks the request. A upstream crash mid-stream ends the stream (already-relayed events are not re-sent); upstream errors ≥ 400 are observed in the ledger and relayed to the sidecar so its pending turns never leak. Malformed or non-object request bodies get an Anthropic-shaped 400 `invalid_request_error`.

Headroom's invariants are the proxy's: the frozen conversation prefix is never mutated mid-conversation (only the live delta is compressed), messages are never dropped or reordered, responses are never rewritten, and everything degrades to passthrough.

## Tests

```
.venv\Scripts\python.exe -m pytest
```

One module per proxy module, following the repository style: fake-sidecar and fake-gateway transports where needed; no test ever starts a real sidecar or downloads anything.

## Known limits

- Compression is marker-free lossless by default (no CCR retrieval/re-drive; `can_redrive` is forced off in the gateway block).
- Codex traffic does not go through this proxy yet; only its session extraction exists.
- Savings on short sessions are small (2–3%): most input tokens are the cached system prompt. The ML compressor needs the model cache warm (see above) to do better than structural compression.
- The Responses translation drops `temperature`/`top_p`/`stop_sequences`/`metadata`, and only maps tools that carry `input_schema`; SSE output omits `ping` events.
- `/v1/messages/count_tokens` on the responses route is a chars/4 estimate, not a gateway call.
- Client credentials are forwarded verbatim; with a dummy client token the configured gateway key is not substituted — point Claude Code's `ANTHROPIC_AUTH_TOKEN` at a real key, or have the gateway accept the proxy's injected key.
