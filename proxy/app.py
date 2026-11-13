from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import claude_config
from . import features as features_settings
from . import headroom_settings
from . import ledger
from . import model_cache
from . import usage as usage_log
from .config import (
    CONFIG_PATH,
    DASHBOARD_DIR,
    LEDGER_DIR,
    LISTEN_HOST,
    LISTEN_PORT,
    SIDECAR_LOG_PATH,
    USAGE_DB_PATH,
    Settings,
    load_settings,
    save_settings,
)
from .routing import ROUTE_ANTHROPIC, ROUTE_RESPONSES, VALID_ROUTES, normalize_model, route_for, upstream_base
from .session_key import extract_session
from .sidecar import Sidecar
from .sidecar_client import CompressError, Outcome, SidecarClient
from .translate_request import to_responses_request
from .translate_response import StreamConverter, reported_cost, to_anthropic_message, usage_from_responses
from .usage import StreamUsage

logger = logging.getLogger("token_proxy")


class SettingsUpdate(BaseModel):
    upstream_url: Optional[str] = None
    api_key: Optional[str] = None
    exposed_models: Optional[list[str]] = None
    route_overrides: Optional[dict[str, str]] = None


class HeadroomUpdate(BaseModel):
    values: dict[str, Any]


@dataclass
class RequestContext:
    """State threaded through the pipeline stages of one /v1/messages round trip."""

    request_id: str
    requested_model: str
    model: str
    route: str
    stream: bool
    body: dict[str, Any]
    features: dict[str, Any]
    session_key: str | None = None
    session_source: str | None = None


def error_json(status: int, message: str, kind: str = "api_error") -> JSONResponse:
    return JSONResponse(status_code=status, content={"type": "error", "error": {"type": kind, "message": message}})


def safe_json(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def upstream_error(resp: httpx.Response) -> JSONResponse:
    message = resp.text[:1000]
    err = safe_json(resp).get("error")
    if isinstance(err, dict):
        message = str(err.get("message", message))
    elif isinstance(err, str):
        message = err
    return error_json(resp.status_code, message)


def header_cost(headers: httpx.Headers) -> str | None:
    for name, value in headers.items():
        if "cost" in name.lower():
            return f"{name}: {value}"
    return None


def _sidecar_settings(settings: Settings) -> dict[str, Any]:
    return settings.headroom


def create_app(
    settings_path: Path = CONFIG_PATH,
    db_path: Path = USAGE_DB_PATH,
    claude_settings_path: Path = claude_config.DEFAULT_SETTINGS_PATH,
    ledger_dir: Path = LEDGER_DIR,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
    sidecar_transport: httpx.AsyncBaseTransport | None = None,
    manage_sidecar: bool = True,
    sidecar_log_path: Path = SIDECAR_LOG_PATH,
) -> FastAPI:
    settings = load_settings(settings_path)
    usage_log.init_db(db_path)
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0), transport=upstream_transport)
    sidecar = Sidecar(sidecar_log_path)
    state: dict[str, Any] = {"sidecar_client": None}
    _background_tasks: set["asyncio.Task[None]"] = set()

    def build_sidecar_client() -> SidecarClient:
        h = _sidecar_settings(settings)
        return SidecarClient(
            settings.sidecar_url(),
            timeout_ms=int(h["request_timeout_ms"]),
            retries=int(h["retries"]),
            backoff_ms=int(h["backoff_ms"]),
            token=str(h.get("HEADROOM_PROXY_TOKEN") or ""),
            transport=sidecar_transport,
        )

    async def rebuild_sidecar_client() -> None:
        old = state["sidecar_client"]
        state["sidecar_client"] = build_sidecar_client()
        if old is not None:
            await old.aclose()

    def sidecar_client() -> SidecarClient:
        if state["sidecar_client"] is None:
            state["sidecar_client"] = build_sidecar_client()
        return state["sidecar_client"]

    def sidecar_env() -> dict[str, str]:
        return headroom_settings.sidecar_env(settings.headroom)

    def start_sidecar() -> None:
        if manage_sidecar and settings.headroom["sidecar_enabled"]:
            sidecar.start(sidecar_env(), int(settings.headroom["sidecar_port"]))
            prefetch_model_cache()

    def prefetch_model_cache() -> None:
        """The sidecar itself is offline, so this process pre-heats its model cache.

        Missing Kompress artefacts are downloaded in the background; when they
        land, a running managed sidecar is restarted so ML compression loads
        instead of the structural-only fallback. Never blocks startup or requests.
        """
        if not manage_sidecar or not settings.headroom["sidecar_enabled"]:
            return
        if settings.headroom.get("HEADROOM_DISABLE_KOMPRESS"):
            return
        model_cache.ensure(background=True, on_ready=on_model_cache_ready)
        logger.info("Kompress model cache: %s", model_cache.status().get("status"))

    def on_model_cache_ready() -> None:
        if manage_sidecar and sidecar.running and settings.headroom["sidecar_enabled"]:
            logger.info("Kompress model cache ready; restarting the sidecar to enable ML compression")
            sidecar.restart(sidecar_env(), int(settings.headroom["sidecar_port"]))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        start_sidecar()
        yield
        if state["sidecar_client"] is not None:
            await state["sidecar_client"].aclose()
        if manage_sidecar:
            sidecar.stop()
        await client.aclose()

    app = FastAPI(title="Token Proxy", lifespan=lifespan)

    @app.exception_handler(httpx.HTTPError)
    async def upstream_unreachable(_request: Request, exc: httpx.HTTPError) -> JSONResponse:
        return error_json(502, f"Could not reach the gateway at {settings.upstream_url}: {exc}")

    def base() -> str:
        return upstream_base(settings.upstream_url)

    def upstream_headers(request: Request, passthrough: bool) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        for name in ("authorization", "x-api-key"):
            if name in request.headers:
                headers[name] = request.headers[name]
        key = settings.effective_key()
        if "authorization" not in headers and "x-api-key" not in headers and key:
            headers["authorization"] = f"Bearer {key}"
        if passthrough:
            headers["anthropic-version"] = request.headers.get("anthropic-version", "2023-06-01")
            if "anthropic-beta" in request.headers:
                headers["anthropic-beta"] = request.headers["anthropic-beta"]
        return headers

    def check_exposed(model: str) -> JSONResponse | None:
        if model in settings.exposed_models:
            return None
        return error_json(
            404,
            f"Model '{model}' is not exposed by the proxy. Enable it in the dashboard at "
            f"http://{LISTEN_HOST}:{LISTEN_PORT}/",
            "not_found_error",
        )

    def log_usage(
        *,
        model: str,
        route: str,
        stream: bool,
        status: int,
        usage: dict[str, Any] | None,
        cost: str | None,
        duration_ms: float,
        compression: dict[str, Any] | None,
    ) -> None:
        usage_log.record(
            db_path,
            model=model,
            route=route,
            stream=stream,
            status=status,
            usage=usage,
            cost=cost,
            duration_ms=int(duration_ms),
            compression=compression,
        )

    # --- Ledger observers (tracing flag gated; every write fails open) ---------------

    def ledger_request(ctx: RequestContext) -> None:
        if not ctx.features["tracing"]:
            return
        ledger.request_event(
            ledger_dir,
            session_key=ctx.session_key,
            session_source=ctx.session_source,
            request_id=ctx.request_id,
            run_label=ctx.features["run_label"],
            requested_model=ctx.requested_model,
            model=ctx.model,
            route=ctx.route,
            stream=ctx.stream,
            body=ctx.body,
        )

    def ledger_stage(ctx: RequestContext, record: dict[str, Any], duration_ms: float, *, error_message: str | None = None) -> None:
        if not ctx.features["tracing"]:
            return
        provenance = dict(record)
        provenance["duration_ms"] = duration_ms
        ledger.stage_event(
            ledger_dir,
            session_key=ctx.session_key,
            session_source=ctx.session_source,
            request_id=ctx.request_id,
            run_label=ctx.features["run_label"],
            feature="compression",
            provenance=provenance,
        )
        if error_message:
            ledger.error_event(
                ledger_dir,
                session_key=ctx.session_key,
                session_source=ctx.session_source,
                request_id=ctx.request_id,
                run_label=ctx.features["run_label"],
                stage="compression",
                message=error_message,
            )

    def ledger_error(ctx: RequestContext, stage: str, message: str) -> None:
        if not ctx.features["tracing"]:
            return
        ledger.error_event(
            ledger_dir,
            session_key=ctx.session_key,
            session_source=ctx.session_source,
            request_id=ctx.request_id,
            run_label=ctx.features["run_label"],
            stage=stage,
            message=message,
        )

    def observe_response(
        ctx: RequestContext,
        record: dict[str, Any],
        *,
        route: str,
        stream: bool,
        status: int,
        usage: dict[str, Any] | None,
        cost: str | None,
        duration_ms: float,
        stop_reason: str | None = None,
        content: Any = None,
    ) -> None:
        """Log the usage row, write the ledger response event, and schedule the sidecar relay."""
        log_usage(model=ctx.model, route=route, stream=stream, status=status, usage=usage,
                  cost=cost, duration_ms=duration_ms, compression=record)
        if ctx.features["tracing"]:
            ledger.response_event(
                ledger_dir,
                session_key=ctx.session_key,
                session_source=ctx.session_source,
                request_id=ctx.request_id,
                run_label=ctx.features["run_label"],
                status=status,
                duration_ms=duration_ms,
                usage=usage,
                stop_reason=stop_reason,
                content=content,
            )
        schedule_relay(record, status=status, latency_ms=duration_ms, usage=usage or {})

    def observe_upstream_error(ctx: RequestContext, record: dict[str, Any], *, status: int, duration_ms: float) -> None:
        """Ledger response event + sidecar relay cleanup for failed upstream calls."""
        if ctx.features["tracing"]:
            ledger.response_event(
                ledger_dir,
                session_key=ctx.session_key,
                session_source=ctx.session_source,
                request_id=ctx.request_id,
                run_label=ctx.features["run_label"],
                status=status,
                duration_ms=duration_ms,
                usage=None,
            )
        schedule_relay(record, status=status, latency_ms=duration_ms, usage={})

    def compression_record(outcome: Outcome | None, *, fallback: str | None, attempts: int) -> dict[str, Any]:
        if outcome is None:
            return {"compressed": False, "fallback": fallback, "attempts": attempts}
        return {
            "compressed": outcome.tokens_saved > 0,
            "tokens_before": outcome.tokens_before,
            "tokens_after": outcome.tokens_after,
            "tokens_saved": outcome.tokens_saved,
            "session_source": outcome.session_source,
            "session_replayed": outcome.session_replayed,
            "attempts": outcome.attempts,
            "fallback": fallback,
            "turn_id": outcome.turn_id,
            "relayed": False,
        }

    def extract_identity(request: Request, ctx: RequestContext) -> None:
        h = settings.headroom
        session = extract_session(
            dict(request.headers),
            ctx.body,
            {
                "claude_header": h["session_claude_header"],
                "claude_metadata": h["session_claude_metadata"],
                "codex_header": h["session_codex_header"],
                "codex_prompt_cache_key": h["session_codex_prompt_cache_key"],
            },
        )
        if session is not None:
            ctx.session_key = session.key
            ctx.session_source = session.source

    async def compression_stage(ctx: RequestContext) -> tuple[dict[str, Any], Outcome | None]:
        """Flag-gated compression stage. Returns (compression record, outcome); mutates ctx.body."""
        h = settings.headroom
        if not ctx.features["compression"]:
            record = compression_record(None, fallback="compression_disabled", attempts=0)
            ledger_stage(ctx, record, 0)
            return record, None
        if not h["sidecar_enabled"]:
            record = compression_record(None, fallback="sidecar_disabled", attempts=0)
            ledger_stage(ctx, record, 0)
            return record, None
        if ctx.session_key is None:
            record = compression_record(None, fallback="no_session_key", attempts=0)
            ledger_stage(ctx, record, 0)
            return record, None
        started = time.monotonic()
        try:
            outcome = await sidecar_client().compress(
                ctx.body,
                model=ctx.model,
                token_budget=int(h["token_budget"]),
                session_key=ctx.session_key,
                session_source=ctx.session_source,
                values=h,
            )
        except CompressError as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            if h["failure_policy"] == "fail_request":
                raise _CompressionFailed(exc) from exc
            logger.warning("sidecar compression failed after %s attempts (%s); forwarding original", exc.attempts, exc.reason)
            record = compression_record(None, fallback=f"forwarded_original: {exc.reason}", attempts=exc.attempts)
            ledger_stage(ctx, record, duration_ms, error_message=exc.reason)
            return record, None
        duration_ms = int((time.monotonic() - started) * 1000)
        ctx.body = outcome.body
        record = compression_record(outcome, fallback=None, attempts=outcome.attempts)
        ledger_stage(ctx, record, duration_ms)
        return record, outcome

    def schedule_relay(record: dict[str, Any], *, status: int, latency_ms: float, usage: dict[str, Any]) -> None:
        turn_id = record.get("turn_id")
        if not turn_id or not settings.headroom["relay_usage"] or not record.get("relay_pending"):
            return
        record["relay_pending"] = False
        task = asyncio.create_task(_relay_turn(turn_id, status, latency_ms, usage))
        _background_tasks.add(task)
        task.add_done_callback(_relay_done)

    def _relay_done(task: "asyncio.Task[None]") -> None:
        _background_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("usage relay task failed: %s", task.exception())

    async def _relay_turn(turn_id: str, status: int, latency_ms: float, usage: dict[str, Any]) -> None:
        payload_usage = {
            "input_tokens": usage.get("input_tokens", 0) or 0,
            "output_tokens": usage.get("output_tokens", 0) or 0,
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0) or 0,
        }
        ok = await sidecar_client().relay(turn_id, status=status, latency_ms=latency_ms, usage=payload_usage)
        if ok:
            usage_log.mark_relayed(db_path, turn_id)

    async def passthrough(request: Request, ctx: RequestContext, record: dict[str, Any], outcome: Outcome | None) -> Response:
        url = f"{base()}/v1/messages"
        headers = upstream_headers(request, passthrough=True)
        stream = ctx.stream
        started = time.monotonic()
        payload = json.dumps(ctx.body).encode("utf-8")
        if outcome is not None and "relay_usage" in outcome.obligations:
            record["relay_pending"] = True

        if not stream:
            resp = await client.post(url, content=payload, headers=headers)
            data = safe_json(resp)
            usage = data.get("usage") if resp.status_code < 400 else None
            observe_response(ctx, record, route=ROUTE_ANTHROPIC, stream=False, status=resp.status_code,
                             usage=usage, cost=header_cost(resp.headers),
                             duration_ms=(time.monotonic() - started) * 1000,
                             stop_reason=data.get("stop_reason") if isinstance(data.get("stop_reason"), str) else None,
                             content=data.get("content"))
            return Response(content=resp.content, status_code=resp.status_code,
                            media_type=resp.headers.get("content-type", "application/json"))

        req = client.build_request("POST", url, content=payload, headers=headers)
        resp = await client.send(req, stream=True)
        media = resp.headers.get("content-type", "text/event-stream")
        if resp.status_code >= 400:
            await resp.aread()
            await resp.aclose()
            observe_upstream_error(ctx, record, status=resp.status_code,
                                   duration_ms=(time.monotonic() - started) * 1000)
            return Response(content=resp.content, status_code=resp.status_code, media_type=media)

        tracker = StreamUsage()

        async def relay():
            try:
                async for chunk in resp.aiter_bytes():
                    tracker.feed(chunk)
                    yield chunk
            finally:
                await resp.aclose()
                observe_response(ctx, record, route=ROUTE_ANTHROPIC, stream=True, status=resp.status_code,
                                 usage=tracker.usage, cost=header_cost(resp.headers),
                                 duration_ms=(time.monotonic() - started) * 1000,
                                 stop_reason=tracker.stop_reason)

        return StreamingResponse(relay(), status_code=resp.status_code, media_type=media)

    async def via_responses(request: Request, ctx: RequestContext, record: dict[str, Any], outcome: Outcome | None) -> Response:
        model = ctx.model
        upstream_body = to_responses_request(ctx.body, model)
        url = f"{base()}/v1/responses"
        headers = upstream_headers(request, passthrough=False)
        stream = ctx.stream
        started = time.monotonic()
        if outcome is not None and "relay_usage" in outcome.obligations:
            record["relay_pending"] = True

        if not stream:
            resp = await client.post(url, json=upstream_body, headers=headers)
            if resp.status_code >= 400:
                observe_upstream_error(ctx, record, status=resp.status_code,
                                       duration_ms=(time.monotonic() - started) * 1000)
                return upstream_error(resp)
            data = safe_json(resp)
            usage = usage_from_responses(data.get("usage"))
            message = to_anthropic_message(data, ctx.requested_model)
            observe_response(ctx, record, route=ROUTE_RESPONSES, stream=False, status=resp.status_code,
                             usage=usage, cost=header_cost(resp.headers) or reported_cost(data.get("usage")),
                             duration_ms=(time.monotonic() - started) * 1000,
                             stop_reason=message.get("stop_reason"),
                             content=message.get("content"))
            return JSONResponse(message)

        req = client.build_request("POST", url, json=upstream_body, headers=headers)
        resp = await client.send(req, stream=True)
        if resp.status_code >= 400:
            await resp.aread()
            error = upstream_error(resp)
            await resp.aclose()
            observe_upstream_error(ctx, record, status=resp.status_code,
                                   duration_ms=(time.monotonic() - started) * 1000)
            return error

        converter = StreamConverter(ctx.requested_model)

        async def relay():
            data_lines: list[str] = []
            try:
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                    elif not line and data_lines:
                        raw = "\n".join(data_lines)
                        data_lines = []
                        try:
                            event = json.loads(raw)
                        except ValueError:
                            continue
                        if isinstance(event, dict):
                            for chunk in converter.feed(event):
                                yield chunk
            finally:
                await resp.aclose()
                observe_response(ctx, record, route=ROUTE_RESPONSES, stream=True, status=resp.status_code,
                                 usage=converter.usage, cost=header_cost(resp.headers) or converter.cost,
                                 duration_ms=(time.monotonic() - started) * 1000,
                                 stop_reason=converter.stop_reason)

        return StreamingResponse(relay(), media_type="text/event-stream")

    @app.post("/v1/messages")
    async def messages(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return error_json(400, "Request body is not valid JSON", "invalid_request_error")
        if not isinstance(body, dict):
            return error_json(400, "Request body must be a JSON object", "invalid_request_error")
        requested = body.get("model", "")
        model = normalize_model(requested)
        blocked = check_exposed(model)
        if blocked is not None:
            return blocked
        body["model"] = model
        ctx = build_context(request, body)

        ledger_request(ctx)
        try:
            record, outcome = await compression_stage(ctx)
        except _CompressionFailed as failed:
            ledger_error(ctx, "compression", f"failed after {failed.inner.attempts} attempts: {failed.inner.reason}")
            return error_json(
                503,
                f"Headroom sidecar unavailable after {failed.inner.attempts} attempts: {failed.inner.reason}",
            )
        if ctx.route == ROUTE_RESPONSES:
            return await via_responses(request, ctx, record, outcome)
        return await passthrough(request, ctx, record, outcome)

    def build_context(request: Request, body: dict[str, Any]) -> RequestContext:
        """Identity stage: read flags from config at request time, compute route, extract session."""
        flags = features_settings.load_features(settings_path)
        ctx = RequestContext(
            request_id=str(uuid.uuid4()),
            requested_model=body.get("model", ""),
            model=body["model"],
            route=route_for(body["model"], settings.route_overrides),
            stream=bool(body.get("stream")),
            body=body,
            features=flags,
        )
        extract_identity(request, ctx)
        return ctx

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return error_json(400, "Request body is not valid JSON", "invalid_request_error")
        if not isinstance(body, dict):
            return error_json(400, "Request body must be a JSON object", "invalid_request_error")
        model = normalize_model(body.get("model", ""))
        blocked = check_exposed(model)
        if blocked is not None:
            return blocked
        body["model"] = model
        if route_for(model, settings.route_overrides) == ROUTE_RESPONSES:
            return JSONResponse({"input_tokens": max(1, len(json.dumps(body)) // 4)})
        resp = await client.post(
            f"{base()}/v1/messages/count_tokens",
            json=body,
            headers=upstream_headers(request, passthrough=True),
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    @app.get("/v1/models")
    async def list_models():
        data = [
            {"type": "model", "id": m, "display_name": m, "created_at": "1970-01-01T00:00:00Z"}
            for m in settings.exposed_models
        ]
        return {
            "data": data,
            "has_more": False,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
        }

    def claude_state() -> dict[str, Any]:
        state_ = claude_config.status(claude_settings_path)
        state_["applied"] = state_.get("base_url") == settings.proxy_url()
        return state_

    @app.get("/", response_model=None)
    async def dashboard():
        return FileResponse(DASHBOARD_DIR / "index.html")

    @app.get("/api/state")
    async def state_view():
        return {
            "proxy_url": settings.proxy_url(),
            "settings": {
                "upstream_url": settings.upstream_url,
                "api_key_set": bool(settings.effective_key()),
                "exposed_models": settings.exposed_models,
                "route_overrides": settings.route_overrides,
            },
            "claude": claude_state(),
        }

    def persist_settings() -> None:
        """Save the in-memory settings, refreshing the features block from disk first.

        Features are owned by config.json (the ablation harness may rewrite them
        at any time), so a dashboard save must not clobber a newer on-disk block.
        """
        settings.features = features_settings.load_features(settings_path)
        save_settings(settings, settings_path)

    @app.put("/api/settings")
    async def update_settings(update: SettingsUpdate):
        new_url = None
        if update.upstream_url is not None:
            new_url = upstream_base(update.upstream_url)
            parts = urlsplit(new_url)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                return error_json(400, "Upstream URL must start with http:// or https://", "invalid_request_error")
            if parts.port == LISTEN_PORT:
                return error_json(400, f"Upstream cannot be the proxy itself (port {LISTEN_PORT})",
                                  "invalid_request_error")
        if update.route_overrides is not None and any(v not in VALID_ROUTES for v in update.route_overrides.values()):
            return error_json(400, "Route must be 'anthropic' or 'responses'", "invalid_request_error")

        if new_url is not None:
            settings.upstream_url = new_url
        if update.api_key is not None:
            settings.api_key = update.api_key.strip()
        if update.exposed_models is not None:
            settings.exposed_models = list(dict.fromkeys(update.exposed_models))
        if update.route_overrides is not None:
            settings.route_overrides = dict(update.route_overrides)
        persist_settings()
        return {"ok": True}

    @app.get("/api/features")
    async def features_view():
        return {
            "schema": features_settings.FEATURES,
            "values": features_settings.load_features(settings_path),
        }

    @app.put("/api/features")
    async def features_update(update: HeadroomUpdate):
        current = features_settings.load_features(settings_path)
        try:
            normalized = features_settings.normalize({**current, **update.values})
        except ValueError as exc:
            return error_json(400, str(exc), "invalid_request_error")
        settings.features = normalized
        save_settings(settings, settings_path)
        return {"ok": True}

    @app.get("/api/sessions")
    async def sessions_view():
        return {"sessions": ledger.list_sessions(ledger_dir)}

    @app.get("/api/sessions/{key}")
    async def session_detail(key: str):
        session = ledger.read_session(ledger_dir, key)
        if session is None:
            return error_json(404, "Session not found", "not_found_error")
        return session

    @app.get("/api/gateway/models")
    async def gateway_models():
        headers = {}
        if settings.effective_key():
            headers["authorization"] = f"Bearer {settings.effective_key()}"
        resp = await client.get(f"{base()}/v1/models", headers=headers)
        if resp.status_code >= 400:
            return upstream_error(resp)
        models = []
        for item in safe_json(resp).get("data", []):
            model_id = item.get("id", "")
            models.append(
                {
                    "id": model_id,
                    "display_name": item.get("display_name", model_id),
                    "exposed": model_id in settings.exposed_models,
                    "route": route_for(model_id, settings.route_overrides),
                }
            )
        return {"models": models}

    @app.post("/api/claude/apply")
    async def apply_claude():
        try:
            claude_config.apply(claude_settings_path, settings.proxy_url(), settings.exposed_models)
        except (OSError, ValueError) as exc:
            return error_json(400, str(exc), "invalid_request_error")
        return claude_state()

    @app.post("/api/claude/restore")
    async def restore_claude():
        try:
            claude_config.restore(claude_settings_path)
        except (OSError, ValueError) as exc:
            return error_json(400, str(exc), "invalid_request_error")
        return claude_state()

    @app.get("/api/usage")
    async def usage_rows(limit: int = 50):
        return {"rows": usage_log.recent(db_path, max(1, min(limit, 500)))}

    @app.get("/api/usage/totals")
    async def usage_totals():
        return usage_log.totals(db_path)

    @app.get("/api/headroom")
    async def headroom_view():
        return {
            "schema": headroom_settings.SCHEMA,
            "values": settings.headroom,
            "sidecar": {**sidecar.status(), "healthy": await sidecar_client().health()},
            "sidecar_url": settings.sidecar_url(),
        }

    @app.put("/api/headroom")
    async def headroom_update(update: HeadroomUpdate):
        previous_env = headroom_settings.sidecar_env(settings.headroom)
        previous_port = settings.headroom["sidecar_port"]
        previous_client = (
            settings.headroom["request_timeout_ms"],
            settings.headroom["retries"],
            settings.headroom["backoff_ms"],
            settings.headroom["HEADROOM_PROXY_TOKEN"],
        )
        try:
            merged = {**settings.headroom, **update.values}
            normalized = headroom_settings.normalize(merged)
        except ValueError as exc:
            return error_json(400, str(exc), "invalid_request_error")
        settings.headroom = normalized
        persist_settings()
        env_changed = headroom_settings.sidecar_env(normalized) != previous_env or normalized["sidecar_port"] != previous_port
        client_changed = previous_client != (
            normalized["request_timeout_ms"],
            normalized["retries"],
            normalized["backoff_ms"],
            normalized["HEADROOM_PROXY_TOKEN"],
        )
        if client_changed or normalized["sidecar_port"] != previous_port:
            await rebuild_sidecar_client()
        restarted = False
        if env_changed and sidecar.running and manage_sidecar:
            sidecar.restart(sidecar_env(), int(normalized["sidecar_port"]))
            restarted = True
        if normalized["sidecar_enabled"] and not sidecar.running and manage_sidecar:
            sidecar.start(sidecar_env(), int(normalized["sidecar_port"]))
        return {"ok": True, "restarted": restarted, "restart_required": False}

    @app.post("/api/headroom/start")
    async def headroom_start():
        if not manage_sidecar:
            return error_json(400, "Sidecar management is disabled in this process", "invalid_request_error")
        sidecar.start(sidecar_env(), int(settings.headroom["sidecar_port"]))
        prefetch_model_cache()
        return sidecar.status()

    @app.post("/api/headroom/stop")
    async def headroom_stop():
        if not manage_sidecar:
            return error_json(400, "Sidecar management is disabled in this process", "invalid_request_error")
        sidecar.stop()
        return sidecar.status()

    @app.post("/api/headroom/restart")
    async def headroom_restart():
        if not manage_sidecar:
            return error_json(400, "Sidecar management is disabled in this process", "invalid_request_error")
        sidecar.restart(sidecar_env(), int(settings.headroom["sidecar_port"]))
        prefetch_model_cache()
        return sidecar.status()

    @app.get("/api/headroom/model")
    async def kompress_model_view():
        return model_cache.current_status()

    @app.post("/api/headroom/model")
    async def kompress_model_download():
        """Check the Kompress cache now; download in the background when missing.

        The sidecar restarts by itself once the cache is warm, so ML compression
        turns on without any further action. Repeated posts while a download is
        already running are no-ops.
        """
        if settings.headroom.get("HEADROOM_DISABLE_KOMPRESS"):
            return model_cache.current_status()
        model_cache.ensure(background=True, on_ready=on_model_cache_ready)
        return model_cache.current_status()

    @app.get("/api/headroom/stats")
    async def headroom_stats():
        try:
            return await sidecar_client().stats()
        except (httpx.HTTPError, ValueError) as exc:
            return error_json(502, f"Sidecar stats unavailable: {exc}")

    return app


class _CompressionFailed(Exception):
    def __init__(self, inner: CompressError) -> None:
        super().__init__(inner.reason)
        self.inner = inner
