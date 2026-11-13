"""HTTP client for the Headroom sidecar's gateway contract.

Request half: POST /v1/compress returns the provider body to send. Response half:
POST /v1/compress/response relays the provider's usage so the sidecar can track cache hits.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from .headroom_settings import PROXY_TOKEN_HEADER, gateway_block, request_config

logger = logging.getLogger("token_proxy.sidecar")

RETRYABLE_STATUS = {500, 502, 503, 504}


class CompressError(Exception):
    def __init__(self, reason: str, attempts: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.attempts = attempts


@dataclass
class Outcome:
    body: dict[str, Any]
    turn_id: str | None = None
    obligations: list[str] = field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    transforms: list[str] = field(default_factory=list)
    session_key: str | None = None
    session_source: str | None = None
    session_replayed: bool = False
    attempts: int = 1
    fallback: str | None = None


class SidecarClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_ms: int,
        retries: int,
        backoff_ms: int,
        token: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_ms / 1000, connect=2.0),
            transport=transport,
        )
        self._retries = retries
        self._backoff_ms = backoff_ms
        self._token = token

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._token:
            headers[PROXY_TOKEN_HEADER] = self._token
        return headers

    async def compress(
        self,
        request_body: dict[str, Any],
        *,
        model: str,
        token_budget: int,
        session_key: str,
        session_source: str,
        values: dict[str, Any],
    ) -> Outcome:
        payload = dict(request_body)
        payload["messages"] = request_body.get("messages", [])
        payload["model"] = model
        payload["token_budget"] = token_budget
        payload["config"] = request_config(values, session_key)
        payload["gateway"] = gateway_block(values)

        attempts = 0
        reason = "no attempt made"
        while attempts <= self._retries:
            attempts += 1
            try:
                resp = await self._client.post("/v1/compress", json=payload, headers=self._headers())
                if resp.status_code == 200:
                    return self._outcome(resp.json(), session_key, session_source, attempts)
                reason = f"HTTP {resp.status_code}"
                if resp.status_code not in RETRYABLE_STATUS:
                    break
            except (httpx.HTTPError, ValueError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
            if attempts <= self._retries:
                await asyncio.sleep(self._backoff_ms * (2 ** (attempts - 1)) / 1000)
        raise CompressError(reason, attempts)

    def _outcome(
        self,
        data: dict[str, Any],
        session_key: str,
        session_source: str,
        attempts: int,
    ) -> Outcome:
        body = data.get("body")
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            raise CompressError("response has no body", attempts)
        session = data.get("session") if isinstance(data.get("session"), dict) else {}
        return Outcome(
            body=body,
            turn_id=data.get("turn_id"),
            obligations=list(data.get("obligations") or []),
            tokens_before=int(data.get("tokens_before") or 0),
            tokens_after=int(data.get("tokens_after") or 0),
            tokens_saved=int(data.get("tokens_saved") or 0),
            transforms=list(data.get("transforms_applied") or []),
            session_key=session_key,
            session_source=session_source,
            session_replayed=bool(session.get("cached_prefix_replayed")),
            attempts=attempts,
        )

    async def relay(
        self,
        turn_id: str,
        *,
        status: int,
        latency_ms: float,
        usage: dict[str, int],
    ) -> bool:
        payload = {"turn_id": turn_id, "status": status, "latency_ms": latency_ms, "usage": usage}
        for attempt in range(2):
            try:
                resp = await self._client.post(
                    "/v1/compress/response", json=payload, headers=self._headers()
                )
                if resp.status_code == 200:
                    return True
                if resp.status_code != 409:
                    logger.warning("usage relay for turn %s returned HTTP %s", turn_id, resp.status_code)
                    return False
            except httpx.HTTPError as exc:
                logger.warning("usage relay for turn %s failed: %s", turn_id, exc)
                return False
            await asyncio.sleep(0.2)
        return False

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200 and resp.json().get("status") == "healthy"
        except (httpx.HTTPError, ValueError):
            return False

    async def stats(self) -> dict[str, Any]:
        resp = await self._client.get("/stats")
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}
