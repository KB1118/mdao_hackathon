import asyncio
import json

import httpx
import pytest

from proxy import headroom_settings as hs
from proxy.sidecar_client import CompressError, SidecarClient

VALUES = hs.normalize({})
BODY = {"messages": [{"role": "user", "content": "hi"}], "system": "sys"}
COMPRESSED = {
    "body": {"messages": [{"role": "user", "content": "compressed"}], "system": "sys"},
    "turn_id": "t1",
    "obligations": ["relay_usage"],
    "tokens_before": 100,
    "tokens_after": 60,
    "tokens_saved": 40,
    "transforms_applied": ["router"],
    "session": {"cached_prefix_replayed": False},
}


def _client(handler, retries=2):
    return SidecarClient(
        "http://sidecar.test",
        timeout_ms=1000,
        retries=retries,
        backoff_ms=0,
        transport=httpx.MockTransport(handler),
    )


def _compress(client):
    return client.compress(
        BODY,
        model="claude-haiku-5-5",
        token_budget=1000,
        session_key="claude:s:f",
        session_source="claude_header",
        values=VALUES,
    )


def test_compress_sends_session_gateway_block_and_returns_body():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=COMPRESSED)

    async def run():
        client = _client(handler)
        outcome = await _compress(client)
        await client.aclose()
        return outcome

    outcome = asyncio.run(run())
    assert outcome.body["messages"][0]["content"] == "compressed"
    assert outcome.turn_id == "t1"
    assert outcome.tokens_saved == 40
    assert seen["config"]["session_id"] == "claude:s:f"
    assert seen["gateway"]["can_redrive"] is False
    assert seen["token_budget"] == 1000
    assert seen["system"] == "sys"


def test_retries_transient_failures_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={"error": {"type": "compression_timeout"}})
        return httpx.Response(200, json=COMPRESSED)

    async def run():
        client = _client(handler, retries=2)
        outcome = await _compress(client)
        await client.aclose()
        return outcome

    assert asyncio.run(run()).attempts == 3


def test_gives_up_after_retries_are_spent():
    def handler(request):
        return httpx.Response(503, json={})

    async def run():
        client = _client(handler, retries=1)
        try:
            await _compress(client)
        finally:
            await client.aclose()

    with pytest.raises(CompressError) as info:
        asyncio.run(run())
    assert info.value.attempts == 2


def test_client_errors_are_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"type": "invalid_request"}})

    async def run():
        client = _client(handler, retries=3)
        try:
            await _compress(client)
        finally:
            await client.aclose()

    with pytest.raises(CompressError):
        asyncio.run(run())
    assert calls["n"] == 1


def test_relay_reports_success_and_failure():
    def ok(request):
        return httpx.Response(200, json={"action": "done"})

    def broken(request):
        return httpx.Response(500, json={})

    async def run(handler):
        client = _client(handler)
        try:
            return await client.relay("t1", status=200, latency_ms=12.0, usage={"input_tokens": 1})
        finally:
            await client.aclose()

    assert asyncio.run(run(ok)) is True
    assert asyncio.run(run(broken)) is False
