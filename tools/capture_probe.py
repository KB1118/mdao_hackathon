"""Logging forwarder used to record which headers and body fields coding agents send.

Every request is written as one JSON line to the log file, then forwarded unchanged
to the upstream gateway and relayed back. Use it only for probing session identifiers.
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

UPSTREAM = os.environ.get("PROBE_UPSTREAM", "http://127.0.0.1:18080")
LOG_PATH = Path(os.environ.get("PROBE_LOG", "probe_capture.jsonl"))
PORT = int(os.environ.get("PROBE_PORT", "18099"))
HOP_BY_HOP = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}

app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))


def _record(request: Request, raw: bytes) -> None:
    try:
        body = json.loads(raw) if raw else None
    except ValueError:
        body = None
    entry = {
        "ts": time.time(),
        "path": request.url.path,
        "query": request.url.query,
        "headers": {k.lower(): v for k, v in request.headers.items()},
        "body": body,
    }
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD"])
async def forward(path: str, request: Request):
    raw = await request.body()
    _record(request, raw)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    url = f"{UPSTREAM}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    upstream_req = client.build_request(request.method, url, headers=headers, content=raw)
    upstream = await client.send(upstream_req, stream=True)
    response_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in {"content-encoding", "content-length", "transfer-encoding", "connection"}
    }

    async def relay():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(relay(), status_code=upstream.status_code, headers=response_headers,
                             media_type=upstream.headers.get("content-type"))


if __name__ == "__main__":
    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
