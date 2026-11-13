from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

SCHEMA = """
create table if not exists usage (
    id integer primary key autoincrement,
    ts real not null,
    model text not null,
    route text not null,
    stream integer not null,
    status integer not null,
    input_tokens integer not null default 0,
    output_tokens integer not null default 0,
    cache_read_tokens integer not null default 0,
    cache_write_tokens integer not null default 0,
    cost_reported text,
    duration_ms integer not null default 0
)
"""

# Columns added after the first release. Existing rows keep their values; new ones default to 0/NULL.
MIGRATIONS = {
    "compressed": "integer not null default 0",
    "tokens_before": "integer not null default 0",
    "tokens_after": "integer not null default 0",
    "tokens_saved": "integer not null default 0",
    "session_source": "text",
    "session_replayed": "integer not null default 0",
    "attempts": "integer not null default 0",
    "fallback": "text",
    "turn_id": "text",
    "relayed": "integer not null default 0",
}


def normalize_anthropic_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    return {
        "input_tokens": usage.get("input_tokens") or 0,
        "output_tokens": usage.get("output_tokens") or 0,
        "cache_read_input_tokens": usage.get("cache_read_input_tokens") or 0,
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens") or 0,
    }


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(SCHEMA)
        existing = {row[1] for row in db.execute("pragma table_info(usage)")}
        for column, definition in MIGRATIONS.items():
            if column not in existing:
                db.execute(f"alter table usage add column {column} {definition}")


def record(
    path: Path,
    *,
    model: str,
    route: str,
    stream: bool,
    status: int,
    usage: dict[str, Any] | None,
    cost: str | None,
    duration_ms: int,
    compression: dict[str, Any] | None = None,
) -> None:
    counts = normalize_anthropic_usage(usage)
    comp = compression or {}
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(
            "insert into usage (ts, model, route, stream, status, input_tokens, output_tokens, "
            "cache_read_tokens, cache_write_tokens, cost_reported, duration_ms, compressed, "
            "tokens_before, tokens_after, tokens_saved, session_source, session_replayed, attempts, "
            "fallback, turn_id, relayed) "
            "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                model,
                route,
                int(stream),
                status,
                counts["input_tokens"],
                counts["output_tokens"],
                counts["cache_read_input_tokens"],
                counts["cache_creation_input_tokens"],
                cost,
                duration_ms,
                int(comp.get("compressed", False)),
                comp.get("tokens_before", 0),
                comp.get("tokens_after", 0),
                comp.get("tokens_saved", 0),
                comp.get("session_source"),
                int(comp.get("session_replayed", False)),
                comp.get("attempts", 0),
                comp.get("fallback"),
                comp.get("turn_id"),
                int(comp.get("relayed", False)),
            ),
        )


def mark_relayed(path: Path, turn_id: str) -> None:
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("update usage set relayed = 1 where turn_id = ?", (turn_id,))


def recent(path: Path, limit: int) -> list[dict[str, Any]]:
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("select * from usage order by id desc limit ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def totals(path: Path) -> dict[str, Any]:
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            "select count(*) as requests, "
            "coalesce(sum(compressed), 0) as compressed_requests, "
            "coalesce(sum(tokens_before), 0) as tokens_before, "
            "coalesce(sum(tokens_after), 0) as tokens_after, "
            "coalesce(sum(tokens_saved), 0) as tokens_saved, "
            "coalesce(sum(case when fallback is not null then 1 else 0 end), 0) as fallbacks, "
            "coalesce(sum(cache_read_tokens), 0) as cache_read_tokens, "
            "coalesce(sum(cache_write_tokens), 0) as cache_write_tokens "
            "from usage"
        ).fetchone()
    return dict(row)


class StreamUsage:
    """Collects Anthropic usage from a passthrough SSE stream without altering it."""

    def __init__(self) -> None:
        self._buffer = b""
        self.usage: dict[str, Any] = {}
        self.stop_reason: str | None = None

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            try:
                event = json.loads(line[5:])
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "message_start":
                self._merge(event.get("message", {}).get("usage"))
            elif event.get("type") == "message_delta":
                self._merge(event.get("usage"))
                stop = (event.get("delta") or {}).get("stop_reason")
                if isinstance(stop, str):
                    self.stop_reason = stop

    def _merge(self, usage: Any) -> None:
        if isinstance(usage, dict):
            self.usage.update({key: value for key, value in usage.items() if value is not None})
