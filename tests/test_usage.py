import sqlite3
import time
from contextlib import closing

from proxy import usage as usage_log


def test_migration_keeps_existing_rows(tmp_path):
    path = tmp_path / "usage.db"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(
            "create table usage (id integer primary key autoincrement, ts real not null, "
            "model text not null, route text not null, stream integer not null, status integer not null, "
            "input_tokens integer not null default 0, output_tokens integer not null default 0, "
            "cache_read_tokens integer not null default 0, cache_write_tokens integer not null default 0, "
            "cost_reported text, duration_ms integer not null default 0)"
        )
        db.execute("insert into usage (ts, model, route, stream, status) values (?, 'm', 'anthropic', 0, 200)",
                   (time.time(),))

    usage_log.init_db(path)
    row = usage_log.recent(path, 1)[0]
    assert row["model"] == "m"
    assert row["compressed"] == 0
    assert row["fallback"] is None


def test_record_and_totals(tmp_path):
    path = tmp_path / "usage.db"
    usage_log.init_db(path)
    usage_log.record(path, model="m", route="anthropic", stream=False, status=200,
                     usage={"input_tokens": 10, "cache_read_input_tokens": 4}, cost=None, duration_ms=5,
                     compression={"compressed": True, "tokens_before": 100, "tokens_after": 40,
                                  "tokens_saved": 60, "session_source": "claude_header"})
    usage_log.record(path, model="m", route="anthropic", stream=False, status=200,
                     usage={}, cost=None, duration_ms=5,
                     compression={"fallback": "forwarded_original: HTTP 503"})
    totals = usage_log.totals(path)
    assert totals["requests"] == 2
    assert totals["compressed_requests"] == 1
    assert totals["tokens_saved"] == 60
    assert totals["fallbacks"] == 1
    assert totals["cache_read_tokens"] == 4
