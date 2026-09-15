"""
Shared plumbing for live-data providers: an HTTP GET that returns JSON, and
a small response cache in SQLite.

The cache exists for three reasons, in order of importance:

1. The demo. A flaky network in the presentation room must not turn
   "what's the weather" into silence. Anything fetched once is kept, and
   served -- marked stale -- if the next fetch fails.
2. Latency. A cache hit is a SQLite read; a miss is a network round trip
   on top of STT + LLM + TTS, which the person is already waiting through.
3. Quota. Free API tiers are per-day; a household asking about the weather
   twenty times an hour should not burn through one.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Callable

from .. import config, db

log = logging.getLogger(__name__)

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def fetch_json(url: str, params: dict) -> dict:
    """GET url?params and parse the JSON body. Raises on any failure.

    Module-level so tests can replace it with a function returning canned
    payloads -- none of the provider tests touch the network.
    """
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": "jarvis-memory/0.1"},
    )
    with urllib.request.urlopen(request, timeout=config.http_timeout()) as response:
        return json.load(response)


def cached_json(key: str, ttl: timedelta, fetch: Callable[[], dict],
                now: datetime | None = None) -> tuple[dict | None, bool]:
    """Return (payload, is_stale).

    Fresh cache -> served as is. Otherwise `fetch` is called; on success the
    result is cached and returned. On failure, an expired cache entry is
    returned with is_stale=True -- old data beats no data for a voice
    assistant, as long as the model is told it may be old. (None, False)
    means there is nothing at all.
    """
    now = now or datetime.now()
    cached = _read(key)
    if cached is not None:
        payload, fetched_at = cached
        if now - fetched_at <= ttl:
            return payload, False

    try:
        payload = fetch()
    except Exception as exc:                                   # noqa: BLE001
        # Exception type deliberately broad: DNS, TLS, timeout, HTTP error,
        # malformed JSON -- the caller does not care which, and none of
        # them may propagate up into the voice pipeline.
        log.warning("fetch failed for %s: %s", key, exc)
        if cached is not None:
            return cached[0], True
        return None, False

    _write(key, payload, now)
    return payload, False


def _read(key: str) -> tuple[dict, datetime] | None:
    row = db.connect().execute(
        "SELECT payload, fetched_at FROM api_cache WHERE key = ?", (key,)
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["payload"]), datetime.strptime(row["fetched_at"], _TIME_FORMAT)


def _write(key: str, payload: dict, now: datetime) -> None:
    conn = db.connect()
    conn.execute(
        "INSERT OR REPLACE INTO api_cache (key, payload, fetched_at) VALUES (?, ?, ?)",
        (key, json.dumps(payload, ensure_ascii=False), now.strftime(_TIME_FORMAT)),
    )
    conn.commit()
