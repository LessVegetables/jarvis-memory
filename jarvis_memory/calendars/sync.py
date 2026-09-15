"""
Pull calendar events into the `events` table.

The key decision: sync on a timer, never query a calendar at question time.
The request path stays a SQL lookup -- fast, and it works when the wifi
does not. `store.get_schedule` did not change for this step at all.

Each sync replaces the window it covers, per (user, source): delete what
was there, insert what the calendar says now. That is how deleted and
moved events disappear; an upsert would leave ghosts behind. Hand-seeded
events carry source='seed' and are never touched.

Sources are declared in calendars.json (gitignored; see
calendars.example.json). Passwords may be given by environment variable
name instead of inline, so secrets can live in .env with the API keys.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, time, timedelta
from pathlib import Path

from .. import config, db, store
from .sources import CalDavSource, Event, IcsSource

log = logging.getLogger(__name__)

CONFIG_PATH = config.ROOT / "calendars.json"
WINDOW_BACK = timedelta(days=1)
WINDOW_FORWARD = timedelta(days=14)

_FORMAT = "%Y-%m-%d %H:%M"


def load_sources(path: Path = CONFIG_PATH) -> dict[str, list]:
    """user_id -> [sources]. Missing file means no calendars configured."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    sources: dict[str, list] = {}
    for user_id, entries in raw.items():
        # "_comment" and the like in the example file are notes, not users.
        if user_id.startswith("_"):
            continue
        if not isinstance(entries, list):
            log.warning("calendars.json: %r should map to a list of sources, "
                        "got %s; skipped", user_id, type(entries).__name__)
            continue
        sources[user_id] = []
        for entry in entries:
            if not isinstance(entry, dict):
                log.warning("calendars.json: %s has a non-object entry %r; skipped",
                            user_id, entry)
                continue
            kind = entry.get("type")
            name = entry.get("name") or kind
            if kind == "ics":
                sources[user_id].append(IcsSource(name, entry["url"]))
            elif kind == "caldav":
                password = entry.get("password") or os.environ.get(entry.get("password_env", ""), "")
                if not password:
                    log.warning("%s/%s: no password (set %s in .env)",
                                user_id, name, entry.get("password_env", "password"))
                sources[user_id].append(CalDavSource(
                    name, entry["url"], entry["username"], password, entry.get("calendars"),
                ))
            else:
                log.warning("%s: unknown calendar source type %r", user_id, kind)
    return sources


def sync_all(now: datetime | None = None,
             sources: dict[str, list] | None = None) -> dict[str, dict]:
    """Sync every configured source. Returns per-source outcomes; never raises.

    One failing calendar (expired password, Google having a moment) must
    not stop the others from syncing, so failures are collected, not thrown.
    """
    now = now or datetime.now()
    start = datetime.combine((now - WINDOW_BACK).date(), time.min)
    end = datetime.combine((now + WINDOW_FORWARD).date(), time.min)
    sources = load_sources() if sources is None else sources

    results: dict[str, dict] = {}
    for user_id, user_sources in sources.items():
        if store.get_profile(user_id) is None:
            log.warning("calendars.json names unknown user %r; skipped", user_id)
            continue
        for source in user_sources:
            label = f"{user_id}/{source.id}"
            try:
                events = source.fetch(start, end)
                count = replace_window(user_id, source.id, events, start, end)
                results[label] = {"ok": True, "events": count}
            except Exception as exc:                           # noqa: BLE001
                # One line at warning level: an expired password or a dead
                # URL is routine on a cron log and needs no traceback. The
                # traceback is still there at DEBUG for the unexpected cases.
                log.warning("calendar sync failed for %s: %s: %s",
                            label, type(exc).__name__, exc)
                log.debug("traceback for %s", label, exc_info=True)
                results[label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return results


def replace_window(user_id: str, source_id: str, events: list[Event],
                   start: datetime, end: datetime) -> int:
    """Make the events table match `events` for this user+source+window."""
    conn = db.connect()
    conn.execute(
        "DELETE FROM events WHERE user_id = ? AND source = ? "
        "AND starts_at >= ? AND starts_at < ?",
        (user_id, source_id, start.strftime(_FORMAT), end.strftime(_FORMAT)),
    )
    rows = [
        (user_id, source_id, event.uid, event.starts_at.strftime(_FORMAT),
         event.ends_at.strftime(_FORMAT) if event.ends_at else None,
         event.title, event.location, int(event.all_day))
        for event in events
        if start <= event.starts_at < end
    ]
    conn.executemany(
        "INSERT INTO events (user_id, source, uid, starts_at, ends_at, title, location, all_day) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)
