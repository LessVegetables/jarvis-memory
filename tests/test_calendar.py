"""
Calendar sync tests. No network: ICS sources get a fetch function returning
a fixture, and the CalDAV source gets a fake client.

Needs icalendar and recurring_ical_events; skips with a message otherwise,
so the rest of the suite still runs on a machine without them.

Run:  python3 tests/test_calendar.py   (or python3 -m pytest tests/ -q)
"""

import json
import os
import sys
import tempfile
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = Path(tempfile.mkdtemp(prefix="jarvis-test-"))
_TMP_DB = _TMP / "test.db"
os.environ["JARVIS_DB"] = str(_TMP_DB)

try:
    import icalendar  # noqa: F401
    import recurring_ical_events  # noqa: F401
except ImportError as exc:
    print(f"SKIP calendar tests: {exc} (pip install icalendar recurring_ical_events)")
    sys.exit(0)

from jarvis_memory import db, seed, store  # noqa: E402
from jarvis_memory.calendars import (  # noqa: E402
    CalDavSource, IcsSource, load_sources, parse_ics, sync_all,
)

NOW = datetime(2026, 9, 14, 14, 30)   # a Monday
START = datetime.combine((NOW - timedelta(days=1)).date(), time.min)
END = datetime.combine((NOW + timedelta(days=14)).date(), time.min)
MSK = ZoneInfo("Europe/Moscow")

# Each fixture event separately, so tests can compose feeds with and
# without particular events.
EVENTS = {
    "timed": """BEGIN:VEVENT
UID:timed@test
DTSTART;TZID=Europe/Moscow:20260915T100000
DTEND;TZID=Europe/Moscow:20260915T113000
SUMMARY:Консультация
LOCATION:ауд. 210
END:VEVENT""",
    "allday": """BEGIN:VEVENT
UID:allday@test
DTSTART;VALUE=DATE:20260916
DTEND;VALUE=DATE:20260917
SUMMARY:День рождения мамы
END:VEVENT""",
    "weekly": """BEGIN:VEVENT
UID:weekly@test
DTSTART;TZID=Europe/Moscow:20260901T180000
DTEND;TZID=Europe/Moscow:20260901T190000
RRULE:FREQ=WEEKLY;COUNT=10
SUMMARY:Английский клуб
END:VEVENT""",
    "far": """BEGIN:VEVENT
UID:far@test
DTSTART;TZID=Europe/Moscow:20261101T120000
SUMMARY:Далеко за окном
END:VEVENT""",
}


def ics(*names):
    body = "\n".join(EVENTS[n] for n in names)
    return f"BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//test//EN\n{body}\nEND:VCALENDAR\n"


def local(aware: datetime) -> datetime:
    """What to_local produces for an aware datetime -- whatever this
    machine's zone is, so the test is not tied to one timezone."""
    return aware.astimezone().replace(tzinfo=None)


def setup():
    os.environ["JARVIS_DB"] = str(_TMP_DB)
    db.close()
    seed.seed()


# --- parsing -----------------------------------------------------------------------

def test_parse_expands_recurrences_and_converts_times():
    events = parse_ics(ics("timed", "allday", "weekly", "far"), START, END)
    titles = sorted(e.title for e in events)
    assert titles == ["Английский клуб", "Английский клуб",
                      "День рождения мамы", "Консультация"], titles

    timed = next(e for e in events if e.title == "Консультация")
    assert timed.starts_at == local(datetime(2026, 9, 15, 10, tzinfo=MSK))
    assert timed.ends_at == local(datetime(2026, 9, 15, 11, 30, tzinfo=MSK))
    assert timed.location == "ауд. 210"
    assert not timed.all_day

    allday = next(e for e in events if e.all_day)
    assert allday.starts_at == datetime(2026, 9, 16)
    assert allday.title == "День рождения мамы"

    weekly = sorted(e.starts_at.date() for e in events if e.title == "Английский клуб")
    assert weekly == [date(2026, 9, 15), date(2026, 9, 22)], weekly

    # Instances of one recurring event must not collide on uid.
    assert len({e.uid for e in events}) == 4


# --- sync ----------------------------------------------------------------------------

def _seed_count(user_id="anton"):
    return db.connect().execute(
        "SELECT count(*) FROM events WHERE user_id = ? AND source = 'seed'", (user_id,)
    ).fetchone()[0]


def test_sync_replaces_its_window_and_leaves_seed_alone():
    setup()
    before = _seed_count()
    source = IcsSource("google", "https://x", fetch=lambda url: ics("timed", "allday", "weekly"))
    results = sync_all(now=NOW, sources={"anton": [source]})
    assert results == {"anton/ics:google": {"ok": True, "events": 4}}, results
    assert _seed_count() == before

    # Syncing again with one event gone from the feed: it disappears, and
    # nothing is duplicated.
    source = IcsSource("google", "https://x", fetch=lambda url: ics("allday", "weekly"))
    results = sync_all(now=NOW, sources={"anton": [source]})
    assert results["anton/ics:google"]["events"] == 3
    synced = db.connect().execute(
        "SELECT title FROM events WHERE source = 'ics:google' ORDER BY starts_at"
    ).fetchall()
    assert [r["title"] for r in synced] == ["Английский клуб", "День рождения мамы",
                                            "Английский клуб"], synced
    assert _seed_count() == before


def test_schedule_shows_synced_and_all_day_events():
    setup()
    source = IcsSource("google", "https://x", fetch=lambda url: ics("timed", "allday"))
    sync_all(now=NOW, sources={"anton": [source]})

    rows = store.get_schedule("anton", date(2026, 9, 16))
    assert rows[0] == ("весь день", "День рождения мамы"), rows

    rows = store.get_schedule("anton", local(datetime(2026, 9, 15, 10, tzinfo=MSK)).date())
    assert any(title == "Консультация, ауд. 210" for _, title in rows), rows


def test_sync_isolates_failures_and_skips_unknown_users():
    setup()

    class Boom:
        id = "ics:boom"

        def fetch(self, start, end):
            raise ConnectionError("dns failed")

    ok = IcsSource("g", "https://x", fetch=lambda url: ics("timed"))
    results = sync_all(now=NOW, sources={
        "anton": [Boom(), ok],
        "nobody": [IcsSource("g", "https://x", fetch=lambda url: ics("timed"))],
    })
    assert results["anton/ics:boom"]["ok"] is False
    assert "dns failed" in results["anton/ics:boom"]["error"]
    assert results["anton/ics:g"] == {"ok": True, "events": 1}
    assert "nobody/ics:g" not in results


# --- CalDAV ----------------------------------------------------------------------------

class FakeItem:
    def __init__(self, data):
        self.data = data


class FakeCalendar:
    def __init__(self, name, data, fail=False):
        self.name, self._data, self._fail = name, data, fail

    def search(self, **kwargs):
        if self._fail:
            raise RuntimeError("401 Unauthorized")
        return [FakeItem(self._data)]


class FakeClient:
    def __init__(self, calendars):
        self._calendars = calendars

    def principal(self):
        return self

    def calendars(self):
        return self._calendars


def test_caldav_filters_calendars_and_survives_one_failing():
    calendars = [
        FakeCalendar("Учёба", ics("timed")),
        FakeCalendar("Праздники", ics("allday")),          # not selected
        FakeCalendar("Сломанный", ics("weekly"), fail=True),
    ]
    source = CalDavSource("icloud", "https://caldav.icloud.com/", "u", "p",
                          calendars=["Учёба", "Сломанный"],
                          client_factory=lambda: FakeClient(calendars))
    events = source.fetch(START, END)
    assert [e.title for e in events] == ["Консультация"], events
    assert source.id == "caldav:icloud"


# --- configuration ---------------------------------------------------------------------

def test_load_sources_reads_password_from_env():
    os.environ["TEST_ICLOUD_PW"] = "abcd-efgh"
    path = _TMP / "calendars.json"
    path.write_text(json.dumps({
        "anton": [{"type": "ics", "name": "google", "url": "https://x/basic.ics"}],
        "masha": [{"type": "caldav", "name": "icloud", "url": "https://caldav.icloud.com/",
                   "username": "m@icloud.com", "password_env": "TEST_ICLOUD_PW",
                   "calendars": ["Учёба"]}],
    }), encoding="utf-8")
    sources = load_sources(path)
    assert isinstance(sources["anton"][0], IcsSource)
    icloud = sources["masha"][0]
    assert isinstance(icloud, CalDavSource)
    assert icloud.password == "abcd-efgh"
    assert icloud.calendars == {"Учёба"}
    assert load_sources(_TMP / "missing.json") == {}


# --- migration ---------------------------------------------------------------------------

def test_migration_adds_columns_to_an_old_database():
    """A jarvis.db created before step 9 has no source/uid/ends_at/all_day."""
    import sqlite3
    old = _TMP / "old.db"
    conn = sqlite3.connect(old)
    conn.executescript("""
        CREATE TABLE users (user_id TEXT PRIMARY KEY, name TEXT NOT NULL, age INTEGER);
        CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            starts_at TEXT NOT NULL, title TEXT NOT NULL, location TEXT);
        INSERT INTO users VALUES ('anton', 'Антон', 21);
        INSERT INTO events (user_id, starts_at, title) VALUES ('anton', '2026-09-14 16:00', 'Старая лекция');
    """)
    conn.commit()
    conn.close()

    os.environ["JARVIS_DB"] = str(old)
    db.close()
    try:
        columns = {r["name"] for r in db.connect().execute("PRAGMA table_info(events)")}
        assert {"source", "uid", "ends_at", "all_day"} <= columns, columns
        rows = store.get_schedule("anton", date(2026, 9, 14))
        assert rows == [("16:00", "Старая лекция")], rows
        row = db.connect().execute("SELECT source, all_day FROM events").fetchone()
        assert (row["source"], row["all_day"]) == ("seed", 0)
    finally:
        os.environ["JARVIS_DB"] = str(_TMP_DB)
        db.close()


TESTS = [
    test_parse_expands_recurrences_and_converts_times,
    test_sync_replaces_its_window_and_leaves_seed_alone,
    test_schedule_shows_synced_and_all_day_events,
    test_sync_isolates_failures_and_skips_unknown_users,
    test_caldav_filters_calendars_and_survives_one_failing,
    test_load_sources_reads_password_from_env,
    test_migration_adds_columns_to_an_old_database,
]

if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"  ok  {test.__name__}")
    print(f"OK — {len(TESTS)} calendar tests")
