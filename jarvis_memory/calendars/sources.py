"""
Where calendar events come from, and how they are turned into rows.

Two kinds of source, one output:

- ICS: a URL that serves an iCalendar file. Google Calendar's "secret
  address in iCal format" is this, and so is any public calendar link.
  A plain HTTPS GET, no authentication dance. This is the route for Google:
  its CalDAV endpoint wants OAuth, which costs a day of setup for nothing
  the ICS address does not already give us for reading.
- CalDAV: iCloud. Needs an app-specific password from appleid.apple.com
  (the account password will not work with two-factor on), then the
  `caldav` library does discovery and fetching.

Both produce `Event`s inside a date window. Times are converted to the
device's local time and stored naive, because that is what the rest of the
module already speaks -- make sure the board's timezone is set
(`timedatectl set-timezone Europe/Moscow` or wherever the kitchen is).

Recurring events are expanded here, client-side, with recurring_ical_events.
iCloud can expand server-side; Google's ICS feed cannot, and doing it in one
place for both is simpler than trusting either.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable

log = logging.getLogger(__name__)

# Sync runs on a timer, not in the voice path, so it may wait longer than
# a live request. Google's ICS feed in particular can be slow to generate.
SYNC_TIMEOUT = 20


@dataclass(frozen=True)
class Event:
    uid: str
    starts_at: datetime           # device-local, naive
    ends_at: datetime | None
    title: str
    location: str | None
    all_day: bool


def parse_ics(text: str | bytes, start: datetime, end: datetime) -> list[Event]:
    """Every event instance in [start, end), recurrences expanded."""
    import icalendar

    calendar = icalendar.Calendar.from_ical(text)
    events = []
    for component in _instances(calendar, start, end):
        dtstart = component.get("DTSTART")
        if dtstart is None:
            continue
        begin = dtstart.dt
        dtend = component.get("DTEND")
        starts_at = to_local(begin)
        events.append(Event(
            # Instances of a recurring event share a UID; the start time
            # is what tells them apart.
            uid=f"{component.get('UID', '')}@{starts_at:%Y-%m-%dT%H:%M}",
            starts_at=starts_at,
            ends_at=to_local(dtend.dt) if dtend is not None else None,
            title=str(component.get("SUMMARY") or "").strip() or "(без названия)",
            location=str(component.get("LOCATION") or "").strip() or None,
            all_day=not isinstance(begin, datetime),
        ))
    return events


def _instances(calendar, start: datetime, end: datetime):
    """Expanded event instances in the window, with a plain fallback.

    recurring_ical_events handles RRULE, EXDATE and RECURRENCE-ID overrides.
    If it chokes on a feed (they are not all well-formed), fall back to the
    non-recurring events rather than losing the whole calendar.
    """
    try:
        import recurring_ical_events
        return list(recurring_ical_events.of(calendar).between(start.date(), end.date()))
    except Exception as exc:                                   # noqa: BLE001
        log.warning("recurrence expansion failed (%s); using plain events only", exc)
        plain = []
        for component in calendar.walk("VEVENT"):
            dtstart = component.get("DTSTART")
            if dtstart is None or component.get("RRULE"):
                continue
            when = to_local(dtstart.dt)
            if start <= when < end:
                plain.append(component)
        return plain


def to_local(value: date | datetime) -> datetime:
    """Device-local naive datetime. All-day dates become midnight."""
    if not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day)
    if value.tzinfo is not None:
        return value.astimezone().replace(tzinfo=None)
    return value


class IcsSource:
    """An iCalendar file at a URL. Google's secret address, or any .ics link."""

    def __init__(self, name: str, url: str,
                 fetch: Callable[[str], bytes] | None = None):
        self.id = f"ics:{name}"
        self.url = url
        self._fetch = fetch or _http_get

    def fetch(self, start: datetime, end: datetime) -> list[Event]:
        return parse_ics(self._fetch(self.url), start, end)


class CalDavSource:
    """A CalDAV account -- iCloud in practice.

    `calendars` optionally restricts sync to calendars with those display
    names; by default every calendar on the account is read. Shared and
    holiday calendars tend to be noise, so restricting is usually right.
    """

    def __init__(self, name: str, url: str, username: str, password: str,
                 calendars: list[str] | None = None,
                 client_factory: Callable[[], object] | None = None):
        self.id = f"caldav:{name}"
        self.url, self.username, self.password = url, username, password
        self.calendars = set(calendars) if calendars else None
        self._client_factory = client_factory or self._connect

    def fetch(self, start: datetime, end: datetime) -> list[Event]:
        principal = self._client_factory().principal()
        events: list[Event] = []
        for calendar in principal.calendars():
            if self.calendars is not None and calendar.name not in self.calendars:
                continue
            try:
                items = calendar.search(start=start, end=end, event=True, expand=True)
            except Exception as exc:                           # noqa: BLE001
                # One broken calendar must not take the others down with it.
                log.warning("%s: calendar %r failed: %s", self.id, calendar.name, exc)
                continue
            for item in items:
                events.extend(parse_ics(item.data, start, end))
        return events

    def _connect(self):
        import caldav
        client = caldav.DAVClient(url=self.url, username=self.username,
                                  password=self.password, timeout=SYNC_TIMEOUT)
        _force_http1(client)
        return client


def _force_http1(client) -> None:
    """Make the CalDAV client speak plain HTTP/1.1.

    caldav 3.x uses niquests when it is installed, which tries HTTP/3 (QUIC
    over UDP) and HTTP/2 before falling back. On the board that produced
    "Connection aborted, OSError(5, Input/output error)" on the very first
    request to iCloud -- a low-level failure from the QUIC attempt, not a
    credentials problem. CalDAV has never needed anything beyond HTTP/1.1,
    so disable both. Auth is kept on the client, not the session, so
    replacing the session loses nothing.
    """
    try:
        import niquests
    except ImportError:
        return                      # plain requests: already HTTP/1.1 only
    client.session = niquests.Session(disable_http2=True, disable_http3=True)


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "jarvis-memory/0.1"})
    with urllib.request.urlopen(request, timeout=SYNC_TIMEOUT) as response:
        return response.read()
