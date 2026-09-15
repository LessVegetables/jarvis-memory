"""
Nearby places via the 2GIS Catalog API.

The search term comes out of the transcript by stripping the words that are
about asking rather than about the place: "до скольки работает аптека" ->
"аптека". 2GIS's own search is forgiving, so this does not need to be
precise, only to not send it the whole sentence.

Results are cached for a day per query. Shops do not move.

One 2GIS quirk worth knowing before touching this: `point` is "lon,lat",
longitude first, the opposite of nearly everything else.
"""

from __future__ import annotations

import logging
import math
import re
import urllib.error
from datetime import datetime, timedelta

from .. import config, prompts, router
from . import base

log = logging.getLogger(__name__)

URL = "https://catalog.api.2gis.com/3.0/items"
TTL = timedelta(hours=24)
RADIUS_M = 1500
MAX_RESULTS = 4

# Words that are part of asking, not part of what is being asked about.
_ASKING = re.compile(
    r"\b(где|что|есть|рядом|поблизости|недалеко|тут|здесь|там|работает|"
    r"до\s+скольк[иу]|открыт[аоы]?|закрыт[аоы]?|закрывается|открывается|"
    r"когда|найди|найти|подскажи|скажи|мне|нам|какой|какая|какие|какое|"
    r"ближайш\w*|сегодня|сейчас|можно|купить|сходить|поесть|а|и|ли|бы|"
    r"у нас|нас|адрес)\b"
)
_PUNCT = re.compile(r"[?!.,;:«»\"']")

_WEEKDAY_KEYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def query_from(transcript: str) -> str:
    """'до скольки работает аптека?' -> 'аптека'."""
    text = _PUNCT.sub(" ", router.normalise(transcript))
    text = _ASKING.sub(" ", text)
    return " ".join(text.split())


def block(transcript: str, now: datetime) -> str:
    """The nearby-places block for the prompt. Never raises."""
    try:
        return _block(transcript, now)
    except Exception:                                          # noqa: BLE001
        log.exception("places block failed")
        return prompts.PLACES_UNAVAILABLE.format(query=query_from(transcript) or "?")


def _block(transcript: str, now: datetime) -> str:
    key = config.get("DGIS_KEY")
    if not key:
        return prompts.PLACES_NO_KEY

    query = query_from(transcript)
    if not query:
        return prompts.PLACES_NO_QUERY

    lat, lon = config.location()
    payload, stale = base.cached_json(
        key=f"places:{lat:.3f},{lon:.3f}:{query}",
        ttl=TTL,
        fetch=lambda: _fetch(key, query, lat, lon),
        now=now,
    )
    if payload is None:
        return prompts.PLACES_UNAVAILABLE.format(query=query)

    items = (payload.get("result") or {}).get("items") or []
    if not items:
        return prompts.PLACES_NONE.format(query=query)

    lines = "\n".join(_render_item(item, lat, lon, now) for item in items[:MAX_RESULTS])
    text = prompts.PLACES_BLOCK.format(query=query, lines=lines)
    return f"{text}\n{prompts.PLACES_STALE}" if stale else text


def _fetch(key: str, query: str, lat: float, lon: float) -> dict:
    try:
        return base.fetch_json(URL, {
            "q": query,
            "point": f"{lon},{lat}",            # longitude first -- see module doc
            "radius": RADIUS_M,
            "key": key,
            "fields": "items.point,items.schedule,items.address_name",
            "locale": "ru_RU",
            "page_size": MAX_RESULTS,
        })
    except urllib.error.HTTPError as exc:
        # 2GIS answers "nothing found" with a 404, which is a result, not an
        # outage. Anything else really is a failure and is re-raised so the
        # cache layer can serve stale data or report unavailability.
        if exc.code == 404:
            return {"result": {"items": []}}
        raise


def _render_item(item: dict, lat: float, lon: float, now: datetime) -> str:
    name = item.get("name") or "(без названия)"
    parts = [name]
    if item.get("address_name"):
        parts.append(item["address_name"])
    point = item.get("point") or {}
    if "lat" in point and "lon" in point:
        parts.append(_distance_phrase(lat, lon, point["lat"], point["lon"]))
    hours = _hours_today(item.get("schedule"), now)
    if hours:
        parts.append(hours)
    return "- " + ", ".join(parts)


def _hours_today(schedule: dict | None, now: datetime) -> str | None:
    if not schedule:
        return None
    if schedule.get("is_24x7"):
        return "круглосуточно"
    today = schedule.get(_WEEKDAY_KEYS[now.weekday()])
    spans = (today or {}).get("working_hours") or []
    if not spans:
        return "сегодня закрыто"
    return "сегодня " + ", ".join(f"{s.get('from', '?')}–{s.get('to', '?')}" for s in spans)


def _distance_phrase(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """Haversine distance, phrased the way a person would say it."""
    radius = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    metres = 2 * radius * math.asin(math.sqrt(a))
    if metres < 50:
        return "прямо у дома"
    if metres < 950:
        return f"в {int(round(metres, -1))} м"
    return f"в {metres / 1000:.1f} км".replace(".0 км", " км")
