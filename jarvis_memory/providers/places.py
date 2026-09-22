"""
Nearby places via the 2GIS Catalog API.

The search term comes out of the transcript by stripping the words that are
about asking rather than about the place: "до скольки работает аптека" ->
"аптека". 2GIS's own search is forgiving, so this does not need to be
precise, only to not send it the whole sentence.

Results are cached for a day per query. Shops do not move.

One 2GIS quirk worth knowing before touching this: `point` is "lon,lat",
longitude first, the opposite of nearly everything else.

--- why this file selects instead of listing ---------------------------------

It used to render every result as "name, address, distance, hours" and hand
the model all four rows. Two things went wrong, both in the logs:

  1. The model answered with the name of the second pharmacy, the address of
     the first and the distance of the third. It was not inventing anything --
     it was reading across the rows of a table nobody had told it was a table.

  2. Asked at 02:19 which pharmacy was open, it named one that had closed at
     22:00. Nothing in the prompt had asked it to compare the clock against
     opening hours, so it did not; it treated the question as "user wants a
     pharmacy" and returned the first one.

Neither is fixable by instruction. A 0.5B rephrases what it is given; it does
not filter, sort or cross-check. So this file does the filtering, sorting and
cross-checking, decides which single business answers the question, and puts
only the fields that answer it into the prompt. Every field that does not
answer the question is one more thing that can end up attached to the wrong
name, so the shapes below are deliberately sparse -- the list shape carries
names and nothing else, because names alone cannot be mismatched.

`router.place_shape()` decides which shape the question wants.
"""

from __future__ import annotations

import logging
import math
import re
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timedelta

from .. import config, num_to_words, prompts, router
from . import base

log = logging.getLogger(__name__)

URL = "https://catalog.api.2gis.com/3.0/items"
TTL = timedelta(hours=24)
RADIUS_M = 1500

# Ask for more than we show. A question about what is open now can eliminate
# every result at 2am, and a disliked place eliminates another; four rows used
# to be the whole budget, so both filters could empty the list outright.
PAGE_SIZE = 10
LIST_RESULTS = 3

_METRES = ("метр", "метра", "метров")
_KILOMETRES = ("километр", "километра", "километров")
_REVIEWS = ("отзыв", "отзыва", "отзывов")

# Words that are part of asking, not part of what is being asked about.
_ASKING = re.compile(
    r"\b(где|что|есть|рядом|поблизости|недалеко|тут|здесь|там|работает|"
    r"до\s+скольк[иу]|открыт[аоы]?|закрыт[аоы]?|закрывается|открывается|"
    r"когда|найди|найти|подскажи|скажи|мне|нам|какой|какая|какие|какое|"
    r"ближайш\w*|сегодня|сейчас|можно|купить|сходить|поесть|а|и|ли|бы|"
    r"у нас|нас|у|адрес|"
    # The shape words: they say which answer is wanted, never what to look for.
    r"раньше|всех|всего|пораньше|ранн\w*|утром|рейтинг|оценка|оценки|отзыв\w*|"
    r"далеко|близко|ближе|список|варианты|перечисли|лучш\w*|получше|"
    r"режим|часы|находится|адрес\w*)\b"
)
_PUNCT = re.compile(r"[?!.,;:«»\"']")

# 2GIS writes addresses as "<улица>, <дом>". Spoken, that comma is a pause in
# the middle of a single name: "Морской проспект... шесть". Nobody says the
# street and the house number as two things. The last segment must start with
# a digit, so "Морской проспект, 6, офис 12" keeps the comma it needs.
_HOUSE_NUMBER = re.compile(r",\s*(\d[^,]*)$")
# "6к1" / "6с2" -- corpus and building, written short and said long.
_HOUSE_SUFFIX = re.compile(r"(\d)\s*(к|корп\.?|с|стр\.?)\s*(\d)", re.IGNORECASE)
_SUFFIX_WORDS = {"к": "корпус", "корп": "корпус", "с": "строение", "стр": "строение"}

_WEEKDAY_KEYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass(frozen=True)
class Place:
    """One business, parsed out of 2GIS's payload.

    Frozen and plain: it is handed to `history.set_last_place` so that "а это
    далеко?" can be answered next turn without another search.
    """

    name: str
    address: str | None = None
    distance_m: float | None = None
    rating: float | None = None
    reviews: int | None = None
    is_24x7: bool = False
    # Seven tuples of ("09:00", "21:00") pairs, indexed by weekday.
    week: tuple[tuple[tuple[str, str], ...], ...] = ((),) * 7


@dataclass(frozen=True)
class PlaceAnswer:
    """The block, plus the place it named -- if it named exactly one.

    The second field is what makes a distance follow-up possible. `block()`
    below keeps returning a bare string so nothing outside this module has
    to care.
    """

    text: str
    place: Place | None = None


def query_from(transcript: str) -> str:
    """'до скольки работает аптека?' -> 'аптека'."""
    text = _PUNCT.sub(" ", router.normalise(transcript))
    text = _ASKING.sub(" ", text)
    return " ".join(text.split())


def block(transcript: str, now: datetime) -> str:
    """The nearby-places block for the prompt. Never raises."""
    return answer(transcript, now).text


def answer(transcript: str, now: datetime, last_place: Place | None = None,
           dislikes: tuple[str, ...] = ()) -> PlaceAnswer:
    """The block plus the place it named. Never raises.

    `dislikes` are names the speaker has asked not to be offered; they are
    dropped before anything is selected, so a disliked business cannot come
    back as "the nearest" either.
    """
    try:
        return _answer(transcript, now, last_place, dislikes)
    except Exception:                                          # noqa: BLE001
        log.exception("places block failed")
        return PlaceAnswer(
            prompts.PLACES_UNAVAILABLE.format(query=query_from(transcript) or "?"))


def _answer(transcript: str, now: datetime, last_place: Place | None,
            dislikes: tuple[str, ...]) -> PlaceAnswer:
    shape = router.place_shape(transcript)

    # "А это далеко?" is about a place already named. It needs no search at
    # all -- and must not start one, or the answer would silently be about a
    # different business than the one the question meant.
    if shape == router.PLACE_DISTANCE:
        if last_place is None:
            return PlaceAnswer(prompts.PLACE_DISTANCE_UNKNOWN)
        return PlaceAnswer(
            prompts.PLACE_DISTANCE_BLOCK.format(
                name=last_place.name, distance=_distance_words(last_place)),
            last_place)

    key = config.get("DGIS_KEY")
    if not key:
        return PlaceAnswer(prompts.PLACES_NO_KEY)

    query = query_from(transcript)
    if not query:
        return PlaceAnswer(prompts.PLACES_NO_QUERY)

    lat, lon = config.location()
    payload, stale = base.cached_json(
        # v2: the cached payloads from before this rewrite have no `reviews`
        # key, and a 24-hour TTL means they would outlive the deploy.
        key=f"places:v2:{lat:.3f},{lon:.3f}:{query}",
        ttl=TTL,
        fetch=lambda: _fetch(key, query, lat, lon),
        now=now,
    )
    if payload is None:
        return PlaceAnswer(prompts.PLACES_UNAVAILABLE.format(query=query))

    items = (payload.get("result") or {}).get("items") or []
    places = [_parse_item(item, lat, lon) for item in items]
    places = [p for p in places if not _disliked(p.name, dislikes)]
    if not places:
        return PlaceAnswer(prompts.PLACES_NONE.format(query=query))

    result = _render(shape, places, now)
    if stale:
        result = PlaceAnswer(f"{result.text}\n{prompts.PLACES_STALE}", result.place)
    return result


def _render(shape: str, places: list[Place], now: datetime) -> PlaceAnswer:
    """One shape, one record, only the fields that answer it."""
    nearest = sorted(places, key=_distance_key)

    if shape == router.PLACE_LIST:
        # Names only. Four businesses with four fields each is what let the
        # model build a sentence out of three different listings; with nothing
        # but names there is nothing to cross-wire.
        names = ", ".join(f"«{p.name}»" for p in nearest[:LIST_RESULTS])
        return PlaceAnswer(prompts.PLACE_LIST_BLOCK.format(names=names))

    if shape == router.PLACE_OPEN_NOW:
        open_now = [p for p in nearest if _is_open(p, now)]
        if open_now:
            place = open_now[0]
            return PlaceAnswer(
                prompts.PLACE_OPEN_NOW_BLOCK.format(
                    name=place.name, address=place.address or "адрес неизвестен"),
                place)
        # Everything is closed. This branch is the whole point of the shape:
        # the old code answered this question with a business that shut four
        # hours earlier, because nothing ever compared the clock to the hours.
        soonest = _soonest_opening(nearest, now)
        if soonest is None:
            return PlaceAnswer(prompts.PLACE_NONE_OPEN_UNKNOWN)
        place, offset, at = soonest
        return PlaceAnswer(
            prompts.PLACE_NONE_OPEN_BLOCK.format(
                name=place.name, opens=_opening_words(offset, at)),
            place)

    if shape == router.PLACE_OPENS_EARLIEST:
        soonest = _soonest_opening(nearest, now)
        if soonest is None:
            # Nothing has hours -- but something may be open around the clock.
            always = [p for p in nearest if p.is_24x7]
            if always:
                return PlaceAnswer(
                    prompts.PLACE_ALWAYS_OPEN_BLOCK.format(name=always[0].name),
                    always[0])
            return PlaceAnswer(prompts.PLACE_NO_HOURS.format(name=nearest[0].name),
                               nearest[0])
        place, offset, at = soonest
        return PlaceAnswer(
            prompts.PLACE_OPENS_EARLIEST_BLOCK.format(
                name=place.name, opens=_opening_words(offset, at)),
            place)

    if shape == router.PLACE_RATING:
        rated = [p for p in places if p.rating is not None]
        if not rated:
            return PlaceAnswer(prompts.PLACE_NO_RATING.format(name=nearest[0].name),
                               nearest[0])
        place = max(rated, key=lambda p: p.rating)
        return PlaceAnswer(
            prompts.PLACE_RATING_BLOCK.format(
                name=place.name,
                rating=_rating_words(place.rating),
                reviews=num_to_words.count(place.reviews or 0, _REVIEWS)),
            place)

    if shape == router.PLACE_HOURS:
        place = nearest[0]
        hours = _hours_words(place, now)
        if hours is None:
            return PlaceAnswer(prompts.PLACE_NO_HOURS.format(name=place.name), place)
        return PlaceAnswer(
            prompts.PLACE_HOURS_BLOCK.format(name=place.name, hours=hours), place)

    if shape == router.PLACE_ADDRESS:
        place = nearest[0]
        if not place.address:
            return PlaceAnswer(prompts.PLACE_NO_ADDRESS.format(name=place.name), place)
        return PlaceAnswer(
            prompts.PLACE_ADDRESS_BLOCK.format(name=place.name,
                                               address=place.address), place)

    # PLACE_NEAREST -- the default. "Where is a pharmacy" is asking which one
    # to walk to, and the nearest is the only honest answer to that.
    place = nearest[0]
    return PlaceAnswer(
        prompts.PLACE_NEAREST_BLOCK.format(name=place.name,
                                           distance=_distance_words(place)), place)


def _fetch(key: str, query: str, lat: float, lon: float) -> dict:
    try:
        return base.fetch_json(URL, {
            "q": query,
            "point": f"{lon},{lat}",            # longitude first -- see module doc
            "radius": RADIUS_M,
            "key": key,
            "fields": ("items.point,items.schedule,items.address_name,"
                       "items.reviews"),
            "locale": "ru_RU",
            "page_size": PAGE_SIZE,
        })
    except urllib.error.HTTPError as exc:
        # 2GIS answers "nothing found" with a 404, which is a result, not an
        # outage. Anything else really is a failure and is re-raised so the
        # cache layer can serve stale data or report unavailability.
        if exc.code == 404:
            return {"result": {"items": []}}
        raise


# --- parsing ------------------------------------------------------------------

def _parse_item(item: dict, lat: float, lon: float) -> Place:
    point = item.get("point") or {}
    distance = None
    if "lat" in point and "lon" in point:
        distance = _haversine(lat, lon, point["lat"], point["lon"])

    # `reviews` is tier-dependent on the Catalog API: on a key without it the
    # field is simply absent, and every shape but PLACE_RATING is unaffected.
    reviews = item.get("reviews") or {}
    rating = reviews.get("general_rating")

    schedule = item.get("schedule") or {}
    week = tuple(
        tuple((s["from"], s["to"])
              for s in (schedule.get(day) or {}).get("working_hours") or []
              if s.get("from") and s.get("to"))
        for day in _WEEKDAY_KEYS
    )

    return Place(
        name=item.get("name") or "(без названия)",
        address=_address(item["address_name"]) if item.get("address_name") else None,
        distance_m=distance,
        rating=float(rating) if rating is not None else None,
        reviews=reviews.get("general_review_count"),
        is_24x7=bool(schedule.get("is_24x7")),
        week=week,
    )


def _address(name: str) -> str:
    """2GIS's address, as a person would read it out."""
    name = _HOUSE_NUMBER.sub(r" \1", name.strip())
    return _HOUSE_SUFFIX.sub(
        lambda m: f"{m.group(1)} {_SUFFIX_WORDS[m.group(2).lower().rstrip('.')]} "
                  f"{m.group(3)}",
        name)


def _disliked(name: str, dislikes: tuple[str, ...]) -> bool:
    """Has the speaker asked not to be offered this one?

    Substring match through `normalise`, because what gets stored is what was
    said out loud -- "экона" against "Аптека Экона".
    """
    haystack = router.normalise(name)
    return any(d and router.normalise(d) in haystack for d in dislikes)


# --- opening hours ------------------------------------------------------------

def _minutes(hhmm: str) -> int:
    hours, _, mins = hhmm.partition(":")
    return int(hours) * 60 + int(mins)


def _is_open(place: Place, now: datetime) -> bool:
    """Is it open at this exact moment?

    Spans that wrap past midnight ("22:00"-"06:00") are the reason this is not
    a one-line comparison, and they are exactly the businesses a question at
    2am is about.
    """
    if place.is_24x7:
        return True
    minute = now.hour * 60 + now.minute
    today = now.weekday()

    for start, end in place.week[today]:
        first, last = _minutes(start), _minutes(end)
        if last <= first:                      # wraps past midnight
            if minute >= first:
                return True
        elif first <= minute < last:
            return True

    # Yesterday's wrapping span may still be running: at 02:19 a place open
    # "22:00"-"06:00" is open, and it is filed under yesterday.
    for start, end in place.week[(today - 1) % 7]:
        first, last = _minutes(start), _minutes(end)
        if last <= first and minute < last:
            return True
    return False


def _next_opening(place: Place, now: datetime) -> tuple[int, str] | None:
    """(days from today, "HH:MM") of the next time it opens, or None."""
    if place.is_24x7:
        return None
    minute = now.hour * 60 + now.minute
    for offset in range(8):
        day = (now.weekday() + offset) % 7
        for start, _ in sorted(place.week[day]):
            if offset == 0 and _minutes(start) <= minute:
                continue
            return offset, start
    return None


def _soonest_opening(places: list[Place],
                     now: datetime) -> tuple[Place, int, str] | None:
    """Whichever of them opens first, with when."""
    candidates = []
    for place in places:
        nxt = _next_opening(place, now)
        if nxt is not None:
            candidates.append((nxt[0], _minutes(nxt[1]), place, nxt[1]))
    if not candidates:
        return None
    offset, _, place, at = min(candidates, key=lambda c: (c[0], c[1]))
    return place, offset, at


# --- spoken forms -------------------------------------------------------------
# Everything the model is handed is already spoken Russian: it is read to a
# person through a speaker, and a small model repeats the shape it is given.
# "+14°" in the prompt comes back as "+14°" in the answer, whatever the
# instructions say, so the digits never get written in the first place.

_OPENING_DAYS = {0: "сегодня", 1: "завтра", 2: "послезавтра"}


def _opening_words(offset: int, at: str) -> str:
    """'завтра в девять часов'."""
    day = _OPENING_DAYS.get(offset)
    if day is None:
        day = f"через {num_to_words.count(offset, ('день', 'дня', 'дней'))}"
    return f"{day} в {num_to_words.time_words(at)}"


def _hours_words(place: Place, now: datetime) -> str | None:
    if place.is_24x7:
        return "круглосуточно"
    spans = place.week[now.weekday()]
    if not spans:
        return None
    return ", ".join(num_to_words.time_range(start, end) for start, end in spans)


def _rating_words(value: float) -> str:
    """'четыре и пять' -- how a rating is said out loud.

    Deliberately not spell()'s generic decimal path, which gives "четыре целых
    пять десятых". That is right for a measurement and wrong for a rating:
    nobody reads a star rating out as a decimal fraction.
    """
    whole, frac = divmod(int(round(float(value) * 10)), 10)
    words = num_to_words.cardinal(whole)
    return f"{words} и {num_to_words.cardinal(frac)}" if frac else words


def _distance_key(place: Place) -> float:
    """Sort key. Places with no coordinates go last rather than first."""
    return place.distance_m if place.distance_m is not None else float("inf")


def _distance_words(place: Place) -> str:
    if place.distance_m is None:
        return "расстояние неизвестно"
    return _distance_phrase(place.distance_m)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * radius * math.asin(math.sqrt(a))


def _distance_phrase(metres: float) -> str:
    """Distance, phrased the way a person would say it out loud.

    Nominative, and no preposition: the template around it supplies the
    grammar, so "триста метров" slots into whatever sentence is built. "в
    трёхстах метрах" would need the prepositional case for every number, to
    say exactly the same thing.
    """
    if metres < 50:
        return "прямо у дома"
    if metres < 950:
        return num_to_words.count(int(round(metres, -1)), _METRES)
    kilometres, rest = divmod(int(round(metres, -2)), 1000)
    phrase = num_to_words.count(kilometres, _KILOMETRES)
    return f"{phrase} {num_to_words.count(rest, _METRES)}" if rest else phrase
