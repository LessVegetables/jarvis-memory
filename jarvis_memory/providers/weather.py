"""
Weather via weatherapi.com.

One request covers three days, so "сегодня", "завтра" and "послезавтра" are
all served from the same cached payload; the question only decides which
part of it gets rendered. Cached for 15 minutes -- the weather does not
change faster than that, and the free tier is not infinite.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from .. import config, num_to_words, prompts, router
from . import base

log = logging.getLogger(__name__)

URL = "https://api.weatherapi.com/v1/forecast.json"
TTL = timedelta(minutes=15)
DAYS = 3


def block(transcript: str, now: datetime) -> str:
    """The weather block for the prompt. Never raises."""
    try:
        return _block(transcript, now)
    except Exception:                                          # noqa: BLE001
        log.exception("weather block failed")
        return prompts.WEATHER_UNAVAILABLE


def _block(transcript: str, now: datetime) -> str:
    key = config.get("WEATHERAPI_KEY")
    if not key:
        return prompts.WEATHER_NO_KEY

    today = now.date()
    day = router.resolve_day(transcript, today)
    offset = (day - today).days
    if offset < 0:
        # History is a paid endpoint on weatherapi; not worth it for a
        # question nobody asks a kitchen speaker.
        return prompts.WEATHER_PAST
    if offset >= DAYS:
        return prompts.WEATHER_TOO_FAR.format(day=router.day_label(day, today))

    lat, lon = config.location()
    payload, stale = base.cached_json(
        key=f"weather:{lat:.3f},{lon:.3f}",
        ttl=TTL,
        fetch=lambda: base.fetch_json(URL, {
            "key": key, "q": f"{lat},{lon}", "days": DAYS,
            "lang": "ru", "aqi": "no", "alerts": "no",
        }),
        now=now,
    )
    if payload is None:
        return prompts.WEATHER_UNAVAILABLE

    text = _render(payload, offset, router.day_label(day, today))
    return f"{text} {prompts.WEATHER_STALE}" if stale else text


def _render(payload: dict, offset: int, label: str) -> str:
    """One or two sentences the model can read out as they are."""
    forecast = payload["forecast"]["forecastday"]
    if offset >= len(forecast):
        return prompts.WEATHER_TOO_FAR.format(day=label)
    day = forecast[offset]["day"]

    parts = [f"Погода ({label}):"]
    if offset == 0:
        current = payload["current"]
        parts.append(
            f"сейчас {_deg(current['temp_c'])}, {current['condition']['text'].lower()}, "
            f"ветер {_wind(current['wind_kph'])}."
        )
    parts.append(
        f"Днём {_deg(day['maxtemp_c'])}, ночью {_deg(day['mintemp_c'])}, "
        f"{day['condition']['text'].lower()}, "
        f"вероятность дождя {_percent(day.get('daily_chance_of_rain', 0))}."
    )
    return " ".join(parts)


# Everything the model is handed is already spoken Russian: it is read to a
# person through a speaker, and a small model repeats the shape it is given.
# "+14°" in the prompt comes back as "+14°" in the answer, whatever the
# instructions say, so the digits never get written in the first place.
def _deg(value) -> str:
    """'плюс четырнадцать градусов' -- the sign is spoken, because
    "три градуса" alone does not say which side of zero it is on."""
    rounded = round(float(value))
    words = num_to_words.count(rounded, ("градус", "градуса", "градусов"))
    return f"плюс {words}" if rounded > 0 else words


def _wind(kph) -> str:
    return num_to_words.count(round(float(kph)),
                              ("километр", "километра", "километров")) + " в час"


def _percent(value) -> str:
    return num_to_words.count(int(float(value)),
                              ("процент", "процента", "процентов"))
