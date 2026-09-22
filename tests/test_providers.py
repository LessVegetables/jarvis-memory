"""
Weather and places provider tests. No network: `base.fetch_json` is
replaced with a function returning canned payloads shaped like the real
APIs, and a counter shows when it was (not) called.

Run:  python3 tests/test_providers.py   (or python3 -m pytest tests/ -q)
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP_DB = Path(tempfile.mkdtemp(prefix="jarvis-test-")) / "test.db"
os.environ["JARVIS_DB"] = str(_TMP_DB)
os.environ["JARVIS_LAT"] = "55.7539"
os.environ["JARVIS_LON"] = "37.6208"

import jarvis_memory as memory  # noqa: E402
from jarvis_memory import db, prompts, seed  # noqa: E402
from jarvis_memory.providers import base, places, weather  # noqa: E402

NOW = datetime(2026, 9, 14, 14, 30)   # a Monday

WEATHER_PAYLOAD = {
    "current": {"temp_c": 14.3, "wind_kph": 11.5,
                "condition": {"text": "Облачно"}},
    "forecast": {"forecastday": [
        {"date": "2026-09-14", "day": {"maxtemp_c": 18.2, "mintemp_c": 9.1,
                                       "daily_chance_of_rain": 10,
                                       "condition": {"text": "Переменная облачность"}}},
        {"date": "2026-09-15", "day": {"maxtemp_c": 12.0, "mintemp_c": 6.0,
                                       "daily_chance_of_rain": 80,
                                       "condition": {"text": "Дождь"}}},
        {"date": "2026-09-16", "day": {"maxtemp_c": 15.0, "mintemp_c": 7.0,
                                       "daily_chance_of_rain": 20,
                                       "condition": {"text": "Ясно"}}},
    ]},
}

PLACES_PAYLOAD = {"result": {"items": [
    {"name": "Аптека Вита", "address_name": "ул. Ленина, 5",
     "point": {"lat": 55.7560, "lon": 37.6230},
     "schedule": {"Mon": {"working_hours": [{"from": "08:00", "to": "22:00"}]}}},
    {"name": "Аптека 24", "address_name": "пр. Мира, 10",
     "point": {"lat": 55.7600, "lon": 37.6300},
     "schedule": {"is_24x7": True}},
    {"name": "Аптека без графика", "point": {"lat": 55.7539, "lon": 37.6208}},
]}}


class FakeFetch:
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, 0

    def __call__(self, url, params):
        self.calls += 1
        if self.error:
            raise self.error
        return self.payload


def setup(weather_key="k", places_key="k"):
    for name, value in (("WEATHERAPI_KEY", weather_key), ("DGIS_KEY", places_key)):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    db.close()
    seed.seed()
    db.connect().execute("DELETE FROM api_cache")
    db.connect().commit()


# --- weather ------------------------------------------------------------------

def test_weather_without_key_says_so():
    setup(weather_key=None)
    base.fetch_json = FakeFetch(WEATHER_PAYLOAD)
    assert weather.block("какая погода?", NOW) == prompts.WEATHER_NO_KEY
    assert base.fetch_json.calls == 0


def test_weather_today_renders_current_and_forecast():
    setup()
    base.fetch_json = FakeFetch(WEATHER_PAYLOAD)
    text = weather.block("какая сегодня погода?", NOW)
    # Spoken Russian, not glyphs: the model repeats whatever shape it is
    # given, and "+14°" comes back out of the speaker as a string of symbols.
    assert text.startswith("Погода (сегодня): сейчас плюс четырнадцать градусов, "
                           "облачно, ветер двенадцать километров в час."), text
    assert "Днём плюс восемнадцать градусов, ночью плюс девять градусов" in text
    assert "вероятность дождя десять процентов" in text
    assert not any(char.isdigit() for char in text), text


def test_weather_tomorrow_uses_second_day_without_current():
    setup()
    base.fetch_json = FakeFetch(WEATHER_PAYLOAD)
    text = weather.block("а завтра будет дождь?", NOW)
    assert text.startswith("Погода (завтра): Днём плюс двенадцать градусов"), text
    assert "сейчас" not in text
    assert "дождя восемьдесят процентов" in text


def test_weather_is_cached_within_ttl():
    setup()
    base.fetch_json = FakeFetch(WEATHER_PAYLOAD)
    weather.block("погода", NOW)
    weather.block("погода завтра", NOW + timedelta(minutes=5))
    assert base.fetch_json.calls == 1, base.fetch_json.calls
    weather.block("погода", NOW + timedelta(minutes=20))
    assert base.fetch_json.calls == 2


def test_weather_serves_stale_when_fetch_fails():
    setup()
    base.fetch_json = FakeFetch(WEATHER_PAYLOAD)
    weather.block("погода", NOW)
    base.fetch_json = FakeFetch(error=TimeoutError("timed out"))
    text = weather.block("погода", NOW + timedelta(hours=2))
    assert text.startswith("Погода (сегодня): сейчас плюс четырнадцать градусов"), text
    assert prompts.WEATHER_STALE in text


def test_weather_unavailable_when_nothing_cached():
    setup()
    base.fetch_json = FakeFetch(error=ConnectionError("no network"))
    assert weather.block("погода", NOW) == prompts.WEATHER_UNAVAILABLE


def test_weather_out_of_range_days():
    setup()
    base.fetch_json = FakeFetch(WEATHER_PAYLOAD)
    assert weather.block("какая вчера была погода", NOW) == prompts.WEATHER_PAST
    assert base.fetch_json.calls == 0


# --- places -------------------------------------------------------------------

def test_query_extraction():
    cases = [
        ("до скольки работает аптека?", "аптека"),
        ("где купить молоко", "молоко"),
        ("что есть поблизости", ""),
        ("какие кафе рядом открыты сейчас", "кафе"),
        ("найди ближайший банкомат сбербанка", "банкомат сбербанка"),
    ]
    for phrase, expected in cases:
        assert places.query_from(phrase) == expected, (phrase, places.query_from(phrase))


def test_places_renders_distance_and_hours():
    setup()
    base.fetch_json = FakeFetch(PLACES_PAYLOAD)
    text = places.block("до скольки работает аптека?", NOW)
    assert text.startswith("Места поблизости по запросу «аптека»"), text
    # The house number joins the street without a comma: spoken, that pause
    # splits one name in two -- "Морской проспект... шесть".
    assert ("- Аптека Вита, ул. Ленина 5, двести семьдесят метров, "
            "сегодня с восьми до двадцати двух") in text, text
    assert ("Аптека 24, пр. Мира 10, восемьсот девяносто метров, "
            "круглосуточно") in text, text
    assert "- Аптека без графика, прямо у дома" in text, text


def test_places_without_query_asks_back():
    setup()
    base.fetch_json = FakeFetch(PLACES_PAYLOAD)
    assert places.block("что есть поблизости?", NOW) == prompts.PLACES_NO_QUERY
    assert base.fetch_json.calls == 0


def test_places_404_means_nothing_found():
    import urllib.error
    setup()
    base.fetch_json = FakeFetch(error=urllib.error.HTTPError(
        "u", 404, "Not Found", {}, None))
    assert places.block("аптека", NOW) == prompts.PLACES_NONE.format(query="аптека")


def test_places_sends_lon_lat_in_that_order():
    setup()
    seen = {}

    def capture(url, params):
        seen.update(params)
        return PLACES_PAYLOAD
    base.fetch_json = capture
    places.block("аптека", NOW)
    assert seen["point"] == "37.6208,55.7539", seen["point"]


# --- through build_context ---------------------------------------------------

def test_guest_gets_weather_but_not_schedule():
    setup()
    base.fetch_json = FakeFetch(WEATHER_PAYLOAD)
    ctx = memory.build_context(None, "какая сегодня погода?", now=NOW)
    assert prompts.WHO_GUEST in ctx.system_prompt
    assert "Погода (сегодня)" in ctx.system_prompt
    ctx = memory.build_context(None, "какое у меня расписание?", now=NOW)
    assert prompts.WHO_UNKNOWN in ctx.system_prompt
    assert "Расписание" not in ctx.system_prompt


def test_known_user_places_question_keeps_facts():
    setup()
    base.fetch_json = FakeFetch(PLACES_PAYLOAD)
    ctx = memory.build_context("anton", "какие кафе рядом?", now=NOW)
    assert "Места поблизости по запросу «кафе»" in ctx.system_prompt
    assert "Что важно помнить о собеседнике" in ctx.system_prompt


TESTS = [
    test_weather_without_key_says_so,
    test_weather_today_renders_current_and_forecast,
    test_weather_tomorrow_uses_second_day_without_current,
    test_weather_is_cached_within_ttl,
    test_weather_serves_stale_when_fetch_fails,
    test_weather_unavailable_when_nothing_cached,
    test_weather_out_of_range_days,
    test_query_extraction,
    test_places_renders_distance_and_hours,
    test_places_without_query_asks_back,
    test_places_404_means_nothing_found,
    test_places_sends_lon_lat_in_that_order,
    test_guest_gets_weather_but_not_schedule,
    test_known_user_places_question_keeps_facts,
]

if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"  ok  {test.__name__}")
    print(f"OK — {len(TESTS)} provider tests")
