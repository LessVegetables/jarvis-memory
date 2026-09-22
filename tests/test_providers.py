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

# 02:19 on the Tuesday. The hour is not arbitrary: it is when the assistant
# was asked which pharmacy was open and answered with one that had closed at
# 22:00, because nothing ever compared the clock against the hours.
NIGHT = datetime(2026, 9, 15, 2, 19)


def _every_day(start, end):
    return {day: {"working_hours": [{"from": start, "to": end}]}
            for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]}


PLACES_PAYLOAD = {"result": {"items": [
    {"name": "Аптека Вита", "address_name": "ул. Ленина, 5",
     "point": {"lat": 55.7560, "lon": 37.6230},              # ~270 m, nearest
     "reviews": {"general_rating": 4.3, "general_review_count": 128},
     "schedule": _every_day("08:00", "22:00")},
    {"name": "Аптека 24", "address_name": "пр. Мира, 10",
     "point": {"lat": 55.7600, "lon": 37.6300},              # ~890 m
     "reviews": {"general_rating": 4.8, "general_review_count": 37},
     "schedule": {"is_24x7": True}},
    # No schedule, no address, no reviews: every shape has to survive a
    # listing that is missing the field it wants.
    {"name": "Аптека без графика", "point": {"lat": 55.7700, "lon": 37.6400}},
]}}

# Nothing open around the clock, so at 02:19 every one of them is shut.
CLOSED_PAYLOAD = {"result": {"items": [
    {"name": "Аптека Экона", "address_name": "Морской проспект, 6",
     "point": {"lat": 55.7560, "lon": 37.6230},
     "schedule": _every_day("09:00", "21:00")},
    {"name": "Аптека Академическая", "address_name": "ул. Ленина, 5",
     "point": {"lat": 55.7600, "lon": 37.6300},
     "schedule": _every_day("08:00", "22:00")},
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


def _names_in(text):
    """Which of the fixture's businesses a block mentions."""
    return {name for name in ("Аптека Вита", "Аптека 24", "Аптека без графика")
            if name in text}


def test_block_names_exactly_one_place():
    """The bug this whole rewrite exists for.

    Four businesses listed with four fields each let the model build one
    sentence out of three different listings. Every shape but the list names
    a single business, so there is nothing left to cross-wire.
    """
    setup()
    for question in ("какая аптека ближе всего", "какая аптека сейчас открыта",
                     "до скольки работает аптека", "адрес аптеки",
                     "у какой аптеки рейтинг лучше"):
        base.fetch_json = FakeFetch(PLACES_PAYLOAD)
        db.connect().execute("DELETE FROM api_cache")
        text = places.block(question, NOW)
        assert len(_names_in(text)) == 1, (question, text)


def test_nearest_gives_distance_and_no_hours():
    setup()
    base.fetch_json = FakeFetch(PLACES_PAYLOAD)
    text = places.block("какая аптека ближе всего", NOW)
    assert "«Аптека Вита»" in text, text
    assert "двести семьдесят метров" in text, text
    # The fields that do not answer "which is nearest" are simply absent.
    assert "восьми" not in text and "Ленина" not in text, text


def test_hours_shape_says_hours_only():
    setup()
    base.fetch_json = FakeFetch(PLACES_PAYLOAD)
    text = places.block("до скольки работает аптека?", NOW)
    assert "«Аптека Вита»" in text, text
    assert "с восьми до двадцати двух" in text, text
    assert "метров" not in text, text


def test_address_shape_joins_house_number():
    setup()
    base.fetch_json = FakeFetch(PLACES_PAYLOAD)
    text = places.block("какой адрес у аптеки", NOW)
    # The house number joins the street without a comma: spoken, that pause
    # splits one name in two -- "Морской проспект... шесть".
    assert "ул. Ленина 5" in text, text


def test_rating_picks_the_best_not_the_nearest():
    setup()
    base.fetch_json = FakeFetch(PLACES_PAYLOAD)
    text = places.block("у какой аптеки рейтинг лучше", NOW)
    assert "«Аптека 24»" in text, text          # 4.8, though it is the farther one
    # A rating is said "четыре и восемь", not "четыре целых восемь десятых".
    assert "четыре и восемь" in text, text
    assert "тридцать семь отзывов" in text, text


def test_open_now_at_night_skips_the_closed_one():
    """02:19, straight from the log. Вита shut at 22:00; Аптека 24 has not."""
    setup()
    base.fetch_json = FakeFetch(PLACES_PAYLOAD)
    text = places.block("какая аптека сейчас работает", NIGHT)
    assert "«Аптека 24»" in text, text
    assert "Вита" not in text, text


def test_nothing_open_says_so_and_gives_the_next_opening():
    """The other half of the log: when everything is shut, say so."""
    setup()
    base.fetch_json = FakeFetch(CLOSED_PAYLOAD)
    text = places.block("какая аптека сейчас работает", NIGHT)
    assert "не работает ничего" in text, text
    # Академическая opens at 08:00, an hour before Экона.
    assert "«Аптека Академическая»" in text, text
    assert "сегодня в восемь часов" in text, text


def test_list_shape_carries_names_and_nothing_else():
    setup()
    base.fetch_json = FakeFetch(PLACES_PAYLOAD)
    text = places.block("какие аптеки рядом", NOW)
    assert len(_names_in(text)) == 3, text
    # Names cannot be mismatched with each other; fields can.
    for leak in ("метров", "Ленина", "восьми", "рейтинг"):
        assert leak not in text, (leak, text)


def test_missing_fields_get_a_sentence_not_a_hole():
    setup()
    payload = {"result": {"items": [
        {"name": "Аптека без всего", "point": {"lat": 55.7560, "lon": 37.6230}},
    ]}}
    for question, expected in (("до скольки работает аптека", "Часы работы"),
                               ("какой адрес у аптеки", "Адрес места"),
                               ("у какой аптеки рейтинг лучше", "Оценок и отзывов нет")):
        setup()
        base.fetch_json = FakeFetch(payload)
        text = places.block(question, NOW)
        assert expected in text, (question, text)


def test_distance_followup_uses_the_remembered_place():
    """"А это далеко?" is about the place named a turn ago, not a new search."""
    setup()
    base.fetch_json = FakeFetch(PLACES_PAYLOAD)
    first = places.answer("какая аптека ближе всего", NOW)
    assert first.place is not None

    # No fetch at all on the follow-up: searching again could quietly answer
    # about a different business than the question meant.
    base.fetch_json = FakeFetch(error=AssertionError("must not search again"))
    text = places.answer("а это далеко?", NOW, last_place=first.place).text
    assert "«Аптека Вита»" in text, text
    assert "двести семьдесят метров" in text, text


def test_distance_followup_without_a_place_asks_back():
    setup()
    assert places.block("а это далеко?", NOW) == prompts.PLACE_DISTANCE_UNKNOWN


def test_dislikes_are_dropped_before_anything_is_chosen():
    """A disliked place must not come back as "the nearest" either."""
    setup()
    base.fetch_json = FakeFetch(PLACES_PAYLOAD)
    text = places.answer("какая аптека ближе всего", NOW,
                         dislikes=("вита",)).text
    assert "Вита" not in text, text
    assert "«Аптека 24»" in text, text


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
    ctx = memory.build_context("daniil", "какие кафе рядом?", now=NOW)
    assert "Поблизости есть" in ctx.system_prompt
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
    test_block_names_exactly_one_place,
    test_nearest_gives_distance_and_no_hours,
    test_hours_shape_says_hours_only,
    test_address_shape_joins_house_number,
    test_rating_picks_the_best_not_the_nearest,
    test_open_now_at_night_skips_the_closed_one,
    test_nothing_open_says_so_and_gives_the_next_opening,
    test_list_shape_carries_names_and_nothing_else,
    test_missing_fields_get_a_sentence_not_a_hole,
    test_distance_followup_uses_the_remembered_place,
    test_distance_followup_without_a_place_asks_back,
    test_dislikes_are_dropped_before_anything_is_chosen,
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
