"""
"Запомни, я терпеть не могу аптеку Экона" -> it stops being offered.

The point of these tests is the difference between remembering a preference
and honouring one. Storing the sentence was already possible: RAG would
retrieve it and the model would recite it back, which looks like it worked
and changes nothing. What has to be true is that the next question about
pharmacies comes back without that pharmacy in it.

Run:  python3 tests/test_preferences.py   (or python3 -m pytest tests/ -q)
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP_DB = Path(tempfile.mkdtemp(prefix="jarvis-test-")) / "test.db"
os.environ["JARVIS_DB"] = str(_TMP_DB)
os.environ["JARVIS_LAT"] = "55.7539"
os.environ["JARVIS_LON"] = "37.6208"
os.environ["DGIS_KEY"] = "k"

import jarvis_memory as memory  # noqa: E402
from jarvis_memory import db, history, llm_intent, router, seed, store  # noqa: E402
from jarvis_memory.providers import base  # noqa: E402

NOW = datetime(2026, 9, 14, 14, 30)

PAYLOAD = {"result": {"items": [
    {"name": "Аптека Экона", "address_name": "Морской проспект, 6",
     "point": {"lat": 55.7545, "lon": 37.6215}},          # the nearest one
    {"name": "Аптека Вита", "address_name": "ул. Ленина, 5",
     "point": {"lat": 55.7600, "lon": 37.6300}},
]}}


class FakeAsk:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def __call__(self, system, user, max_tokens=llm_intent.MAX_TOKENS):
        self.calls += 1
        return self.reply


def setup(subject_reply="Экона"):
    os.environ["JARVIS_LLM_INTENT"] = "1"
    llm_intent._ask = FakeAsk(subject_reply)
    db.close()
    seed.seed()
    history.clear()
    db.connect().execute("DELETE FROM facts")
    db.connect().execute("DELETE FROM api_cache")
    db.connect().commit()
    base.fetch_json = lambda url, params: PAYLOAD


def ask(phrase):
    db.connect().execute("DELETE FROM api_cache")
    db.connect().commit()
    return memory.build_context("daniil", phrase, now=NOW).system_prompt


def test_a_dislike_changes_behaviour_not_just_memory():
    setup()
    # Before: the nearest pharmacy is the one he is about to object to.
    assert "Экона" in ask("какая аптека ближе всего")

    said = ask("запомни, я терпеть не могу аптеку Экона")
    assert "предлагать не нужно" in said, said

    after = ask("какая аптека ближе всего")
    assert "Экона" not in after, after
    assert "Вита" in after, after


def test_the_filter_survives_inflection():
    """What was said is inflected; what 2GIS returns is not."""
    for reply in ("Экона", "аптеку Экона", "экона"):
        setup(subject_reply=reply)
        ask("запомни, не предлагай мне аптеку Экона")
        assert "Экона" not in ask("какая аптека ближе всего"), reply


def test_it_is_stored_as_a_fact_as_well():
    """Filtering is the new part; being recited when relevant still matters."""
    setup()
    ask("запомни, я терпеть не могу аптеку Экона")
    rows = db.connect().execute(
        "SELECT text, polarity, subject FROM facts WHERE user_id = 'daniil'"
    ).fetchall()
    assert len(rows) == 1, rows
    assert "терпеть не могу" in rows[0]["text"]
    assert rows[0]["polarity"] == store.NEGATIVE
    assert rows[0]["subject"] == "Экона"


def test_an_ordinary_fact_is_not_a_dislike():
    setup()
    said = ask("запомни, что я вегетарианец")
    assert "предлагать не нужно" not in said, said
    row = db.connect().execute(
        "SELECT polarity, subject FROM facts WHERE user_id = 'daniil'").fetchone()
    assert row["polarity"] is None and row["subject"] is None
    # And the extraction call was never made for it.
    assert llm_intent._ask.calls == 0


def test_a_failed_extraction_still_stores_the_fact():
    """A None from the model is cheap: the fact stays, it just does not filter.

    The alternative -- believing an invented name -- would silently hide a
    business nobody objected to, and look like a bug in the 2GIS search.
    """
    for reply in (None, "Магнит", "это такое место, которое ему не нравится"):
        setup(subject_reply=reply)
        said = ask("запомни, я терпеть не могу аптеку Экона")
        assert "предлагать не нужно" not in said, reply
        row = db.connect().execute(
            "SELECT polarity, subject FROM facts WHERE user_id = 'daniil'").fetchone()
        assert row["polarity"] == store.NEGATIVE, reply
        assert row["subject"] is None, reply
        # Nothing was filtered on a guess.
        assert "Экона" in ask("какая аптека ближе всего"), reply


def test_one_persons_dislike_is_not_anothers():
    setup()
    ask("запомни, я терпеть не могу аптеку Экона")
    assert store.get_dislikes("daniil") == ("Экона",)
    assert store.get_dislikes("fedor") == ()
    fedor = memory.build_context("fedor", "какая аптека ближе всего",
                                 now=NOW).system_prompt
    assert "Экона" in fedor, fedor


def test_a_guest_is_filtered_by_nobodys_preferences():
    setup()
    ask("запомни, я терпеть не могу аптеку Экона")
    guest = memory.build_context(None, "какая аптека ближе всего",
                                 now=NOW).system_prompt
    assert "Экона" in guest, guest


TESTS = [
    test_a_dislike_changes_behaviour_not_just_memory,
    test_the_filter_survives_inflection,
    test_it_is_stored_as_a_fact_as_well,
    test_an_ordinary_fact_is_not_a_dislike,
    test_a_failed_extraction_still_stores_the_fact,
    test_one_persons_dislike_is_not_anothers,
    test_a_guest_is_filtered_by_nobodys_preferences,
]

if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"  ok  {test.__name__}")
    print(f"OK — {len(TESTS)} preference tests")
