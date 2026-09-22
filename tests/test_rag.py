"""
RAG plumbing tests.

These do not use the real neural model. They inject a deterministic
stand-in embedder built from character trigrams, which is enough to verify
everything around the model: that vectors are stored, that search is scoped
to one user, that a fact said out loud is searchable immediately, and that
everything degrades to a non-semantic query when the model is missing.

The stand-in is genuinely similarity-preserving for these cases -- strings
sharing trigrams land near each other -- so a query about орехи really does
retrieve the allergy fact. What it cannot test is semantic matching across
different words ("миндаль" -> "орехи"); that needs the real model, and is
what tools/ram_spike.py exercises on the board.

Run:  python3 tests/test_rag.py   (or python3 -m pytest tests/ -q)
"""

import math
import os
import sys
import tempfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point at a scratch database before anything opens the real one.
_TMP_DB = Path(tempfile.mkdtemp(prefix="jarvis-test-")) / "test.db"
os.environ["JARVIS_DB"] = str(_TMP_DB)

from datetime import datetime, timedelta  # noqa: E402

import jarvis_memory as memory  # noqa: E402
from jarvis_memory import db, embeddings, history, prompts, seed, store  # noqa: E402

NOW = datetime(2026, 9, 14, 14, 30)
DAYS_AGO = lambda n: NOW - timedelta(days=n)  # noqa: E731


def trigram_embed(texts):
    """Deterministic stand-in for the real model.

    crc32 rather than hash(): Python randomises string hashing per process,
    which would make stored vectors and query vectors disagree between runs.
    """
    vectors = []
    for text in texts:
        vector = [0.0] * embeddings.DIM
        lowered = f"  {text.lower()}  "
        for i in range(len(lowered) - 2):
            bucket = zlib.crc32(lowered[i:i + 3].encode()) % embeddings.DIM
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


def setup():
    embeddings.set_backend(trigram_embed)
    db.close()
    seed.seed()
    assert db.vec_available(), "sqlite-vec did not load; cannot test search"


def test_vectors_are_built():
    setup()
    facts = db.connect().execute("SELECT count(*) FROM facts").fetchone()[0]
    vectors = db.connect().execute("SELECT count(*) FROM vec_facts").fetchone()[0]
    assert vectors == facts, f"{facts} facts but {vectors} vectors"


def test_search_finds_the_relevant_fact():
    setup()
    found = store.get_facts("daniil", "а орехи мне можно?", limit=3)
    assert any("орех" in f.lower() for f in found), found

    found = store.get_facts("daniil", "он с собакой гуляет?", limit=3)
    assert any("Бобик" in f for f in found), found


def test_search_is_scoped_to_one_user():
    setup()
    # Fedor's facts mention фортепиано; Daniil's never should, whatever
    # Daniil asks. A partition key enforces this inside the index -- filtering
    # after a global search could return k rows that are all hers.
    for question in ["играет на фортепиано?", "что там с музыкой",
                     "аллергия", "расскажи о себе"]:
        for fact in store.get_facts("daniil", question, limit=5):
            assert "фортепиано" not in fact.lower(), (question, fact)


def test_new_fact_is_searchable_immediately():
    setup()
    store.add_fact("daniil", "терпеть не может громкую музыку по утрам")
    found = store.get_facts("daniil", "громкую музыку любит?", limit=3)
    assert any("громкую музыку" in f for f in found), found


def test_duplicate_fact_is_not_indexed_twice():
    setup()
    first = store.add_fact("daniil", "пьёт чай с молоком")
    second = store.add_fact("daniil", "Пьёт чай с молоком")
    assert first == second
    rows = db.connect().execute(
        "SELECT count(*) FROM vec_facts WHERE fact_id = ?", (first,)
    ).fetchone()[0]
    assert rows == 1, f"{rows} vectors for one fact"


def test_falls_back_when_model_is_missing():
    setup()
    embeddings.set_backend(None)
    try:
        found = store.get_facts("daniil", "а орехи мне можно?", limit=3)
        # No semantic search, but the assistant must still get something.
        assert len(found) == 3, found
    finally:
        embeddings.set_backend(trigram_embed)


# --- dialogue archive (step 7) ------------------------------------------------

def test_exchange_is_archived_with_vector():
    setup()
    memory.record_answer("daniil", "как зовут мою собаку?", "Бобик.", now=NOW)
    conn = db.connect()
    assert conn.execute("SELECT count(*) FROM dialogue").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM vec_dialogue").fetchone()[0] == 1


def test_unknown_speaker_is_not_archived():
    setup()
    memory.record_answer(None, "какая погода?", "Солнечно.", now=NOW)
    # The live buffer gets it (so "повтори" works for a guest) ...
    assert history.get_last_answer(None, now=NOW) == "Солнечно."
    # ... but nothing is written to anyone's permanent memory.
    assert db.connect().execute("SELECT count(*) FROM dialogue").fetchone()[0] == 0


def test_recall_finds_relevant_past_exchange():
    setup()
    memory.record_answer("daniil", "Бобик заболел, что делать?",
                         "Отвези Бобика к ветеринару на Ленина.", now=DAYS_AGO(3))
    memory.record_answer("daniil", "сколько будет два плюс два?",
                         "Четыре.", now=DAYS_AGO(2))
    found = store.get_dialogue("daniil", "что там с Бобиком?",
                               limit=1, before=NOW - history.TTL)
    assert found and "Бобик" in found[0][0], found


def test_recent_turns_are_not_recalled():
    """A turn inside the live buffer is already in the messages; recalling
    it from the archive as well would put it in the prompt twice."""
    setup()
    memory.record_answer("daniil", "Бобик заболел", "К ветеринару.", now=NOW)
    found = store.get_dialogue("daniil", "что с Бобиком?",
                               limit=2, before=NOW - history.TTL)
    assert found == [], found


def test_recall_is_scoped_to_user():
    setup()
    memory.record_answer("fedor", "когда репетиция на фортепиано?",
                         "В среду в шесть.", now=DAYS_AGO(2))
    found = store.get_dialogue("daniil", "фортепиано репетиция",
                               limit=5, before=NOW - history.TTL)
    assert found == [], found


def test_archive_is_pruned_to_cap():
    setup()
    original = store.MAX_DIALOGUE_ROWS
    store.MAX_DIALOGUE_ROWS = 5
    try:
        for i in range(8):
            memory.record_answer("daniil", f"вопрос номер {i}", f"ответ {i}",
                                 now=DAYS_AGO(30) + timedelta(minutes=i))
        conn = db.connect()
        rows = conn.execute("SELECT question FROM dialogue ORDER BY id").fetchall()
        assert len(rows) == 5, len(rows)
        # The oldest three are the ones that went.
        assert rows[0]["question"] == "вопрос номер 3", rows[0]["question"]
        # Vectors were pruned in step with the rows -- no orphans.
        assert conn.execute("SELECT count(*) FROM vec_dialogue").fetchone()[0] == 5
    finally:
        store.MAX_DIALOGUE_ROWS = original


def test_context_includes_past_dialogue_for_general_questions():
    setup()
    memory.clear()
    memory.record_answer("daniil", "Бобик заболел, что делать?",
                         "Отвези Бобика к ветеринару.", now=DAYS_AGO(3))
    ctx = memory.build_context("daniil", "что там с Бобиком, как он?", now=NOW)
    assert ctx.intent == "general"
    assert prompts.DIALOGUE_BLOCK.split(":")[0] in ctx.system_prompt, ctx.system_prompt
    assert "ветеринару" in ctx.system_prompt


def test_context_omits_past_dialogue_for_schedule_questions():
    setup()
    memory.clear()
    memory.record_answer("daniil", "Бобик заболел", "К ветеринару.", now=DAYS_AGO(3))
    ctx = memory.build_context("daniil", "что у меня сегодня?", now=NOW)
    assert ctx.intent == "schedule"
    assert "прошлых разговоров" not in ctx.system_prompt


TESTS = [
    test_vectors_are_built,
    test_search_finds_the_relevant_fact,
    test_search_is_scoped_to_one_user,
    test_new_fact_is_searchable_immediately,
    test_duplicate_fact_is_not_indexed_twice,
    test_falls_back_when_model_is_missing,
    test_exchange_is_archived_with_vector,
    test_unknown_speaker_is_not_archived,
    test_recall_finds_relevant_past_exchange,
    test_recent_turns_are_not_recalled,
    test_recall_is_scoped_to_user,
    test_archive_is_pruned_to_cap,
    test_context_includes_past_dialogue_for_general_questions,
    test_context_omits_past_dialogue_for_schedule_questions,
]

if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"  ok  {test.__name__}")
    print(f"OK — {len(TESTS)} RAG tests")
