# jarvis-memory — memory and context module (task C)

Turns **(who asked, what they asked)** into context for the LLM: a system
prompt plus the last few turns of dialogue.

This module does **not** call a language model and does **not** touch audio.
Inside it is only data lookups and string substitution into a template —
ordinary code you can step through line by line in a debugger.

## Interface for the orchestrator (app.py)

```python
import jarvis_memory as memory

ctx = memory.build_context(user_id, transcript)   # user_id=None if the voice was not recognised
answer = llm.generate(ctx.to_messages())
memory.record_answer(user_id, transcript, answer)
```

**The second call is required.** Without it neither "repeat" nor multi-turn
dialogue works: the module never learns what was said back.

### `build_context(user_id, transcript, now=None) -> PromptContext`

| Field | Type | What it is |
|---|---|---|
| `.system_prompt` | `str` | System prompt: rules plus this user's data |
| `.history` | `list[dict]` | Recent turns: `{"role": "user"/"assistant", "content": ...}` |
| `.transcript` | `str` | The current question |
| `.intent` | `str` | What the router matched: `schedule` / `weather` / `places` / `repeat` / `remember` / `general` |
| `.to_messages()` | `list[dict]` | **Usually the only thing you need** — system + history + question |
| `.estimate_tokens()` | `int` | Rough estimate of the context budget used |

### Why it returns an object rather than a string

The draft contract in the spec said `{user_id, transcript} -> system_prompt`.
Dialogue history does not fit into that. Qwen2.5-Instruct is trained on ChatML
dialogue markup, and it understands past turns passed as separate `user` /
`assistant` messages noticeably better than the same text glued into the
system prompt as a paragraph. So the module returns a ready-made message list
instead. If the orchestrator really does want a single string, it is still
there as `.system_prompt` — nothing is lost.

## What works and what is a stub

| Part | Status |
|---|---|
| Interface, prompt templates, assembly | Done |
| Dialogue history, "repeat" | Done |
| Access control for an unrecognised speaker | Done |
| User data in SQLite | Done |
| Intent router, follow-ups, date resolution | Done |
| "Повтори" and "запомни, что…" | Done |
| Fact lookup (RAG) | Done — semantic search, falls back to most-recent |
| Dialogue archive and recall | Done — capped per user, semantic only |
| Weather, 2GIS | Not started — step 6 |

The function signatures in `store.py` do not change — only their bodies do.
Step 2 swapped hardcoded dicts for SQL queries without touching a single
caller, and step 5 will swap the fact lookup for vector search the same way.

## Database

A single SQLite file, `jarvis.db`, no server process. Schema is in
[`jarvis_memory/schema.sql`](jarvis_memory/schema.sql):

| Table | Holds |
|---|---|
| `users` | `user_id`, name, age |
| `events` | one row per calendar event: `starts_at` (ISO 8601 text), title, location |
| `facts` | one short fact per row, retrieved by meaning |
| `vec_facts` | fact embeddings (virtual table, sqlite-vec) |
| `dialogue` | every exchange with a recognised speaker, capped at 500 per user |
| `vec_dialogue` | exchange embeddings, with `created_at` for excluding recent turns |

Fill it with development data:

```
python3 -m jarvis_memory.seed
```

Re-running wipes and refills. Events are generated **relative to today**,
covering the current week and the next one — seeded with fixed dates they
would go stale and the demo would quietly stop demonstrating anything.

`jarvis.db` is gitignored. It is generated, not source.

Three notes for whoever touches the schema next:

- SQLite has no date type. `starts_at` is ISO 8601 text (`2026-09-21 16:00`),
  which sorts and compares correctly as plain text and works with `date()`.
- `vec_facts` and `vec_dialogue` are virtual tables, so no foreign key
  reaches them. Anything that deletes rows must clear them too, or searches
  join against dead ids.
- Changing the embedding model or its pooling invalidates every vector.
  Run `python3 tools/rebuild_vectors.py` afterwards; mixed vectors give
  silent nonsense, not errors.
- Voice embeddings are not in the schema yet: their dimension depends on
  which extractor task B picks (ECAPA-TDNN is 192, WeSpeaker models vary).
  Ask B before adding that table.

## Semantic search

Fact lookup embeds the question and compares it against stored fact vectors,
so «можно мне печенье с миндалём?» can retrieve «аллергия на орехи» despite
sharing no words with it.

Setup is two steps, because the export tooling must not go on the board:

```bash
# On a LAPTOP (installs torch, ~2-3 GB):
python3 tools/export_embedding_model.py     # writes models/

# Copy models/ to the board, then there:
pip install sqlite-vec onnxruntime tokenizers numpy
python3 -m jarvis_memory.seed               # builds the vectors
python3 tools/ram_spike.py                  # confirms it fits in RAM
```

Model: `cointegrated/rubert-tiny2`, 312-dim, ~29M parameters, **CLS pooling**
(its `1_Pooling/config.json` on HuggingFace says so — mean pooling gives a
self-consistent but worse space). Measured on the board: 28 MB on disk,
+65 MB resident, ~4 ms per sentence. Chosen over the more accurate
`multilingual-e5-small` (~470 MB fp32) because of the 4 GB budget shared
with a 1.7 GB LLM; with that much headroom left, it could be revisited if
retrieval quality disappoints.

**Without the model everything still runs** — `get_facts` falls back to the
most recent facts and logs why. `demo.py` prints which path is active. This
matters so teammates can work without copying a model around.

Search is scoped per user by a sqlite-vec *partition key*, not by filtering
afterwards: the nearest k rows globally could easily be someone else's, which
would leak one housemate's facts into another's answer.

## Recalling past conversations

Every exchange with a recognised speaker is archived (`record_answer` does
this; guests are not archived — their conversation is nobody's memory). For
open-ended questions, the two most relevant past exchanges are put in the
context, so «как там Бобик?» can surface a vet visit discussed days ago.

Three deliberate limits:

- **Only for `general` questions.** A schedule or weather question gains
  nothing from old chatter, and a small model will answer the old question
  instead of the new one if both are in view.
- **Turns still in the live buffer are excluded** — they are already in the
  messages. The exclusion happens inside the vector search (`created_at` is
  a sqlite-vec metadata column), not by filtering afterwards.
- **No non-semantic fallback**, unlike facts. Random old chatter is worse
  than nothing. Without the model, the assistant simply does not recall.

The archive is capped at `store.MAX_DIALOGUE_ROWS = 500` per user, oldest
deleted first. On a 4 GB / 15 GB device an unbounded table is a slow-motion
outage.

## Context budget

A local Qwen2.5-1.5B has a context of 2048 or 4096 tokens. That number is
fixed when the model is converted to `.rkllm` and cannot be changed
afterwards. **Ask task G which one it actually is** — how many turns of
history we can afford depends on it.

Russian text runs about 2.5 characters per token. A typical request currently
costs ~200 tokens, of which history is ~50. The knobs are in `history.py`:
`MAX_TURNS = 3` (exchanges) and `TTL = 5 minutes`.

## Intents

| Intent | Example | What the context gets |
|---|---|---|
| `schedule` | «что у меня завтра?» | that day's events, with the day named |
| `weather` | «зонт нужен?» | step 8 |
| `places` | «до скольки работает аптека» | step 8 |
| `repeat` | «повтори, я не расслышал» | the previous answer, verbatim |
| `remember` | «запомни, что я не ем острое» | writes a fact, asks for confirmation |
| `general` | «что приготовить на ужин?» | facts about the user |

A short follow-up («а завтра?») keeps the previous turn's intent, so it stays
on topic instead of falling through to `general`. An explicit match always
wins over stickiness: «а какая погода?» after a schedule question is a
weather question.

`tests/test_router.py` is the specification of what the assistant understands.
When someone reports "it didn't get me", add a row there first, then a
pattern in `router.py`.

## Known gaps

- History for unrecognised speakers is shared between all of them.
- Only relative days are understood (сегодня / завтра / послезавтра /
  вчера). «в пятницу» and «на выходных» are not.
- Fact extraction is a prefix strip, so «запомни, что мне не нравится X»
  is stored in first person as said, not normalised to the user's name.
- Recalled dialogue is matched on the exchange as a whole; a very long
  answer dilutes the vector. Answers are meant to be short, so this is
  tolerated rather than solved.

## Language

Code, comments and documentation are in English. Prompt templates, seed data
and router patterns stay in Russian: those are model-facing content, not
commentary.

## Running it

```
python3 demo.py                # prints real prompts for every intent
python3 tests/test_router.py   # routing and fact extraction
python3 tests/test_rag.py      # retrieval plumbing (uses a stand-in embedder)
# or, if you have it:  python3 -m pytest tests/ -q
```

Prints the prompts for several scenarios, including two different users
asking the same question, and a multi-turn dialogue. Seeds the database on
first run. No external dependencies.
