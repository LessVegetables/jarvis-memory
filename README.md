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
| Fact lookup (RAG) | **Stub**: returns 3 most recent facts, ignores the question — step 5 |
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
| `facts` | one short fact per row, for step 5 to retrieve by meaning |

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
- Step 5 does not change these tables. Embeddings go into a separate
  `sqlite-vec` virtual table keyed by `facts.id`.
- Voice embeddings are not in the schema yet: their dimension depends on
  which extractor task B picks (ECAPA-TDNN is 192, WeSpeaker models vary).
  Ask B before adding that table.

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

## Language

Code, comments and documentation are in English. Prompt templates, seed data
and router patterns stay in Russian: those are model-facing content, not
commentary.

## Running it

```
python3 demo.py            # prints real prompts for every intent
python3 tests/test_router.py   # or: python3 -m pytest tests/ -q
```

Prints the prompts for several scenarios, including two different users
asking the same question, and a multi-turn dialogue. Seeds the database on
first run. No external dependencies.
