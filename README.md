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
| `.intent` | `str` | `schedule` / `weather` / `places` / `music` / `repeat` / `remember` / `general` |
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
| Weather (weatherapi.com), places (2GIS) | Done — cached, degrade to a sentence on failure |
| Calendar sync: Google (ICS), iCloud (CalDAV) | Done — on a timer, into `events` |

The function signatures in `store.py` do not change — only their bodies do.
Step 2 swapped hardcoded dicts for SQL queries without touching a single
caller, and step 5 will swap the fact lookup for vector search the same way.

## Database

A single SQLite file, `jarvis.db`, no server process. Schema is in
[`jarvis_memory/schema.sql`](jarvis_memory/schema.sql):

| Table | Holds |
|---|---|
| `users` | `user_id`, name, age |
| `events` | one row per calendar event: `starts_at` (ISO 8601 text), title, location, `source` (`seed` or a calendar id), `all_day` |
| `facts` | one short fact per row, retrieved by meaning. `polarity` + `subject` mark a preference the code enforces rather than merely recites |
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

## Users

`seed.py` creates two fake demo users, `anton` and `masha`. Real people are
added with:

```bash
python3 tools/add_user.py daniel "Даниил" 21
```

The `user_id` must be **exactly the id module B emits** for that voice —
it is the key everything else hangs off. Nothing creates users implicitly:
a calendar or a fact for an unknown id is logged and skipped, because a
mismatch with speaker identification is worth noticing, not hiding.

## Calendar sync

Google and iCloud calendars are pulled into the `events` table on a timer.
**The request path never talks to a calendar server** — `get_schedule` is
the same SQL lookup it was in step 2, fast, and working when the wifi is
not. Sources go in a gitignored `calendars.json` (see the example file):

| Provider | How | What you need |
|---|---|---|
| Google | its *secret address in iCal format* — a plain HTTPS GET | Calendar settings → Integrate calendar → copy the private `.ics` URL |
| iCloud | CalDAV via the `caldav` library | an **app-specific password** from appleid.apple.com — the account password will not work with two-factor on |

Google's CalDAV endpoint wants OAuth, which costs a day of setup for
nothing the ICS address does not already give for reading. So Google is
ICS, iCloud is CalDAV, and both produce the same `Event` rows.

```bash
python3 tools/sync_calendars.py            # once, by hand
*/15 * * * * cd ~/jarvis-memory && python3 tools/sync_calendars.py >> sync.log 2>&1
```

Each sync **replaces** the two-week window for its own `(user, source)`:
deleted and moved events disappear, hand-seeded rows (`source='seed'`)
are never touched, and one calendar failing does not stop the others.
Recurring events are expanded client-side for both providers. Times are
converted to the device's local zone, so set it:
`sudo timedatectl set-timezone Europe/Moscow` (or wherever the kitchen is).

## Live data: weather and places

Keys go in a gitignored `.env` (see `.env.example`), along with the device's
coordinates — the kitchen does not move, so location is configuration, not
something the request carries.

```
WEATHERAPI_KEY=...     # weatherapi.com, free tier
DGIS_KEY=...           # 2GIS Catalog API
JARVIS_LAT=55.75  JARVIS_LON=37.62
```

Anyone may ask, recognised or not — a guest gets the weather and is told
they are being answered as a guest. Personal data still needs a profile.

Every response is cached in SQLite (`api_cache`): weather for 15 minutes,
places for a day. If a fetch fails and something is cached, the stale copy
is served with a note saying so. Old data beats silence in a demo room with
bad wifi, as long as the model is told it may be old. Every failure mode
has its own sentence, so the model reports what happened instead of
inventing a forecast: no key, no network, nothing cached, nothing found.

HTTP timeout is 3 seconds (`JARVIS_HTTP_TIMEOUT`). The person is already
waiting through STT + LLM + TTS.

One 2GIS quirk: its `point` parameter is `lon,lat`, longitude first. There
is a test pinning that.

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
| `weather` | «зонт нужен?» | today's/tomorrow's forecast from weatherapi.com |
| `places` | «до скольки работает аптека» | **one** business from 2GIS, with only the fields that answer the question |
| `music` | «включи мой любимый плейлист» | starts playback, then asks for confirmation |
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

### Places: one answer shape per question

A places question is routed twice — to `places`, and then by
`router.place_shape()` to one of eight shapes: `open_now`, `nearest`,
`opens_earliest`, `hours`, `address`, `rating`, `list`, `distance`. The shape
decides which business ends up in the prompt and which of its fields come
with it.

This is not tidiness. The block used to list four businesses with four fields
each, and the model answered with the name of one, the address of a second and
the distance of a third — it was reading across the rows of a table nobody had
told it was a table. Every field that does not answer the question is one more
thing that can end up attached to the wrong name, so `list` carries names and
nothing else, and every other shape names exactly one business.

The same reasoning produced the `open_now` "everything is shut" branch: asked
at 02:19 which pharmacy was open, the old block answered with one that had
closed at 22:00, because nothing in it ever compared the clock against the
hours. `places.py` does that comparison now; the model is never asked to.

### The one language-model call

`router.route()` runs first. Only when it returns `general` — nothing matched,
not a topic, not a follow-up — does `llm_intent.classify()` get a turn, and on
that path the alternative is not a faster right answer but no answer at all.
So canonical phrasings cost nothing extra, every row of the router's table
still passes by construction, and a timeout, a dead server or an unparseable
reply all fall back to the `general` the caller would have used anyway.

On the intent fixture, coarse intents: **88.5% from the regexes alone**, with
canonical, confusable and off-domain rows at 100%. About one turn in eight
reaches the model.

`llm_intent.extract_subject()` is the only other call, and only when someone
says they dislike something. Both are extraction, not reasoning: the answer is
already in the input and the model has to find it, not work it out. Set
`JARVIS_LLM_INTENT=0` to switch both off — worth knowing during a shared
demo, since it reverts this module to pure regexes without touching code.

### Preferences that change behaviour

«Запомни, я терпеть не могу аптеку Экона» stores the sentence *and* the
subject, and `places.py` drops matching results before it selects anything —
so the place cannot come back as "the nearest" either. Reciting a preference
is not honouring one, and only the `subject` column makes the assistant act.

The extracted name is checked against what was said before it is believed: an
extraction that invents a word has not extracted anything, and acting on one
would silently hide a business nobody objected to. If it fails, the fact is
still stored and still recited — it just does not filter.

Matching is on five-character stems, because what was said is inflected and
what 2GIS returns is not: «аптеку Экона» shares no substring with «Аптека
Экона».

## Known gaps

- History for unrecognised speakers is shared between all of them.
- Only relative days are understood (сегодня / завтра / послезавтра /
  вчера). «в пятницу» and «на выходных» are not.
- Calendar events are matched to a day by their local start time, so an
  event starting at 01:00 Moscow time on a board set to UTC lands on the
  previous day. Set the board's timezone.
- Fact extraction is a prefix strip, so «запомни, что мне не нравится X»
  is stored in first person as said, not normalised to the user's name.
- Dislikes filter by name only. «Не предлагай мне аптеки» would be stored
  with `аптеки` as the subject and would then filter every pharmacy, which
  is what was asked but probably not what was meant.
- `opens_earliest` ignores businesses open around the clock: they never
  "open", so at 02:19 «какая аптека откроется раньше всех» names one that
  opens in the morning rather than the one already open. «Какая аптека
  сейчас работает» is the question that finds it.
- Spotify needs the app already open on some device; the Web API will not
  start playback otherwise. This has its own spoken sentence rather than the
  generic failure one, because it is the likeliest thing to go wrong.
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
python3 tests/test_providers.py   # weather and places, with canned API payloads
python3 tests/test_llm_intent.py  # the fallback classifier, with a fake model
python3 tests/test_spotify.py     # playback commands, with a fake Web API
python3 tests/test_preferences.py # a dislike stops a place being offered
python3 tests/test_calendar.py    # ICS parsing, sync, migration (needs icalendar + recurring_ical_events)
# or, if you have it:  python3 -m pytest tests/ -q
```

Prints the prompts for several scenarios, including two different users
asking the same question, and a multi-turn dialogue. Seeds the database on
first run. No external dependencies.
