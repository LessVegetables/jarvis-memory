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
| `.intent` | `str` | What the router matched: `schedule` / `weather` / `repeat` / `places` / `general` |
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
| User data | **Hardcoded** — step 2: SQLite |
| Intent router | **Stub**, four regexes — step 4 |
| Fact lookup (RAG) | **Stub**: returns the first 3 facts — step 5 |
| Weather, 2GIS | Not started — step 6 |

The function signatures in `store.py` will not change — only their bodies
will. Code written against them today keeps working after the move to a real
database.

## Context budget

A local Qwen2.5-1.5B has a context of 2048 or 4096 tokens. That number is
fixed when the model is converted to `.rkllm` and cannot be changed
afterwards. **Ask task G which one it actually is** — how many turns of
history we can afford depends on it.

Russian text runs about 2.5 characters per token. A typical request currently
costs ~200 tokens, of which history is ~50. The knobs are in `history.py`:
`MAX_TURNS = 3` (exchanges) and `TTL = 5 minutes`.

## Known gaps

- A short follow-up question (`"а тренировка?"`) routes to `general` rather
  than `schedule`: the keyword router cannot see that the previous turn was
  about the schedule. Fix belongs in step 4 — carry the previous intent
  forward when a short phrase matches nothing.
- History for unrecognised speakers is shared between all of them.

## Language

Code, comments and documentation are in English. Prompt templates, seed data
and router patterns stay in Russian: those are model-facing content, not
commentary.

## Running it

```
python3 demo.py
```

Prints the prompts for several scenarios, including two different users
asking the same question, and a multi-turn dialogue. No external dependencies.
