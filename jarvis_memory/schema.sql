-- Jarvis memory: relational schema (step 2).
--
-- One file, no server: everything lives in jarvis.db next to the code.
-- Step 5 adds a vector table for RAG over `facts`; the tables below will
-- not change when that happens -- the embeddings go in a separate virtual
-- table keyed by facts.id.

CREATE TABLE IF NOT EXISTS users (
    user_id   TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    age       INTEGER
);

-- starts_at is TEXT in ISO 8601 ('2026-09-21 16:00'). SQLite has no date
-- type; ISO strings are the standard workaround because they sort and
-- compare correctly as plain text, and date() understands them.
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY,
    user_id   TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    starts_at TEXT NOT NULL,
    title     TEXT NOT NULL,
    location  TEXT
);

CREATE INDEX IF NOT EXISTS events_by_user_day ON events(user_id, starts_at);

-- One fact per row, one short sentence each. Deliberately not a "profile
-- blob": step 5 retrieves these individually by meaning, and a paragraph
-- cannot be retrieved in pieces.
CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS facts_by_user ON facts(user_id);

-- Voice embeddings (task C also owns these, per the spec) are NOT here yet:
-- their dimension depends on which extractor module B settles on
-- (ECAPA-TDNN is 192, WeSpeaker models vary). Ask B before adding a table.

-- Every exchange, kept so the assistant can recall things said weeks ago.
-- Capped per user (see store.MAX_DIALOGUE_ROWS): on a 4 GB / 15 GB device an
-- unbounded table is a slow-motion outage. Vectors live in vec_dialogue,
-- created in db.py, with created_at duplicated there as an integer so the
-- vector search itself can exclude turns that are still in the live buffer.
CREATE TABLE IF NOT EXISTS dialogue (
    id         INTEGER PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS dialogue_by_user_time ON dialogue(user_id, created_at);

-- Responses from live-data APIs (weather, 2GIS), keyed by provider+query.
-- Served fresh within each provider's TTL, and served stale if the next
-- fetch fails: old data beats silence, as long as the model is told.
CREATE TABLE IF NOT EXISTS api_cache (
    key        TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
