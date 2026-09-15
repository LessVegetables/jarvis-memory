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
