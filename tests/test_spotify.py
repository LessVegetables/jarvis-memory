"""
Spotify provider tests. No network and no account: spotify._api is replaced
with a fake that records the calls it was asked to make.

What is worth asserting here is not the HTTP -- it is that the command runs
BEFORE the model is asked to speak, and that every way it can fail produces a
sentence saying what actually happened. A block that quietly said "включаю"
when nothing started playing would be the worst outcome available, and it is
the easy one to write by accident.

Run:  python3 tests/test_spotify.py   (or python3 -m pytest tests/ -q)
"""

import os
import sys
import tempfile
import urllib.error
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP_DB = Path(tempfile.mkdtemp(prefix="jarvis-test-")) / "test.db"
os.environ["JARVIS_DB"] = str(_TMP_DB)
os.environ["JARVIS_LLM_INTENT"] = "0"          # no model in these tests

import jarvis_memory as memory  # noqa: E402
from jarvis_memory import db, prompts, router, seed  # noqa: E402
from jarvis_memory.providers import spotify  # noqa: E402

NOW = datetime(2026, 9, 14, 14, 30)

SEARCH = {
    "tracks": {"items": [{
        "name": "Группа крови", "uri": "spotify:track:abc",
        "artists": [{"name": "Кино"}],
    }]},
    "playlists": {"items": [{"name": "Чужой плейлист",
                             "uri": "spotify:playlist:stranger"}]},
}
OWN_PLAYLISTS = {"items": [
    {"name": "Любимое", "uri": "spotify:playlist:mine"},
    {"name": "Для бега", "uri": "spotify:playlist:run"},
]}


class FakeApi:
    """Stands in for the Web API. Records (method, path, body)."""

    def __init__(self, playlists=OWN_PLAYLISTS, search=SEARCH, error_on=None):
        self.playlists, self.search, self.error_on = playlists, search, error_on
        self.calls = []

    def __call__(self, method, path, body=None, params=None):
        self.calls.append((method, path, body))
        if self.error_on and self.error_on in path:
            raise urllib.error.HTTPError(path, 404, "Not Found", {}, None)
        if path == "/me/playlists":
            return self.playlists
        if path == "/search":
            return self.search
        return {}

    def played(self):
        return [(m, p, b) for m, p, b in self.calls if "/player/" in p]


def setup(api=None, credentials=True):
    for name, value in (("SPOTIFY_CLIENT_ID", "id"),
                        ("SPOTIFY_CLIENT_SECRET", "secret"),
                        ("SPOTIFY_REFRESH_TOKEN", "refresh")):
        if credentials:
            os.environ[name] = value
        else:
            os.environ.pop(name, None)
    fake = api or FakeApi()
    spotify._api = fake
    spotify._command.__globals__["_api"] = fake
    db.close()
    seed.seed()
    return fake


# --- the commands -------------------------------------------------------------

def test_named_track_is_played_and_confirmed():
    api = setup()
    text = spotify.block("поставь Группа крови", NOW)
    assert ("PUT", "/me/player/play", {"uris": ["spotify:track:abc"]}) in api.calls
    assert "Группа крови — Кино" in text, text
    # The block describes something already done, so the model confirms
    # rather than deciding.
    assert "подтверди" in text, text


def test_own_playlist_beats_a_stranger_with_the_same_name():
    """The whole point of a personal assistant playing music."""
    api = setup()
    text = spotify.block("включи мой любимый плейлист", NOW)
    assert ("PUT", "/me/player/play",
            {"context_uri": "spotify:playlist:mine"}) in api.calls
    assert "Любимое" in text, text
    assert "Чужой" not in text, text
    # It must not have fallen through to the public catalogue at all.
    assert not any(path == "/search" for _, path, _ in api.calls), api.calls


def test_named_own_playlist_is_matched_by_the_distinguishing_words():
    api = setup()
    spotify.block("включи мой плейлист для бега", NOW)
    assert ("PUT", "/me/player/play",
            {"context_uri": "spotify:playlist:run"}) in api.calls


def test_bare_music_request_resumes_instead_of_searching():
    """"Включи музыку" must not go looking for a band called Музыка."""
    api = setup()
    text = spotify.block("включи музыку", NOW)
    assert api.played() == [("PUT", "/me/player/play", None)], api.calls
    assert not any(p in ("/search", "/me/playlists") for _, p, _ in api.calls)
    assert text == prompts.MUSIC_RESUMED


def test_pause_next_previous():
    for phrase, expected in (("поставь на паузу", ("PUT", "/me/player/pause", None)),
                             ("следующая песня", ("POST", "/me/player/next", None)),
                             ("предыдущий трек", ("POST", "/me/player/previous", None))):
        api = setup()
        spotify.block(phrase, NOW)
        assert api.played() == [expected], (phrase, api.calls)


# --- the failures -------------------------------------------------------------

def test_no_active_device_says_to_open_the_app():
    """The likeliest demo failure, and the one a generic message would hide."""
    api = setup(FakeApi(error_on="/player/"))
    text = spotify.block("включи музыку", NOW)
    assert text == prompts.MUSIC_NO_DEVICE
    assert "Spotify не открыт" in text


def test_missing_credentials_say_so_rather_than_failing():
    setup(credentials=False)
    assert spotify.block("включи музыку", NOW) == prompts.MUSIC_NO_KEY


def test_nothing_found_is_not_reported_as_success():
    setup(FakeApi(playlists={"items": []},
                  search={"tracks": {"items": []}, "playlists": {"items": []}}))
    text = spotify.block("поставь что-то чего нет", NOW)
    assert text == prompts.MUSIC_NOT_FOUND.format(query="что-то чего нет")


def test_any_other_failure_is_admitted():
    class Boom:
        def __call__(self, *a, **kw):
            raise RuntimeError("spotify is on fire")
    setup(Boom())
    assert spotify.block("включи музыку", NOW) == prompts.MUSIC_UNAVAILABLE


# --- through build_context ----------------------------------------------------

def test_music_reaches_the_prompt_and_works_for_a_guest():
    """A shared kitchen speaker should take "поставь музыку" from anyone."""
    setup()
    ctx = memory.build_context(None, "включи мой любимый плейлист", now=NOW)
    assert ctx.intent == router.MUSIC
    assert "Любимое" in ctx.system_prompt
    assert prompts.WHO_GUEST in ctx.system_prompt


def test_a_song_about_rain_is_not_a_forecast():
    api = setup()
    ctx = memory.build_context("anton", "поставь песню про дождь", now=NOW)
    assert ctx.intent == router.MUSIC, ctx.intent
    assert "Погода" not in ctx.system_prompt


TESTS = [
    test_named_track_is_played_and_confirmed,
    test_own_playlist_beats_a_stranger_with_the_same_name,
    test_named_own_playlist_is_matched_by_the_distinguishing_words,
    test_bare_music_request_resumes_instead_of_searching,
    test_pause_next_previous,
    test_no_active_device_says_to_open_the_app,
    test_missing_credentials_say_so_rather_than_failing,
    test_nothing_found_is_not_reported_as_success,
    test_any_other_failure_is_admitted,
    test_music_reaches_the_prompt_and_works_for_a_guest,
    test_a_song_about_rain_is_not_a_forecast,
]

if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"  ok  {test.__name__}")
    print(f"OK — {len(TESTS)} spotify tests")
