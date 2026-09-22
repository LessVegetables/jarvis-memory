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

# setup() below swaps _api out for a fake and never puts it back, so the one
# test that exercises the real response parsing has to restore this.
_REAL_API = spotify._api
_REAL_REQUEST = spotify._request

# Shaped like a real answer to "ЛСП": the artist is there, and so are tracks
# by them. Which one is right depends entirely on what was asked.
# Taken from what the real API returns. The trap is "Группа Крови": there is
# an obscure band by that name AND a famous Кино song, and only popularity
# tells them apart.
SEARCH = {
    "artists": {"items": [
        {"name": "ЛСП", "uri": "spotify:artist:lsp", "popularity": 62},
        {"name": "Группа Крови", "uri": "spotify:artist:gk", "popularity": 12},
        {"name": "Кинотеатр", "uri": "spotify:artist:kinoteatr", "popularity": 30},
    ]},
    "tracks": {"items": [
        None,                                    # the API really does pad with these
        {"name": "Группа крови", "uri": "spotify:track:abc", "popularity": 68,
         "artists": [{"name": "Кино"}]},
        {"name": "ЛСП", "uri": "spotify:track:lsptrack", "popularity": 4,
         "artists": [{"name": "Кто-то"}]},
    ]},
    "albums": {"items": [{"name": "Свежая кровь", "uri": "spotify:album:blood"}]},
    "playlists": {"items": [{"name": "Чужой плейлист",
                             "uri": "spotify:playlist:stranger"}]},
}
# Deliberately named in English, like a real library: a Russian request for
# "мой любимый плейлист" matches none of them by name, which is exactly the
# case that used to fall through to the public catalogue.
OWN_PLAYLISTS = {"items": [
    {"name": "Night driving", "uri": "spotify:playlist:night"},
    {"name": "Basketball Mix", "uri": "spotify:playlist:ball"},
    {"name": "Для бега", "uri": "spotify:playlist:run"},
]}
LIKED = {"items": [
    {"track": {"name": "Группа крови", "uri": "spotify:track:liked1"}},
    {"track": {"name": "Пачка сигарет", "uri": "spotify:track:liked2"}},
]}


class FakeApi:
    """Stands in for the Web API. Records (method, path, body-or-params).

    Two entry points, because the module has two: _api for calls whose JSON
    matters, and _request for player commands, which do not read the body at
    all. A fake that only covered _api let the command path reach the real
    network.
    """

    def __init__(self, playlists=OWN_PLAYLISTS, search=SEARCH, error_on=None,
                 liked=LIKED, volume=40, playing=None):
        self.playlists, self.search, self.error_on = playlists, search, error_on
        self.liked, self.volume = liked, volume
        self.supports_volume = True
        self.playing = {"item": {"name": "Группа крови",
                                 "artists": [{"name": "Кино"}]}} if playing is None \
            else playing
        self.calls = []

    def __call__(self, method, path, body=None, params=None):
        self.calls.append((method, path, body if params is None else params))
        if self.error_on and self.error_on in path:
            raise urllib.error.HTTPError(path, 404, "Not Found", {}, None)
        if path == "/me/playlists":
            return self.playlists
        if path == "/me/tracks":
            return self.liked
        if path == "/search":
            return self.search
        if path == "/me/player":
            return {"device": {"volume_percent": self.volume,
                               "supports_volume": self.supports_volume}}
        if path == "/me/player/currently-playing":
            return self.playing
        return {}

    def request(self, method, path, body=None, params=None):
        """_request: records the call and returns an empty 204, as Spotify does."""
        self(method, path, body, params)
        return 204, b""

    def played(self):
        """Only the calls that CHANGE something.

        Reads go to /me/player/... too -- currently-playing, the device's
        volume -- so filtering on the path alone counts a question as a
        command.
        """
        return [(m, p, b) for m, p, b in self.calls
                if m != "GET" and "/player/" in p]


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
    spotify._request = fake.request
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


def test_my_favourites_means_liked_songs():
    """"Мой любимый плейлист" is not a playlist -- it is the saved library.

    The failure this replaces: a library whose playlists are all named in
    English matches nothing, and the request falls through to /v1/search,
    which answers with a stranger's track called "мой любимый плейлист".
    """
    api = setup()
    text = spotify.block("включи мой любимый плейлист", NOW)
    assert ("PUT", "/me/player/play",
            {"uris": ["spotify:track:liked1", "spotify:track:liked2"]}) in api.calls
    assert "Любимые треки" in text, text
    # It must not have reached the public catalogue at all.
    assert not any(path == "/search" for _, path, _ in api.calls), api.calls


def test_bare_my_music_also_means_liked_songs():
    api = setup()
    spotify.block("поставь мою музыку", NOW)
    assert ("PUT", "/me/player/play",
            {"uris": ["spotify:track:liked1", "spotify:track:liked2"]}) in api.calls


def test_empty_library_falls_back_to_a_playlist():
    """An empty Liked Songs is not an error, just nothing to play from."""
    api = setup(FakeApi(liked={"items": []}))
    spotify.block("включи мой любимый плейлист", NOW)
    assert ("PUT", "/me/player/play",
            {"context_uri": "spotify:playlist:night"}) in api.calls


def test_a_named_playlist_still_wins_over_the_catalogue():
    api = setup()
    text = spotify.block("включи мой плейлист для бега", NOW)
    assert ("PUT", "/me/player/play",
            {"context_uri": "spotify:playlist:run"}) in api.calls
    assert "Чужой" not in text, text
    assert not any(path == "/search" for _, path, _ in api.calls), api.calls


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


def test_an_english_playlist_named_in_russian():
    """The normal case for a Russian speaker with an English library.

    "Найт драйвинг" and "Night driving" share not one character, so stems
    cannot bridge it and the request used to fall through to the public
    catalogue.
    """
    for phrase, uri in (("включи найт драйвинг", "spotify:playlist:night"),
                        ("поставь баскетбол", "spotify:playlist:ball"),
                        ("включи night driving", "spotify:playlist:night")):
        api = setup()
        spotify.block(phrase, NOW)
        assert ("PUT", "/me/player/play", {"context_uri": uri}) in api.calls, phrase
        assert not any(p == "/search" for _, p, _ in api.calls), phrase


def test_sounding_like_nothing_still_reaches_the_catalogue():
    """The cutoff has to let genuinely unknown things through."""
    api = setup()
    spotify.block("включи This is America", NOW)
    assert any(p == "/search" for _, p, _ in api.calls), api.calls


def test_an_artist_is_played_as_an_artist():
    """"Включи ЛСП" is a request for the act, not for one song of theirs.

    Searching tracks only -- which is what this did at first -- answers it
    with whichever song ranked first and never plays the artist at all.
    """
    api = setup()
    text = spotify.block("включи ЛСП", NOW)
    assert ("PUT", "/me/player/play",
            {"context_uri": "spotify:artist:lsp"}) in api.calls
    assert "ЛСП" in text, text


def test_a_famous_song_beats_an_obscure_band_of_the_same_name():
    """Found against the real catalogue: there IS a band called "Группа
    Крови", and preferring the artist outright played their top track
    instead of the Кино song everyone means."""
    api = setup()
    spotify.block("включи Группа крови", NOW)
    assert ("PUT", "/me/player/play",
            {"uris": ["spotify:track:abc"]}) in api.calls, api.calls


def test_a_famous_band_beats_an_obscure_song_of_the_same_name():
    """The mirror case, which is why this is popularity and not a rule that
    tracks always win."""
    api = setup()
    spotify.block("включи ЛСП", NOW)
    assert ("PUT", "/me/player/play",
            {"context_uri": "spotify:artist:lsp"}) in api.calls, api.calls


def test_an_artist_must_match_exactly():
    """"Кино" is a prefix of "Кинотеатр". A loose match here would replace
    the song someone asked for with a different act's back catalogue."""
    api = setup()
    spotify.block("включи кинотеатр повторного фильма", NOW)
    played = api.played()[0]
    assert played[2] != {"context_uri": "spotify:artist:kinoteatr"}, played


def test_album_is_used_when_there_is_no_track():
    api = setup(FakeApi(search={"artists": {"items": []},
                                "tracks": {"items": []},
                                "albums": SEARCH["albums"],
                                "playlists": SEARCH["playlists"]}))
    text = spotify.block("включи свежая кровь", NOW)
    assert ("PUT", "/me/player/play",
            {"context_uri": "spotify:album:blood"}) in api.calls
    assert "Свежая кровь" in text, text


def test_quieter_turns_it_down_rather_than_off():
    """"Тише" used to route to pause, which silenced the music entirely --
    the one thing the word cannot mean."""
    api = setup(FakeApi(volume=40))
    text = spotify.block("сделай потише", NOW)
    assert ("PUT", "/me/player/volume", {"volume_percent": 20}) in api.calls
    assert not any("/pause" in path for _, path, _ in api.calls), api.calls
    assert text == prompts.MUSIC_QUIETER_BLOCK


def test_louder_is_relative_to_where_it_is_and_clamps():
    for start, expected in ((40, 60), (90, 100), (0, 20)):
        api = setup(FakeApi(volume=start))
        spotify.block("погромче", NOW)
        assert ("PUT", "/me/player/volume",
                {"volume_percent": expected}) in api.calls, start


def test_volume_without_a_device_says_to_open_the_app():
    api = setup(FakeApi(volume=None))
    assert spotify.block("погромче", NOW) == prompts.MUSIC_NO_DEVICE


def test_what_is_playing():
    api = setup()
    text = spotify.block("что сейчас играет", NOW)
    assert "Группа крови — Кино" in text, text
    # A question, so nothing should have been commanded.
    assert not api.played(), api.calls


def test_nothing_playing_is_not_a_failure():
    api = setup(FakeApi(playing={}))
    assert spotify.block("что сейчас играет", NOW) == prompts.MUSIC_NOTHING_PLAYING


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

        def request(self, *a, **kw):
            raise RuntimeError("spotify is on fire")
    setup(Boom())
    assert spotify.block("включи музыку", NOW) == prompts.MUSIC_UNAVAILABLE


def test_a_device_that_refuses_volume_says_so():
    """Phones and web players answer 403 to a volume change. Saying "не
    получается" would be true and useless; this says what to do instead."""
    api = setup(FakeApi(volume=40))
    api.supports_volume = False
    assert spotify.block("погромче", NOW) == prompts.MUSIC_VOLUME_UNSUPPORTED


# --- the HTTP layer itself ----------------------------------------------------
# FakeApi above replaces _api wholesale, so nothing so far exercises the
# response parsing -- which is exactly where the bug was that reported
# failure for commands Spotify had already carried out.

class FakeResponse:
    def __init__(self, status, body):
        self.status, self._body = status, body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_empty_responses_are_success_not_a_parse_error():
    """pause and next answer 204, sometimes with a newline for a body.

    Reported as a failure, this is the worst outcome available: the music
    stopped and the assistant said it could not stop it.
    """
    import urllib.request
    spotify._api, spotify._request = _REAL_API, _REAL_REQUEST
    spotify._token = ("tok", datetime(2099, 1, 1))
    original = urllib.request.urlopen
    try:
        for status, body in ((204, b""), (200, b""), (200, b"\n"), (200, b"   ")):
            urllib.request.urlopen = lambda *a, **kw: FakeResponse(status, body)
            assert spotify._api("PUT", "/me/player/pause") == {}, (status, body)

        urllib.request.urlopen = lambda *a, **kw: FakeResponse(
            200, b'{"ok": true}')
        assert spotify._api("GET", "/me/player") == {"ok": True}
    finally:
        urllib.request.urlopen = original
        spotify._token = None


# --- through build_context ----------------------------------------------------

def test_music_reaches_the_prompt_and_works_for_a_guest():
    """A shared kitchen speaker should take "поставь музыку" from anyone."""
    setup()
    ctx = memory.build_context(None, "включи мой любимый плейлист", now=NOW)
    assert ctx.intent == router.MUSIC
    assert "Любимые треки" in ctx.system_prompt
    assert prompts.WHO_GUEST in ctx.system_prompt


def test_a_song_about_rain_is_not_a_forecast():
    api = setup()
    ctx = memory.build_context("anton", "поставь песню про дождь", now=NOW)
    assert ctx.intent == router.MUSIC, ctx.intent
    assert "Погода" not in ctx.system_prompt


TESTS = [
    test_named_track_is_played_and_confirmed,
    test_my_favourites_means_liked_songs,
    test_bare_my_music_also_means_liked_songs,
    test_empty_library_falls_back_to_a_playlist,
    test_a_named_playlist_still_wins_over_the_catalogue,
    test_an_english_playlist_named_in_russian,
    test_sounding_like_nothing_still_reaches_the_catalogue,
    test_an_artist_is_played_as_an_artist,
    test_a_famous_song_beats_an_obscure_band_of_the_same_name,
    test_a_famous_band_beats_an_obscure_song_of_the_same_name,
    test_an_artist_must_match_exactly,
    test_album_is_used_when_there_is_no_track,
    test_quieter_turns_it_down_rather_than_off,
    test_louder_is_relative_to_where_it_is_and_clamps,
    test_volume_without_a_device_says_to_open_the_app,
    test_what_is_playing,
    test_nothing_playing_is_not_a_failure,
    test_bare_music_request_resumes_instead_of_searching,
    test_pause_next_previous,
    test_no_active_device_says_to_open_the_app,
    test_a_device_that_refuses_volume_says_so,
    test_empty_responses_are_success_not_a_parse_error,
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
