"""
Playback control via the Spotify Web API.

This is the module's first provider that *does* something rather than
reporting something, and it follows the shape REMEMBER already established in
context.py: perform the side effect, then hand the model a block saying it has
already happened. The model confirms out loud; it never decides whether to
play, what to play, or whether it worked.

That keeps the change inside module Ц entirely. The orchestrator's contract is
still build_context(user_id, transcript) -> ctx, so nothing in app.py, the
audio module or the LLM module has to know that music exists.

--- what it plays ------------------------------------------------------------

The speaker's own playlists are searched before the public catalogue. "Включи
мой любимый плейлист" is the request this project exists to demonstrate, and
/v1/search would answer it with a stranger's playlist that happens to be
called that. A personal assistant that plays someone else's music is not the
demo.

--- auth ---------------------------------------------------------------------

The board holds a refresh token and nothing else. tools/spotify_auth.py is run
once, on a laptop, to obtain it; the board only ever exchanges it for an
access token. There is no callback server and no browser on the device.

--- the failure everyone hits ------------------------------------------------

Spotify will not start playback on an account with no active device: the app
has to be open somewhere first. It answers 404 NOT_FOUND for that, which reads
like "no such endpoint" and is really "nothing to play on". It has its own
sentence in prompts.py because it is by far the likeliest thing to go wrong
during a demo, and "сервис не отвечает" would send someone debugging the
network instead of unlocking their phone.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

from .. import config, prompts, router

log = logging.getLogger(__name__)

TOKEN_URL = "https://accounts.spotify.com/api/token"
API = "https://api.spotify.com/v1"

# Refreshed a minute early rather than on the second: the access token is
# checked and used in two separate requests, and expiring between them would
# fail a playback command for no reason anyone could reproduce.
_EXPIRY_MARGIN = timedelta(seconds=60)
_token: tuple[str, datetime] | None = None


class NoActiveDevice(Exception):
    """Spotify has nowhere to play. Not an outage -- open the app."""


def block(transcript: str, now: datetime) -> str:
    """Do it, then describe what was done. Never raises."""
    try:
        return _block(transcript, now)
    except NoActiveDevice:
        # Caught ahead of the generic handler on purpose: "сервис не отвечает"
        # would send someone debugging the network when the real fix is to
        # open Spotify on a phone.
        log.info("spotify: no active device")
        return prompts.MUSIC_NO_DEVICE
    except Exception:                                          # noqa: BLE001
        log.exception("spotify block failed")
        return prompts.MUSIC_UNAVAILABLE


def _block(transcript: str, now: datetime) -> str:
    if not _credentials():
        return prompts.MUSIC_NO_KEY

    action = router.music_action(transcript)

    if action == router.MUSIC_PAUSE:
        _command("PUT", "/me/player/pause")
        return prompts.MUSIC_PAUSED
    if action == router.MUSIC_NEXT:
        _command("POST", "/me/player/next")
        return prompts.MUSIC_NEXT_BLOCK
    if action == router.MUSIC_PREVIOUS:
        _command("POST", "/me/player/previous")
        return prompts.MUSIC_PREVIOUS_BLOCK

    query = router.music_query(transcript)
    if not query:
        # "Включи музыку" with nothing more: resume whatever was last on.
        _command("PUT", "/me/player/play")
        return prompts.MUSIC_RESUMED

    found = _find(query)
    if found is None:
        return prompts.MUSIC_NOT_FOUND.format(query=query)
    name, body = found
    _command("PUT", "/me/player/play", body)
    return prompts.MUSIC_PLAYING.format(name=name)


# --- finding something to play ------------------------------------------------

def _find(query: str) -> tuple[str, dict] | None:
    """(spoken name, play-request body) for the best match, or None."""
    own = _own_playlist(query)
    if own is not None:
        return own

    results = _api("GET", "/search", params={
        "q": query, "type": "track,playlist", "limit": 5, "market": "from_token",
    })

    tracks = ((results.get("tracks") or {}).get("items")) or []
    for track in tracks:
        if track and track.get("uri"):
            artists = ", ".join(a["name"] for a in track.get("artists") or []
                                if a.get("name"))
            name = f"{track['name']} — {artists}" if artists else track["name"]
            return name, {"uris": [track["uri"]]}

    playlists = ((results.get("playlists") or {}).get("items")) or []
    for playlist in playlists:
        if playlist and playlist.get("uri"):
            return playlist["name"], {"context_uri": playlist["uri"]}
    return None


def _own_playlist(query: str) -> tuple[str, dict] | None:
    """The speaker's own playlist whose name best matches, if any.

    Checked before /v1/search so that "мой любимый плейлист" plays theirs and
    not a stranger's with the same title.
    """
    items = (_api("GET", "/me/playlists", params={"limit": 50}).get("items")) or []
    if not items:
        return None

    wanted = _strip_possessive(router.normalise(query))
    if not wanted:
        # "Мой плейлист", with nothing to tell one from another. Spotify
        # returns them in the order the owner keeps them, so the first is the
        # closest thing to "mine" that the request actually specified.
        first = items[0]
        return first["name"], {"context_uri": first["uri"]}

    for playlist in items:
        if _name_matches(wanted, router.normalise(playlist.get("name") or "")):
            return playlist["name"], {"context_uri": playlist["uri"]}
    return None


# Possessives and the word "playlist" itself: never part of the name, always
# part of the asking. "Любимый" deliberately stays -- it is usually the name.
_POSSESSIVE = re.compile(r"\b(мой|моя|мои|мою|моё|мое|моего|моих|"
                         r"плейлист\w*|список|альбом)\b")


def _strip_possessive(text: str) -> str:
    """'мой плейлист для бега' -> 'для бега'."""
    return " ".join(_POSSESSIVE.sub(" ", text).split())


def _name_matches(wanted: str, name: str) -> bool:
    """Does this playlist name answer that request?

    Matched on five-character stems rather than whole words, because the
    request is inflected and the name is not: someone asks for "мой любимый
    плейлист" and the playlist is called "Любимое". Russian morphology is a
    dependency (pymorphy2 and a dictionary) for what is, here, a prefix
    comparison -- the board has 4 GB of RAM and 20 GB of disk shared with
    other teams, and this is not what to spend either on.
    """
    stems = [word[:5] for word in wanted.split() if len(word) >= 3]
    return bool(stems) and all(stem in name for stem in stems)


# --- HTTP ---------------------------------------------------------------------

def _credentials() -> tuple[str, str, str] | None:
    client = config.get("SPOTIFY_CLIENT_ID")
    secret = config.get("SPOTIFY_CLIENT_SECRET")
    refresh = config.get("SPOTIFY_REFRESH_TOKEN")
    return (client, secret, refresh) if client and secret and refresh else None


def _access_token() -> str:
    """A live access token, refreshed when the cached one is about to expire."""
    global _token
    if _token is not None and datetime.now() < _token[1] - _EXPIRY_MARGIN:
        return _token[0]

    client, secret, refresh = _credentials()
    auth = base64.b64encode(f"{client}:{secret}".encode()).decode()
    data = urllib.parse.urlencode({
        "grant_type": "refresh_token", "refresh_token": refresh,
    }).encode()
    request = urllib.request.Request(TOKEN_URL, data=data, method="POST", headers={
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/x-www-form-urlencoded",
    })
    with urllib.request.urlopen(request, timeout=config.http_timeout()) as response:
        payload = json.loads(response.read().decode("utf-8"))

    _token = (payload["access_token"],
              datetime.now() + timedelta(seconds=int(payload.get("expires_in", 3600))))
    return _token[0]


def _api(method: str, path: str, body: dict | None = None,
         params: dict | None = None) -> dict:
    """One Web API call. Returns {} for the empty 204s the player endpoints give."""
    url = f"{API}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {_access_token()}",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(request, timeout=config.http_timeout()) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8")) if raw else {}


def _command(method: str, path: str, body: dict | None = None) -> None:
    """A player command, with 404 read as what it actually means."""
    try:
        _api(method, path, body)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise NoActiveDevice from exc
        raise
