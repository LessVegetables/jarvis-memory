#!/usr/bin/env python3
"""
One-time Spotify authorisation. Run this on a laptop, never on the board.

It opens a browser, catches the redirect on loopback, exchanges the code and
prints a refresh token. That token is the only Spotify secret the device ever
holds: providers/spotify.py exchanges it for an access token and nothing else,
so there is no callback server, no browser and no interactive step on the
Firefly.

Setup, once:

  1. https://developer.spotify.com/dashboard -> Create app
  2. Add this exact redirect URI to the app:   http://127.0.0.1:8888/callback
     (Spotify rejects plain http anywhere except loopback, which is why it is
     the IP literal and not "localhost".)
  3. Put the client id and secret in memory_module/.env, then run:

       python3 tools/spotify_auth.py

  4. Paste the printed SPOTIFY_REFRESH_TOKEN line into .env as well, and copy
     .env to the board.

Playback also needs a Premium account and the Spotify app open somewhere --
the Web API will not start playback on a device that is not already running.
"""

import http.server
import json
import secrets
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis_memory import config  # noqa: E402

REDIRECT = "http://127.0.0.1:8888/callback"
PORT = 8888

# Only what the assistant actually does: read the playlists to find "мой
# любимый плейлист", read the player to know there is a device, and control
# playback. No library writes, no profile, no email.
SCOPES = " ".join([
    "user-modify-playback-state",
    "user-read-playback-state",
    "playlist-read-private",
    # Liked Songs, which is what "включи мой любимый плейлист" resolves to.
    "user-library-read",
])

_result: dict[str, str] = {}
_done = threading.Event()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):                                          # noqa: N802
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _result.update({k: v[0] for k, v in query.items()})
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        ok = "code" in _result
        self.wfile.write(
            ("<h2>Готово, можно закрыть вкладку.</h2>" if ok else
             f"<h2>Не получилось: {_result.get('error')}</h2>").encode("utf-8"))
        _done.set()

    def log_message(self, *args):
        pass                                                   # quiet


def main() -> int:
    client = config.get("SPOTIFY_CLIENT_ID")
    secret = config.get("SPOTIFY_CLIENT_SECRET")
    if not client or not secret:
        print("SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET are not set.\n"
              f"Put them in {config.ENV_PATH} first (see the docstring).")
        return 1

    state = secrets.token_urlsafe(16)
    url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode({
        "client_id": client, "response_type": "code", "redirect_uri": REDIRECT,
        "scope": SCOPES, "state": state,
    })

    server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"Открываю браузер. Если он не открылся — перейди сюда:\n\n{url}\n")
    webbrowser.open(url)

    if not _done.wait(timeout=300):
        print("Ответа не дождались за пять минут.")
        return 1
    server.shutdown()

    if "code" not in _result:
        print(f"Spotify вернул ошибку: {_result.get('error', '?')}")
        return 1
    if _result.get("state") != state:
        # Someone else's redirect landed here. Nothing good comes of
        # exchanging a code that did not come from the request we made.
        print("state не совпал — ответ пришёл не на наш запрос.")
        return 1

    data = urllib.parse.urlencode({
        "grant_type": "authorization_code", "code": _result["code"],
        "redirect_uri": REDIRECT, "client_id": client, "client_secret": secret,
    }).encode()
    request = urllib.request.Request(
        "https://accounts.spotify.com/api/token", data=data, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    token = payload.get("refresh_token")
    if not token:
        print(f"Токен не пришёл: {payload}")
        return 1

    print("\n" + "=" * 62)
    print("Добавь эту строку в memory_module/.env :\n")
    print(f"SPOTIFY_REFRESH_TOKEN={token}")
    print("=" * 62)
    print("\nОна не истекает. Больше этот скрипт запускать не нужно.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
