"""
Configuration: API keys, device location, timeouts.

Everything is read from the environment. A gitignored `.env` file in the
repository root is the convenient place to put values -- it is loaded once,
on import, and never overrides a variable that is already set in the real
environment. `.env.example` lists every key.

Values are read at call time, not import time, so tests can change them and
so a key added to the environment after startup is picked up.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

# Default location if none is configured: Moscow, Red Square. Wrong for
# nearly everyone, which is the point -- an obviously-wrong default gets
# fixed; a plausible one gets shipped.
_DEFAULT_LAT = 55.7539
_DEFAULT_LON = 37.6208


def load_dotenv(path: Path = ENV_PATH) -> int:
    """Load KEY=VALUE lines into os.environ without overriding. Returns count."""
    if not path.exists():
        return 0
    loaded = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


load_dotenv()


def get(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def location() -> tuple[float, float]:
    """(lat, lon) of the device. The kitchen does not move, so this is config."""
    return (float(get("JARVIS_LAT", _DEFAULT_LAT)),
            float(get("JARVIS_LON", _DEFAULT_LON)))


def http_timeout() -> float:
    """Seconds. Short on purpose: a voice assistant that hangs for ten seconds
    waiting on a weather API is broken, whatever it says afterwards."""
    return float(get("JARVIS_HTTP_TIMEOUT", "3"))
