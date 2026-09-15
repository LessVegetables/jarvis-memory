#!/usr/bin/env python3
"""
Sync every calendar in calendars.json into the events table.

    python3 tools/sync_calendars.py

Run it on a timer. The request path never talks to a calendar server, so
the assistant knows what the last sync saw; every 15 minutes is plenty:

    crontab -e
    */15 * * * * cd /home/firefly/jarvis-memory && python3 tools/sync_calendars.py >> sync.log 2>&1

Exit code is non-zero if any source failed, so a cron mail or a log grep
will show it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis_memory import calendars  # noqa: E402


def main() -> int:
    sources = calendars.load_sources()
    if not sources:
        print(f"No calendars configured ({calendars.sync.CONFIG_PATH} missing). "
              "See calendars.example.json.")
        return 0

    results = calendars.sync_all(sources=sources)
    failed = 0
    for label, outcome in results.items():
        if outcome["ok"]:
            print(f"  ok    {label}: {outcome['events']} events")
        else:
            failed += 1
            print(f"  FAIL  {label}: {outcome['error']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
