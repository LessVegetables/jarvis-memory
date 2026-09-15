#!/usr/bin/env python3
"""
Add (or rename) a user.

    python3 tools/add_user.py daniel "Даниил" 21
    python3 tools/add_user.py            # lists users

user_id must match what module B (speaker identification) emits for that
voice. Running again with the same id updates the name and age.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis_memory import db, store  # noqa: E402


def main(argv: list[str]) -> int:
    if not argv:
        rows = db.connect().execute("SELECT user_id, name, age FROM users ORDER BY user_id")
        for row in rows:
            print(f"  {row['user_id']:12} {row['name']}  {row['age'] or ''}")
        return 0
    if len(argv) < 2:
        print(__doc__)
        return 2
    user_id, name = argv[0], argv[1]
    age = int(argv[2]) if len(argv) > 2 else None
    store.add_user(user_id, name, age)
    print(f"ok: {user_id} = {name}" + (f", {age}" if age else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
