"""
Calendar sync: Google (via its secret ICS address) and iCloud (via CalDAV)
into the events table, on a timer. See sync.py for the design and
sources.py for the two source types.
"""

from .sources import CalDavSource, Event, IcsSource, parse_ics
from .sync import load_sources, replace_window, sync_all

__all__ = ["CalDavSource", "Event", "IcsSource", "parse_ics",
           "load_sources", "replace_window", "sync_all"]
