"""
Live data fetched at question time: weather, nearby places.

These are the one kind of information that cannot live in the database --
it changes hourly, or there is too much of it to store. Each provider turns
a transcript into a block of text for the prompt, caches what it fetched,
and never raises: a failed lookup becomes a sentence saying so.
"""

from . import places, weather

__all__ = ["places", "weather"]
