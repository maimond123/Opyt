"""
pipeline/frontier/source_search.py — the `Adapter` contract: what Frontier stage 2 needs from a
source it searches.

One job per adapter: take a standing query plus "everything since T", return candidates. No
judgement, storage or dedup; the searches source (`pipeline/frontier/searching.py`) owns those.
The transport is platforms'; an adapter maps what it returns to `Candidate`s. `adapters` builds
the registry.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pipeline.frontier.candidates import Candidate


class Adapter(Protocol):
    """A search adapter: a standing query plus "everything since T" → `Candidate`s.

    What every adapter must have is `slug`, `host` (the API host its requests go to, which the
    web pool runs one unit at a time on) and `search`. The searches source also reads four
    optional declarations, each with the default it applies when an adapter leaves one out:

      `min_interval_s`     the courtesy gap between two requests to the source (0: no pacing);
      `min_lookback_days`  how far back the window must reach, whatever the watermark says
                           (0: the watermark alone — see `cadence.lookback_floor`);
      `max_window_days`    how far back the source can usefully be asked at all, whatever window
                           the caller wants (0: any window — see `lanes.window_days`);
      `available()`        whether the host is worth asking now (True).

    """
    slug: str
    host: str

    def search(self, query: str, *, since: datetime | None, limit: int, now: datetime,
               context: tuple[str, ...] | list[str] = ()) -> list[Candidate]:
        """Search a lower-bound window using the execute pass clock as its current time.
        `context` is the watch's context words (`queries.context`): words that narrow a name on a
        source that searches loose text. Only X reads them; every other source takes and ignores
        them."""
        ...


def available(adapter) -> bool:
    """Whether a source is worth asking right now: its `available()`, read fail-open, and True for
    an adapter that declares none. The daily pulls and the searches made now (`lanes`) both
    ask it."""
    probe = getattr(adapter, "available", None)
    try:
        return True if probe is None else bool(probe())
    except Exception:
        return True
