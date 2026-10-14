"""
pipeline/curation/still_scoring.py — the notice that a connected platform's collectors have not
read the user yet, so the Oracle roster's scores are still being filled in.

Curation's notice, asked for by the server's rider flow (`mcp_server/riders.py`) on
`oracle(action="screen")`, so a thin, shifting roster is not read as the final ranking.
"""
from __future__ import annotations

from pipeline.curation import collectors
from pipeline.platforms import connections


def still_scoring_notice() -> dict | None:
    """The candidates' scores are still being filled in: a connected platform's collectors have
    never succeeded (`collectors.curation_pending`). None otherwise.

    Not "a collector's walk is under way": a refresh of a platform already read changes counts,
    not who is a candidate, and a walk whose page is backing off would read as scoring forever."""
    connected = {p for p, on in (("x", connections._connected_x()),
                                 ("substack", connections._connected_substack())) if on}
    pending = collectors.curation_pending(connected)
    if not pending:
        return None
    return {"pending": pending,
            "message": ("Still scoring: OPYT is still reading who the user follows and saves"
                        f" on {' and '.join(p.capitalize() for p in pending)}"
                        ", so this list will change as it finishes — people can move up or gain "
                        "a tick. Show it, and say it is still filling in.")}
