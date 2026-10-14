"""
pipeline/topics/checked.py — flow: record the host's check of the finds `latest` handed it.

Kept finds are owed admission into the library (stage 3, `frontier/admitting.py`, which this wakes);
passed ones keep their row, marked `passed`, so they are never shown again. Only a waiting find
moves (`candidates.check`), so a verdict sent twice changes nothing.
"""
from __future__ import annotations

from pipeline.frontier import candidates, chain
from pipeline.kb.writes import after_commit, write_tx

from . import lists

# What to say about the check, which the user never hears of.
SAID = ("Say nothing about the finds you passed over; the user only hears about what you kept. "
        "Kept finds join their library within the hour, where search finds them.")


def run(conn, *, keep, passed) -> dict:
    keep, passed = lists.strings(keep), lists.strings(passed)
    if not (keep or passed):
        return {"status": "refused",
                "reason": "Send the finds you checked: `keep` and `passed`, by candidate_id."}
    with write_tx(conn, site="topics.checked"):
        moved = candidates.check(conn, kept=keep, passed=passed)
        if moved[candidates.KEPT]:
            after_commit(conn, chain.finds_kept)
    return {"status": "ok", "kept": moved[candidates.KEPT], "passed": moved[candidates.PASSED],
            "note": SAID}
