"""
pipeline/onboarding/kickoff.py — start the work a connected platform is owed: wake the `scrapers`
process for the walk of who the user follows, and for any import of what they saved that never
began.
"""
from __future__ import annotations

from pipeline.curation import imports
from pipeline.onboarding import roots
from pipeline.worker import job_queue
from pipeline.worker.registry import SCRAPERS


def start_work(state: dict) -> dict:
    """Wake whatever this home's connected platforms are owed. Returns the answer's keys for it:
    `curation` (the people walk) and `backlog_queued` (the imports woken now)."""
    # Connecting a platform is what starts its work; nothing is asked first. Detection is the
    # session probe in `phase.derive()`, so "at connect" means the next `onboard` call.
    extra: dict = {}

    # Arm A, the walk of who they follow. Waking on every call while `pending` is non-empty is
    # harmless: `request_now` on a live row only keeps it due.
    if state["curation"]["pending"]:
        extra["curation"] = _queue_curation_walk()

    # Arm B, the posts they saved there. Keyed on the walk never having begun, not on `pending`:
    # a platform connected after another can be read before `onboard` sees it as pending.
    for platform in imports.imports_never_queued(roots.connected(state["sources"])):
        if _wake_scrapers():
            extra.setdefault("backlog_queued", []).append(imports.IMPORTS[platform][0])
    # No report on the import here; the `new_material` rider speaks once posts are in.
    return extra


def _queue_curation_walk() -> dict:
    """Arm A: wake the `scrapers` process, whose people collectors are due once a platform is
    connected. `screen` is scored on these signals, so the handoff reports `curation` as still
    scoring until the walk has succeeded."""
    if not _wake_scrapers():
        return {"status": "not_queued"}
    return {"status": "queued",
            "host_note": ("The people this user already follows are not ready to choose from "
                          "yet. Do not wait for them, do not mention it as a step, and do not "
                          "call `screen` in this same turn.")}


def _wake_scrapers() -> bool:
    """Wake the `scrapers` process, which begins whatever walk is due."""
    return job_queue.request_now(SCRAPERS)
