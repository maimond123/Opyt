"""
pipeline/onboarding/worker_check.py — the background worker as `onboard` reports it.
"""
from __future__ import annotations

import sys

from pipeline.worker import liveness


def report_applies(answer: dict) -> bool:
    """Does this answer carry the worker report? Not the off-macOS refusal, which already says
    why; not an error, which opened nothing."""
    return answer.get("phase") != "unsupported" and answer.get("status") != "error"


def refuse_off_macos() -> dict | None:
    """Refuse a local install off macOS, and send the user to a Mac. None everywhere else.

    The worker ships only on macOS, so a local install elsewhere would finish setup with nothing
    running. Only a home the worker report calls `unsupported` is refused, so a hand-run
    `opyt-worker` is let through. A probe that raises lets the call through.
    """
    if sys.platform == "darwin":
        return None
    try:
        if liveness.repair_or_report_cached().get("status") != "unsupported":
            return None
    except Exception:
        return None
    return {"status": "unsupported_platform", "phase": "unsupported",
            "message": ("OPYT's local install runs on macOS only for now: everything it collects "
                        "runs in a background helper, and that helper ships for macOS alone. "
                        "Tell the user that plainly. " + liveness.USE_A_MAC + " Nothing was set up "
                        "here.")}


def ensure() -> dict:
    """Make sure the resident worker is there and claiming, or get back exactly why not.

    Every install gets the worker, since all ingest runs there; this repairs, never falls back.
    A probe that raises is reported, never raised.
    """
    try:
        return liveness.repair_or_report()
    except Exception as e:
        return _unknown(e)


def cached_report() -> dict:
    """The worker check this call already ran (`ensure`), from `liveness`'s cache.

    Every sentence that promises background work reads this and nothing else, so the reply
    cannot contradict itself. A check that raises reads as not running."""
    try:
        return liveness.repair_or_report_cached()
    except Exception as e:
        return _unknown(e)


def _unknown(e: Exception) -> dict:
    """The report when the check itself raised: read as not running, with the reason kept."""
    return {"ok": False, "status": "unknown",
            "cause": "OPYT couldn't check its background helper.",
            "step": "Call `onboard` again in a minute.",
            "detail": {"error": f"{type(e).__name__}: {e}"}}


def until_running(worker: dict) -> str:
    """The clause a promise of background work carries when the worker is not running."""
    return " ".join(x for x in ("as soon as OPYT's background worker is running again.",
                                worker.get("cause"), worker.get("step")) if x)


def report(worker: dict) -> dict | None:
    """What the host is told about the worker on this call. None when there is nothing to say.

    Names no file and no process: paths and log lines stay in `detail`. A worker that is not
    running gets `worker_problem`, the one key a reader checks to know the user's work is waiting.
    """
    if not worker.get("ok"):
        return {"worker_problem": {
            "status": worker.get("status"), "cause": worker.get("cause"),
            "step": worker.get("step"), "detail": worker.get("detail"),
            "message": ("Nothing the user asks for will run until this is fixed, and nothing is "
                        "lost meanwhile — the work waits. Do NOT say anything is importing, "
                        "pulling or updating. Tell them `cause` in one plain sentence and give "
                        "them `step`, if there is one. Name no file and no process.")}}
    # A running or just-installed worker gets no sentence: there is nothing for the user to do.
    return None
