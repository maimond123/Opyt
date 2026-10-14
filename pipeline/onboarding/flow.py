"""
pipeline/onboarding/flow.py — one `onboard` call, as named steps.

Each call recomputes where setup stands from disk, answers a `start` word if there is one, then
takes the one step the phase and the `source` word call for (`phase.next_step`).
"""
from __future__ import annotations

from pipeline.onboarding import (connect, handoff, kickoff, optional_keys, phase, roots,
                                 worker_check)
from pipeline.onboarding.phase import Step

# Every word `start` takes: the optional keys.
_STARTS = optional_keys.STARTS


def onboard(source: str | None, start: str | None) -> dict:
    answer = _answer(source, start)
    # The worker check this call ran, read back from its cache — so what this reply says about
    # the worker is the same answer its own sentences were built from.
    if worker_check.report_applies(answer) and (
            worker := worker_check.report(worker_check.cached_report())) is not None:
        answer.update(worker)
    return answer


def _answer(source: str | None, start: str | None) -> dict:
    if (refused := worker_check.refuse_off_macos()) is not None:
        return refused
    # A sign-in window was opened and the user is back: wait for the login to reach the disk
    # before reading where setup stands, or this call answers as if nobody signed in.
    if source is None and start is None and (held := connect.hold_for_signin()) is not None:
        return held
    state = phase.derive()
    if start is not None and start not in _STARTS:
        # Refuse, never guess: every step a `start` word begins opens a browser tab.
        return {"status": "error", "phase": state["phase"],
                "message": (f"`start={start!r}` is not one of "
                            f"{', '.join(f'`{w}`' for w in _STARTS)}. Nothing was opened.")}
    if start in optional_keys.STARTS:
        return optional_keys.run(start, state["phase"])
    step = phase.next_step(state["phase"], source)
    if step is Step.REFUSE_SOURCE:
        return roots.refuse(source)

    # Every call that takes a step makes sure the worker is there, since nothing ingests
    # without it. Run once, here, so every sentence below reads the same answer.
    worker_check.ensure()

    if step in (Step.SKIP, Step.NAME_ROOT):
        connect.abandon_signin()
    if step is Step.SKIP:
        state, step = phase.skip_sources()
    if step is Step.CONNECT:
        return connect.start(source)
    if step is Step.NAME_ROOT:
        return roots.name_question(source)
    if step is Step.SOURCES_MENU:
        return roots.menu()

    # Hold while a just-signed-in session does not answer yet, or the curation pull skips every
    # collector and reports "you follow nobody".
    if step is Step.CURATION and (held := connect.hold_if_not_ready(state)) is not None:
        return held

    # What landed while nobody was looking (an Oracle pull's milestones, new material) rides
    # this handoff through the server's rider flow (`mcp_server/riders.py`).
    return handoff.answer(state, kickoff.start_work(state))
