"""Copy that promises a schedule must consult a check that can say no.

"Resumes on its own… about every 10 minutes that their computer is awake" and "fills in over
the following days; nothing needs to be run by hand" must not be unconditional: with the worker
disabled nothing drains any rail while the tools keep promising a schedule. The one worker check
(`liveness.repair_or_report_cached`) is the gate: its no stops the promise; an unknown keeps it,
because an unknown must not manufacture an under-promise.
"""
from __future__ import annotations

import pytest

from pipeline.worker import liveness


# ── the candidate coverage note (candidates._coverage_note) ──────────────

def test_a_dead_worker_stops_the_fills_in_promise(monkeypatch):
    from pipeline.oracles import candidates
    monkeypatch.setattr(liveness, "repair_or_report_cached", lambda **kw: {
        "ok": False, "status": "refused", "cause": "the worker is switched off",
        "step": "Run opyt-install-client."})

    note = candidates._coverage_note(10, 12)

    assert "nothing needs to be run by hand" not in note
    assert "not running" in note and "the worker is switched off" in note
    assert "Run opyt-install-client." in note
    assert "PARTIAL COVERAGE" in note


@pytest.mark.parametrize("ok", [True, None])
def test_a_running_or_unknown_worker_keeps_the_fills_in_promise(monkeypatch, ok):
    from pipeline.oracles import candidates
    monkeypatch.setattr(liveness, "repair_or_report_cached", lambda **kw: {"ok": ok})

    note = candidates._coverage_note(10, 12)

    assert "fills them in over the following days" in note
