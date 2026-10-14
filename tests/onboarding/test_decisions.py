"""
tests/onboarding/test_decisions.py

One test per row of each onboarding decision's table. Plain values in,
answer out: no store, no mock.
"""
from __future__ import annotations

import pytest

from pipeline.onboarding import handoff, phase, worker_check
from pipeline.onboarding.handoff import Branch
from pipeline.onboarding.phase import Step


@pytest.mark.parametrize("connected, skipped, pending, expected", [
    (False, False, [], "sources"),
    (True, False, ["x"], "curation"),
    (False, True, ["substack"], "curation"),
    (True, False, [], "done"),
    (False, True, [], "done"),
])
def test_the_phase(connected, skipped, pending, expected):
    assert phase.decide(connected=connected, skipped=skipped, pending=pending) == expected


def test_the_optional_keys_are_no_phase_and_no_step():
    """Asked on the first Oracle confirmation (`optional_keys.question`, a
    rider), not before sources."""
    assert not hasattr(Step, "OFFER_KEYS")
    assert "keys_offered" not in phase.decide.__code__.co_varnames


@pytest.mark.parametrize("source, at, expected", [
    ("twitter", "done", Step.REFUSE_SOURCE),
    ("skip", "sources", Step.SKIP),
    ("x", "done", Step.CONNECT),
    ("substack", "sources", Step.CONNECT),
    ("research", "curation", Step.NAME_ROOT),
    ("blog", "sources", Step.NAME_ROOT),
    (None, "sources", Step.SOURCES_MENU),
    (None, "curation", Step.CURATION),
    (None, "done", Step.HANDOFF),
])
def test_the_next_step(source, at, expected):
    assert phase.next_step(at, source) is expected


@pytest.mark.parametrize("answer, expected", [
    ({"status": "unsupported_platform", "phase": "unsupported"}, False),
    ({"status": "error", "phase": "sources"}, False),
    ({"status": "needs_source", "phase": "sources"}, True),
    ({"status": "ok", "phase": "done"}, True),
])
def test_the_worker_report_applies(answer, expected):
    assert worker_check.report_applies(answer) is expected


@pytest.mark.parametrize("at, candidates, collectors, has_oracles, atoms, expected", [
    ("curation", 5, True, True, 9, Branch.STILL_SCORING),
    ("done", 7, True, False, 0, Branch.CHOOSE),
    ("done", 0, True, False, 0, Branch.FOUND_NOBODY),
    ("done", 0, False, True, 0, Branch.NO_DISCOVERY),
    ("done", 0, False, False, 3, Branch.NO_DISCOVERY),
    ("done", 0, False, False, None, Branch.NO_DISCOVERY),
    ("done", 0, False, False, 0, Branch.BLIND_STORE),
])
def test_the_handoff_branch(at, candidates, collectors, has_oracles, atoms, expected):
    assert handoff.branch(phase=at, candidates=candidates, collectors=collectors,
                          has_oracles=has_oracles, atoms=atoms) is expected
