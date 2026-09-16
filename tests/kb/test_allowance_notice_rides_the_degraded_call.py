"""Where the notice attaches, and — just as importantly — where it does not.

D2 of the handoff: this rider REPEATS, unlike `pull_runs.completion_notice` and `new_material`,
because it announces a STATE that persists until the user acts rather than an EVENT that is over
once mentioned. What keeps it from becoming furniture is not a stamp but the GATE: it rides only
a call that actually came back degraded, so it is always the explanation for the thin result in
front of the reader. These tests are that gate.
"""
from __future__ import annotations

import pytest

from mcp_server import oracle_tools, sitting_tools
from pipeline.circuit_breaker import CircuitBreaker
from pipeline.kb import allowance_notice as an
from pipeline.kb import pull_runs, screen

PICKS = [{"canonical_id": "c:a", "name": "Andrej"}, {"canonical_id": "c:d", "name": "Dean"}]


@pytest.fixture()
def conn(kb_home):
    c = pull_runs.connect()
    yield c
    c.close()


@pytest.fixture()
def spent(monkeypatch):
    """The live 2026-09-15 state: the provider circuit open on repeated `403 Key limit exceeded`,
    and `readiness` confirming the allowance rather than an outage."""
    monkeypatch.setattr(an, "_CACHE", None)
    monkeypatch.setattr("opyt_core.readiness.openrouter",
                        lambda: {"state": "trial_over",
                                 "message": "The starter allowance that came with OPYT is used up."})
    breaker = CircuitBreaker("openrouter")
    for _ in range(breaker.threshold):
        breaker.record_failure("HTTP 403: Key limit exceeded (total limit)")
    return breaker


@pytest.fixture()
def healthy(monkeypatch):
    monkeypatch.setattr(an, "_CACHE", None)
    monkeypatch.setattr("opyt_core.readiness.openrouter",
                        lambda: {"state": "ok", "message": "live and has credit"})


# ── screen: the roster that came back thin ────────────────────────────────────────────────────

def test_a_screen_whose_classify_was_refused_says_why(conn, spent, monkeypatch):
    """THE MEASURED SURFACE. `[screen] classify batch of 100 skipped (degrade-open)` went to a
    log; the user got a shorter, less-labelled roster and concluded their X login was broken."""
    monkeypatch.setattr(screen, "classify_kinds",
                        lambda *a, **k: {"ran": False, "reason": "HTTP 403", "classified": 0})

    out = oracle_tools._screen(conn, floor=15, limit=40)

    assert out["model_provider"]["state"] == "trial_over"
    assert out["model_provider"]["next_call"] == "onboard(start='openrouter')"


def test_a_screen_with_nothing_left_to_classify_is_not_a_degrade(conn, spent, monkeypatch):
    """`classify_kinds` returns `ran: True` when every candidate was ALREADY classified. That is
    a complete answer, not a thin one, and a notice on it would be the furniture D2 avoids."""
    monkeypatch.setattr(screen, "classify_kinds",
                        lambda *a, **k: {"ran": True, "classified": 0,
                                         "note": "every candidate already classified"})

    assert "model_provider" not in oracle_tools._screen(conn, floor=15, limit=40)


def test_a_healthy_screen_stays_clean(conn, healthy, monkeypatch):
    monkeypatch.setattr(screen, "classify_kinds", lambda *a, **k: {"ran": False, "classified": 0})
    assert "model_provider" not in oracle_tools._screen(conn, floor=15, limit=40)


# ── ingest: the pull that fetched everything and wrote nothing ────────────────────────────────

def test_a_pull_that_wrote_nothing_says_why(conn, spent, monkeypatch):
    """With the provider refused, every atom reaches the sink and none survives the embed —
    `AtomSink._flush_isolated` skips each one WITHOUT writing, correctly and silently. What comes
    back is a pull that reached its writers and added zero pieces, which reads as "there was
    nothing there" about people who post daily."""
    monkeypatch.setattr(oracle_tools, "_results_from",
                        lambda *a, **k: [{"name": "Andrej", "atoms_added": 0, "results": []},
                                         {"name": "Dean", "atoms_added": 0, "results": []}])
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)

    out = oracle_tools._report(conn, run_id)

    assert out["model_provider"]["state"] == "trial_over"


def test_a_pull_that_landed_material_is_not_degraded(conn, spent, monkeypatch):
    """Atoms were written, so the provider was working for this pull whatever the breaker says
    about some later minute. Nothing here needs explaining."""
    monkeypatch.setattr(oracle_tools, "_results_from",
                        lambda *a, **k: [{"name": "Andrej", "atoms_added": 57, "results": []}])
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)

    assert "model_provider" not in oracle_tools._report(conn, run_id)


# ── sitting: a read that failed in its own words ──────────────────────────────────────────────

def test_a_failed_sitting_read_gains_the_remedy(spent):
    """`sitting_reader._fail` already says what happened — "provider rejected the prompt
    (HTTP 402)" — which tells a reader the transport and nothing about what to do."""
    out = sitting_tools._say_why_if_blocked(
        {"status": "failed", "reason": "provider rejected the prompt (HTTP 402)"})

    assert out["reason"].startswith("provider rejected")      # untouched
    assert out["model_provider"]["next_call"] == "onboard(start='openrouter')"


def test_a_sitting_read_that_worked_gains_nothing(spent):
    out = sitting_tools._say_why_if_blocked({"status": "ok", "queries": ["a", "b"]})
    assert "model_provider" not in out


def test_explaining_a_failure_can_never_cause_one(monkeypatch):
    """Fail-safe: this is a courtesy on somebody else's answer. A rider that raises would turn a
    recorded, recoverable failure into a tool crash."""
    monkeypatch.setattr(an, "allowance_notice", lambda: 1 / 0)
    res = {"status": "failed", "reason": "whatever"}
    assert sitting_tools._say_why_if_blocked(res) == res
