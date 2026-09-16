"""`pipeline/kb/enrichment.py` — the metered upgrade loop over X bookmarks.

R2 (2026-09-13): background ⟺ blocked by a rate meter. Enrichment is the only part of the bookmark
path that qualifies, and the only part allowed to SLEEP — `_refuse_if_spent` refuses rather than
sleeps because a rail is a one-shot detached child holding a lease and a connection, and neither
is true of this loop.

`sleep` is injected throughout, so these drive real multi-window behaviour in milliseconds.
"""
from __future__ import annotations

import pytest

from pipeline.kb import enrichment


@pytest.fixture(autouse=True)
def consented(monkeypatch, kb_home):
    from pipeline.kb import bookmark_catchup
    monkeypatch.setattr(bookmark_catchup, "consented", lambda: True)
    monkeypatch.setattr(enrichment, "models_unroutable", lambda rail: None)


@pytest.fixture()
def runs(monkeypatch, kb_home):
    """Drive `sync_bookmarks` from a scripted list of summaries, one per pass, and record the
    kwargs each pass was called with."""
    from pipeline.kb import ingest_x, schema
    from pipeline.kb import embed as embed_mod

    monkeypatch.setattr(embed_mod, "get_kb_embedder", lambda: object())
    monkeypatch.setattr(schema, "connect", lambda *a, **k: _FakeConn())
    box: dict = {"script": [], "calls": []}

    def _sync(conn, embedder, **kw):
        box["calls"].append(kw)
        return box["script"].pop(0) if box["script"] else {"added": 0, "deferred": 0}

    monkeypatch.setattr(ingest_x, "sync_bookmarks", _sync)
    return box


class _FakeConn:
    def close(self):
        pass


def _slept(monkeypatch, waits):
    """Fake the meter: `waits` is the answer for each between-pass check, in order."""
    seq = list(waits)
    monkeypatch.setattr(enrichment, "seconds_until_window_resets",
                        lambda *a, **k: seq.pop(0) if seq else 0.0)
    naps: list = []
    return naps


def test_it_keeps_paying_windows_until_nothing_is_owed(runs, monkeypatch):
    """The headline behaviour, and the thing the old code could not do at all: 966 ÷ 150 is seven
    windows, and `_ConvoFetcher` disabling itself for the run meant only the first was ever paid."""
    runs["script"] = [{"added": 150, "deferred": 816},
                      {"added": 150, "deferred": 666},
                      {"added": 666, "deferred": 0}]
    naps = _slept(monkeypatch, [900.0, 900.0])

    out = enrichment.run_enrichment(sleep=naps.append)

    assert out["status"] == "done"
    assert out["passes"] == 3 and out["added"] == 966 and out["deferred"] == 0
    assert naps == [900.0, 900.0]            # it slept out two windows, and only two
    assert all(c["enrich"] is True for c in runs["calls"])


def test_a_spent_meter_is_waited_out_even_when_a_pass_gained_nothing(runs, monkeypatch):
    """The pass that starts with the bucket already empty resolves NOTHING — `_ConvoFetcher` is
    disabled by its first refusal. Reading that as no-progress would abandon the backlog at exactly
    the moment waiting is the whole answer."""
    runs["script"] = [{"added": 0, "deferred": 900},
                      {"added": 900, "deferred": 0}]
    naps = _slept(monkeypatch, [900.0])

    out = enrichment.run_enrichment(sleep=naps.append)

    assert out["status"] == "done" and out["passes"] == 2
    assert naps == [900.0]


def test_no_progress_with_a_free_meter_stops_instead_of_spinning(runs, monkeypatch):
    """An X-Article whose body x.com never ships, or a deleted conversation. The meter is open, so
    the shortfall is not waiting on a window — another fifteen minutes only re-learns that."""
    runs["script"] = [{"added": 5, "deferred": 3}, {"added": 0, "deferred": 3}]
    naps = _slept(monkeypatch, [0.0, 0.0])

    out = enrichment.run_enrichment(sleep=naps.append)

    assert out["status"] == "no_progress"
    assert out["passes"] == 2 and out["deferred"] == 3
    assert naps == []                         # it never slept, so it never pretended to wait


def test_a_dead_cookie_stops_the_loop_rather_than_waiting_it_out(runs, monkeypatch):
    """`error` on the summary is the walk itself failing. A rate limit is `blocked` and comes back;
    an expired session needs a person, and no number of windows produces one."""
    runs["script"] = [{"added": 0, "deferred": 900, "error": "SyncAuthError: session rejected"}]
    naps = _slept(monkeypatch, [900.0])

    out = enrichment.run_enrichment(sleep=naps.append)

    assert out["status"] == "error" and out["passes"] == 1
    assert naps == []


def test_the_runaway_guard_reports_rather_than_loops(runs, monkeypatch):
    runs["script"] = [{"added": 1, "deferred": n} for n in (30, 20, 10)]
    naps = _slept(monkeypatch, [900.0, 900.0])

    out = enrichment.run_enrichment(max_passes=2, sleep=naps.append)

    assert out["status"] == "max_passes" and out["passes"] == 2
    assert out["deferred"] == 20               # and it SAYS what is still owed


def test_a_cancel_lands_between_passes_never_mid_write(runs, monkeypatch):
    runs["script"] = [{"added": 5, "deferred": 5}, {"added": 5, "deferred": 0}]
    naps = _slept(monkeypatch, [0.0])
    stop = {"now": False}

    def _should_stop():
        was, stop["now"] = stop["now"], True
        return was

    out = enrichment.run_enrichment(should_stop=_should_stop, sleep=naps.append)

    assert out["status"] == "cancelled"
    assert out["passes"] == 1                  # the pass that started ran to completion
    assert len(runs["calls"]) == 1


def test_the_rails_lease_holder_wins_and_enrichment_yields(runs, monkeypatch):
    """Single-flight, shared with `bookmark_catchup` on purpose: the rail walks the same corpus
    against the same 150/15-min bucket, and its own pass does this work."""
    from pipeline import sync_lock

    class _Held:
        acquired = False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def lost(self):
            return False

    monkeypatch.setattr(sync_lock, "CatchupLock", lambda *a, **k: _Held())

    out = enrichment.run_enrichment()

    assert out["status"] == "already_running"
    assert runs["calls"] == []


def test_state_is_readable_while_it_runs_and_cleared_after(runs, monkeypatch):
    """`atoms_tools._import_outstanding` reads `rail_jobs.db`, and Enrichment is a THREAD, not a
    rail row — so this snapshot is the only thing that keeps `import_incomplete` from going blind
    and `thin_coverage` from blaming the user's corpus."""
    seen: list = []
    runs["script"] = [{"added": 4, "deferred": 2}, {"added": 2, "deferred": 0}]
    _slept(monkeypatch, [900.0])

    def _peek(_):
        seen.append(enrichment.state())

    enrichment.run_enrichment(sleep=_peek)

    assert seen and seen[0]["running"] is True
    assert seen[0]["passes"] == 1 and seen[0]["deferred"] == 2
    assert enrichment.is_running() is False
    assert enrichment.state()["status"] == "done"


# ── the meter read itself ──────────────────────────────────────────────────────

def test_an_unknown_meter_reads_as_go_not_as_wait(monkeypatch):
    """The same rule as `_refuse_if_spent`: the budget is process-local and a
    fresh process is blind. Reading no-evidence as spent would make a restarted server sleep
    fifteen minutes before its first request."""
    from pipeline.ingestion import x_graphql_core as core
    monkeypatch.setattr(core, "rate_budget", lambda op: None)
    assert enrichment.seconds_until_window_resets() == 0.0


def test_a_spent_window_waits_past_the_reset_not_onto_it(monkeypatch):
    from pipeline.ingestion import x_graphql_core as core
    monkeypatch.setattr(core, "rate_budget", lambda op: (0, 1_000.0))
    assert enrichment.seconds_until_window_resets(now=400.0) == 600.0 + enrichment._RESET_SLACK


def test_a_window_that_already_rolled_over_is_not_waited_on(monkeypatch):
    from pipeline.ingestion import x_graphql_core as core
    monkeypatch.setattr(core, "rate_budget", lambda op: (0, 1_000.0))
    assert enrichment.seconds_until_window_resets(now=1_001.0) == 0.0
    monkeypatch.setattr(core, "rate_budget", lambda op: (42, 1_000.0))
    assert enrichment.seconds_until_window_resets(now=400.0) == 0.0
