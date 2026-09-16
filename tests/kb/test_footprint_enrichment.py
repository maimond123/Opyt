"""`pipeline/kb/footprint_enrichment.py` — the metered deepening loop over Oracle X timelines.

R2 (2026-09-13): background ⟺ blocked by a rate meter. A foreground ingest over four Oracles
CANNOT cover four Oracles — `UserTweets` is 50 per 15 minutes against a measured 2-26 requests
each — so the remainder is definitionally background work, and asking the user to re-trigger it
asks twice for one decision they already made when they confirmed these writers.

A sibling of `enrichment.py`, not a branch inside it: different bucket, different corpus,
different lease. One sleep cannot serve two unrelated meters.

`sleep` is injected throughout, so these drive real multi-window behaviour in milliseconds.
"""
from __future__ import annotations

import pytest

from pipeline.kb import footprint_enrichment as fe


@pytest.fixture(autouse=True)
def startable(monkeypatch, kb_home):
    from pipeline.kb import oracle_refresh
    monkeypatch.setattr(oracle_refresh, "consented", lambda: True)
    monkeypatch.setattr(fe, "models_unroutable", lambda rail: None)


class _FakeConn:
    def close(self):
        pass


@pytest.fixture()
def passes(monkeypatch, kb_home):
    """Drive `backfill_pass` from a scripted list of summaries, one per pass."""
    from pipeline.kb import embed as embed_mod, oracle_refresh, oracle_refresh_state as st

    monkeypatch.setattr(embed_mod, "get_kb_embedder", lambda: object())
    monkeypatch.setattr(st, "connect", lambda *a, **k: _FakeConn())
    box: dict = {"script": [], "calls": 0}

    def _pass(conn, embedder, **kw):
        box["calls"] += 1
        return (box["script"].pop(0) if box["script"]
                else {"status": "ok", "considered": 0, "deferred": 0, "new_atoms": 0})

    monkeypatch.setattr(oracle_refresh, "backfill_pass", _pass)
    # `owed` is read from the STORE after every pass, so the script drives it too — one entry per
    # pass, defaulting to 0 (finished) once the script runs out.
    box["owed"] = []
    monkeypatch.setattr(fe, "owed",
                        lambda conn=None: box["owed"].pop(0) if box["owed"] else 0)
    return box


def _meter(monkeypatch, waits):
    """Fake the meter: `waits` is the answer for each between-pass check, in order."""
    seq = list(waits)
    monkeypatch.setattr(fe, "seconds_until_window_resets",
                        lambda *a, **k: seq.pop(0) if seq else 0.0)


def _run(**kw):
    naps: list = []
    out = fe.run_footprint_enrichment(sleep=naps.append, **kw)
    return out, naps


# ── it pays as many windows as the work takes ─────────────────────────────────

def test_it_keeps_paying_windows_until_every_window_is_met(passes, monkeypatch):
    """The headline behaviour, and the thing no foreground call can do: four Oracles do not fit in
    one 15-minute bucket, and nothing here makes them fit. It waits instead."""
    passes["script"] = [
        {"status": "rate_paused", "considered": 4, "deferred": 3, "new_atoms": 20},
        {"status": "rate_paused", "considered": 3, "deferred": 1, "new_atoms": 14},
        {"status": "ok", "considered": 1, "deferred": 0, "new_atoms": 9},
    ]
    passes["owed"] = [3, 1, 0]
    _meter(monkeypatch, [900.0, 900.0])

    out, naps = _run()

    assert out["status"] == "done" and out["passes"] == 3
    assert out["added"] == 43
    assert naps == [900.0, 900.0]              # one sleep between passes, none after the last


def test_nothing_owed_finishes_without_sleeping(passes, monkeypatch):
    """Every window met. `considered == 0` is the finished case and must not buy a nap."""
    passes["script"] = [{"status": "ok", "considered": 0, "deferred": 0, "new_atoms": 0}]
    passes["owed"] = [0]

    out, naps = _run()

    assert out["status"] == "done" and out["passes"] == 1 and naps == []


# ── the things a window cannot fix ────────────────────────────────────────────

def test_a_dead_session_stops_the_loop_instead_of_sleeping_on_it(passes, monkeypatch):
    """A reconnect needs a person. Fifteen minutes changes nothing about it."""
    passes["script"] = [{"status": "needs_reconnect", "considered": 2, "deferred": 2,
                         "new_atoms": 0}]
    passes["owed"] = [2]
    _meter(monkeypatch, [900.0])

    out, naps = _run()

    assert out["status"] == "needs_reconnect" and naps == []


def test_no_ground_gained_off_the_meter_stops_rather_than_spins(passes, monkeypatch):
    """A dead handle or an open breaker is owed forever and waiting cannot pay it. The meter is
    NOT the blocker here — that is what makes another window pointless rather than premature."""
    passes["script"] = [{"status": "ok", "considered": 2, "deferred": 2, "new_atoms": 0},
                        {"status": "ok", "considered": 2, "deferred": 2, "new_atoms": 0}]
    passes["owed"] = [2, 2]
    _meter(monkeypatch, [0.0, 0.0])

    out, naps = _run()

    assert out["status"] == "no_progress" and out["passes"] == 2 and naps == []


def test_the_same_owed_count_behind_a_LIVE_meter_keeps_waiting(passes, monkeypatch):
    """The other half of that rule, and the one that would silently break the loop if dropped: a
    pass that gained no ground BECAUSE the bucket was empty is exactly what another window fixes."""
    passes["script"] = [{"status": "rate_paused", "considered": 2, "deferred": 2, "new_atoms": 0},
                        {"status": "ok", "considered": 2, "deferred": 0, "new_atoms": 5}]
    passes["owed"] = [2, 0]
    _meter(monkeypatch, [900.0])

    out, naps = _run()

    assert out["status"] == "done" and naps == [900.0]


def test_max_passes_is_a_guard_that_reports_rather_than_a_silent_cap(passes, monkeypatch):
    _meter(monkeypatch, [1.0] * 10)
    passes["script"] = [{"status": "rate_paused", "considered": 5, "deferred": 5 - i,
                         "new_atoms": 1} for i in range(5)]
    passes["owed"] = [5, 4, 3, 2, 1]

    out, _naps = _run(max_passes=3)

    assert out["status"] == "max_passes" and out["passes"] == 3
    assert out["owed"] == 3                    # and it says what is still outstanding


def test_a_reclaimed_lease_stops_the_loop(passes, monkeypatch):
    passes["script"] = [{"status": "lease_lost", "considered": 3, "deferred": 3, "new_atoms": 2}]
    passes["owed"] = [3]

    out, _naps = _run()

    assert out["status"] == "lease_lost"


def test_should_stop_is_checked_between_passes_never_mid_write(passes, monkeypatch):
    out, _naps = _run(should_stop=lambda: True)

    assert out["status"] == "cancelled" and passes["calls"] == 0


# ── it never takes anything down ──────────────────────────────────────────────

def test_it_cannot_start_without_consent(monkeypatch, kb_home):
    """The same marker the rail asks for: this reads the user's X session in the background and
    spends model credits on what it pulls. A caller must read this as "could not start"."""
    from pipeline.kb import oracle_refresh
    monkeypatch.setattr(oracle_refresh, "consented", lambda: False)

    assert fe.run_footprint_enrichment()["status"] == "needs_consent"


def test_a_pass_that_raises_is_reported_not_propagated(passes, monkeypatch):
    from pipeline.kb import oracle_refresh
    monkeypatch.setattr(oracle_refresh, "backfill_pass",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))

    out, _naps = _run()

    assert out["status"] == "error" and "boom" in out["error"]
    assert fe.is_running() is False             # …and the probe does not stay stuck on


# ── the two buckets ───────────────────────────────────────────────────────────

def test_the_wait_is_the_LONGER_of_the_two_buckets(monkeypatch):
    """⚠️ A footprint pull walks BOTH timelines and they meter independently. Waking while either
    is still spent buys a pass that finishes one timeline and is cut off on the other — which
    `_walk_frontier` correctly refuses to claim any frontier for, so the window is wasted."""
    import time
    from pipeline.ingestion import x_graphql_core as core

    now = time.time()
    monkeypatch.setattr(core, "rate_budget", lambda op: {
        core.USERTWEETS_OP: (0, now + 60), core.USERREPLIES_OP: (0, now + 600)}[op])

    assert fe.seconds_until_window_resets(now) == pytest.approx(600 + fe._RESET_SLACK, abs=1)


def test_an_unknown_meter_reads_as_go(monkeypatch):
    """The budget is process-local and a fresh process starts blind. Treating no-evidence as spent
    would make a restarted server sleep fifteen minutes before its first request."""
    from pipeline.ingestion import x_graphql_core as core
    monkeypatch.setattr(core, "rate_budget", lambda op: None)

    assert fe.seconds_until_window_resets() == 0.0


def test_a_bucket_with_headroom_does_not_wait_on_the_other_s_reset(monkeypatch):
    import time
    from pipeline.ingestion import x_graphql_core as core

    now = time.time()
    monkeypatch.setattr(core, "rate_budget", lambda op: {
        core.USERTWEETS_OP: (12, now + 600), core.USERREPLIES_OP: (7, now + 900)}[op])

    assert fe.seconds_until_window_resets(now) == 0.0


# ── the start seam, which the message layer reads ─────────────────────────────

def test_nothing_owed_starts_no_thread(monkeypatch, kb_home):
    """So a caller can tell "finished" from "running" — and never promises a pass that had no
    work to do."""
    monkeypatch.setattr(fe, "owed", lambda conn=None: 0)

    assert fe.start_background() == {"status": "nothing_owed"}


def test_a_spawn_failure_is_reported_as_not_started(monkeypatch, kb_home):
    """Fail-safe, and load-bearing for the copy: "filling in on its own" is a claim about a thread
    that exists. A caller told otherwise stops waiting for a pull that cannot start."""
    import threading
    monkeypatch.setattr(fe, "owed", lambda conn=None: 3)
    monkeypatch.setattr(threading, "Thread",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no threads")))

    assert fe.start_background()["status"] == "not_started"


def test_owed_counts_only_x_pairs_with_an_unmet_window(kb_home, monkeypatch):
    from datetime import timedelta
    from pipeline.ingestion import x_graphql
    from pipeline.kb import oracle_refresh, oracle_refresh_state as st
    from pipeline.timeparse import utc_now

    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: True)
    conn = st.connect()
    now = utc_now()
    for key, covered in (("deep", now - timedelta(days=400)),     # already past the target
                         ("shallow", now - timedelta(days=30))):  # owed
        st.upsert_source(conn, st.SourceRow(canonical_id=f"x:user:{key}", source_type="x",
                                            source_key=key, status="trusted"))
        row = next(r for r in st.list_sources(conn) if r.source_key == key)
        st.record_pull(conn, row, last_status="ingested", stamp=True,
                       covered_from=covered.isoformat())

    assert fe.owed(conn) == 1
    assert oracle_refresh.deepen_target(now) < now                # sanity on the target itself
    conn.close()


def test_owed_is_zero_without_an_x_session(kb_home, monkeypatch):
    """No session, no metered timeline, nothing for this loop to drain."""
    from pipeline.ingestion import x_graphql
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: False)

    assert fe.owed() == 0


def test_an_unreadable_store_reports_nothing_owed(monkeypatch, kb_home):
    """Fail-safe direction: every caller uses this to decide whether to PROMISE something."""
    from pipeline.kb import oracle_refresh_state as st
    monkeypatch.setattr(st, "connect", lambda *a, **k: 1 / 0)

    assert fe.owed() == 0


# ── the foreground ingest hands off what it could not reach ───────────────────
#
# ⚠️ R2 inverted is what this replaces. The old message ended in a command the reader would never
# type — "call `oracle(action='ingest')` again once the window refills" — which turned meter-
# blocked work into a user decision. They already chose these writers when they confirmed them.

def _ingest_results(*rows):
    return [{"name": "A", "atoms_added": 1, "results": list(rows)}]


def test_a_deferred_oracle_hands_off_to_the_background(monkeypatch):
    from mcp_server import oracle_tools
    monkeypatch.setattr(fe, "start_background", lambda: {"status": "running"})

    out = oracle_tools._finish_in_background(
        _ingest_results({"type": "x", "action": "deferred"}))

    assert out["status"] == "running"


def test_a_partial_oracle_hands_off_too(monkeypatch):
    """A partial walk landed atoms, so it is not `deferred` — and it still owes the rest. Reading
    only `deferred` here would leave exactly the writers this build exists for behind."""
    from mcp_server import oracle_tools
    monkeypatch.setattr(fe, "start_background", lambda: {"status": "running"})

    out = oracle_tools._finish_in_background(
        _ingest_results({"type": "x", "action": "ingested", "partial": True}))

    assert out["status"] == "running"


def test_a_fully_covered_run_starts_nothing(monkeypatch):
    from mcp_server import oracle_tools
    monkeypatch.setattr(fe, "start_background",
                        lambda: pytest.fail("nothing was owed; nothing should have started"))

    out = oracle_tools._finish_in_background(
        _ingest_results({"type": "x", "action": "ingested"}))

    assert out["status"] == "nothing_owed"


def test_a_failed_start_is_reported_rather_than_raised(monkeypatch):
    """The message layer chooses between "filling in on its own" and "not moving right now" on
    this value, so it must be a fact and never an exception."""
    from mcp_server import oracle_tools
    monkeypatch.setattr(fe, "start_background",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    out = oracle_tools._finish_in_background(
        _ingest_results({"type": "x", "action": "deferred"}))

    assert out["status"] == "not_started" and "boom" in out["error"]


# ── `deferred` is not "still owed" (found by the live run, 2026-09-14) ────────
#
# ⚠️ MEASURED FAILURE. A real pass reported `2 shallow, 1 deepened, 0 deferred, 218 new atoms` and
# this loop STOPPED — with both those pairs still owed. `deferred` counts pairs the pass REFUSED
# to try; it says nothing about a pair that was tried, landed atoms, and still did not meet its
# window. Since durable partial walks that is the COMMON case: a one-sided walk deliberately
# claims no frontier, so `covered_from` does not move however many atoms land.
#
# The two counts agree only for an all-or-nothing walk — exactly what this build removed. The
# condition was inherited from the bookmark sibling, where `deferred` really does mean "still
# owed", which is the second time "same shape as bookmark Enrichment" has been wrong in this file.

def test_a_pass_that_defers_nothing_but_owes_everything_keeps_going(passes, monkeypatch):
    """The regression. 218 atoms landed, nothing was refused, and the windows are still unmet —
    so there is more to do and the loop must wait for the next one."""
    passes["script"] = [{"status": "ok", "considered": 2, "deferred": 0, "new_atoms": 218},
                        {"status": "ok", "considered": 2, "deferred": 0, "new_atoms": 40}]
    passes["owed"] = [2, 0]                  # the store still says owed after pass 1
    _meter(monkeypatch, [900.0])

    out, naps = _run()

    assert out["passes"] == 2, "stopped on deferred==0 while the store still owed two windows"
    assert out["status"] == "done" and out["added"] == 258
    assert naps == [900.0]


def test_atoms_landing_count_as_progress_even_when_no_frontier_moves(passes, monkeypatch):
    """⚠️ The no-progress check needs all THREE conditions. A partial walk lands real content
    without advancing any frontier, so judging on the owed count alone would abandon exactly the
    case this pass exists to finish."""
    passes["script"] = [{"status": "ok", "considered": 2, "deferred": 0, "new_atoms": 218},
                        {"status": "ok", "considered": 2, "deferred": 0, "new_atoms": 97},
                        {"status": "ok", "considered": 2, "deferred": 0, "new_atoms": 0}]
    passes["owed"] = [2, 2, 2]               # frontier never moves…
    _meter(monkeypatch, [0.0, 0.0, 0.0])     # …and the meter is NOT the blocker

    out, _naps = _run()

    # Passes 1 and 2 landed atoms, so they are progress. Pass 3 landed none — that is the one
    # that means nothing more is coming.
    assert out["passes"] == 3 and out["status"] == "no_progress"
    assert out["added"] == 315


def test_owed_is_re_read_from_the_store_not_taken_from_the_pass(passes, monkeypatch):
    """The fix in one assertion: the pass can say whatever it likes about `deferred`; what ends
    this loop is the store reporting no unmet window."""
    passes["script"] = [{"status": "ok", "considered": 9, "deferred": 7, "new_atoms": 1}]
    passes["owed"] = [0]                     # the store disagrees with `deferred` — store wins

    out, naps = _run()

    assert out["passes"] == 1 and out["status"] == "done" and naps == []
