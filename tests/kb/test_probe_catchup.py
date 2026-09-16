"""The candidate-probe rail — the spawner, the DAILY ceiling, single-flight, fail-safety.

Same brief as `test_curation_catchup.py`: everything here fails silently in production. This rail
runs detached, at session open, with nobody watching, and the two halves fail in opposite
directions. A trigger that never fires leaves `oracle(action='candidates')` empty forever — the
exact state the live store was in on 2026-08-16, 923 candidates due and zero probed, for five days
after the pull itself was finished and tested. A ceiling that does not hold turns an hourly spawn
into 24 runs a day against ONE cookie session, which is the multi-hour drain the ceiling exists to
spread out.

⚠️ NOTHING HERE MAY REACH A LIVE THIRD PARTY. `probe_candidates` and `get_kb_embedder` are stubbed
in every test that runs the rail. That leak has already cost this tree once: `05e1888a` records six
consent tests silently doing live cookie reads at ~9s each, 58s → 1s once fixed.
"""
from __future__ import annotations

import subprocess

import pytest

from pipeline.kb import candidate_probe as cp
from pipeline.kb import probe_catchup as pc
from pipeline.kb import probe_store, schema


@pytest.fixture()
def rail_home(kb_home):
    return kb_home


@pytest.fixture()
def probe(monkeypatch):
    """Stub the pull itself and hand back the `max_candidates` each call received.

    That list IS the assertion surface for the whole ceiling: the rail's only job is to decide what
    number goes in, and `max_candidates=0` means "the WHOLE due queue" one layer down — so an empty
    list and a `[0]` are opposite outcomes, not near-misses."""
    budgets: list[int] = []

    def _fake(conn, embedder, *, max_candidates=0, **kw):
        budgets.append(max_candidates)
        return {"source": "candidate-probe", "queued": 3, "requests": 3, "atoms": 7,
                "by_status": {"ok": 3}, "stopped": None, "remaining": 900}

    monkeypatch.setattr(cp, "probe_candidates", _fake)
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **k: object())
    return budgets


@pytest.fixture()
def lock_names(monkeypatch):
    """Every lease name the rail actually ACQUIRES, in order. Real locking underneath — only the
    construction is observed, so single-flight keeps working while a test can still prove that a
    refused run never took a lease at all."""
    from pipeline import sync_lock

    names: list[str] = []
    real = sync_lock.CatchupLock

    class _Spy(real):
        def __init__(self, name="catchup", *a, **kw):
            names.append(name)
            super().__init__(name, *a, **kw)

    monkeypatch.setattr(sync_lock, "CatchupLock", _Spy)
    return names


def _spend(n: int, *, day_offset: int = 0) -> None:
    """Record `n` pull attempts against the meter, `day_offset` days in the past.

    The back-date is scoped to the ledger rows this call wrote. Current pull state is deliberately
    not the meter: it overwrites when the same candidate retries."""
    ids = [f"x:user:{day_offset}-{i}" for i in range(n)]
    conn = schema.connect()
    try:
        for who in ids:
            probe_store.record_attempt(conn, who)
        if day_offset:
            conn.execute(
                f"UPDATE probe_attempts SET attempted_at = datetime('now', ?) "
                f"WHERE who_id IN ({','.join('?' * len(ids))})",
                [f"-{day_offset} days", *ids])
            conn.commit()
    finally:
        conn.close()


def _boom(*a, **kw):
    raise RuntimeError("dead X session")


# ── the daily ceiling ───────────────────────────────────────────────────────────
def test_a_fresh_day_spends_the_whole_ceiling(rail_home, probe):
    out = pc.run_candidate_probe(daily_ceiling=60)
    assert out["status"] == "ok"
    assert probe == [60]
    assert (out["probed_today"], out["budget"], out["daily_ceiling"]) == (0, 60, 60)
    assert out["atoms"] == 7 and out["remaining"] == 900      # the run summary rides through


def test_a_partially_spent_day_passes_only_the_remainder(rail_home, probe):
    _spend(25)
    out = pc.run_candidate_probe(daily_ceiling=60)
    assert probe == [35]
    assert (out["probed_today"], out["budget"]) == (25, 35)


def test_a_spent_day_refuses_without_taking_the_lock(rail_home, probe, lock_names):
    """⚠️ TWO failures in one, and the second is the dangerous one. The lease must not be taken to
    report a free refusal — `run_bookmark_catchup`'s ordering, so a no-op never looks like
    contention. And the refusal must be a REFUSAL: `max_candidates=0` means "the whole due queue"
    one layer down, so a spent day falling through as a 0 budget would turn the ceiling into its
    exact opposite — a ~9-hour unbounded drain, triggered BY being out of budget."""
    _spend(60)
    out = pc.run_candidate_probe(daily_ceiling=60)
    assert out["status"] == "daily_ceiling"
    assert (out["probed_today"], out["budget"]) == (60, 0)
    assert probe == []                     # nothing pulled — and specifically not an unbounded run
    assert lock_names == []                # and no lease taken to say so


def test_overspending_yesterday_does_not_bind_today(rail_home, probe):
    """The ceiling resets at UTC midnight. It reads `date(pulled_at) = date('now')`, so a rail that
    ran long yesterday starts today with a full allowance rather than a permanent debt."""
    _spend(200, day_offset=1)
    out = pc.run_candidate_probe(daily_ceiling=60)
    assert out["probed_today"] == 0
    assert probe == [60]


def test_the_meter_counts_attempts_not_successes(rail_home, probe):
    """A `failed` candidate still spent the X request the ceiling rations, so it counts. Gating on
    successes would let a broken session burn the whole day's budget and then ask for another."""
    conn = schema.connect()
    try:
        for _ in range(10):
            probe_store.record_attempt(conn, "x:user:1")
            probe_store.record_pull(conn, "x:user:1", probe_store.STATUS_FAILED,
                                    detail="fetch blew up")
    finally:
        conn.close()
    pc.run_candidate_probe(daily_ceiling=60)
    assert probe == [50]


def test_a_non_positive_ceiling_is_the_explicit_unbounded_run(rail_home, probe):
    """The ONE way to ask for the whole queue, and it has to be typed. `max_candidates=0` is
    `probe_candidates`' own vocabulary for "everyone", so the rail passes it straight through —
    but only when a human wrote the 0, never as the result of subtraction."""
    _spend(60)
    out = pc.run_candidate_probe(daily_ceiling=0)
    assert out["status"] == "ok"
    assert probe == [0]


# ── single-flight ───────────────────────────────────────────────────────────────
def test_a_second_pass_skips_while_one_holds_the_lease(rail_home, probe):
    """Two paced walkers interleaved against ONE cookie session double its request rate, against a
    budget shared with every other GraphQL consumer on the machine."""
    from pipeline.sync_lock import CatchupLock

    with CatchupLock("candidate-probe") as held:
        assert held.acquired
        out = pc.run_candidate_probe()

    assert out["status"] == "already_running"
    assert probe == []


def test_force_does_not_bypass_single_flight(rail_home, probe):
    """The one thing force must not buy. Ignoring a ceiling costs a day's requests; running two
    passes at once costs the session."""
    from pipeline.sync_lock import CatchupLock

    with CatchupLock("candidate-probe") as held:
        assert held.acquired
        assert pc.run_candidate_probe()["status"] == "already_running"
    assert probe == []


def test_the_lease_is_not_shared_with_the_bookmark_or_curation_rails(rail_home, probe):
    """Different names, different leases. Sharing one would make this rail — the slowest of the
    three, up to ~22 minutes a run — block a free list refresh for its whole duration."""
    from pipeline.sync_lock import CatchupLock

    with CatchupLock("bookmark-catchup") as a, CatchupLock("curation-catchup") as b:
        assert a.acquired and b.acquired
        out = pc.run_candidate_probe()
    assert out["status"] == "ok"
    assert probe == [pc.PROBE_DAILY_CANDIDATES]


# ── fail-safety ─────────────────────────────────────────────────────────────────
def test_it_never_raises(rail_home, monkeypatch):
    """Fail-safe invariant. This runs in a detached child whose only caller is a `-m` entrypoint,
    so a propagated exception is a traceback in a log file nobody opens."""
    monkeypatch.setattr(cp, "probe_candidates", _boom)
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **k: object())
    out = pc.run_candidate_probe()
    assert out["status"] == "error" and "dead X session" in out["error"]


def test_a_dead_meter_never_raises_either(rail_home, probe, monkeypatch):
    """The ceiling read happens before anything else, outside the lock — a failure there must
    degrade to a reported error, not to a traceback with the lease left dangling."""
    monkeypatch.setattr(probe_store, "probed_today", _boom)
    assert pc.run_candidate_probe()["status"] == "error"
    assert probe == []


def test_an_empty_queue_exits_clean_and_touches_nothing(rail_home, monkeypatch):
    """The normal case for most sessions once the fill completes: the rail takes the lease, finds
    nobody due, and exits. It must not read as a failure, because it happens every day."""
    monkeypatch.setattr(cp, "probe_candidates",
                        lambda *a, **k: {"source": "candidate-probe", "queued": 0,
                                         "note": "no candidate is due"})
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **k: object())
    out = pc.run_candidate_probe()
    assert out["status"] == "ok" and out["queued"] == 0
    assert pc.main(["--once"]) == 0


def test_a_stopped_run_is_not_reported_as_ok(rail_home, monkeypatch):
    """A dead session, a spent X rate budget or a dead embedder each leave the queue longer than
    the ceiling implies. The only human who reads this is looking at a log to answer "did this pass
    do what it set out to do", so `stopped` must not exit 0."""
    monkeypatch.setattr(cp, "probe_candidates",
                        lambda *a, **k: {"source": "candidate-probe", "queued": 60, "requests": 0,
                                         "atoms": 0, "stopped": "auth", "error": "no X session"})
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **k: object())
    assert pc.run_candidate_probe()["status"] == "stopped"
    assert pc.main(["--once"]) == 1


# ── the meter itself ────────────────────────────────────────────────────────────
def test_the_meter_on_a_never_probed_store_creates_no_tables(rail_home):
    """⚠️ The read that happens on EVERY session open, on stores that will never probe. A single
    `count_probe_atoms` once created nine tables (three of ours plus FTS5's six shadow tables) just
    by asking a question — this is the same trap with far more traffic through it."""
    conn = schema.connect()
    try:
        assert probe_store.probed_today(conn) == 0
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert not [t for t in tables if t.startswith("probe_")]


def test_the_meter_counts_only_today(rail_home):
    _spend(4)
    _spend(9, day_offset=3)
    conn = schema.connect()
    try:
        assert probe_store.probed_today(conn) == 4
    finally:
        conn.close()


# ── the CLI ─────────────────────────────────────────────────────────────────────
def test_a_successful_run_exits_zero(rail_home, probe):
    assert pc.main(["--once"]) == 0
    assert probe == [pc.PROBE_DAILY_CANDIDATES]


def test_the_ceiling_flag_reaches_the_run(rail_home, probe):
    assert pc.main(["--once", "--daily-ceiling", "5"]) == 0
    assert probe == [5]


def test_a_refused_run_still_exits_zero(rail_home, probe):
    """A spent ceiling is the rail working, not failing. Exiting 1 would make a healthy day look
    broken to anyone tailing the log."""
    _spend(pc.PROBE_DAILY_CANDIDATES)
    assert pc.main(["--once"]) == 0
    assert probe == []


def test_bare_invocation_prints_help_and_exits_two(rail_home, capsys):
    assert pc.main([]) == 2
