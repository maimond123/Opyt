"""Frontier stage 2 — EXECUTE, proven offline against fake adapters.

The contract, in the order it matters:

  • NOTHING REACHES `atoms`. Stage 2 stages candidates; stage 3 admits. That separation is the
    entire license for stage 1 generating queries agentically, so it is asserted directly.
  • A FAILED PULL NEVER STAMPS. Stamping on error buys a full TTL of silence on that pair for one
    bad night, with nothing recorded to say why it went quiet.
  • THE WATERMARK ONLY MOVES FORWARD, and a second run with nothing new upstream costs nothing.
  • NOTHING IS SILENTLY DROPPED — an unknown source and a budget-deferred pair are both counted.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pipeline.kb import frontier_execute as fe
from pipeline.kb import frontier_queries as fq
from pipeline.kb import schema
from pipeline.kb.frontier_sources import Candidate, SourceError

_NOW = datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _query(conn, text="gated DeltaNet attention", sources=("arxiv",)):
    fq.upsert_queries(conn, [{"text": text, "target_sources": list(sources),
                              "atom_ids": ["x:1"], "rationale": "r"}],
                      generator="bookmark-reader")
    return fq.query_id_for(fq.normalize(text))


def _cand(cid="arxiv:2501.00001", published="2026-08-09", source="arxiv", kind="paper"):
    return Candidate(candidate_id=cid, source=source, kind=kind, title=f"paper {cid}",
                     url=f"https://example/{cid}", published=published, summary="s")


class FakeAdapter:
    """Records the `since` it was handed — the windowing is the thing most likely to silently
    regress, because a dropped `since` looks identical to a working pull."""
    def __init__(self, slug="arxiv", results=None, error=None):
        self.slug, self._results, self._error = slug, results, error
        self.calls: list[tuple[str, datetime | None]] = []

    def search(self, query, *, since, limit=25):
        self.calls.append((query, since))
        if self._error:
            raise self._error
        return list(self._results if self._results is not None else [_cand()])


# ── the boundary that licenses the whole rail ───────────────────────────────────
def test_execution_never_writes_an_atom(conn):
    """Stage 2 stages; stage 3 admits. A bad query must cost inbox noise, never KB pollution."""
    _query(conn)
    before = conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
    res = fe.run_frontier_execute(conn, registry={"arxiv": FakeAdapter()}, now=_NOW)
    assert res["status"] == "ok" and res["candidates_new"] == 1
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == before == 0
    assert conn.execute(
        "SELECT status FROM frontier_candidates").fetchone()[0] == "new"   # never 'admitted'


# ── the watermark ───────────────────────────────────────────────────────────────
def test_a_second_run_inside_the_ttl_pulls_nothing(conn):
    _query(conn)
    a = FakeAdapter()
    fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW)
    fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW + timedelta(hours=2))
    assert len(a.calls) == 1                                    # still inside the 24h arXiv TTL


def test_the_pair_comes_due_again_after_its_ttl(conn):
    _query(conn)
    a = FakeAdapter()
    fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW)
    fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW + timedelta(hours=30))
    assert len(a.calls) == 2


def test_the_second_pull_resumes_from_the_watermark_with_an_overlap(conn):
    """A hard boundary at the last pull time drops anything published while the previous request
    was in flight, and that gap is invisible — nothing reports the paper you never saw."""
    _query(conn)
    a = FakeAdapter()
    fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW)
    fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW + timedelta(hours=30))
    second_since = a.calls[1][1]
    assert second_since == _NOW - timedelta(hours=fe.OVERLAP_HOURS)


def test_a_never_pulled_pair_looks_back_a_bounded_window(conn):
    """Not to the beginning of time: a first pull that re-reads all of arXiv is indistinguishable
    from a working one until you look at the volume."""
    _query(conn)
    a = FakeAdapter()
    fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW)
    assert a.calls[0][1] == _NOW - timedelta(days=fe.FIRST_PULL_DAYS)


def test_the_cursor_never_rewinds(conn):
    qid = _query(conn)
    fe.record_pull(conn, qid, "arxiv", last_status="ok", cursor_ts="2026-08-09", now=_NOW)
    fe.record_pull(conn, qid, "arxiv", last_status="ok", cursor_ts="2026-01-01", now=_NOW)
    assert fe.get_pair(conn, qid, "arxiv")["cursor_ts"] == "2026-08-09"


# ── failure must not stamp ──────────────────────────────────────────────────────
def test_a_failed_pull_leaves_the_watermark_untouched(conn):
    """Otherwise one bad night buys a full TTL of silence on that pair."""
    qid = _query(conn)
    bad = FakeAdapter(error=SourceError("upstream 503"))
    res = fe.run_frontier_execute(conn, registry={"arxiv": bad}, now=_NOW)
    row = fe.get_pair(conn, qid, "arxiv")
    assert row["last_pulled_at"] is None and row["last_status"] == "error"
    assert row["error_count"] == 1
    assert res["status"] == "ok" and res.get("pairs_pulled", 0) == 0
    # ...and because it never stamped, the very next run retries instead of waiting out the TTL.
    good = FakeAdapter()
    fe.run_frontier_execute(conn, registry={"arxiv": good}, now=_NOW + timedelta(minutes=1))
    assert len(good.calls) == 1


def test_an_adapter_bug_is_contained_to_its_own_pair(conn):
    """A crash in one source must not take the run down or block the others."""
    _query(conn, "q one", sources=("arxiv", "github"))
    boom = FakeAdapter(slug="arxiv", error=ValueError("adapter bug"))
    fine = FakeAdapter(slug="github", results=[_cand("repo:a/b", source="github")])
    res = fe.run_frontier_execute(conn, registry={"arxiv": boom, "github": fine}, now=_NOW)
    assert res["status"] == "ok" and res["candidates_new"] == 1
    assert len(fine.calls) == 1


def test_an_absurd_window_is_refused_before_the_request(conn):
    """The realistic failure is a threading bug, not volume: if `since` never reaches the adapter
    the source applies its own default and every pull becomes a full-history re-read."""
    qid = _query(conn)
    fe.record_pull(conn, qid, "arxiv", last_status="ok", now=_NOW - timedelta(days=400))
    a = FakeAdapter()
    fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW)
    assert a.calls == []
    assert fe.get_pair(conn, qid, "arxiv")["last_status"] == "window_refused"


# ── nothing silently dropped ────────────────────────────────────────────────────
def test_an_unregistered_source_is_counted_not_dropped(conn):
    _query(conn, "bio thing", sources=("biorxiv",))
    res = fe.run_frontier_execute(conn, registry={"arxiv": FakeAdapter()}, now=_NOW)
    assert "no adapter" in (res.get("reason") or "")
    assert fe.get_pair(conn, fq.query_id_for("bio thing"), "biorxiv")["last_status"] == "no_adapter"


def test_the_request_budget_defers_rather_than_truncates(conn, monkeypatch):
    monkeypatch.setattr(fe, "MAX_REQUESTS_PER_RUN", 2)
    for i in range(5):
        _query(conn, f"query number {i}")
    res = fe.run_frontier_execute(conn, registry={"arxiv": FakeAdapter()}, now=_NOW)
    assert res["pairs_due"] == 5 and res["requests"] == 2 and res["pairs_deferred"] == 3


def test_a_free_outcome_does_not_consume_a_request_slot(conn, monkeypatch):
    """A refused window costs nothing, so it must not starve a pair that could have run."""
    monkeypatch.setattr(fe, "MAX_REQUESTS_PER_RUN", 1)
    stale = _query(conn, "stale one")
    fe.record_pull(conn, stale, "arxiv", last_status="ok", now=_NOW - timedelta(days=400))
    _query(conn, "healthy one")
    a = FakeAdapter()
    fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW)
    assert [c[0] for c in a.calls] == ["healthy one"]           # the refused pair did not eat it


# ── candidates + provenance ─────────────────────────────────────────────────────
def test_the_same_artifact_from_two_queries_is_one_row_with_two_links(conn):
    """Multi-query hits are the signal stage 4 needs; one `query_id` column would keep only the
    query that happened to run first."""
    _query(conn, "query a")
    _query(conn, "query b")
    fe.run_frontier_execute(conn, registry={"arxiv": FakeAdapter()}, now=_NOW)
    assert conn.execute("SELECT COUNT(*) FROM frontier_candidates").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM frontier_candidate_queries").fetchone()[0] == 2


# ── the relevance intake cut (RULED 2026-08-27) ─────────────────────────────────
def _scored(cid, score):
    return Candidate(candidate_id=cid, source="openalex", kind="paper", title=cid,
                     url=f"https://example/{cid}", published="2026-08-09", summary="s",
                     payload={"relevance_score": score})


def test_the_intake_cut_drops_the_tail_below_the_fraction_of_top(conn):
    """The measured OpenAlex page shape: a head, a cliff, then a plateau of roughly-equal,
    roughly-irrelevant matches. At `INTAKE_KEEP_FRAC` of the page's OWN top the plateau goes and
    the head stays. The cut is ingest-time and one-way, so a dropped work leaves NO candidate row
    and NO link row — it never existed as far as stage 3 can see."""
    _query(conn, sources=("openalex",))
    page = [_scored(f"openalex:W{i}", s) for i, s in enumerate([34, 31, 15, 5, 5, 4])]
    res = fe.run_frontier_execute(conn, registry={"openalex": FakeAdapter("openalex", page)},
                                  now=_NOW)
    assert res["candidates_new"] == 2
    kept = {r[0] for r in conn.execute("SELECT candidate_id FROM frontier_candidates")}
    assert kept == {"openalex:W0", "openalex:W1"}
    links = {r[0] for r in conn.execute("SELECT candidate_id FROM frontier_candidate_queries")}
    assert links == kept, "a dropped work must not leave a link row either"


def test_a_scoreless_page_passes_through_uncut(conn):
    """arXiv is date-sorted by deliberate design and GitHub star-sorted; neither emits a
    relevance score, so the cut must not touch them. A missing score means NOT COMPARABLE,
    never zero — treating it as zero would drop every scoreless candidate on a scored page."""
    _query(conn)
    page = [_cand(f"arxiv:2501.0000{i}") for i in range(3)]
    res = fe.run_frontier_execute(conn, registry={"arxiv": FakeAdapter(results=page)}, now=_NOW)
    assert res["candidates_new"] == 3


def test_a_uniformly_scored_page_is_not_cut(conn):
    """The known limit, pinned so it reads as designed rather than broken: a fraction-of-top rule
    cannot see a page that is uniformly mediocre (measured live: top 6.80 / min 4.28 keeps all
    25). Only an absolute floor could, and that was ruled out — the score is BM25-shaped, so no
    absolute number means the same thing on two queries."""
    _query(conn, sources=("openalex",))
    page = [_scored(f"openalex:W{i}", s) for i, s in enumerate([6.8, 5.5, 4.9, 4.3])]
    res = fe.run_frontier_execute(conn, registry={"openalex": FakeAdapter("openalex", page)},
                                  now=_NOW)
    assert res["candidates_new"] == 4


def test_reseeing_a_candidate_updates_last_seen_and_counts_as_seen(conn):
    _query(conn)
    a = FakeAdapter()
    r1 = fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW)
    r2 = fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW + timedelta(hours=30))
    assert r1["candidates_new"] == 1
    assert r2.get("candidates_new", 0) == 0 and r2["candidates_seen"] == 1
    row = conn.execute("SELECT first_seen_at, last_seen_at FROM frontier_candidates").fetchone()
    assert row["last_seen_at"] > row["first_seen_at"]


def test_dry_run_searches_but_writes_no_candidates(conn):
    _query(conn)
    res = fe.run_frontier_execute(conn, registry={"arxiv": FakeAdapter()}, dry_run=True, now=_NOW)
    assert res["status"] == "dry-run"
    assert conn.execute("SELECT COUNT(*) FROM frontier_candidates").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM frontier_exec_runs").fetchone()[0] == 0


def test_a_hand_retired_query_is_not_executed(conn):
    """Retirement has to mean something. It is also the ONLY thing that stops a query costing
    requests — a demoted one keeps running, forever, just slower."""
    _query(conn, "done with this")
    fq.retire_query(conn, "done with this")
    a = FakeAdapter()
    res = fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW)
    assert a.calls == [] and res["status"] == "skipped"


def test_no_active_queries_is_a_skip_not_a_failure(conn):
    assert fe.run_frontier_execute(conn, registry={}, now=_NOW)["status"] == "skipped"


# ── decay tiers: a dropped query slows down, it never stops ─────────────────────
@pytest.mark.parametrize("misses, mult", [
    (0, 1.0), (1, 1.0), (2, 1.0),        # daily on arXiv
    (3, 7.0), (5, 7.0), (9, 7.0),        # weekly
    (10, 30.0), (11, 30.0),              # monthly...
    (500, 30.0),                         # ...and the floor is HARD, never slower
])
def test_the_tier_steps_and_then_floors(misses, mult):
    """A tier that kept doubling would reach "never" and quietly reinvent the deletion this
    replaced. Steps rather than a curve so a surfacing months later is explainable."""
    assert fe.tier_for(misses) == mult


def test_a_demoted_query_still_comes_due_at_its_stretched_interval(conn):
    """The whole design rests on this: demotion is a SPEED, not an exit. A dropped query that is
    never asked again cannot ever surface the thing that would revive it."""
    qid = _query(conn, "quiet thread")
    conn.execute("UPDATE frontier_queries SET miss_count=12")     # deep in the monthly tier
    conn.commit()
    fe.record_pull(conn, qid, "arxiv", last_status="ok", now=_NOW)
    row = fe.get_pair(conn, qid, "arxiv")
    # Still silent a week later — where an undemoted query would already have run twice...
    assert fe.is_due(row, "arxiv", query_id=qid, miss_count=12,
                     now=_NOW + timedelta(days=7)) is False
    assert fe.is_due(row, "arxiv", query_id=qid, miss_count=0,
                     now=_NOW + timedelta(days=7)) is True
    # ...and back on its own beat a month later. It never stops.
    assert fe.is_due(row, "arxiv", query_id=qid, miss_count=12,
                     now=_NOW + timedelta(days=35)) is True


def test_the_tier_multiplies_the_sources_own_ttl(conn):
    """Decay stretches a source's cadence rather than overriding it: GitHub's 48h beat stays twice
    arXiv's at every tier, because how fast an upstream publishes is a fact about the upstream."""
    qid = _query(conn, "a thread", sources=("github",))
    fe.record_pull(conn, qid, "github", last_status="ok", now=_NOW)
    row = fe.get_pair(conn, qid, "github")
    assert fe.is_due(row, "github", query_id=qid, miss_count=4,
                     now=_NOW + timedelta(days=13)) is False      # 7 x 48h = 14 days
    assert fe.is_due(row, "github", query_id=qid, miss_count=4,
                     now=_NOW + timedelta(days=16)) is True


def test_the_run_reads_each_querys_own_miss_count(conn):
    """Wiring test: the tier is useless if `_run` never passes the counter it is keyed on."""
    hot = _query(conn, "hot thread")
    cold = _query(conn, "cold thread")
    for qid in (hot, cold):
        fe.record_pull(conn, qid, "arxiv", last_status="ok", now=_NOW)
    conn.execute("UPDATE frontier_queries SET miss_count=15 WHERE query_id=?", (cold,))
    conn.commit()
    a = FakeAdapter()
    fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW + timedelta(days=2))
    assert [c[0] for c in a.calls] == ["hot thread"]              # the cold one is not due yet


# ── fairness: the request budget delays, it must not starve ────────────────────
def test_the_budget_serves_the_stalest_pair_first(conn, monkeypatch):
    """`active_queries()` order is STABLE, so once demand exceeds the budget the same head wins
    every run and the tail is deferred forever — silently. Ordering by real staleness makes the
    budget a delay for everyone instead of a permanent cut for some."""
    monkeypatch.setattr(fe, "MAX_REQUESTS_PER_RUN", 1)
    fresh = _query(conn, "recently pulled")
    stale = _query(conn, "long neglected")
    fe.record_pull(conn, fresh, "arxiv", last_status="ok", now=_NOW - timedelta(days=2))
    fe.record_pull(conn, stale, "arxiv", last_status="ok", now=_NOW - timedelta(days=30))
    a = FakeAdapter()
    fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW)
    assert [c[0] for c in a.calls] == ["long neglected"]


def test_a_never_pulled_pair_outranks_every_pulled_one(conn, monkeypatch):
    """Never-pulled is infinitely stale — a brand-new query must not sit behind the whole
    backlog waiting for its first look."""
    monkeypatch.setattr(fe, "MAX_REQUESTS_PER_RUN", 1)
    old = _query(conn, "pulled long ago")
    fe.record_pull(conn, old, "arxiv", last_status="ok", now=_NOW - timedelta(days=30))
    _query(conn, "brand new")
    a = FakeAdapter()
    fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW)
    assert [c[0] for c in a.calls] == ["brand new"]


# ── the scheduler ───────────────────────────────────────────────────────────────
@pytest.fixture()
def spawn_env(tmp_path, monkeypatch):
    """Point the stamp and log at a tmp dir, and never actually fork."""
    monkeypatch.delenv("OPYT_NO_FRONTIER_EXEC", raising=False)
    monkeypatch.setenv("OPYT_FRONTIER_EXEC_STAMP", str(tmp_path / "stamp"))
    monkeypatch.setenv("OPYT_FRONTIER_EXEC_LOG", str(tmp_path / "exec.log"))
    calls = []
    monkeypatch.setattr(fe.subprocess, "Popen", lambda cmd, **kw: calls.append((cmd, kw)))
    return calls


def test_the_spawn_is_detached_and_never_writes_to_stdout(spawn_env):
    """The server's stdout IS the JSON-RPC channel, so an inherited handle from a child would
    corrupt the protocol. It also has to outlive the session that started it."""
    assert fe.spawn_frontier_execute() is True
    cmd, kw = spawn_env[0]
    assert cmd[1:] == ["-m", "pipeline.kb.frontier_execute", "--once"]
    assert kw["stdout"] is kw["stderr"] and kw["stdout"] is not None   # to the log, not inherited
    assert kw["stdin"] == fe.subprocess.DEVNULL
    assert kw["start_new_session"] is True
    assert (Path(kw["cwd"]) / "pipeline" / "kb").is_dir()             # repo root, so -m resolves


def test_the_coalesce_window_stops_every_session_firing_a_pass(spawn_env):
    assert fe.spawn_frontier_execute() is True
    assert fe.spawn_frontier_execute() is False                       # inside the window
    assert fe.spawn_frontier_execute(force=True) is True              # ...but force gets through
    assert len(spawn_env) == 2


def test_the_kill_switch_stops_the_rail_without_touching_stage_one(spawn_env, monkeypatch):
    """Each rail owns its own switch: stage 1 and stage 2 fail for different reasons, so either
    must be disableable alone."""
    monkeypatch.setenv("OPYT_NO_FRONTIER_EXEC", "1")
    assert fe.spawn_frontier_execute(force=True) is False
    assert spawn_env == []


def test_a_broken_spawn_is_swallowed_rather_than_raised(spawn_env, monkeypatch):
    """It is called from the MCP server's startup path. A scheduler hiccup must never stop the
    server serving."""
    def boom(*a, **kw):
        raise OSError("no fork for you")
    monkeypatch.setattr(fe.subprocess, "Popen", boom)
    assert fe.spawn_frontier_execute() is False


# ── politeness ──────────────────────────────────────────────────────────────────
def test_a_rate_limited_source_is_paced_between_requests(conn):
    """arXiv asks for one request every three seconds and enforces it — a live run firing 22 in
    twelve seconds took two 429s. A rail that reliably loses a slice of every pass to self-inflicted
    throttling is just quietly incomplete."""
    for i in range(3):
        _query(conn, f"query {i}")
    slept = []
    slow = FakeAdapter()
    slow.min_interval_s = 3.0
    fe.run_frontier_execute(conn, registry={"arxiv": slow}, now=_NOW, sleep=slept.append)
    assert len(slow.calls) == 3
    assert len(slept) == 2                       # paced between calls, never before the first
    assert all(0 < s <= 3.0 for s in slept)


def test_a_fast_source_is_not_paced(conn):
    for i in range(3):
        _query(conn, f"query {i}", sources=("github",))
    slept = []
    fast = FakeAdapter(slug="github", results=[_cand("repo:a/b", source="github")])
    fast.min_interval_s = 0.0
    fe.run_frontier_execute(conn, registry={"github": fast}, now=_NOW, sleep=slept.append)
    assert slept == []


# ── jitter ──────────────────────────────────────────────────────────────────────
def test_pairs_are_jittered_so_a_batch_does_not_fall_due_together():
    """28 queries seeded on the same day would otherwise come due in the same second forever."""
    vals = {fe._jitter(f"q{i}", "arxiv") for i in range(50)}
    assert len(vals) > 40                                        # spread, not constant
    assert all(1 - fe.JITTER <= v <= 1 + fe.JITTER for v in vals)
    assert fe._jitter("q1", "arxiv") == fe._jitter("q1", "arxiv")     # stable, never random


def test_an_unavailable_source_costs_neither_a_request_slot_nor_a_sleep(conn):
    """A source whose breaker is open costs nothing, so it must not consume budget or make the run
    wait out a politeness delay for a call that never leaves the process — measured at 19 x 3s of
    pure sleeping on a blocked arXiv."""
    for i in range(3):
        _query(conn, f"query {i}")
    slept = []
    down = FakeAdapter()
    down.min_interval_s = 3.0
    down.available = lambda: False
    res = fe.run_frontier_execute(conn, registry={"arxiv": down}, now=_NOW, sleep=slept.append)
    assert down.calls == [] and slept == []
    assert res["requests"] == 0 and res.get("pairs_pulled", 0) == 0
    assert fe.get_pair(conn, fq.query_id_for("query 0"), "arxiv")["last_status"] == "breaker_open"
    assert fe.get_pair(conn, fq.query_id_for("query 0"), "arxiv")["last_pulled_at"] is None


def test_an_adapter_without_an_available_probe_is_always_asked(conn):
    """A new adapter should stay a one-method object."""
    _query(conn)
    plain = FakeAdapter()
    assert not hasattr(plain, "available")
    fe.run_frontier_execute(conn, registry={"arxiv": plain}, now=_NOW, sleep=lambda s: None)
    assert len(plain.calls) == 1


# ── The declared look-back ──────────────────────────────────────────────────────
class _WideAdapter(FakeAdapter):
    """An adapter that cannot window on its own index date and says so (OpenAlex's case)."""
    min_lookback_days = 30


def test_a_declared_lookback_widens_the_window_past_the_cursor(conn):
    """The gap this closes is invisible by construction: a source that windows on PUBLICATION date
    while indexing LATE never reports the record it skipped. Measured on OpenAlex over 400 works,
    8-9% are indexed more than 7 days after publication — every one of those is published before a
    cursor-width window opens and indexed after it closes, so a plain cursor misses them forever.
    `OVERLAP_HOURS` is the wrong dial: it is hours, and the gap is weeks.
    """
    _query(conn, sources=("arxiv",))
    a = _WideAdapter(results=[_cand()])
    fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW)

    _, since = a.calls[0]
    assert since <= _NOW - timedelta(days=30)


def test_an_adapter_without_one_still_gets_exactly_the_cursor(conn):
    """The widening is opt-in per adapter. A source whose window means what the loop thinks it
    means must be untouched, or every source silently starts re-reading a month of history."""
    _query(conn, sources=("arxiv",))
    a = FakeAdapter(results=[_cand()])
    fe.run_frontier_execute(conn, registry={"arxiv": a}, now=_NOW)

    _, since = a.calls[0]
    assert since >= _NOW - timedelta(days=fe.FIRST_PULL_DAYS + 1)


def test_the_window_assertion_sees_the_widened_window_not_the_cursor():
    """Ordering, asserted directly. `window_ok` exists to catch a `since` that failed to reach the
    adapter; validating a 30-hour window and then sending a 30-day one would defeat it. So the
    floor is applied BEFORE the check, and a look-back past the ceiling is refused rather than
    quietly sent."""
    absurd = type("A", (), {"min_lookback_days": fe.MAX_WINDOW_DAYS + 10})()
    widened = fe._lookback_floor(absurd, _NOW - timedelta(hours=1), _NOW)
    assert not fe.window_ok(widened, _NOW)


def test_the_kind_that_decides_the_minter_is_persisted(conn):
    """Stage 3 dispatches on `kind` and reads it off the row, so a candidate staged without one is
    an artifact no minter will ever claim."""
    _query(conn, sources=("arxiv",))
    fe.run_frontier_execute(conn, registry={"arxiv": FakeAdapter(results=[_cand()])}, now=_NOW)

    assert conn.execute("SELECT kind FROM frontier_candidates").fetchone()[0] == "paper"
