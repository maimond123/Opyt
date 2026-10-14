"""Frontier stage 2 — the searches source (`pipeline/frontier/searching.py`), proven offline
against fake adapters.

The contract, in the order it matters:

  • NOTHING REACHES `atoms`. Stage 2 stages candidates; stage 3 admits. That separation is the
    entire license for stage 1 generating queries agentically, so it is asserted directly.
  • A FAILED PULL NEVER STAMPS. Stamping on error buys a full TTL of silence on that pair for one
    bad night, with nothing recorded to say why it went quiet. It waits an hour, then retries.
  • THE WATERMARK ONLY MOVES FORWARD, and a second look with nothing new upstream costs nothing.
  • NOTHING IS SILENTLY DROPPED — an unknown source is recorded and counted, and a pair the
    hour's allowance holds back is still due, with `next_due` saying when it frees.
  • NOTHING WAITS IN A SLOT. The allowance, the courtesy gap and the breaker are decided before a
    unit is offered, never by sleeping inside one.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from pipeline.platforms.github import search as github_client
from pipeline.frontier import allowance, cadence, candidates, github_candidates, pulls, searching
from pipeline.frontier import queries as fq
from pipeline.frontier.candidates import Candidate
from pipeline.frontier.searching import Searches
from pipeline.platforms.errors import SourceError
from pipeline.kb import schema
from pipeline.worker.quota_loop.retries import PLATFORM_PAUSES_S, Outages
from pipeline.worker.quota_loop.source import Changed, Context

_NOW = datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)
_HOSTS = {"arxiv": "export.arxiv.org", "github": "api.github.com",
          "openalex": "api.openalex.org"}


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _query(conn, text="gated DeltaNet attention", sources=("arxiv",), *, status="active"):
    """A standing query, watched by a topic of its own in `status`."""
    qid = fq.query_id_for(fq.normalize(text))
    fq.add(conn, text, list(sources), now="2026-08-01T00:00:00Z")
    _topic(conn, f"t-{qid}", qid, status=status)
    return qid


def _topic(conn, topic_id, *query_ids, status="active"):
    conn.execute("INSERT INTO user_topics (topic_id, name, status, created_at, updated_at) "
                 "VALUES (?, 't', ?, 'a', 'a')", (topic_id, status))
    conn.executemany("INSERT INTO topic_watches (topic_id, query_id, name, added_at) "
                     "VALUES (?, ?, 'w', 'a')", [(topic_id, q) for q in query_ids])
    conn.commit()


def _cand(cid="arxiv:2501.00001", published="2026-08-09", source="arxiv", kind="paper"):
    return Candidate(candidate_id=cid, source=source, kind=kind, title=f"paper {cid}",
                     url=f"https://example/{cid}", published=published, summary="s")


class FakeAdapter:
    """Records the `since` it was handed — the windowing is the thing most likely to silently
    regress, because a dropped `since` looks identical to a working pull. Like a real source, what
    it returns names the query it was asked, unless `naming=False`."""
    def __init__(self, slug="arxiv", results=None, error=None, naming=True):
        self.slug, self._results, self._error, self._naming = slug, results, error, naming
        self.host = _HOSTS.get(slug, slug)
        self.calls: list[tuple[str, datetime | None]] = []
        self.contexts: list[list[str]] = []

    def search(self, query, *, since, now, limit=25, context=()):
        assert isinstance(now, datetime)
        self.calls.append((query, since))
        self.contexts.append(list(context))
        if self._error:
            raise self._error
        found = list(self._results if self._results is not None else [_cand()])
        return [replace(c, summary=f"{c.summary} {query}") for c in found] if self._naming else found


def _source(registry, now=_NOW) -> Searches:
    return Searches(Outages(PLATFORM_PAUSES_S), registry=registry, clock=lambda: now)


def _run(conn, registry, now=_NOW, *, source=None) -> list[str]:
    """Every unit the source offers at `now`, run one at a time as the web pool runs them, until
    it offers none. What each came to."""
    src = source or _source(registry, now)
    out = []
    for _ in range(100):                             # a source that never stops offering fails
        unit = next(iter(src.candidates(conn, now)), None)
        if unit is None:
            return out
        out.append(src.record(conn, unit, src.fetch(conn, unit, Context("web"))).status)
    raise AssertionError("the source kept offering units")


def _staged(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM frontier_candidates").fetchone()[0]


# ── the boundary that licenses the whole rail ───────────────────────────────────
def test_searching_never_writes_an_atom(conn):
    """Stage 2 stages; stage 3 admits. A bad query must cost inbox noise, never KB pollution."""
    _query(conn)
    before = conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
    assert _run(conn, {"arxiv": FakeAdapter()}) == ["ok"]
    assert _staged(conn) == 1
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == before == 0
    assert conn.execute(
        "SELECT status FROM frontier_candidates").fetchone()[0] == "new"   # never 'admitted'


# ── the watermark ───────────────────────────────────────────────────────────────
def test_a_second_look_inside_the_ttl_pulls_nothing(conn):
    _query(conn)
    a = FakeAdapter()
    _run(conn, {"arxiv": a})
    _run(conn, {"arxiv": a}, _NOW + timedelta(hours=2))
    assert len(a.calls) == 1                                    # still inside the 24h arXiv TTL


def test_the_pair_comes_due_again_after_its_ttl(conn):
    _query(conn)
    a = FakeAdapter()
    _run(conn, {"arxiv": a})
    _run(conn, {"arxiv": a}, _NOW + timedelta(hours=30))
    assert len(a.calls) == 2


def test_the_second_pull_resumes_from_the_watermark_with_an_overlap(conn):
    """A hard boundary at the last pull time drops anything published while the previous request
    was in flight, and that gap is invisible — nothing reports the paper you never saw."""
    _query(conn)
    a = FakeAdapter()
    _run(conn, {"arxiv": a})
    _run(conn, {"arxiv": a}, _NOW + timedelta(hours=30))
    second_since = a.calls[1][1]
    assert second_since == _NOW - timedelta(hours=cadence.OVERLAP_HOURS)


def test_a_never_pulled_pair_looks_back_a_bounded_window(conn):
    """Not to the beginning of time: a first pull that re-reads all of arXiv is indistinguishable
    from a working one until you look at the volume."""
    _query(conn)
    a = FakeAdapter()
    _run(conn, {"arxiv": a})
    assert a.calls[0][1] == _NOW - timedelta(days=cadence.FIRST_PULL_DAYS)


# ── failure must not stamp ──────────────────────────────────────────────────────
def test_a_failed_pull_leaves_the_watermark_untouched_and_retries_an_hour_later(conn):
    """Otherwise one bad night buys a full TTL of silence on that pair. The hourly pass retried it
    at the next pass; the source retries it an hour after the try, never sooner."""
    qid = _query(conn)
    bad = FakeAdapter(error=SourceError("upstream 503"))
    assert _run(conn, {"arxiv": bad}) == ["error"]
    row = pulls.get_pair(conn, qid, "arxiv")
    assert row["last_pulled_at"] is None and row["last_status"] == "error"

    good = FakeAdapter()
    _run(conn, {"arxiv": good}, _NOW + timedelta(minutes=1))
    assert good.calls == []                                     # no hot retry
    _run(conn, {"arxiv": good}, _NOW + cadence.RETRY_AFTER)
    assert len(good.calls) == 1


def test_an_adapter_bug_is_contained_to_its_own_pair(conn):
    """A crash in one source must not take the others down."""
    _query(conn, "q one", sources=("arxiv", "github"))
    boom = FakeAdapter(slug="arxiv", error=ValueError("adapter bug"))
    fine = FakeAdapter(slug="github", results=[_cand("repo:a/b", source="github")])
    assert sorted(_run(conn, {"arxiv": boom, "github": fine})) == ["error", "ok"]
    assert _staged(conn) == 1 and len(fine.calls) == 1


def test_an_absurd_window_is_refused_before_the_request(conn):
    """The realistic failure is a threading bug, not volume: if `since` never reaches the adapter
    the source applies its own default and every pull becomes a full-history re-read."""
    qid = _query(conn)
    pulls.record_pull(conn, qid, "arxiv", last_status="ok", now=_NOW - timedelta(days=400))
    a = FakeAdapter()
    assert _run(conn, {"arxiv": a}) == ["window_refused"]
    assert a.calls == []
    assert pulls.get_pair(conn, qid, "arxiv")["last_status"] == "window_refused"


def test_a_lost_lease_writes_neither_the_page_nor_the_stamp(conn):
    qid = _query(conn)
    src = _source({"arxiv": FakeAdapter()})
    unit = src.candidates(conn, _NOW)[0]
    with pytest.raises(Changed):
        src.fetch(conn, unit, Context("web", lease_lost=lambda: True))
    assert _staged(conn) == 0 and pulls.get_pair(conn, qid, "arxiv") is None


# ── nothing silently dropped ────────────────────────────────────────────────────
def test_an_unregistered_source_is_recorded_once_and_never_pulled(conn):
    _query(conn, "bio thing", sources=("biorxiv",))
    assert _run(conn, {"arxiv": FakeAdapter()}) == ["no_adapter"]
    assert pulls.get_pair(conn, fq.query_id_for("bio thing"), "biorxiv")["last_status"] == "no_adapter"
    assert _run(conn, {"arxiv": FakeAdapter()}, _NOW + timedelta(days=3)) == []


def test_the_hours_allowance_holds_pairs_back_rather_than_truncating(conn, monkeypatch):
    """The pairs it holds back are still due, and `next_due` says when the hour frees."""
    monkeypatch.setattr(allowance, "SEARCHES_PER_HOUR", 2)
    for i in range(5):
        _query(conn, f"query number {i}")
    a = FakeAdapter()
    src = _source({"arxiv": a})
    assert _run(conn, {}, source=src) == ["ok", "ok"]
    assert len(a.calls) == 2
    assert src.next_due(conn, _NOW) == _NOW + allowance.HOUR
    later = _source({"arxiv": a}, _NOW + allowance.HOUR)
    assert _run(conn, {}, _NOW + allowance.HOUR, source=later) == ["ok", "ok"]


def test_a_free_outcome_does_not_consume_the_allowance(conn, monkeypatch):
    """A refused window costs nothing, so it must not starve a pair that could have run."""
    monkeypatch.setattr(allowance, "SEARCHES_PER_HOUR", 1)
    stale = _query(conn, "stale one")
    pulls.record_pull(conn, stale, "arxiv", last_status="ok", now=_NOW - timedelta(days=400))
    _query(conn, "healthy one")
    a = FakeAdapter()
    _run(conn, {"arxiv": a})
    assert [c[0] for c in a.calls] == ["healthy one"]           # the refused pair did not eat it


# ── candidates + provenance ─────────────────────────────────────────────────────
def test_the_same_artifact_from_two_queries_is_one_row_with_two_links(conn):
    """Multi-query hits are the signal stage 4 needs; one `query_id` column would keep only the
    query that happened to run first."""
    _query(conn, "query a")
    _query(conn, "query b")
    _run(conn, {"arxiv": FakeAdapter()})
    assert _staged(conn) == 1
    assert conn.execute("SELECT COUNT(*) FROM frontier_candidate_queries").fetchone()[0] == 2


# ── the relevance intake cut ─────────────────────────────────
def test_a_find_that_does_not_name_the_watch_leaves_no_row(conn):
    """The name check is ingest-time and one-way: a dropped find leaves NO candidate row and NO
    link row, so it never existed as far as the host's check can see."""
    _query(conn, "OpenFold3", sources=("openalex",))
    page = [_cand("openalex:W1", source="openalex"), _cand("openalex:W2", source="openalex")]
    page[0] = replace(page[0], title="OpenFold3 reproduces AlphaFold3")
    _run(conn, {"openalex": FakeAdapter("openalex", page, naming=False)})
    assert {r[0] for r in conn.execute("SELECT candidate_id FROM frontier_candidates")} == {
        "openalex:W1"}
    assert {r[0] for r in conn.execute("SELECT candidate_id FROM frontier_candidate_queries")} == {
        "openalex:W1"}


def test_reseeing_a_candidate_updates_last_seen_and_is_not_new(conn):
    qid = _query(conn)
    pair = cadence.DuePair(conn.execute("SELECT * FROM frontier_queries").fetchone(), "arxiv",
                           None)
    found = searching.Pulled("ok", [_cand()])
    assert searching.store(conn, pair, found, _NOW) == 1
    assert searching.store(conn, pair, found, _NOW + timedelta(hours=30)) == 0
    row = conn.execute("SELECT first_seen_at, last_seen_at FROM frontier_candidates").fetchone()
    assert row["last_seen_at"] > row["first_seen_at"]
    assert pulls.get_pair(conn, qid, "arxiv")["last_status"] == "ok"


def test_a_query_no_followed_topic_owns_is_not_searched(conn):
    """A dropped topic's watches, and a watch its topic dropped, stop costing requests."""
    _query(conn, "done with this", status="dropped")
    dropped = _query(conn, "dropped")
    conn.execute("DELETE FROM topic_watches WHERE query_id = ?", (dropped,))
    conn.commit()
    a = FakeAdapter()
    assert _run(conn, {"arxiv": a}) == [] and a.calls == []


def test_a_query_runs_while_any_followed_topic_owns_it(conn):
    """Two topics can watch the same name: one dropped leaves the other's watch running."""
    qid = _query(conn, "shared thread", status="dropped")
    _topic(conn, "t-other", qid)
    a = FakeAdapter()
    _run(conn, {"arxiv": a})
    assert [c[0] for c in a.calls] == ["shared thread"]


def test_no_active_queries_offers_nothing_and_owes_nothing(conn):
    src = _source({"arxiv": FakeAdapter()})
    assert src.candidates(conn, _NOW) == []
    assert src.next_due(conn, _NOW) is None


# ── cadence: every watch runs once a day on every source ──────────────────────
def test_a_watch_comes_due_again_a_day_later_on_every_source(conn):
    """No decay, and no source runs slower than another: every source is due at arXiv's 24h."""
    on_arxiv = _query(conn, "a thread")
    on_github = _query(conn, "a repo thread", sources=("github",))
    pulls.record_pull(conn, on_arxiv, "arxiv", last_status="ok", now=_NOW)
    pulls.record_pull(conn, on_github, "github", last_status="ok", now=_NOW)
    a, g = FakeAdapter(), FakeAdapter(slug="github")
    _run(conn, {"arxiv": a, "github": g}, _NOW + timedelta(hours=20))
    assert a.calls == [] and g.calls == []
    _run(conn, {"arxiv": a, "github": g}, _NOW + timedelta(hours=30))
    assert [c[0] for c in a.calls] == ["a thread"] and [c[0] for c in g.calls] == ["a repo thread"]


# ── fairness: the allowance delays, it must not starve ─────────────────────────
def test_the_allowance_serves_the_stalest_pair_first(conn, monkeypatch):
    """`active_queries()` order is STABLE, so once demand exceeds the allowance the same head
    would win every hour and the tail would wait forever — silently. Ordering by real staleness
    makes the allowance a delay for everyone instead of a permanent cut for some."""
    monkeypatch.setattr(allowance, "SEARCHES_PER_HOUR", 1)
    fresh = _query(conn, "recently pulled")
    stale = _query(conn, "long neglected")
    pulls.record_pull(conn, fresh, "arxiv", last_status="ok", now=_NOW - timedelta(days=2))
    pulls.record_pull(conn, stale, "arxiv", last_status="ok", now=_NOW - timedelta(days=30))
    a = FakeAdapter()
    _run(conn, {"arxiv": a})
    assert [c[0] for c in a.calls] == ["long neglected"]


def test_a_never_pulled_pair_outranks_every_pulled_one(conn, monkeypatch):
    """Never-pulled is infinitely stale — a brand-new query must not sit behind the whole
    backlog waiting for its first look."""
    monkeypatch.setattr(allowance, "SEARCHES_PER_HOUR", 1)
    old = _query(conn, "pulled long ago")
    pulls.record_pull(conn, old, "arxiv", last_status="ok", now=_NOW - timedelta(days=30))
    _query(conn, "brand new")
    a = FakeAdapter()
    _run(conn, {"arxiv": a})
    assert [c[0] for c in a.calls] == ["brand new"]


# ── politeness, held in admission ───────────────────────────────────────────────
def test_a_rate_limited_source_is_held_between_requests_without_sleeping(conn):
    """arXiv asks for one request every three seconds and enforces it — a live run firing 22 in
    twelve seconds took two 429s. On the runner the gap is held in admission: a source inside it
    offers no unit, so no pool slot waits it out, and `next_due` says when it frees."""
    for i in range(3):
        _query(conn, f"query {i}")
    slow = FakeAdapter()
    slow.min_interval_s = 3.0
    src = _source({"arxiv": slow})
    assert _run(conn, {}, source=src) == ["ok"]                 # then held
    assert src.next_due(conn, _NOW) == _NOW + timedelta(seconds=3)

    at = _NOW + timedelta(seconds=3)
    src.clock = lambda: at
    assert _run(conn, {}, at, source=src) == ["ok"]
    assert len(slow.calls) == 2


def test_a_fast_source_is_not_held(conn):
    for i in range(3):
        _query(conn, f"query {i}", sources=("github",))
    fast = FakeAdapter(slug="github", results=[_cand("repo:a/b", source="github")])
    fast.min_interval_s = 0.0
    assert _run(conn, {"github": fast}) == ["ok", "ok", "ok"]


def test_a_paused_platform_offers_no_search(conn):
    """The web pool's outage pause on a platform holds frontier's searches too."""
    _query(conn, sources=("github",))
    src = _source({"github": FakeAdapter(slug="github")})
    until = src.outages.pause("github", _NOW, 1.0)
    assert src.candidates(conn, _NOW) == []
    assert src.next_due(conn, _NOW) == until


# ── jitter ──────────────────────────────────────────────────────────────────────
def test_pairs_are_jittered_so_a_batch_does_not_fall_due_together():
    """28 queries seeded on the same day would otherwise come due in the same second forever."""
    vals = {cadence.jitter(f"q{i}", "arxiv") for i in range(50)}
    assert len(vals) > 40                                        # spread, not constant
    assert all(1 - cadence.JITTER <= v <= 1 + cadence.JITTER for v in vals)
    assert cadence.jitter("q1", "arxiv") == cadence.jitter("q1", "arxiv")     # stable, never random


def test_an_unavailable_source_is_not_offered_and_costs_nothing(conn):
    """A source whose breaker is open costs nothing, so it must not take from the allowance or
    hold a slot for a call that never leaves the process — measured at 19 x 3s of pure sleeping
    on a blocked arXiv."""
    for i in range(3):
        _query(conn, f"query {i}")
    down = FakeAdapter()
    down.min_interval_s = 3.0
    down.available = lambda: False
    assert _run(conn, {"arxiv": down}) == []
    assert down.calls == []
    assert pulls.get_pair(conn, fq.query_id_for("query 0"), "arxiv") is None


def test_an_adapter_without_an_available_probe_is_always_asked(conn):
    """A new adapter should stay a one-method object."""
    _query(conn)
    plain = FakeAdapter()
    assert not hasattr(plain, "available")
    _run(conn, {"arxiv": plain})
    assert len(plain.calls) == 1


# ── next_due ────────────────────────────────────────────────────────────────────
def test_next_due_is_the_soonest_ttl(conn):
    qid = _query(conn)
    pulls.record_pull(conn, qid, "arxiv", last_status="ok", now=_NOW)
    ttl = (cadence.TTL_HOURS * cadence.jitter(qid, "arxiv"))
    assert _source({"arxiv": FakeAdapter()}).next_due(conn, _NOW) == \
        _NOW + timedelta(hours=ttl)


def test_next_due_after_a_failure_is_the_retry_wait(conn):
    _query(conn)
    src = _source({"arxiv": FakeAdapter(error=SourceError("503"))})
    _run(conn, {}, source=src)
    assert src.next_due(conn, _NOW) == _NOW + cadence.RETRY_AFTER


# ── the hourly allowance ────────────────────────────────────────────────────────
def test_an_hour_with_room_frees_nothing():
    spent = [_NOW - timedelta(minutes=m) for m in (10, 20)]
    assert allowance.frees_at(spent, 3, _NOW) is None


def test_a_full_hour_frees_when_its_oldest_counted_use_leaves_it():
    spent = [_NOW - timedelta(minutes=m) for m in (5, 30, 50)]
    assert allowance.frees_at(spent, 2, _NOW) == _NOW - timedelta(minutes=30) + allowance.HOUR
    assert allowance.frees_at(spent, 3, _NOW) == _NOW - timedelta(minutes=50) + allowance.HOUR


def test_uses_older_than_the_hour_are_not_counted():
    spent = [_NOW - timedelta(hours=2), _NOW - timedelta(minutes=61)]
    assert allowance.frees_at(spent, 1, _NOW) is None


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
    _run(conn, {"arxiv": a})

    _, since = a.calls[0]
    assert since <= _NOW - timedelta(days=30)


def test_an_adapter_without_one_still_gets_exactly_the_cursor(conn):
    """The widening is opt-in per adapter. A source whose window means what the loop thinks it
    means must be untouched, or every source silently starts re-reading a month of history."""
    _query(conn, sources=("arxiv",))
    a = FakeAdapter(results=[_cand()])
    _run(conn, {"arxiv": a})

    _, since = a.calls[0]
    assert since >= _NOW - timedelta(days=cadence.FIRST_PULL_DAYS + 1)


def test_the_window_assertion_sees_the_widened_window_not_the_cursor():
    """Ordering, asserted directly. `window_ok` exists to catch a `since` that failed to reach the
    adapter; validating a 30-hour window and then sending a 30-day one would defeat it. So the
    floor is applied BEFORE the check, and a look-back past the ceiling is refused rather than
    quietly sent."""
    absurd = type("A", (), {"min_lookback_days": cadence.MAX_WINDOW_DAYS + 10})()
    widened = cadence.lookback_floor(absurd, _NOW - timedelta(hours=1), _NOW)
    assert not cadence.window_ok(widened, _NOW)


def test_the_kind_that_decides_the_minter_is_persisted(conn):
    """Stage 3 dispatches on `kind` and reads it off the row, so a candidate staged without one is
    an artifact no minter will ever claim."""
    _query(conn, sources=("arxiv",))
    _run(conn, {"arxiv": FakeAdapter(results=[_cand()])})

    assert conn.execute("SELECT kind FROM frontier_candidates").fetchone()[0] == "paper"


def test_a_page_that_fails_partway_commits_none_of_it(conn, monkeypatch):
    """Staging a page and stamping its pair are one transaction. When the second of three
    candidates failed to stage, the first must not reach the store with the pair unstamped, or the next
    look would count it `seen`, never `new`."""
    qid = _query(conn)
    real = candidates._stage_one
    calls = []

    def _second_fails(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("disk full")
        return real(*a, **kw)

    monkeypatch.setattr(candidates, "_stage_one", _second_fails)
    page = [_cand(f"arxiv:2501.0000{i}") for i in range(3)]
    src = _source({"arxiv": FakeAdapter(results=page)})
    unit = src.candidates(conn, _NOW)[0]
    with pytest.raises(RuntimeError, match="disk full"):
        src.fetch(conn, unit, Context("web"))

    assert _staged(conn) == 0
    mark = pulls.get_pair(conn, qid, "arxiv")
    assert mark is None or mark["last_pulled_at"] is None


def test_a_candidate_two_queries_find_is_new_once(conn):
    """Newness is whether the insert inserted. Read as `first_seen_at = last_seen_at`, it would stay
    true for the whole pass, and the second query's sighting would count the same row new again."""
    _query(conn, "q one")
    _query(conn, "q two")
    rows = list(conn.execute("SELECT * FROM frontier_queries ORDER BY text"))
    found = searching.Pulled("ok", [_cand()])
    news = [searching.store(conn, cadence.DuePair(q, "arxiv", None), found, _NOW) for q in rows]

    assert news == [1, 0]
    assert _staged(conn) == 1
    assert conn.execute("SELECT COUNT(*) FROM frontier_candidate_queries").fetchone()[0] == 2


def test_a_failed_pull_stores_only_its_status(conn):
    qid = _query(conn)
    pair = cadence.DuePair(conn.execute("SELECT * FROM frontier_queries").fetchone(), "arxiv",
                           None)
    assert searching.store(conn, pair, searching.Pulled("error"), _NOW) == 0
    row = pulls.get_pair(conn, qid, "arxiv")
    assert (row["last_status"], row["last_pulled_at"]) == ("error", None)
    assert row["tried_at"] is not None and _staged(conn) == 0


def test_a_watchs_context_words_reach_its_source(conn):
    """Only X reads them, but every source is handed
    them: the pull does not know which source needs what."""
    qid = _query(conn, "Chai-2")
    fq.set_context(conn, qid, ["antibody"])
    conn.commit()
    adapter = FakeAdapter()
    assert _run(conn, {"arxiv": adapter}) == ["ok"]
    assert adapter.contexts == [["antibody"]]


def test_the_sources_clock_reaches_the_adapter(conn):
    _query(conn)
    seen = []

    class ClockAdapter:
        host = "export.arxiv.org"

        def search(self, _query, *, since, limit, now, context=()):
            assert since is not None and limit > 0
            seen.append(now)
            return []

    assert _run(conn, {"arxiv": ClockAdapter()}) == ["empty"]
    assert seen == [_NOW]


class _GitHubResponse:
    def __init__(self, status_code, items=()):
        self.status_code, self._items = status_code, list(items)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise github_client.requests.HTTPError(f"{self.status_code} Server Error")

    def json(self):
        return {"items": self._items}


def _live_github(kb_home, monkeypatch, responses):
    """The real client, adapter and breaker, with only `requests.get` replaced."""
    monkeypatch.setattr(github_client, "_gh_breaker", None)
    calls = []

    def get(_url, **kw):
        calls.append(kw["params"]["q"])
        return responses.pop(0)
    monkeypatch.setattr(github_client.requests, "get", get)
    client = github_client.GitHubApiClient(token="", cache_path=kb_home / "github_cache.json")
    return github_candidates.GitHubAdapter(client=client), client, calls


def test_a_failed_github_request_is_neither_cached_nor_stamped(conn, kb_home, monkeypatch):
    """A 503 must not become a cached `[]` and an `empty` stamp, or the pair loses its retry for a
    full TTL and the identical search stays blank for seven days. A failed
    request writes no cache; the next try actually requests and returns records."""
    qid = _query(conn, sources=("github",))
    repo = {"full_name": "o/n", "html_url": "https://github.com/o/n", "pushed_at": "2026-08-09",
            "description": "a gated DeltaNet attention kernel"}
    adapter, client, calls = _live_github(
        kb_home, monkeypatch, [_GitHubResponse(503), _GitHubResponse(200, [repo])])

    assert _run(conn, {"github": adapter}) == ["error"]
    row = pulls.get_pair(conn, qid, "github")
    assert row["last_status"] == "error" and row["last_pulled_at"] is None
    assert client._cache == {} and not (kb_home / "github_cache.json").exists()

    assert _run(conn, {"github": adapter}, _NOW + cadence.RETRY_AFTER) == ["ok"]
    row = pulls.get_pair(conn, qid, "github")
    assert len(calls) == 2 and _staged(conn) == 1
    assert row["last_status"] == "ok" and row["last_pulled_at"] is not None


def test_a_successful_empty_github_search_is_cached(kb_home, monkeypatch):
    """An empty page that GitHub actually answered is a result, not a failure: it stays cached."""
    adapter, _client, calls = _live_github(kb_home, monkeypatch, [_GitHubResponse(200, [])])
    assert adapter.search("kernel", since=None, now=_NOW) == []
    assert adapter.search("kernel", since=None, now=_NOW) == []
    assert len(calls) == 1


def test_a_github_404_is_an_empty_result_not_a_failure(kb_home, monkeypatch):
    """The documented 404 rule: not a failure, no breaker failure recorded, an empty page."""
    adapter, _client, calls = _live_github(kb_home, monkeypatch, [_GitHubResponse(404)])
    assert adapter.search("kernel", since=None, now=_NOW) == []
    assert github_client._github_breaker().allow()
    assert len(calls) == 1
