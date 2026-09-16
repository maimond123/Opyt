"""`_pull_own_timeline` keeps what it walked — and claims only what it can defend.

RULED 2026-09-14, from a measurement taken twice against the live store. A foreground
`oracle(action='ingest')` over four confirmed Oracles covered two of them, PERMANENTLY: the other
two got 0 requests and 0 atoms on a cold meter and again on a refilled one.

The root cause was one property of this function — it buffered every tweet and returned, so an
`XRateLimited` anywhere discarded the lot. A partial walk therefore spent requests and wrote
nothing, which made refusing pre-emptively the correct move, which is what starved the later
Oracles. Remove the property and the reserve stops earning its keep.

⚠️ THE RISK LIVES IN `test_a_one_sided_walk_claims_no_frontier`. `UserTweets` and
`UserRepliesTimeline` are INDEPENDENT 50/15-min buckets, and the Posts tab omits standalone
replies — over one 108-day window, 200 of the 214 tweets found only by the paid API were replies.
So a walk that finished posts and never started replies is a BIASED sample, not a shorter one, and
recording a frontier from it would permanently hide the gap: `covered_from` only ever widens, so
nothing would ever go back for the replies. Atoms still land; the claim does not.
"""
from __future__ import annotations

import pytest

from pipeline.ingestion import x_graphql_core as core
from pipeline.kb import ingest_x_footprint as fp, schema


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _tweet(tid: str, day: int) -> dict:
    """A tweet on 2026-01-{day}, in X's own date format."""
    return {"id": tid, "createdAt": f"Mon Jan {day:02d} 10:00:00 +0000 2026",
            "author": {"id": "99", "userName": "carol"}, "text": "t"}


def _ts(day: int) -> float:
    from pipeline.ingestion import x_render as xt
    return xt._parse_twitter_date(f"Mon Jan {day:02d} 10:00:00 +0000 2026").timestamp()


def _walk(monkeypatch, per_timeline: dict, *, since_day: int = 1, cap: int = 100):
    """Drive the real `_pull_own_timeline` with a faked transport. Each `per_timeline` value is
    either a list of tweets (a walk that finishes) or an `XRateLimited` (one that is cut short
    after handing back whatever pages are in the list beside it)."""
    def _fetch(cookies, headers, uid, *, pages, timeline, after_page=None, **kw):
        landed, outcome = per_timeline[timeline]
        if after_page is not None and landed:
            after_page(list(landed))          # the page lands, THEN the next request is refused
        if outcome is not None:
            raise outcome
        return list(landed)

    monkeypatch.setattr(core, "fetch_user_tweets", _fetch)
    return fp._pull_own_timeline({}, {}, "99", int(_ts(since_day)), cap)


# ── it keeps what it walked ────────────────────────────────────────────────────

def test_a_complete_walk_is_unchanged(monkeypatch):
    """The regression guard. Both timelines finish, so the union comes back trimmed and deduped
    and the frontier is the bound the caller asked for — exactly the old contract."""
    out = _walk(monkeypatch, {"posts": ([_tweet("1", 10)], None),
                              "replies": ([_tweet("2", 11)], None)})

    assert sorted(t["id"] for t in out.tweets) == ["1", "2"]
    assert out.complete == frozenset({"posts", "replies"})
    assert out.reached == _ts(1)


def test_a_cut_short_walk_keeps_the_pages_it_paid_for(monkeypatch):
    """The whole change. `fetch_user_tweets` builds its accumulator and discards it on a raise,
    so `after_page` is the only place the partial survives — and those requests were spent."""
    out = _walk(monkeypatch, {"posts": ([_tweet("1", 20), _tweet("2", 15)],
                                        core.XRateLimited("spent")),
                              "replies": ([], core.XRateLimited("spent"))})

    assert sorted(t["id"] for t in out.tweets) == ["1", "2"]
    assert out.complete == frozenset()


def test_one_timeline_running_dry_does_not_cost_the_other(monkeypatch):
    """Two INDEPENDENT buckets. Re-raising out of the posts walk would throw away a replies walk
    that x.com was still perfectly willing to answer."""
    out = _walk(monkeypatch, {"posts": ([_tweet("1", 20)], core.XRateLimited("spent")),
                              "replies": ([_tweet("2", 21)], None)})

    assert sorted(t["id"] for t in out.tweets) == ["1", "2"]
    assert out.complete == frozenset({"replies"})


def test_a_walk_that_reached_nothing_reports_no_frontier(monkeypatch):
    """Refused before the first page landed. Nothing walked, so there is nothing to report — and
    None is the value `record_pull` already reads as "no lower bound to report"."""
    out = _walk(monkeypatch, {"posts": ([], core.XRateLimited("spent")),
                              "replies": ([], core.XRateLimited("spent"))})

    assert out.tweets == [] and out.reached is None


# ── the frontier is measured BEFORE the trim ──────────────────────────────────

def test_the_frontier_is_how_far_back_we_got_not_what_we_kept(monkeypatch):
    """The trim answers "what did the caller ask to keep"; the frontier answers "how far back did
    we actually get". They differ exactly when a walk overshoots — its date stop fires only after
    a page has landed — or is cut short."""
    out = _walk(monkeypatch,
                {"posts": ([_tweet("1", 20), _tweet("2", 5)], core.XRateLimited("spent")),
                 "replies": ([], core.XRateLimited("spent"))},
                since_day=10)

    assert [t["id"] for t in out.tweets] == ["1"]      # day 5 is outside the window, trimmed
    assert out.reached == _ts(5)                       # …but the walk still went that far back


def test_a_complete_walk_reports_the_bound_not_its_oldest_tweet(monkeypatch):
    """A quiet account that finished both walks covers the WHOLE window, including the silent
    part of it. Reporting its oldest tweet would understate coverage and re-pull forever."""
    out = _walk(monkeypatch, {"posts": ([_tweet("1", 20)], None),
                              "replies": ([_tweet("2", 21)], None)},
                since_day=1)

    assert out.reached == _ts(1)


# ── §C: a one-sided walk claims NO frontier ───────────────────────────────────
#
# ⚠️ THIS IS THE RISK STEP OF THE WHOLE BUILD. `covered_from` only ever WIDENS (it takes the MIN),
# so a frontier claimed once is a claim nothing ever revisits. Record one off a posts-only walk
# and the missing replies — the majority of the population, measured — become permanently
# invisible: `backfill_pair` reads the frontier, sees the window met, and never goes back.
#
# The fail-safe invariant is explicit that a failed external call must not mark unfinished work
# done. Atoms landing from a partial walk is not that; a frontier claimed from one is.

def _frontier(complete, reached):
    return fp._walk_frontier(fp.TimelineWalk(tweets=[], complete=frozenset(complete),
                                             reached=reached))


def test_a_one_sided_walk_claims_no_frontier():
    """Posts finished, replies never started. That is a BIASED sample, not a shorter one — the
    Posts tab omits standalone replies, and 200 of 214 tweets found only by the paid API over one
    108-day window were replies. Claiming a floor here hides that gap forever."""
    assert _frontier({"posts"}, _ts(1)) is None
    assert _frontier({"replies"}, _ts(1)) is None
    assert _frontier(set(), _ts(1)) is None


def test_both_timelines_complete_report_the_honest_frontier():
    """The other half of the rule. A walk that finished BOTH timelines to the same bound holds
    what it says it holds, and refusing to say so would re-pull it forever."""
    out = _frontier({"posts", "replies"}, _ts(1))

    assert out is not None and out.startswith("2026-01-01T")


def test_a_walk_that_reached_nothing_claims_nothing_either():
    """`None` is already `record_pull`'s word for "this pull had no lower bound to report", which
    is exactly what an empty walk has."""
    assert _frontier({"posts", "replies"}, None) is None


# ── the adapter reports both facts on its own summary ─────────────────────────

def _sync(conn, fake_embedder, monkeypatch, walk):
    """Run the REAL `sync_x_footprint` over a given walk outcome, network stubbed."""
    from pipeline.ingestion import x_render as xt
    monkeypatch.setattr(core, "read_x_cookies", lambda: {"auth_token": "t", "ct0": "c"})
    monkeypatch.setattr(core, "auth_headers", lambda *a, **k: {})
    monkeypatch.setattr(core, "x_session", lambda url: {})
    monkeypatch.setattr(core, "fetch_user_profile", lambda c, h, u: {
        "user_id": "99", "handle": u, "display_name": u, "bio": "", "website": "",
        "bio_urls": [], "verified": False, "followers": 1})
    monkeypatch.setattr(xt, "_article_tweet_id", lambda t: None)
    monkeypatch.setattr(fp, "_pull_own_timeline", lambda *a, **kw: walk)
    return fp.sync_x_footprint(conn, fake_embedder, handle="carol")


def test_a_complete_run_reports_its_frontier_and_is_not_partial(conn, fake_embedder, monkeypatch):
    out = _sync(conn, fake_embedder, monkeypatch,
                fp.TimelineWalk(tweets=[], complete=frozenset({"posts", "replies"}),
                                reached=_ts(1)))

    assert out["partial"] is False
    assert out["covered_from"].startswith("2026-01-01T")


def test_a_partial_run_says_so_and_reports_no_frontier(conn, fake_embedder, monkeypatch):
    """Both halves in one summary, and they are independent: `partial` is what the message layer
    and the background pass read, `covered_from` is what the store records."""
    out = _sync(conn, fake_embedder, monkeypatch,
                fp.TimelineWalk(tweets=[], complete=frozenset({"posts"}), reached=_ts(5)))

    assert out["partial"] is True
    assert out["covered_from"] is None


def test_neither_key_rides_the_stats_passthrough(conn, fake_embedder, monkeypatch):
    """⚠️ `run_stats` copies int counters and dict diagnostics — and `isinstance(True, int)` is
    True in Python, so a bool named in `RUN_STAT_KEYS` would silently become a user-facing
    counter. These two are lifted onto the record explicitly instead."""
    from pipeline.kb import ingest_common

    out = _sync(conn, fake_embedder, monkeypatch,
                fp.TimelineWalk(tweets=[], complete=frozenset({"posts"}), reached=_ts(5)))

    stats = ingest_common.run_stats(out)
    assert "partial" not in stats and "covered_from" not in stats


# ── §B: the store records what was REACHED, not what was asked ────────────────
#
# `_record_coverage` wrote the requested window unconditionally. That was harmless for as long as
# only a complete walk could reach it — asked == reached — and became a lie the moment a partial
# walk could write atoms. No schema work: `record_pull` has always documented `covered_from` as
# "the oldest instant this pull reached", only ever widening (MIN), with None meaning "no lower
# bound to report". Only the caller was wrong.

def _pair(conn, cid="x:user:99", handle="carol"):
    from pipeline.kb import oracle_refresh_state as st, resolve
    schema.upsert_entity(conn, cid, name="Carol", profile={"handle": handle})
    resolve.resolve_entities(conn)
    st.upsert_source(conn, st.SourceRow(canonical_id=cid, source_type="x",
                                        source_key=handle, status="trusted"))
    return cid


def _stored(conn, cid):
    from pipeline.kb import oracle_refresh_state as st
    return next(r for r in st.list_sources(conn, canonical_ids=[cid])).covered_from


def _record(conn, cid, rec, *, asked_days=183):
    from datetime import timedelta
    from pipeline.timeparse import utc_now
    from pipeline.kb import oracles
    oracles._record_coverage(conn, cid, [rec], x_since=utc_now() - timedelta(days=asked_days),
                             web_since=None)


def _x_rec(**kw):
    return {"url": "https://x.com/carol", "type": "x", "action": "ingested", **kw}


def test_a_measured_reach_beats_the_window_that_was_asked_for(conn):
    cid = _pair(conn)

    _record(conn, cid, _x_rec(covered_from="2026-03-01T00:00:00+00:00"))

    assert _stored(conn, cid).startswith("2026-03-01")


def test_a_partial_walk_records_no_frontier_at_all(conn):
    """⚠️ PRESENCE, NOT TRUTHINESS. `rec["covered_from"] or asked` would fall straight back to
    the requested window here — which is the claim the whole of §C exists to refuse."""
    cid = _pair(conn)

    _record(conn, cid, _x_rec(covered_from=None))

    assert _stored(conn, cid) is None


def test_a_source_that_measures_nothing_still_falls_back_to_the_asked_window(conn):
    """Substack, blog and OpenAlex are bounded by a snapshot hash, not a meter, so asked and
    reached cannot diverge for them and the old behaviour is still right."""
    from pipeline.kb import oracle_refresh_state as st, resolve
    schema.upsert_entity(conn, "x:user:98", name="Dan", profile={"handle": "dan"})
    resolve.resolve_entities(conn)
    st.upsert_source(conn, st.SourceRow(canonical_id="x:user:98", source_type="x",
                                        source_key="dan", status="trusted"))

    _record(conn, "x:user:98",
            {"url": "https://x.com/dan", "type": "x", "action": "ingested"}, asked_days=10)

    assert _stored(conn, "x:user:98") is not None


def test_a_shallow_partial_never_raises_an_existing_deeper_floor(conn):
    """The MIN merge, which `record_pull` already implements — this is the caller proving it uses
    it. A background pass that only reached March must not undo a walk that reached January."""
    cid = _pair(conn)
    _record(conn, cid, _x_rec(covered_from="2026-01-01T00:00:00+00:00"))

    _record(conn, cid, _x_rec(covered_from="2026-06-01T00:00:00+00:00"))

    assert _stored(conn, cid).startswith("2026-01-01")


def test_a_partial_after_a_complete_walk_leaves_the_floor_where_it_was(conn):
    """The case that would otherwise erase coverage: a later partial reports None, and None means
    "nothing to report", never "reset"."""
    cid = _pair(conn)
    _record(conn, cid, _x_rec(covered_from="2026-01-01T00:00:00+00:00"))

    _record(conn, cid, _x_rec(covered_from=None))

    assert _stored(conn, cid).startswith("2026-01-01")


def test_the_ingest_engine_carries_the_adapter_s_own_frontier_to_the_store(conn, monkeypatch):
    """End to end through `_ingest_oracle`: the adapter measures, the engine carries, the store
    records. Three seams, and the middle one is the one that used to drop the measurement."""
    import importlib
    from pipeline.kb import oracles, resolve
    dp = importlib.import_module("pipeline.ingestion.discover_profile")
    from pipeline.ingestion import x_graphql

    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: True)
    monkeypatch.setattr(dp, "discover_profile",
                        lambda seed, seed_type="x", **kw: {"username": seed, "sources": []})
    monkeypatch.setattr(fp, "sync_x_footprint", lambda c, e, **kw: {
        "source": "x-footprint", "added": 2, "fetched": 2,
        "partial": True, "covered_from": None})

    schema.upsert_entity(conn, "x:user:77", name="Carol", profile={"handle": "carol"})
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=["x:user:77"])
    o = [x for x in oracles.confirmed_oracles(conn) if x["canonical_id"] == "x:user:77"][0]

    oracles._ingest_oracle(conn, None, o)

    assert _stored(conn, "x:user:77") is None            # a partial claims nothing…
    from pipeline.kb import oracle_refresh_state as st
    row = next(r for r in st.list_sources(conn, canonical_ids=["x:user:77"])
               if r.source_type == "x")
    assert row.last_pulled_at is not None                 # …but a real observation still stamps


# ── §D: `partial` is an ANNOTATION, never a new `action` ──────────────────────
#
# ⚠️ Three readers key off `action == "ingested"` — `_stamp_source` (via `_OBSERVED`),
# `_ingest_presentation`, and the counter roll-up in `_merge_source_result`. An
# `action = "partial"` would make every one of them drop the record SILENTLY: the writer would
# vanish from `completed` and `last_pulled_at` would stop being stamped for an observation that
# really happened. Additive keeps an unaware reader correct.

def _partial_result(name="Andrej"):
    return {"name": name, "atoms_added": 41,
            "results": [{"type": "x", "action": "ingested", "partial": True,
                         "covered_from": None}]}


def test_a_partial_record_still_counts_as_an_ingest_everywhere(conn):
    """The three readers, checked as one: the action is unchanged, so the roll-up counts it, the
    stamp fires, and nothing has to learn a new string to stay correct."""
    from pipeline.kb import oracles

    result: dict = {}
    oracles._merge_source_result(result, {"results": _partial_result()["results"],
                                          "ingested": 1, "atoms_added": 41})

    assert result["ingested"] == 1 and result["atoms_added"] == 41
    assert result["results"][0]["action"] == "ingested"


def test_a_partial_writer_is_listed_as_here_AND_as_still_arriving():
    """Both halves. Half of somebody's writing is still a library — burying them until the rest
    lands is the failure mode — and the rest really is still owed."""
    from mcp_server import oracle_tools

    out = oracle_tools._ingest_presentation([_partial_result()])

    assert out["completed"] == [{"oracle": "Andrej", "sources": ["x"], "atoms_added": 41,
                                 "partial": True}]
    assert out["in_progress"]["oracles"] == ["Andrej"]


def test_a_complete_writer_carries_no_partial_flag_and_stays_out_of_in_progress():
    """The regression guard: the ordinary case is untouched."""
    from mcp_server import oracle_tools

    out = oracle_tools._ingest_presentation([
        {"name": "Soren", "atoms_added": 37,
         "results": [{"type": "x", "action": "ingested"}]}])

    assert out["completed"] == [{"oracle": "Soren", "sources": ["x"], "atoms_added": 37}]
    assert "in_progress" not in out


def test_one_pass_completing_does_not_cancel_the_other_coming_back_short():
    """`_merge_passes` keeps a row from BOTH passes, so a person can be covered recently and
    still owed their older window — which is exactly the state `partial` names."""
    from mcp_server import oracle_tools

    out = oracle_tools._ingest_presentation([
        {"name": "Andrej", "atoms_added": 41,
         "results": [{"type": "x", "action": "ingested", "pass": "breadth"},
                     {"type": "x", "action": "ingested", "partial": True, "pass": "depth"}]}])

    assert out["completed"][0]["partial"] is True
    assert out["in_progress"]["oracles"] == ["Andrej"]


# ── nothing observed is still a refusal ───────────────────────────────────────
#
# ⚠️ The fail-safe invariant: a failed external call must SKIP — no write, no mark-processed. A
# walk keeping what it got does not change that when it got NOTHING. Letting a zero-observation
# walk return an ordinary summary would classify as `ingested`, stamp `last_pulled_at`, restart
# the pair's TTL and hide it for a full window, having read not one page.

def test_a_walk_that_finished_neither_timeline_raises_rather_than_stamping(conn, fake_embedder,
                                                                          monkeypatch):
    with pytest.raises(core.XRateLimited):
        _sync(conn, fake_embedder, monkeypatch,
              fp.TimelineWalk(tweets=[], complete=frozenset(), reached=None))


def test_one_finished_timeline_is_an_observation_even_when_it_found_nothing(conn, fake_embedder,
                                                                           monkeypatch):
    """The bar is `complete`, never the tweet count. One walk that ran to the end and found
    nothing is a real reading of that timeline — it just claims no frontier."""
    out = _sync(conn, fake_embedder, monkeypatch,
                fp.TimelineWalk(tweets=[], complete=frozenset({"posts"}), reached=None))

    assert out["partial"] is True and out["covered_from"] is None


def test_a_quiet_account_that_finished_both_walks_is_not_a_refusal(conn, fake_embedder,
                                                                   monkeypatch):
    """An account that genuinely posted nothing in the window looks exactly like a cut-off walk
    if you count tweets. It is the opposite: complete coverage of an empty window."""
    out = _sync(conn, fake_embedder, monkeypatch,
                fp.TimelineWalk(tweets=[], complete=frozenset({"posts", "replies"}),
                                reached=_ts(1)))

    assert out["partial"] is False and out["covered_from"] is not None


def test_the_refresh_rail_records_a_partial_walk_the_same_way(conn, monkeypatch):
    """⚠️ TWO PATHS, ONE RULE. `_pull_pair` had the same `summary.get(...) or since` shape as
    `_record_coverage`, and the same bug hid in it: a partial walk reports `covered_from: None`
    deliberately, and `or` sent exactly that case down the fallback — stamping the window we asked
    for onto a pull that never covered it. The rail is where a partial walk lands MOST often, so
    fixing only the foreground path would have left the common case lying."""
    from datetime import timedelta
    from pipeline.kb import oracle_refresh as orf, oracle_refresh_state as st
    from pipeline.timeparse import utc_now

    cid = _pair(conn, cid="x:user:88", handle="dana")
    row = next(r for r in st.list_sources(conn, canonical_ids=[cid]) if r.source_type == "x")
    monkeypatch.setattr(orf, "ingest_x_footprint_sync", lambda c, e, **kw: {
        "source": "x-footprint", "added": 4, "partial": True, "covered_from": None})

    out = orf._pull_pair(conn, None, row, since=utc_now() - timedelta(days=183))

    assert out["status"] == "ingested" and out["new_atoms"] == 4
    assert _stored(conn, cid) is None                 # atoms landed; no frontier was claimed
