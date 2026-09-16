"""`oracle(action='ingest')` — breadth for everyone, then depth. And the unreviewed-source count.

RULED 2026-09-13. The ingest used to be a serial list comprehension in `schema.list_oracles` order,
which is `ORDER BY confirmed_at DESC` — and a single `confirm` stamps every pick in the same
second, so the order was arbitrary. When `UserTweets` (50/15 min against a measured 2-26 requests
per Oracle) ran out part-way, arbitrary order decided who got a corpus and who got nothing: one
deep, six empty.

Breadth is expressed as a WINDOW so the whole thing stays a loop OVER `_ingest_oracle`, which is
what `.guards.py`'s `retired-expand-cli-engine` requires — never a second copy of the engine.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from pipeline.kb import oracles, schema
from pipeline.timeparse import parse_ts, utc_now
from mcp_server import oracle_tools


@pytest.fixture()
def ingests(kb_home, monkeypatch):
    """Record every `_ingest_oracle` call as `(canonical_id, x_since)` and return a plausible
    result, so these tests are about ORDER and WINDOWS rather than about adapters."""
    calls: list[tuple[str, object]] = []

    def _fake(conn, embedder, oracle, *, force=False, x_since=None, web_since=None,
              scholar_since=None, scholar_topics=None, limit=0, extra_source_urls=None,
              x_only=False):
        cid = oracle["canonical_id"]
        calls.append((cid, x_since))
        return {"oracle_id": cid, "name": oracle.get("name"), "atoms_added": 2, "ingested": 1,
                "results": [{"url": f"https://x.com/{cid}", "type": "x", "action": "ingested",
                             "detail": ""}],
                "lookback": {"x": "window"}}

    monkeypatch.setattr(oracles, "_ingest_oracle", _fake)
    from pipeline.kb import embed
    monkeypatch.setattr(embed, "get_kb_embedder", lambda: object())
    return calls


def _oracle(conn, cid, name):
    schema.upsert_entity(conn, cid, name=name, profile={"handle": name.lower()})
    schema.upsert_oracle(conn, cid, name=name)
    return {"canonical_id": cid, "name": name}


def _run(conn, picks, *, x_connected=True, x_lookback=None, monkeypatch=None):
    from pipeline.ingestion import x_graphql
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: x_connected)
    monkeypatch.setattr(oracles, "confirmed_oracles", lambda c: picks)
    return oracle_tools._ingest(conn, canonical_ids=[p["canonical_id"] for p in picks],
                                force=False, x_lookback=x_lookback, web_lookback=None,
                                scholar_lookback=None, scholar_topics=None)


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


# ── the pass order IS the reservation ──────────────────────────────────────────

def test_every_oracle_gets_breadth_before_anyone_gets_depth(conn, ingests, monkeypatch):
    """The whole point. Depth starting early is what stranded the later picks' breadth — so
    breadth for ALL of them completes first, and no separate budget bookkeeping is needed."""
    picks = [_oracle(conn, f"x:user:{i}", f"P{i}") for i in range(3)]
    cutoff = utc_now() - timedelta(days=31)

    _run(conn, picks, monkeypatch=monkeypatch, x_lookback="1yr")

    windows = [since for _cid, since in ingests]
    assert len(windows) == 6
    # The first three are the SHALLOW window, the last three the deep one — no interleaving.
    assert all(w > cutoff for w in windows[:3]), windows
    assert all(w < cutoff for w in windows[3:]), windows


def test_the_order_is_the_screens_own_ranking_not_confirmed_at(conn, ingests, monkeypatch):
    """`list_oracles` is `ORDER BY confirmed_at DESC` and one `confirm` stamps them all in the same
    second. Reuses `screen.rank_candidates` rather than adding a second scoring function beside
    `Candidate.sort_key` — two orderings of the same people is how two surfaces disagree."""
    from pipeline.kb import screen

    picks = [_oracle(conn, "x:user:1", "First"), _oracle(conn, "x:user:2", "Second")]

    class _C:
        def __init__(self, cid):
            self.canonical_id = cid

    monkeypatch.setattr(screen, "rank_candidates", lambda c, **kw: [_C("x:user:2"), _C("x:user:1")])
    monkeypatch.setattr(screen, "interleave_tiers", lambda r: r)

    _run(conn, picks, monkeypatch=monkeypatch, x_lookback="1yr")

    assert [cid for cid, _s in ingests] == ["x:user:2", "x:user:1"] * 2


def test_an_oracle_with_no_curation_signals_sorts_after_the_vouched_for(conn, ingests, monkeypatch):
    """`add_handles` mints an Oracle with no signals, so it is absent from `ranked` entirely. It
    goes last rather than being given an invented position among the vouched-for."""
    from pipeline.kb import screen

    picks = [_oracle(conn, "x:user:named", "Named"), _oracle(conn, "x:user:1", "Vouched")]

    class _C:
        canonical_id = "x:user:1"

    monkeypatch.setattr(screen, "rank_candidates", lambda c, **kw: [_C()])
    monkeypatch.setattr(screen, "interleave_tiers", lambda r: r)

    _run(conn, picks, monkeypatch=monkeypatch, x_lookback="1yr")

    assert [cid for cid, _s in ingests][:2] == ["x:user:1", "x:user:named"]


def test_an_unreadable_ranking_leaves_the_order_alone(conn, ingests, monkeypatch):
    """Fail-safe: ordering is an optimisation, and a broken read of it must not stop the ingest."""
    from pipeline.kb import screen
    monkeypatch.setattr(screen, "rank_candidates", lambda c, **kw: 1 / 0)

    picks = [_oracle(conn, "x:user:1", "A"), _oracle(conn, "x:user:2", "B")]
    out = _run(conn, picks, monkeypatch=monkeypatch, x_lookback="1yr")

    assert out["ingested_oracles"] == 2
    assert [cid for cid, _s in ingests][:2] == ["x:user:1", "x:user:2"]


# ── when a second pass would buy nothing ───────────────────────────────────────

def test_a_window_already_inside_the_breadth_window_runs_once(conn, ingests, monkeypatch):
    """`since_last` on a person pulled yesterday is narrower than 30 days, so a "breadth" pass
    would be a DEEPER and more expensive depth pass."""
    picks = [_oracle(conn, "x:user:1", "A")]
    monkeypatch.setattr(oracles, "x_since_last",
                        lambda c, cid: utc_now() - timedelta(days=2))

    _run(conn, picks, monkeypatch=monkeypatch, x_lookback=oracles.X_SINCE_LAST)

    assert len(ingests) == 1


def test_no_x_session_means_no_breadth_pass(conn, ingests, monkeypatch):
    """Breadth protects a metered timeline. Without a session there is no timeline to protect, and
    the off-X archives run to completion either way (R4)."""
    picks = [_oracle(conn, "x:user:1", "A"), _oracle(conn, "x:user:2", "B")]

    _run(conn, picks, x_connected=False, monkeypatch=monkeypatch)

    assert len(ingests) == 2
    assert [s for _c, s in ingests] == [None, None]


# ── one Oracle, one result row ─────────────────────────────────────────────────

def test_two_passes_report_as_one_oracle_with_both_windows_visible(conn, monkeypatch):
    """`_ingest_presentation` groups by name and reads `atoms_added` off the top level, so two
    rows for one person would list them twice and split their count."""
    breadth = {"oracle_id": "x:user:1", "name": "A", "atoms_added": 3, "ingested": 1,
               "results": [{"type": "x", "action": "ingested", "detail": ""}],
               "lookback": {"x": "30 days"}}
    depth = {"oracle_id": "x:user:1", "name": "A", "atoms_added": 9, "ingested": 1, "deferred": 1,
             "results": [{"type": "x", "action": "deferred", "detail": ""}],
             "lookback": {"x": "1 year"}}

    merged = oracle_tools._merge_passes(breadth, depth)

    assert merged["atoms_added"] == 12 and merged["ingested"] == 2 and merged["deferred"] == 1
    # BOTH rows survive, tagged — "covered recently, still owed the older window" is one Oracle's
    # honest state, and it takes two rows to say it.
    assert [(r["action"], r["pass"]) for r in merged["results"]] == [("ingested", "breadth"),
                                                                     ("deferred", "depth")]
    assert merged["lookback"] == {"x": "1 year"}            # the deep one: the consent surface
    assert merged["breadth_lookback"] == {"x": "30 days"}

    out = oracle_tools._ingest_presentation([merged])
    assert out["completed"] == [{"oracle": "A", "sources": ["x"], "atoms_added": 12}]
    assert out["in_progress"]["oracles"] == ["A"]           # the deferral still surfaces


# ── the needs-review queue stops being a silent drop (2026-09-13) ──────────────
#
# ⚠️ `oracle_reviews.list_open` has ONE caller in the whole repo, no rail reads the table, and both
# `_coverage_report` and `oracle_refresh.status_summary` iterate `list_sources` only. A needs-review
# source never reaches a registered pair — the adapter did not run, so no `blog:`/`substack:`/
# `github:` entity was minted — so an Oracle with an unreviewed Substack reported X-only coverage
# as COMPLETE. Measured: 21 Oracles → 21 needs-review sources, and the audit's own per-row reason
# for nearly all of them is "no trusted source links this", on names like Karpathy's GitHub and
# Taleb's Substack. So `possible_sources`' "nothing is lost by leaving them out" is measurably
# false, and dropping the confirm step at onboarding raises the question of when the user ever
# sees one. The answer was: never.

def _queue_review(conn, cid, url, stype="substack"):
    from pipeline.kb import oracle_reviews
    oracle_reviews.record_outcomes(conn, cid, [{"url": url, "type": stype,
                                                "action": "needs-review", "detail": "untrusted"}])


def test_open_counts_is_per_oracle_and_counts_only_what_waits(conn):
    from pipeline.kb import oracle_reviews

    _oracle(conn, "x:user:1", "A")
    _oracle(conn, "x:user:2", "B")
    _queue_review(conn, "x:user:1", "https://a.substack.com")
    _queue_review(conn, "x:user:1", "https://a.dev", stype="blog")
    _queue_review(conn, "x:user:2", "https://b.substack.com")

    assert oracle_reviews.open_counts(conn) == {"x:user:1": 2, "x:user:2": 1}
    assert oracle_reviews.open_counts(conn, canonical_ids=["x:user:2"]) == {"x:user:2": 1}

    # `approved` has had its decision and `dismissed` is a terminal no — neither waits on anybody.
    item = next(r for r in oracle_reviews.list_open(conn) if r["canonical_id"] == "x:user:2")
    oracle_reviews.dismiss(conn, item["review_id"])
    assert oracle_reviews.open_counts(conn) == {"x:user:1": 2}


def test_the_ingest_result_says_how_many_sources_are_waiting(conn, kb_home, monkeypatch):
    """`coverage` structurally cannot say this: an unreviewed source has no pair to report on."""
    import importlib
    from pipeline.kb import resolve
    dp = importlib.import_module("pipeline.ingestion.discover_profile")

    schema.upsert_entity(conn, "x:user:1", name="A", profile={"handle": "a"})
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=["x:user:1"])
    o = [x for x in oracles.confirmed_oracles(conn) if x["canonical_id"] == "x:user:1"][0]
    monkeypatch.setattr(dp, "discover_profile",
                        lambda seed, seed_type="x", **kw: {
                            "username": seed,
                            "sources": [{"source_type": "substack", "url": "https://a.substack.com",
                                         "metadata": {}, "trust": {"trusted": False}}]})
    from pipeline.kb import expand
    monkeypatch.setattr(expand, "_x_handle_to_pull", lambda root, profile: None)

    out = oracles._ingest_oracle(conn, None, o)

    assert out["unreviewed_sources"] == 1
    # …and the coverage report still says nothing about it, which is exactly why the count exists.
    assert not any(k.startswith("substack:") for k in out.get("coverage", {}))


def test_status_summary_carries_the_count_but_never_escalates_it(conn, kb_home):
    """⚠️ REPORTED, NEVER NAGGED. `needs_attention` drives a proactive `oracles_stale` search
    notice; the ruling is available and honest. A decision waiting is not a fault."""
    from pipeline.kb import oracle_refresh, oracle_refresh_state as st

    # `needs_attention`'s OTHER condition is "never consented", which would mask the assertion
    # below. Granted so the only thing left that could raise the flag is the review item.
    oracle_refresh.grant_consent()
    _oracle(conn, "x:user:1", "A")
    st.upsert_source(conn, st.SourceRow(canonical_id="x:user:1", source_type="x",
                                        source_key="a", status="trusted",
                                        last_pulled_at=utc_now().isoformat()))
    _queue_review(conn, "x:user:1", "https://a.substack.com")

    summary = oracle_refresh.status_summary(conn)

    entry = next(e for e in summary["oracles"] if e["canonical_id"] == "x:user:1")
    assert entry["unreviewed"] == 1
    assert summary["needs_attention"] is False


# ── the depth pass stops redoing the off-X work (§I, 2026-09-14) ───────────────
#
# ⚠️ Measured 2026-09-14: the depth loop over four Oracles spent 16 seconds re-running discovery
# and a sitemap crawl and produced ZERO atoms. Every line of it was work the breadth pass had done
# minutes earlier — because only the X timeline is bounded by the window the two passes differ in.
# Everything else is unbounded by design (R4), so the "shallow" pass already fetched the whole
# archive and the "deep" pass fetched it again. The cost scales with the roster.

@pytest.fixture()
def real_ingest(conn, kb_home, monkeypatch):
    """Run the REAL `_ingest_oracle`, with only the network boundaries stubbed, and count the
    off-X stages. A faked engine cannot answer this question: it is about what the engine skips."""
    import importlib
    from pipeline.kb import embed, onboard_footprint as of
    dp = importlib.import_module("pipeline.ingestion.discover_profile")

    seen = {"discover": [], "footprint": [], "x": []}
    monkeypatch.setattr(dp, "discover_profile",
                        lambda seed, seed_type="x", **kw: seen["discover"].append(seed)
                        or {"username": seed, "sources": []})
    monkeypatch.setattr(of, "onboard_footprint",
                        lambda c, e, cid, srcs, **kw: seen["footprint"].append(cid) or {})
    from pipeline.kb import ingest_x_footprint
    monkeypatch.setattr(ingest_x_footprint, "sync_x_footprint",
                        lambda c, e, **kw: seen["x"].append(kw.get("handle"))
                        or {"source": "x-footprint", "added": 1, "fetched": 1})
    monkeypatch.setattr(embed, "get_kb_embedder", lambda: object())
    return seen


def _x_rooted(conn, cid, handle):
    from pipeline.kb import resolve
    schema.upsert_entity(conn, cid, name=handle, profile={"handle": handle})
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=[cid])
    return {"canonical_id": cid, "name": handle}


def test_an_oracle_that_got_breadth_discovers_once_not_twice(conn, real_ingest, monkeypatch):
    """Two passes, ONE discovery and ONE footprint route — and still two X pulls, because the X
    window is the only thing the second pass can add."""
    picks = [_x_rooted(conn, "x:user:1", "deep")]
    monkeypatch.setattr(oracles, "x_since_last", lambda c, cid: utc_now() - timedelta(days=200))

    _run(conn, picks, monkeypatch=monkeypatch, x_lookback=oracles.X_SINCE_LAST)

    assert real_ingest["discover"] == ["deep"]
    assert real_ingest["footprint"] == ["x:user:1"]
    assert real_ingest["x"] == ["deep", "deep"]            # breadth, then depth


def test_an_oracle_that_got_no_breadth_still_does_its_own_off_x_work(conn, real_ingest,
                                                                     monkeypatch):
    """`x_only` says "another pass already did these", and for a single-pass Oracle that is false.
    A person pulled two days ago never gets a breadth pass, so nothing else will do their archive."""
    picks = [_x_rooted(conn, "x:user:2", "shallow")]
    monkeypatch.setattr(oracles, "x_since_last", lambda c, cid: utc_now() - timedelta(days=2))

    _run(conn, picks, monkeypatch=monkeypatch, x_lookback=oracles.X_SINCE_LAST)

    assert real_ingest["discover"] == ["shallow"]
    assert real_ingest["footprint"] == ["x:user:2"]
    assert real_ingest["x"] == ["shallow"]


def test_a_substack_rooted_oracle_still_discovers_on_the_deep_pass(conn, real_ingest, monkeypatch):
    """⚠️ THE RISK CASE. A Substack-rooted Oracle's X handle exists ONLY in the discovered
    sources, so skipping discovery there would silently drop the timeline the deep pass is FOR."""
    from pipeline.kb import expand
    picks = [_x_rooted(conn, "x:user:3", "subby")]
    monkeypatch.setattr(expand, "_root_profile",
                        lambda c, o: {"seed": "https://subby.substack.com", "seed_type": "substack"})
    monkeypatch.setattr(expand, "_x_handle_to_pull", lambda root, profile: "subby")
    monkeypatch.setattr(oracles, "x_since_last", lambda c, cid: utc_now() - timedelta(days=200))

    _run(conn, picks, monkeypatch=monkeypatch, x_lookback=oracles.X_SINCE_LAST)

    assert len(real_ingest["discover"]) == 2               # both passes — the handle needs it
    assert real_ingest["footprint"] == ["x:user:3"]        # but the ARCHIVE still runs once
    assert real_ingest["x"] == ["subby", "subby"]


# ── breadth asks for what it is MISSING, not a flat 30 days (§J, 2026-09-14) ───
#
# ⚠️ The flat window made repeating the call an INFINITE LOOP. Second live run, fully refilled
# meter: the two Oracles already covered back to 2026-08-15 re-walked 204 and 400 tweets to
# rediscover it, and the two who had never been pulled were refused right after — the same two,
# because `_ordered_picks` is stable. 67.6s and ~25 requests for 4 atoms and nobody new.

def test_a_recently_pulled_oracle_asks_only_for_what_arrived_since(conn, ingests, monkeypatch):
    """Their `covered_from` already satisfies the breadth window, so re-walking it buys nothing
    and costs the requests the never-pulled Oracles need."""
    picks = [_oracle(conn, "x:user:1", "A")]
    two_days = utc_now() - timedelta(days=2)
    monkeypatch.setattr(oracles, "x_since_last", lambda c, cid: two_days)

    _run(conn, picks, monkeypatch=monkeypatch, x_lookback="1yr")

    breadth_window = ingests[0][1]
    assert breadth_window == two_days


def test_a_never_pulled_oracle_still_gets_the_whole_breadth_window(conn, ingests, monkeypatch):
    """The breadth GUARANTEE. `_breadth_window` is never wider than the floor, so the person with
    no coverage at all is exactly the person it leaves alone."""
    from pipeline.kb.oracle_refresh import BREADTH_WINDOW_DAYS

    picks = [_oracle(conn, "x:user:1", "A")]
    monkeypatch.setattr(oracles, "x_since_last", lambda c, cid: None)

    _run(conn, picks, monkeypatch=monkeypatch, x_lookback="1yr")

    breadth_window = ingests[0][1]
    assert abs((utc_now() - breadth_window).days - BREADTH_WINDOW_DAYS) <= 1


def test_a_pull_older_than_the_breadth_floor_does_not_widen_it(conn, ingests, monkeypatch):
    """`max`, not "whichever we last used". A pull from 200 days ago is not a reason to make the
    breadth pass 200 days deep — that is the depth pass's window, and its budget."""
    picks = [_oracle(conn, "x:user:1", "A")]
    monkeypatch.setattr(oracles, "x_since_last",
                        lambda c, cid: utc_now() - timedelta(days=200))

    _run(conn, picks, monkeypatch=monkeypatch, x_lookback="1yr")

    assert ingests[0][1] > utc_now() - timedelta(days=31)


def test_an_unreadable_last_pull_leaves_the_flat_floor_in_place(conn, ingests, monkeypatch):
    """Fail-safe, and the direction matters: spending requests is recoverable, missing posts the
    walk never asked for is not."""
    picks = [_oracle(conn, "x:user:1", "A")]
    monkeypatch.setattr(oracles, "x_since_last", lambda c, cid: 1 / 0)

    _run(conn, picks, monkeypatch=monkeypatch, x_lookback="1yr")

    assert ingests[0][1] < utc_now() - timedelta(days=29)


# ── the durable run record ─────────────────────────────────────────────────────
# The loop is the only thing that knows a RUN — `_ingest_oracle` is shared with `add_oracle`,
# which has none. These prove what the loop writes down, not what it returns; the two stop being
# the same thing in the commit after next, and the record is the half that survives a call the
# client cut off.

def test_the_whole_roster_is_on_the_record_before_the_first_request(conn, monkeypatch):
    """⚠️ THE PROPERTY THE 60-SECOND WALL DESTROYED. "Who is still waiting" has to be answerable
    one second into a twelve-minute pull, so every pick is written before anything is pulled —
    an Oracle absent from the report reads as an Oracle who failed."""
    from pipeline.kb import pull_runs
    seen: list[list[str]] = []

    def _fake(conn_, embedder, oracle, **kw):
        run = pull_runs.latest_run(conn_)
        seen.append([o.canonical_id for o in pull_runs.oracles_for(conn_, run.run_id)])
        return {"oracle_id": oracle["canonical_id"], "name": oracle.get("name"),
                "atoms_added": 1, "results": []}

    monkeypatch.setattr(oracles, "_ingest_oracle", _fake)
    from pipeline.kb import embed
    monkeypatch.setattr(embed, "get_kb_embedder", lambda: object())
    picks = [_oracle(conn, f"x:user:{i}", f"P{i}") for i in range(3)]

    _run(conn, picks, monkeypatch=monkeypatch, x_lookback="1yr")

    assert seen[0] == ["x:user:0", "x:user:1", "x:user:2"]


def test_the_record_walks_the_same_order_the_loop_does(conn, ingests, monkeypatch):
    """One ranking, computed once. Two orderings of the same people is how two surfaces end up
    disagreeing about who the user cares most about."""
    from pipeline.kb import pull_runs, screen
    picks = [_oracle(conn, "x:user:1", "A"), _oracle(conn, "x:user:2", "B")]
    monkeypatch.setattr(screen, "rank_candidates", lambda c: 1 / 0)   # fall back to given order

    _run(conn, picks, monkeypatch=monkeypatch, x_lookback="1yr")

    run = pull_runs.latest_run(conn)
    assert [o.canonical_id for o in pull_runs.oracles_for(conn, run.run_id)] == \
        [cid for cid, _since in ingests][:2]


def test_one_person_gets_one_row_carrying_the_merged_result(conn, ingests, monkeypatch):
    """Breadth and depth are two visits, and `_merge_passes` is what makes them one result. A
    record written at the breadth pass would count the person twice and split their atoms."""
    from pipeline.kb import pull_runs
    picks = [_oracle(conn, "x:user:1", "A")]

    _run(conn, picks, monkeypatch=monkeypatch, x_lookback="1yr")

    run = pull_runs.latest_run(conn)
    rows = pull_runs.oracles_for(conn, run.run_id)
    assert len(rows) == 1 and rows[0].state == "done"
    assert rows[0].result["oracle_id"] == "x:user:1"


def test_a_finished_loop_closes_its_run(conn, ingests, monkeypatch):
    """`finished_at` on the PARENT is the only thing that means complete — every row done with
    the parent still open is a loop that died on the way out, and that is a different fact."""
    from pipeline.kb import pull_runs
    _run(conn, [_oracle(conn, "x:user:1", "A")], monkeypatch=monkeypatch, x_lookback="1yr")

    run = pull_runs.latest_run(conn)
    assert run.finished_at is not None
    assert pull_runs.run_status(run, alive=False) == "complete"


def test_the_windows_are_on_the_record_from_the_first_second(conn, ingests, monkeypatch):
    """The lookback never depended on the outcome — it was assembled after the pull only because
    that is where the return value was built. On the record, a call that starts a pull can say
    what it is about to ask for."""
    from pipeline.kb import pull_runs
    _run(conn, [_oracle(conn, "x:user:1", "A")], monkeypatch=monkeypatch, x_lookback="1yr")

    recorded = pull_runs.latest_run(conn).lookback
    # The DEPTH window the user asked for, not the breadth floor the loop uses on the way there.
    assert parse_ts(recorded["x_since"]) < utc_now() - timedelta(days=360)
    assert recorded["x"].startswith("since ")


def test_a_record_that_cannot_be_written_does_not_sink_the_pull(conn, ingests, monkeypatch):
    """Fail-safe, and the direction is the whole point: a store that cannot be written loses the
    REPORT. Raising would lose the CONTENT too, which is strictly worse.

    ⚠️ AND IT UNDER-REPORTS RATHER THAN OVER-CLAIMS. There is no in-memory backstop any more —
    the loop's return lives on a thread nobody holds — so a store that will not record says it
    reached nobody, and the writers stay on the roster as unreached. `seed_from_entities` makes
    the same call and says why: a dropped row "self-heals into a re-pull, which is the safe
    direction, not into a coverage claim". Dedup absorbs a re-pull; a claim nothing backs is
    permanent."""
    from pipeline.kb import pull_runs
    monkeypatch.setattr(pull_runs, "start_oracle", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(pull_runs, "finish_oracle", lambda *a, **k: 1 / 0)

    out = _run(conn, [_oracle(conn, "x:user:1", "A")], monkeypatch=monkeypatch, x_lookback="1yr")

    assert ingests                              # the pull really ran, and the atoms landed
    assert out["ingested_oracles"] == 0         # and it claims nothing it cannot see
    assert pull_runs.oracles_for(conn, out["run_id"])[0].state == "waiting"


def test_the_report_is_read_back_rather_than_carried(conn, ingests, monkeypatch):
    """THE WIRE CHANGE, stated as a test. What the loop returns is no longer what anybody
    reports — the record is. A loop that returns nothing while the record holds a result must
    still produce that result, because after `ingest` stops waiting there IS no return value."""
    from pipeline.kb import pull_runs
    real = oracle_tools._breadth_then_depth

    def _amnesiac(*a, **kw):
        real(*a, **kw)
        return []                       # the stack frame nobody will ever reach again

    monkeypatch.setattr(oracle_tools, "_breadth_then_depth", _amnesiac)

    out = _run(conn, [_oracle(conn, "x:user:1", "A")], monkeypatch=monkeypatch, x_lookback="1yr")

    assert out["ingested_oracles"] == 1
    assert out["results"][0]["oracle_id"] == "x:user:1"
    assert "record" not in out


def test_the_ingest_result_names_its_run(conn, ingests, monkeypatch):
    """The handle a later call needs to ask "where did that get to". Useless today, load-bearing
    the moment `ingest` stops waiting."""
    from pipeline.kb import pull_runs
    out = _run(conn, [_oracle(conn, "x:user:1", "A")], monkeypatch=monkeypatch, x_lookback="1yr")

    assert out["run_id"] == pull_runs.latest_run(conn).run_id


# ── one pull at a time, and a second one JOINS ─────────────────────────────────

def test_a_second_ingest_joins_the_running_pull_rather_than_starting_a_rival(conn, ingests,
                                                                             monkeypatch):
    """Two pulls would split the same two x.com buckets — the meter the breadth/depth ordering
    exists to protect. But bouncing the call drops writers the user just asked for."""
    from pipeline.kb import pull_runs
    picks = [_oracle(conn, "x:user:1", "A")]
    running = pull_runs.open_run(conn, kind="ingest",
                                 picks=[{"canonical_id": "x:user:9", "name": "Z"}])

    out = _run(conn, picks, monkeypatch=monkeypatch, x_lookback="1yr")

    assert out["status"] == "joined"
    assert out["run_id"] == running and out["added"] == 1
    assert ingests == []                        # nothing was pulled by this call
    assert [o.canonical_id for o in pull_runs.oracles_for(conn, running)] == \
        ["x:user:9", "x:user:1"]


def test_the_joined_answer_is_a_result_not_an_error(conn, ingests, monkeypatch):
    """⚠️ A host handed "already running" as an error reasons its way into a retry — measured
    2026-09-14, where "do not call it again" was read and rationalised into a smaller call that
    timed out having done less."""
    from pipeline.kb import pull_runs
    pull_runs.open_run(conn, kind="ingest", picks=[{"canonical_id": "x:user:9", "name": "Z"}])

    out = _run(conn, [_oracle(conn, "x:user:1", "A")], monkeypatch=monkeypatch, x_lookback="1yr")

    assert "error" not in out
    assert "progress" in out["message"]
    for word in ("timed out", "failed", "retry", "try again"):
        assert word not in out["message"].lower()


def test_a_dead_pull_is_not_something_to_join(conn, ingests, monkeypatch):
    """An unfinished run whose holder is gone is not a queue — it is the thing the next call
    reports as stopped. Joining it would queue people onto nothing."""
    from pipeline.kb import pull_runs
    stale = pull_runs.open_run(conn, kind="ingest",
                               picks=[{"canonical_id": "x:user:9", "name": "Z"}])
    conn.execute("UPDATE pull_runs SET heartbeat_at = 0.0 WHERE run_id = ?", (stale,))
    conn.commit()

    out = _run(conn, [_oracle(conn, "x:user:1", "A")], monkeypatch=monkeypatch, x_lookback="1yr")

    assert out.get("status") != "joined"
    assert out["run_id"] != stale and out["ingested_oracles"] == 1


def test_a_writer_queued_mid_pull_is_actually_pulled(conn, monkeypatch):
    """⚠️ THE TEST THAT MAKES THE PROMISE TRUE. Without the late-arrival pass the record would say
    somebody was waiting while the loop iterated a fixed list — and the user would have been told
    their newest writer was on the way."""
    from pipeline.kb import pull_runs
    calls: list[tuple[str, object]] = []
    joined = {"done": False}

    def _fake(conn_, embedder, oracle, *, x_since=None, **kw):
        calls.append((oracle["canonical_id"], x_since))
        if not joined["done"]:               # a second `ingest` lands mid-pull
            joined["done"] = True
            run = pull_runs.latest_run(conn_)
            pull_runs.add_picks(conn_, run.run_id, [{"canonical_id": "x:user:2", "name": "B",
                                                     "window": "2024-01-01T00:00:00+00:00"}])
        return {"oracle_id": oracle["canonical_id"], "atoms_added": 1, "results": []}

    monkeypatch.setattr(oracles, "_ingest_oracle", _fake)
    from pipeline.kb import embed
    monkeypatch.setattr(embed, "get_kb_embedder", lambda: object())
    a = _oracle(conn, "x:user:1", "A")
    b = _oracle(conn, "x:user:2", "B")
    monkeypatch.setattr(oracles, "confirmed_oracles", lambda c: [a, b])
    from pipeline.ingestion import x_graphql
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: True)

    out = oracle_tools._ingest(conn, canonical_ids=["x:user:1"], force=False, x_lookback="1yr",
                               web_lookback=None, scholar_lookback=None, scholar_topics=None)

    assert "x:user:2" in [cid for cid, _ in calls]
    assert out["ingested_oracles"] == 2
    late = [o for o in pull_runs.oracles_for(conn, out["run_id"])
            if o.canonical_id == "x:user:2"][0]
    assert late.state == "done"


def test_a_latecomer_is_pulled_with_the_window_they_were_queued_with(conn, monkeypatch):
    """Re-deriving it would silently hand somebody the adapter's 183-day default after the call
    that queued them asked for something else."""
    from pipeline.kb import pull_runs
    calls: list[tuple[str, object]] = []
    joined = {"done": False}

    def _fake(conn_, embedder, oracle, *, x_since=None, **kw):
        calls.append((oracle["canonical_id"], x_since))
        if not joined["done"]:
            joined["done"] = True
            pull_runs.add_picks(conn_, pull_runs.latest_run(conn_).run_id,
                                [{"canonical_id": "x:user:2", "name": "B",
                                  "window": "2019-03-03T00:00:00+00:00"}])
        return {"oracle_id": oracle["canonical_id"], "atoms_added": 1, "results": []}

    monkeypatch.setattr(oracles, "_ingest_oracle", _fake)
    from pipeline.kb import embed
    monkeypatch.setattr(embed, "get_kb_embedder", lambda: object())
    a = _oracle(conn, "x:user:1", "A")
    b = _oracle(conn, "x:user:2", "B")
    monkeypatch.setattr(oracles, "confirmed_oracles", lambda c: [a, b])
    from pipeline.ingestion import x_graphql
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: True)

    oracle_tools._ingest(conn, canonical_ids=["x:user:1"], force=False, x_lookback="1yr",
                         web_lookback=None, scholar_lookback=None, scholar_topics=None)

    window = [since for cid, since in calls if cid == "x:user:2"][0]
    assert window.year == 2019


def test_a_latecomer_unconfirmed_since_being_queued_is_skipped_not_crashed(conn, ingests,
                                                                           monkeypatch):
    """Fail-safe: a roster row whose Oracle is gone has nothing to pull, and one latecomer must
    never sink the run."""
    from pipeline.kb import pull_runs
    a = _oracle(conn, "x:user:1", "A")
    monkeypatch.setattr(oracles, "confirmed_oracles", lambda c: [a])
    real_open = pull_runs.open_run

    def _open(conn_, **kw):
        rid = real_open(conn_, **kw)
        pull_runs.add_picks(conn_, rid, [{"canonical_id": "x:user:404", "name": "Ghost"}])
        return rid

    monkeypatch.setattr(pull_runs, "open_run", _open)

    out = _run(conn, [a], monkeypatch=monkeypatch, x_lookback="1yr")

    assert out["ingested_oracles"] == 1
    ghost = [o for o in pull_runs.oracles_for(conn, out["run_id"])
             if o.canonical_id == "x:user:404"][0]
    assert ghost.state == "waiting"
