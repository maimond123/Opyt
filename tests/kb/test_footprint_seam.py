"""The MERGE SEAM between `main`'s footprint adapters and stage6's onboarding callers.

`main` rewrote the four adapters (three-state fetch contract + producer parallelism); the callers
here — `expand._route_source`, `onboard_footprint`, `oracles._ingest_oracle` — were written on
another branch against the OLD contract. ZERO files overlapped, so git merged clean and both test
suites stayed green: every existing add_oracle/expand test MOCKS the adapters with a stub that
returns `{"adapter": name}`, which can never express "the host stopped us".

These tests feed the callers what `main`'s adapters ACTUALLY return and assert the caller does not
mis-report it. The invariant they lock:

    a run that ingested NOTHING because the host stopped us must never be reported as `ingested`

That is the same fact-about-the-network vs fact-about-the-content confusion `classify_fetch` kills
per POST, reappearing one level up per RUN — in callers that lived on a different branch and so
never got the memo.
"""
from __future__ import annotations

import pytest

from pipeline.kb import (eligibility, expand, ingest_blog, ingest_github, ingest_substack,
                         ingest_x_footprint, onboard_footprint, oracles, resolve, schema)

_SUB = "https://carol.substack.com"
_BLOG = "https://carol.dev"


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


@pytest.fixture(autouse=True)
def managed_x_session(monkeypatch):
    """The X-adapter seam tests exercise an established X connection."""
    from pipeline.ingestion import x_graphql
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: True)


# ── the shapes `main`'s adapters actually return ───────────────────────────────

def _blocked():
    """A Substack archive walk STOPPED by the host — `SubstackListingError` caught inside
    `sync_substack_footprint`. Nothing ingested, nothing marked seen, retried next run.
    Mirrors pipeline/kb/ingest_substack.py's blocked return."""
    return {"source": "substack-footprint", "added": 0, "skipped": 0, "paywalled": 0,
            "no_body": 0, "undetermined": 1, "failed": 0, "gate_rejected": 0,
            "dispatched": 0, "producer_failed": 0, "total": 0,
            "error": "archive listing incomplete: archive listing stopped at offset=50"}


def _bad_input():
    """A CALLER bug, not a transient block: an error with no counters at all. Must read as
    `error` (someone should look) and NOT as `blocked` (which means 'retry, it's fine')."""
    return {"source": "blog", "error": "no blog_url"}


def _ok(**over):
    """A normal successful run."""
    d = {"source": "blog", "added": 7, "skipped": 2, "undetermined": 0, "failed": 0,
         "dispatched": 9, "producer_failed": 0, "gate_rejected": 0, "total": 7}
    d.update(over)
    return d


def _pass_gate(monkeypatch):
    """Patch the MODULE object, not an attribute of a caller: `onboard_footprint` imports its
    adapters inside the function body, so both callers resolve to this same module at call time."""
    monkeypatch.setattr(eligibility, "gate",
                        lambda conn, url, **kw: eligibility.GateDecision("ingest", "stub"))


def _src(stype, url, trusted=True):
    return {"source_type": stype, "url": url, "metadata": {},
            "trust": {"trusted": trusted, "reasons": []}}


# ── Seam 2: expand._route_source ───────────────────────────────────────────────

def test_route_source_blocked_listing_is_not_reported_ingested(conn, monkeypatch):
    """The headline seam. Cloudflare stopped the archive walk, zero atoms were written — the
    caller must not hand the user back an `ingested` record."""
    _pass_gate(monkeypatch)
    monkeypatch.setattr(ingest_substack, "sync_substack_footprint",
                        lambda conn, embedder, **kw: _blocked())

    out = expand._route_source(conn, None, _src("substack", _SUB), author_name="Carol", limit=0)

    assert "ingested" not in out, "a blocked listing must not be labelled ingested"
    assert out["blocked"]["undetermined"] == 1
    assert "listing incomplete" in out["reason"]


def test_route_source_bad_input_is_error_not_blocked(conn, monkeypatch):
    """`blocked` means transient — retry and it clears. A caller bug must stay `error`, or the
    distinction is worthless and every error reads as 'ignore me, it'll retry'."""
    _pass_gate(monkeypatch)
    monkeypatch.setattr(ingest_blog, "sync_blog_footprint",
                        lambda conn, embedder, **kw: _bad_input())

    out = expand._route_source(conn, None, _src("blog", _BLOG), author_name="Carol", limit=0)

    assert "ingested" not in out and "blocked" not in out
    assert "no blog_url" in out["error"]


def test_route_source_successful_run_still_reports_ingested(conn, monkeypatch):
    """Regression guard on the fix: the ordinary path must be untouched."""
    _pass_gate(monkeypatch)
    monkeypatch.setattr(ingest_blog, "sync_blog_footprint",
                        lambda conn, embedder, **kw: _ok())

    out = expand._route_source(conn, None, _src("blog", _BLOG), author_name="Carol", limit=0)

    assert out["ingested"]["added"] == 7
    assert "blocked" not in out and "error" not in out


def test_route_source_partial_run_is_still_ingested(conn, monkeypatch):
    """A run that ingested SOME posts and lost others is a partial SUCCESS, not a block. Only a
    run that produced nothing because we were stopped is `blocked` — otherwise `undetermined`
    (which the blog adapter also increments per-post on a normal run) would flip healthy runs."""
    _pass_gate(monkeypatch)
    monkeypatch.setattr(ingest_blog, "sync_blog_footprint",
                        lambda conn, embedder, **kw: _ok(undetermined=2, producer_failed=1))

    out = expand._route_source(conn, None, _src("blog", _BLOG), author_name="Carol", limit=0)

    assert out["ingested"]["added"] == 7 and "blocked" not in out


# ── Seam 2: onboard_footprint (the LIVE add_oracle path) ───────────────────────

def test_onboard_blocked_source_is_not_counted_as_ingested(conn, monkeypatch):
    """`onboard_footprint`'s summary is COUNTS, and those counts are what the MCP tool hands the
    model, which is what the user reads. `ingested: 1` for a run that wrote zero atoms is the
    user-facing lie this whole file exists to stop."""
    _pass_gate(monkeypatch)
    monkeypatch.setattr(ingest_substack, "sync_substack_footprint",
                        lambda conn, embedder, **kw: _blocked())

    out = onboard_footprint.onboard_footprint(conn, None, "x:user:1", [_src("substack", _SUB)],
                                              author_name="Carol")

    assert out["ingested"] == 0
    assert out["blocked"] == 1
    rec = out["results"][0]
    assert rec["action"] == "blocked" and "listing incomplete" in rec["detail"]


def test_onboard_blocked_source_does_not_sink_the_others(conn, monkeypatch):
    """Fail-safe per source: one blocked Substack must not cost the Oracle their blog."""
    _pass_gate(monkeypatch)
    monkeypatch.setattr(ingest_substack, "sync_substack_footprint",
                        lambda conn, embedder, **kw: _blocked())
    monkeypatch.setattr(ingest_blog, "sync_blog_footprint",
                        lambda conn, embedder, **kw: _ok())

    out = onboard_footprint.onboard_footprint(
        conn, None, "x:user:1", [_src("substack", _SUB), _src("blog", _BLOG)],
        author_name="Carol")

    assert out["ingested"] == 1 and out["blocked"] == 1
    assert {r["type"]: r["action"] for r in out["results"]} == {"substack": "blocked",
                                                                "blog": "ingested"}


def test_onboard_bad_input_summary_is_error_not_ingested(conn, monkeypatch):
    """An adapter that RETURNS an error (rather than raising) must not slip through as ingested —
    the try/except only catches the raising half of the contract."""
    _pass_gate(monkeypatch)
    monkeypatch.setattr(ingest_blog, "sync_blog_footprint",
                        lambda conn, embedder, **kw: _bad_input())

    out = onboard_footprint.onboard_footprint(conn, None, "x:user:1", [_src("blog", _BLOG)],
                                              author_name="Carol")

    assert out["ingested"] == 0
    assert out["results"][0]["action"] == "error"


# ── Seam 3: the counters must SURFACE, not dissolve into a string ──────────────

def test_onboard_surfaces_lost_posts_as_numbers(conn, monkeypatch):
    """`producer_failed` is the ONLY place a post that vanished mid-run appears at all. Stringified
    into `detail` it is unreadable to any caller. It must be a top-level integer."""
    _pass_gate(monkeypatch)
    monkeypatch.setattr(ingest_blog, "sync_blog_footprint",
                        lambda conn, embedder, **kw: _ok(producer_failed=3, undetermined=2))

    out = onboard_footprint.onboard_footprint(conn, None, "x:user:1", [_src("blog", _BLOG)],
                                              author_name="Carol")

    assert out["producer_failed"] == 3
    assert out["undetermined"] == 2


def test_onboard_surfaces_atoms_added_vs_dispatched(conn, monkeypatch):
    """Seam 1's user-visible symptom. `limit` now caps posts DISPATCHED, so `limit=20` can yield
    far fewer atoms. Dispatch-bounded is the honest bound (the consumer lags the producer, so a
    consumer-side cap over-spends before it notices) — but only if the caller REPORTS the gap.
    These two integers are what make the semantics defensible instead of merely surprising."""
    _pass_gate(monkeypatch)
    monkeypatch.setattr(ingest_blog, "sync_blog_footprint",
                        lambda conn, embedder, **kw: _ok(added=4, dispatched=20))

    out = onboard_footprint.onboard_footprint(conn, None, "x:user:1", [_src("blog", _BLOG)],
                                              author_name="Carol")

    assert out["atoms_added"] == 4
    assert out["dispatched"] == 20


# ── Seam 2: the X timeline pull inside `_ingest_oracle` ────────────────────────

def _confirmed_oracle(conn, eid="x:user:1", *, name="Carol", handle="carol"):
    schema.upsert_entity(conn, eid, name=name, profile={"handle": handle})
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=[eid])
    return [o for o in oracles.confirmed_oracles(conn) if o["canonical_id"] == eid][0]


def test_x_timeline_failure_summary_is_not_recorded_as_ingested(conn, monkeypatch):
    """`oracles._ingest_oracle` hardcodes `action: "ingested"` for the X pull regardless of what
    came back — the try/except only sees a RAISE, and `sync_x_footprint` reports bad input by
    RETURNING an error dict. Same seam, third call site."""
    import importlib
    dp = importlib.import_module("pipeline.ingestion.discover_profile")
    monkeypatch.setattr(dp, "discover_profile",
                        lambda seed, seed_type="x", **kw: {"username": seed, "sources": []})
    monkeypatch.setattr(ingest_x_footprint, "sync_x_footprint",
                        lambda conn, embedder, **kw: {"source": "x-footprint", "error": "no handle"})

    o = _confirmed_oracle(conn)
    out = oracles._ingest_oracle(conn, None, o)

    x = [r for r in out["results"] if r["type"] == "x"][0]
    assert x["action"] != "ingested"
    assert "no handle" in x["detail"]


def test_x_timeline_success_still_records_ingested(conn, monkeypatch):
    """Regression guard: the ordinary X pull is untouched."""
    import importlib
    dp = importlib.import_module("pipeline.ingestion.discover_profile")
    monkeypatch.setattr(dp, "discover_profile",
                        lambda seed, seed_type="x", **kw: {"username": seed, "sources": []})
    monkeypatch.setattr(ingest_x_footprint, "sync_x_footprint",
                        lambda conn, embedder, **kw: _ok(source="x-footprint"))

    o = _confirmed_oracle(conn)
    out = oracles._ingest_oracle(conn, None, o)

    x = [r for r in out["results"] if r["type"] == "x"][0]
    assert x["action"] == "ingested"


# ── a thin meter STARTS the walk now (2026-09-14) ─────────────────────────────
#
# ⚠️ THIS BLOCK IS THE INVERSE OF WHAT IT SAID FOR ONE DAY. A reservation here refused an X walk
# whenever the bucket was below a fixed floor, and it was CORRECT given its premise:
# `_pull_own_timeline` was all-or-nothing, so a walk that ran dry mid-way raised with nothing
# written and its requests bought nothing. Refusing first reached the same state for free.
#
# The premise was the defect. Measured twice on 2026-09-14 against the live store, a foreground
# ingest over four confirmed Oracles covered two of them and left ~25 requests unspent — and
# re-running on a refilled meter changed nothing, because `_ordered_picks` is stable and strands
# the same two forever. Durable partial walks removed the premise, so the reserve went with it,
# in both of the places it had been written (`oracle_refresh._budget_spent` was the other).
#
# What remains is `x_graphql_core._refuse_if_spent` — which refuses a request x.com has already
# said it will 429. Evidence, not prediction.

def _x_oracle(conn, monkeypatch, calls, summary=None):
    import importlib
    dp = importlib.import_module("pipeline.ingestion.discover_profile")
    monkeypatch.setattr(dp, "discover_profile",
                        lambda seed, seed_type="x", **kw: {"username": seed, "sources": []})
    monkeypatch.setattr(ingest_x_footprint, "sync_x_footprint",
                        lambda conn, embedder, **kw: calls.append(kw)
                        or (summary if summary is not None else _ok(source="x-footprint")))
    return _confirmed_oracle(conn)


def test_a_thin_meter_starts_the_walk_and_keeps_what_it_gets(conn, monkeypatch):
    """The whole inversion in one assertion: the walk is ATTEMPTED. Nothing in `_ingest_oracle`
    predicts the meter any more, so the requests that used to go unspent are spent."""
    import time
    from pipeline.ingestion import x_graphql_core as core

    calls = []
    o = _x_oracle(conn, monkeypatch, calls,
                  summary={"source": "x-footprint", "added": 7, "fetched": 12,
                           "partial": True, "covered_from": None})
    monkeypatch.setattr(core, "rate_budget", lambda op: (1, time.time() + 600))

    out = oracles._ingest_oracle(conn, None, o)

    assert len(calls) == 1                   # the decisive one: the walk was ATTEMPTED
    x = [r for r in out["results"] if r["type"] == "x"][0]
    assert x["action"] == "ingested" and x["partial"] is True
    assert x["covered_from"] is None         # …and claims nothing it cannot defend


def test_only_x_com_refusing_still_defers(conn, monkeypatch):
    """The one route to `deferred` that remains, and the reason the user-facing wording about a
    window running out stopped being half-wrong: now it really has."""
    from pipeline.ingestion import x_graphql_core as core

    import importlib
    dp = importlib.import_module("pipeline.ingestion.discover_profile")
    monkeypatch.setattr(dp, "discover_profile",
                        lambda seed, seed_type="x", **kw: {"username": seed, "sources": []})

    def _refused(conn, embedder, **kw):
        raise core.XRateLimited("nothing observed")
    monkeypatch.setattr(ingest_x_footprint, "sync_x_footprint", _refused)

    out = oracles._ingest_oracle(conn, None, _confirmed_oracle(conn))

    x = [r for r in out["results"] if r["type"] == "x"][0]
    assert x["action"] == "deferred" and out["deferred"] == 1


def test_the_reserve_exists_in_neither_copy(conn):
    """One bucket, one answer. `_x_budget_spent` and `oracle_refresh._budget_spent` were the same
    function written twice, sharing one justification — so removing one and keeping the other
    would have recreated the split that `test_the_reserve_is_the_refresh_rail_s_own_number` was
    written to catch. They went together."""
    from pipeline.kb import oracle_refresh

    assert not hasattr(oracles, "_x_budget_spent")
    assert not hasattr(oracle_refresh, "_budget_spent")
    assert not hasattr(oracle_refresh, "BACKFILL_MIN_BUDGET")


# ── the UN-MOCKED path: real adapter, only the NETWORK faked ───────────────────
# Everything above hand-copies the adapter's return shape into `_blocked()`/`_ok()`, so it would
# keep passing if the adapter changed what it returns — a contract test whose copy of the contract
# can drift. These two run the REAL `sync_substack_footprint` and stub only the network boundary
# (`_fetch_all_posts` / `_fetch_full_post`), so the caller and the adapter are proven against each
# other. This is the test whose absence let the seam open in the first place.

_REAL_BODY = ("<h2>Intro</h2><p>Autonomous agents compose small tools into larger systems, and "
              "the framework question is which library owns the loop.</p>")


def _real_post(pid, slug="scaling-laws"):
    return {"id": pid, "title": "Scaling Laws", "subtitle": "a deep dive",
            "post_date": "2026-05-01T00:00:00Z", "audience": "everyone", "slug": slug,
            "canonical_url": f"{_SUB}/p/{slug}", "wordcount": 900, "body_html": None}


def test_end_to_end_blocked_archive_reaches_the_caller_as_blocked(conn, fake_embedder,
                                                                  monkeypatch):
    """REAL adapter, faked network: the archive walk is stopped, and the onboarding summary the
    user reads must say `blocked: 1 / ingested: 0` with zero atoms written."""
    from pipeline.ingestion.sources import substack as sub

    _pass_gate(monkeypatch)

    def _stopped(base, since=None):
        raise sub.SubstackListingError("archive listing stopped at offset=50")
    monkeypatch.setattr(sub, "_fetch_all_posts", _stopped)

    out = onboard_footprint.onboard_footprint(conn, fake_embedder, "x:user:1",
                                              [_src("substack", _SUB)], author_name="Carol")

    assert out["ingested"] == 0 and out["blocked"] == 1
    assert out["atoms_added"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM atoms").fetchone()["c"] == 0


def test_end_to_end_successful_archive_reaches_the_caller_as_ingested(conn, fake_embedder,
                                                                      monkeypatch):
    """The other half of the contract — a real archive walk that succeeds still reads as
    `ingested`, and `atoms_added` reflects atoms the adapter actually wrote. Without this, a fix
    that classified everything as `blocked` would pass the test above."""
    from pipeline.ingestion.sources import substack as sub

    _pass_gate(monkeypatch)
    monkeypatch.setattr(sub, "_fetch_all_posts",
                        lambda base, since=None: [_real_post(1), _real_post(2, "second-post")])
    monkeypatch.setattr(sub, "_fetch_full_post",
                        lambda base, slug, cookies: {"body_html": _REAL_BODY})

    out = onboard_footprint.onboard_footprint(conn, fake_embedder, "x:user:1",
                                              [_src("substack", _SUB)], author_name="Carol")

    assert out["blocked"] == 0 and out["ingested"] == 1
    assert out["atoms_added"] > 0
    assert conn.execute("SELECT COUNT(*) c FROM atoms").fetchone()["c"] == out["atoms_added"]


def test_x_dispatched_does_not_pollute_the_post_dispatch_total(conn, monkeypatch):
    """`dispatched` is NOT the same unit across adapters: substack/blog count POSTS handed to the
    pool (what `limit` caps), `sync_x_footprint` counts REFERENCED-LINK backfills. Folding X's
    figure into the cross-source total produces a number in no unit, and corrupts the
    `dispatched` vs `atoms_added` read that makes `limit`'s semantics legible.

    Regression guard for a defect the LIVE run caught and every offline test missed, because the
    stub adapters all returned the same made-up shape."""
    import importlib
    dp = importlib.import_module("pipeline.ingestion.discover_profile")
    monkeypatch.setattr(dp, "discover_profile",
                        lambda seed, seed_type="x", **kw: {"username": seed, "sources": []})
    # A REAL x-footprint shape: 2 atoms added, 9 referenced-LINK dispatches (not posts).
    monkeypatch.setattr(ingest_x_footprint, "sync_x_footprint",
                        lambda conn, embedder, **kw: {"source": "x-footprint", "added": 2,
                                                      "dispatched": 9, "skipped": 0, "failed": 0})

    o = _confirmed_oracle(conn)
    out = oracles._ingest_oracle(conn, None, o)

    assert out["atoms_added"] == 2          # `added` IS comparable across adapters
    assert out.get("dispatched", 0) == 0    # X's link-dispatch count must not leak into the total
    x = [r for r in out["results"] if r["type"] == "x"][0]
    assert x["stats"]["dispatched"] == 9    # still visible where its context disambiguates it


# ── the per-source stamp (2026-09-13) ──────────────────────────────────────────
#
# ⚠️ THE BUG THIS PINS. `_record_coverage` and `oracle_reviews.record_outcomes` both fired ONCE,
# after the last source, while `AtomSink` flushes incrementally. A live `oracle(action='ingest')`
# killed 62s into Dwarkesh's 187-post archive therefore left atoms in the store with zero record of
# what had been attempted and an empty `oracle_sources` — a store that cannot say what it is
# missing, which is the one thing a resumable backfill has to be able to say.

def _stamped(conn):
    return {f"{r['source_type']}:{r['source_key']}"
            for r in conn.execute("SELECT source_type, source_key FROM oracle_sources "
                                  "WHERE last_pulled_at IS NOT NULL")}


def _two_source_oracle(conn, monkeypatch):
    """An Oracle whose cluster owns BOTH a Substack and a blog, so the router routes two sources in
    a known order. The identity links are what `resolve` merges the discovered members on."""
    import importlib
    dp = importlib.import_module("pipeline.ingestion.discover_profile")
    schema.upsert_entity(conn, "x:user:1", name="Carol", profile={"handle": "carol"},
                         identity_links=[_SUB, _BLOG])
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=["x:user:1"])
    monkeypatch.setattr(dp, "discover_profile",
                        lambda seed, seed_type="x", **kw: {
                            "username": seed,
                            "sources": [_src("substack", _SUB), _src("blog", _BLOG)]})
    monkeypatch.setattr(expand, "_x_handle_to_pull", lambda root, profile: None)
    _pass_gate(monkeypatch)
    return [o for o in oracles.confirmed_oracles(conn) if o["canonical_id"] == "x:user:1"][0]


def test_a_finished_source_is_stamped_before_the_next_one_starts(conn, monkeypatch):
    """The kill test. The blog adapter dies the way a killed process does — a BaseException that no
    `except Exception` in the router or the engine catches — and the Substack that already finished
    must ALREADY be recorded. Under the end-of-run stamp, nothing was."""
    def _ran(conn, embedder, **kw):
        # What the real adapter does BEFORE its archive walk: mint the publication entity with its
        # identity link, which is the pair `_record_coverage` joins the outcome back onto.
        from pipeline.kb import derive
        schema.upsert_entity(conn, derive.substack_entity_id(publication_url=_SUB),
                             name="Carol", identity_links=[_SUB])
        return _ok(source="substack-footprint")

    monkeypatch.setattr(ingest_substack, "sync_substack_footprint", _ran)

    def _killed(conn, embedder, **kw):
        raise KeyboardInterrupt("process killed mid-archive")

    monkeypatch.setattr(ingest_blog, "sync_blog_footprint", _killed)

    o = _two_source_oracle(conn, monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        oracles._ingest_oracle(conn, None, o)

    stamped = _stamped(conn)
    assert any(k.startswith("substack:") for k in stamped), stamped
    assert not any(k.startswith("blog:") for k in stamped), stamped


def test_a_needs_review_source_is_queued_before_the_next_one_starts(conn, monkeypatch):
    """`record_outcomes` fired end-of-run too, so a kill also lost the review queue — the record of
    a source discovery FOUND and deliberately did not ingest."""
    from pipeline.kb import oracle_reviews

    def _killed(conn, embedder, **kw):
        raise KeyboardInterrupt("process killed mid-archive")

    monkeypatch.setattr(ingest_blog, "sync_blog_footprint", _killed)

    import importlib
    dp = importlib.import_module("pipeline.ingestion.discover_profile")
    schema.upsert_entity(conn, "x:user:1", name="Carol", profile={"handle": "carol"},
                         identity_links=[_BLOG])
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=["x:user:1"])
    monkeypatch.setattr(dp, "discover_profile",
                        lambda seed, seed_type="x", **kw: {
                            "username": seed,
                            "sources": [_src("substack", _SUB, trusted=False),
                                        _src("blog", _BLOG)]})
    monkeypatch.setattr(expand, "_x_handle_to_pull", lambda root, profile: None)
    _pass_gate(monkeypatch)

    o = [x for x in oracles.confirmed_oracles(conn) if x["canonical_id"] == "x:user:1"][0]
    with pytest.raises(KeyboardInterrupt):
        oracles._ingest_oracle(conn, None, o)

    assert [r["source_url"] for r in oracle_reviews.list_open(conn)] == [_SUB]
