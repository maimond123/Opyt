"""Conversation resolution: `reconstruct_chain` and `_ConvoFetcher`'s resolved-ledger semantics.

Network is monkeypatched — this pins the WIRING, not the live scrape (validated live 2026-07-16).

Until 2026-08-30 `_ConvoFetcher` chose between two backends: twitterapi.io's `walk_thread_context`
when a key was present, and the free `TweetDetail` when it was not. The paid arm is gone and its
tests went with it, along with the backend-pick tests — a switch with one arm is not a choice. What
survived is everything about the LEDGER, which is where the real semantics live: a resolved id must
never be re-fetched, and a transient failure must never be recorded as resolved."""
from __future__ import annotations

import pytest

from pipeline.ingestion import x_graphql_core as core
from pipeline.ingestion.utils import SyncAuthError
from pipeline.kb import ingest_x


def _tw(tid, user, text=""):
    """A normalized tweet. `author.userName` is the same-author key `reconstruct_chain` matches on,
    and it is populated identically by `x_graphql_core.normalize` and by the twitterapi.io shape this
    was first written against — which is why the reconstruction never had to change."""
    return {"id": tid, "text": text, "author": {"userName": user}}


# ── reconstruct_chain (the shared, shape-agnostic reconstruction) ────────────────

def test_reconstruct_self_thread_root_keeps_author_continuation():
    # bookmark the ROOT (gustavokov case): focal first, his own continuation, strangers' replies.
    tweets = [
        _tw("1", "gustavokov", "root take"),
        _tw("2", "gustavokov", "my continuation"),
        _tw("3", "rando", "stranger reply"),          # dropped — not the author
        _tw("4", "gustavokov", "more of my thread"),
    ]
    chain = core.reconstruct_chain(tweets, "1")
    assert [t["id"] for t in chain] == ["1", "2", "4"]   # root + author's self-thread, no stranger


def test_reconstruct_reply_keeps_ancestors():
    # bookmark a REPLY: ancestor (the debate) comes before the focal.
    tweets = [_tw("1", "alice", "question"), _tw("2", "bob", "@alice answer"),
              _tw("3", "rando", "noise")]
    chain = core.reconstruct_chain(tweets, "2")
    assert [t["id"] for t in chain] == ["1", "2"]


def test_reconstruct_no_context_returns_empty():
    tweets = [_tw("1", "bob", "solo"), _tw("2", "rando", "unrelated reply")]
    assert core.reconstruct_chain(tweets, "1") == []     # only strangers reply → no thread


# ── _ConvoFetcher: the resolved-ledger semantics ─────────────────────────────────

def _fetcher(monkeypatch, checked=None):
    monkeypatch.setattr(core, "read_x_cookies", lambda: {"auth_token": "t", "ct0": "c"})
    monkeypatch.setattr(core, "auth_headers", lambda *a, **k: {})
    return ingest_x._ConvoFetcher(checked=checked if checked is not None else set())


def test_fetcher_marks_checked_on_success_even_when_no_thread(monkeypatch):
    """A genuine no-thread is a RESOLVED answer, not a miss. 91% of bookmarks have no conversation,
    so treating that as unresolved would re-fetch nearly the whole corpus every single run."""
    monkeypatch.setattr(core, "fetch_conversation", lambda tid, c, h: [])
    f = _fetcher(monkeypatch)
    assert f.chain("1") == []
    assert "1" in f.checked


def test_fetcher_does_not_mark_checked_on_transient_failure(monkeypatch):
    """The opposite direction, and the fail-safe one: a session that was rejected knows nothing
    about tweet 1, so recording it as resolved would lose that conversation permanently."""
    from pipeline.ingestion.utils import SyncAuthError

    def _dead(tid, c, h):
        raise SyncAuthError("session rejected")

    monkeypatch.setattr(core, "fetch_conversation", _dead)
    f = _fetcher(monkeypatch)
    assert f.chain("1") is None
    assert "1" not in f.checked        # unchecked → retried next run
    assert f.enabled is False          # and the whole run stops asking


def test_fetcher_returns_the_chain(monkeypatch):
    monkeypatch.setattr(core, "fetch_conversation",
                        lambda tid, c, h: [_tw("1", "a", "root"), _tw("2", "a", "cont")])
    f = _fetcher(monkeypatch)
    assert [t["id"] for t in f.chain("1")] == ["1", "2"]


def test_a_rate_limit_disables_the_fetcher_for_the_rest_of_the_run(monkeypatch):
    """`TweetDetail` is 150 per 15 minutes. Once it is spent every remaining bookmark would fail
    identically, so unread saves remain retryable without further conversation requests."""
    def _limited(tid, c, h):
        raise core.XRateLimited("request budget spent")

    monkeypatch.setattr(core, "fetch_conversation", _limited)
    f = _fetcher(monkeypatch)
    assert f.chain("1") is None
    assert f.enabled is False
    assert "1" not in f.checked        # never resolved → retried next run


# ── Step 6: sound thread-skip + ledger consistency (drives sync_bookmarks end-to-end) ──

class _FakeConvo:
    """Stand-in for _ConvoFetcher that RECORDS which tids the loop asked it to fetch, and mirrors
    the real 'resolved → mark checked' bookkeeping + the funnel counters. Returns no thread ([]) so
    rendering stays solo."""
    def __init__(self, checked):
        self.checked = checked
        self.calls: list[str] = []
        self.n_calls = self.n_failed = self.n_chains = 0
        self.enabled = True        # the real fetcher's run-level kill switch; the funnel reports it

    def chain(self, tid):
        self.calls.append(str(tid))
        self.n_calls += 1
        self.checked.add(str(tid))
        return []


def _fake_derive(norm):
    tid = norm["id"]
    return {"who_id": f"x:user:{tid}", "who_name": "U", "who_handle": "u", "who_site": None,
            "when_ts": "2024-01-01T00:00:00Z", "when_precision": "second",
            "source_tags": [], "about_entities": [], "description": "d"}


def test_standalone_bookmark_skips_thread_fetch_but_marks_ledger(kb_home, fake_embedder, monkeypatch):
    """A provably-standalone tweet (no replies AND not a reply) skips the thread fetch, yet is still
    written to the resolved-ledger — so next run's fast-skip fires instead of re-rendering it forever
    (the ledger bug the gate would otherwise introduce). A tweet WITH replies is still fetched.

    `enrich=True`: this is the metered pass's gate. On the free pass NOTHING is fetched, and the
    standalone/replied-to distinction is made all the same — pinned separately below."""
    from pipeline.kb import derive, schema
    from pipeline.ingestion import x_graphql as xg
    import pipeline.ingestion.x_render as twapi_mod
    import pipeline.kb.vision as vision
    from pipeline.ingestion.utils import load_state
    from opyt_core.paths import opyt_home

    norms = [
        {"id": "100", "isReply": False, "replyCount": 0, "text": "solo",
         "url": "https://x.com/u/100", "entities": {"urls": []}, "extendedEntities": {"media": []}},
        {"id": "200", "isReply": False, "replyCount": 5, "text": "popular",
         "url": "https://x.com/u/200", "entities": {"urls": []}, "extendedEntities": {"media": []}},
    ]
    monkeypatch.setattr(xg, "iterate_bookmarks", lambda limit=0: iter(norms))
    monkeypatch.setattr(twapi_mod, "tweet_to_markdown",
                        lambda norm, article=None, thread_tweets=None, source=None,
                        footer_label=None: f"body {norm['id']}")
    monkeypatch.setattr(vision, "enrich_tweet_media", lambda norm, cache, *, describe_all: 0)
    monkeypatch.setattr(derive, "derive_x", _fake_derive)

    captured: dict = {}
    monkeypatch.setattr(ingest_x, "_ConvoFetcher",
                        lambda checked: captured.setdefault("convo", _FakeConvo(checked)))

    conn = schema.connect()
    summary = ingest_x.sync_bookmarks(conn, fake_embedder, enrich=True)

    # The standalone tweet was NOT fetched; the replied-to tweet WAS.
    assert captured["convo"].calls == ["200"]
    # But BOTH are in the persisted ledger — the standalone via the skip-path add, so it fast-skips next run.
    ledger = load_state(opyt_home() / "x_convo_checked.json")
    assert "100" in ledger and "200" in ledger

    assert summary["added"] == 2
    # Step 1: per-stage timings surfaced for Phase 2 (measurement, not the estimated budget).
    assert "stage_seconds" in summary and {"vlm", "render"} <= set(summary["stage_seconds"])
    # Phase-2 instrumentation: per-call latency SHAPE + the stage FUNNEL.
    assert set(summary["stage_latency"]["render"]) == {"count", "mean", "p50", "p95", "max"}
    tf = summary["funnel"]["thread"]
    assert tf["calls"] == 1 and tf["skipped_standalone"] == 1   # only the replied-to tweet was fetched
    assert summary["funnel"]["vlm"]["describe_calls"] == 0      # enrich stubbed → no VLM calls
    conn.close()


@pytest.fixture()
def saved_post(kb_home, monkeypatch):
    from unittest.mock import Mock
    from pipeline.ingestion import x_graphql
    from pipeline.kb import derive, schema, vision

    norm = {"id": "100", "text": "A saved post", "replyCount": 1, "isReply": False,
            "author": {"id": "100", "name": "U", "userName": "u"},
            "url": "https://x.com/u/status/100", "createdAt": "2026-08-01T12:00:00Z",
            "entities": {"urls": []}, "extendedEntities": {"media": []}}
    monkeypatch.setattr(core, "read_x_cookies", lambda: {"auth_token": "t", "ct0": "c"})
    monkeypatch.setattr(core, "auth_headers", lambda *a, **k: {})
    monkeypatch.setattr(core, "fetch_tweets_by_ids", lambda *a, **k: [norm])
    monkeypatch.setattr(x_graphql, "iterate_bookmarks", lambda **k: iter([norm]))
    monkeypatch.setattr(derive, "derive_x", _fake_derive)
    images = Mock(return_value=0)
    monkeypatch.setattr(vision, "enrich_tweet_media", images)
    snapshot = Mock(wraps=ingest_x.snapshot_and_hash)
    monkeypatch.setattr(ingest_x, "snapshot_and_hash", snapshot)
    conn = schema.connect()
    yield conn, norm, images, snapshot
    conn.close()


def _save(path, conn, embedder):
    # The bookmark arm runs at `enrich=True` throughout this section: these tests are about what
    # happens when the METERED read fails, and the free pass never makes one.
    if path == "bookmark":
        return ingest_x.sync_bookmarks(conn, embedder, enrich=True)["added"]
    status, _ = ingest_x.x_atom_from_url(conn, embedder, "https://x.com/u/status/100")
    return int(status == "saved")


@pytest.mark.parametrize("path", ["bookmark", "url"])
@pytest.mark.parametrize("failure", [TimeoutError("timed out"), RuntimeError("unavailable"),
                                      core.XRateLimited("request budget spent"),
                                      SyncAuthError("session rejected")])
def test_failed_context_still_writes_the_atom_bare_and_retries(saved_post, fake_embedder,
                                                               monkeypatch, kb_home, path, failure):
    """⚠️ INVERTED 2026-09-13 (R3). This asserted the OPPOSITE — no atom until the conversation
    read answered — and that reading of CLAUDE.md's fail-safe invariant cost 851 of David's 1,001
    bookmarks their atom permanently, not temporarily: 96.5% of bookmarks want `TweetDetail`,
    `_ConvoFetcher` disables for the whole run on the first refusal, and the re-run needed a
    resident worker that is not installed.

    The invariant is intact, not bent. It says a FAILED external call must skip — no write, no
    mark-processed. The BOOKMARKS call succeeded, and everything written here is derived from it;
    the `TweetDetail` call that failed writes nothing and marks nothing, so `tid` stays out of the
    ledger and the next Enrichment run re-reads exactly this one. Same argument already accepted
    for the `save` signal in `7880d720`, extended from the signal to the body.

    The `url` path still writes nothing at all, and that split is deliberate: it saves ONE post the
    user named, so its failure is reported to the caller in the same breath rather than buried in a
    1,000-item walk."""
    from pipeline.ingestion.utils import load_state
    from pipeline.kb import schema

    conn, _, images, snapshot = saved_post
    bookmark = path == "bookmark"
    calls = []

    def fetch(*args):
        calls.append(args[0])
        if len(calls) == 1:
            raise failure
        return []

    def signals():
        return conn.execute("SELECT COUNT(*) FROM curation_signals").fetchone()[0]

    monkeypatch.setattr(core, "fetch_conversation", fetch)

    assert _save(path, conn, fake_embedder) == (1 if bookmark else 0)
    assert schema.count_atoms(conn, "x") == (1 if bookmark else 0)
    assert signals() == (1 if bookmark else 0)
    assert bool(schema.load_hashes(conn, "x")) is bookmark
    # The bookmark atom was really RENDERED and hashed, not stubbed in — and its images were read,
    # because the VLM is not what failed.
    assert snapshot.called is bookmark
    assert images.called is bookmark
    # Either way `tid` is NOT marked resolved. That is the whole retry mechanism.
    assert "100" not in load_state(kb_home / "x_convo_checked.json")

    # Run 2: the read answers. It answers EMPTY (no conversation), so the bookmark's markdown is
    # unchanged and there is nothing to re-mint — the run's work was marking the ledger.
    assert _save(path, conn, fake_embedder) == (0 if bookmark else 1)
    assert schema.count_atoms(conn, "x") == 1
    # Still ONE signal: `ensure_signal` is idempotent per (who, type, platform), so recording the
    # save on the first pass cannot double-count it on the second.
    assert signals() == 1
    # …and is now marked resolved — on the bookmark path. `x_atom_from_url` fetches through a
    # throwaway ledger it never persists; its own `seen` check is what makes run 3 free.
    assert ("100" in load_state(kb_home / "x_convo_checked.json")) is bookmark
    assert len(calls) == 2

    # Run 3: fully resolved — ingested, conversation-checked, no photo pending — so the gate skips
    # it before any fetch.
    assert _save(path, conn, fake_embedder) == 0
    assert len(calls) == 2


@pytest.mark.parametrize("path", ["bookmark", "url"])
def test_focal_article_teaser_is_written_then_re_minted(saved_post, fake_embedder, monkeypatch,
                                                        kb_home, path):
    """⚠️ INVERTED 2026-09-13 with the failed-context case above, and for the same reason: a teaser
    is not a failed read, it is a SMALLER successful one (X ships the node with a 200, cover image,
    title and preview, ~1.1 KB). 72 of David's 1,001 bookmarks are X-Articles.

    The teaser atom is written and `tid` is left unmarked, so the walk re-reads it; when the body
    finally arrives the markdown grows, the hash changes, and the SAME atom re-mints. The `url`
    path still refuses, because it answers the user about one named post."""
    from pipeline.ingestion.utils import load_state
    from pipeline.kb import schema

    conn, norm, images, snapshot = saved_post
    bookmark = path == "bookmark"
    node = {"title": "Article", "preview_text": "Preview"}
    norm["article"] = {"article_results": {"result": node}}
    norm["replyCount"] = 0
    monkeypatch.setattr(core, "fetch_conversation", lambda *a: [])

    assert _save(path, conn, fake_embedder) == (1 if bookmark else 0)
    assert schema.count_atoms(conn, "x") == (1 if bookmark else 0)
    assert (conn.execute("SELECT COUNT(*) FROM curation_signals").fetchone()[0]
            == (1 if bookmark else 0))
    if bookmark:
        assert "The full article body." not in snapshot.call_args.args[2]   # the teaser, not the body
    # Unmarked even though this tweet wants no conversation read at all — the teaser branch skips
    # the ledger mark precisely so the next walk comes back for the body.
    assert "100" not in load_state(kb_home / "x_convo_checked.json")

    node["content_state"] = {"blocks": [{"type": "unstyled", "text": "The full article body."}]}
    assert _save(path, conn, fake_embedder) == 1
    assert schema.count_atoms(conn, "x") == 1        # RE-MINTED, not a second atom
    assert "The full article body." in snapshot.call_args.args[2]
    if bookmark:
        assert conn.execute("SELECT version FROM atoms WHERE atom_id='x:100'").fetchone()[0] == 2


# ── the truncated backfill (2026-09-13) ─────────────────────────────────────────
#
# ⚠️ THE BUG THIS PINS. `_ConvoFetcher` disables itself for the whole run on the FIRST rate
# refusal, so on a real backlog every bookmark after that one took the `chain is None` branch.
# That branch returned None, `run_concurrent` feeds only non-None results to the consumer, and the
# consumer is where the author entity and the `save` signal are written — so the walk silently
# dropped the corroboration evidence for the entire tail. `funnel.thread.failed` reported 1,
# because the fetcher short-circuits before its own counter once disabled, and `producer_failed`
# counts RAISES, not legitimate Nones. A 500-bookmark run therefore looked exactly like a clean one.
#
# Signals are the scoring input (`screen.CORROBORATION_MIN` = 2 distinct signal/platform pairs), so
# losing them is not a thin corpus — it is a WRONG candidate list, with no symptom.
#
# The signal half was fixed first; the ATOM half followed on the same day (R3). The refusal now
# costs the tail its THREAD CONTEXT, which Enrichment repays, instead of its existence.

def test_a_rate_limit_mid_walk_still_records_every_remaining_save(kb_home, fake_embedder,
                                                                 monkeypatch):
    from pipeline.kb import derive, schema
    from pipeline.ingestion import x_graphql as xg
    import pipeline.ingestion.x_render as twapi_mod
    import pipeline.kb.vision as vision

    # Ten bookmarks, all replies, so every one wants a thread read.
    norms = [{"id": str(i), "isReply": True, "replyCount": 0, "text": f"t{i}",
              "url": f"https://x.com/u/{i}", "entities": {"urls": []},
              "extendedEntities": {"media": []}} for i in range(10)]
    monkeypatch.setattr(xg, "iterate_bookmarks", lambda limit=0: iter(norms))
    monkeypatch.setattr(twapi_mod, "tweet_to_markdown",
                        lambda norm, article=None, thread_tweets=None, source=None,
                        footer_label=None: f"body {norm['id']}")
    monkeypatch.setattr(vision, "enrich_tweet_media", lambda norm, cache, *, describe_all: 0)
    monkeypatch.setattr(derive, "derive_x", _fake_derive)

    # The meter is spent after two reads — the shape of TweetDetail's 150/15-min bucket against a
    # backlog bigger than it.
    calls = []

    def fetch(tid, c, h):
        calls.append(str(tid))
        if len(calls) > 2:
            raise core.XRateLimited("request budget spent")
        return []

    monkeypatch.setattr(core, "read_x_cookies", lambda: {"auth_token": "t", "ct0": "c"})
    monkeypatch.setattr(core, "auth_headers", lambda *a, **k: {})
    monkeypatch.setattr(core, "fetch_conversation", fetch)

    conn = schema.connect()
    summary = ingest_x.sync_bookmarks(conn, fake_embedder, enrich=True)

    # ALL TEN atoms — two with their conversation resolved, eight rendered solo. A spent meter
    # costs fidelity, never coverage.
    assert summary["added"] == 10
    assert schema.count_atoms(conn, "x") == 10
    # And all ten saves: one signal per distinct author, none lost to the meter.
    assert conn.execute("SELECT COUNT(*) FROM curation_signals").fetchone()[0] == 10
    assert conn.execute(
        "SELECT COUNT(*) FROM curation_signals WHERE signal_type='save' AND platform='x'"
    ).fetchone()[0] == 10

    # And the run SAYS what it still owes — eight bookmarks whose atom is in the store but whose
    # thread context is not. `deferred` is the ENRICHMENT backlog now, not a coverage shortfall.
    assert summary["deferred"] == 8
    tf = summary["funnel"]["thread"]
    assert tf["deferred"] == 8 and tf["stopped_early"] is True
    # The old counter cannot see this: the fetcher stops counting once disabled. Pinned so nobody
    # "simplifies" the report back down to it.
    assert tf["failed"] == 1

    # The eight are unresolved, so the next run re-reads exactly them and nothing else.
    from pipeline.ingestion.utils import load_state
    from opyt_core.paths import opyt_home
    assert set(load_state(opyt_home() / "x_convo_checked.json")) == set(calls[:2])
    conn.close()


def test_shared_fetcher_serializes_conversation_requests(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time

    active = peak = 0
    count_lock = threading.Lock()

    def fetch(*args):
        nonlocal active, peak
        with count_lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)  # A slow transport response exposes overlapping requests.
        with count_lock:
            active -= 1
        return []

    monkeypatch.setattr(core, "fetch_conversation", fetch)
    fetcher = _fetcher(monkeypatch)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(fetcher.chain, ["1", "2", "3", "4"])) == [[], [], [], []]
    assert peak == 1
    assert fetcher.n_calls == 4
