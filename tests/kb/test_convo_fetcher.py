"""Conversation resolution: `reconstruct_chain` and `_ConvoFetcher`'s resolved-ledger semantics.

Network is monkeypatched — this pins the WIRING, not the live scrape (validated live 2026-07-16).

Until 2026-08-30 `_ConvoFetcher` chose between two backends: twitterapi.io's `walk_thread_context`
when a key was present, and the free `TweetDetail` when it was not. The paid arm is gone and its
tests went with it, along with the backend-pick tests — a switch with one arm is not a choice. What
survived is everything about the LEDGER, which is where the real semantics live: a resolved id must
never be re-fetched, and a transient failure must never be recorded as resolved."""
from __future__ import annotations

from pipeline.ingestion import x_graphql_core as core
from pipeline.kb import ingest_x


def _tw(tid, user, text=""):
    """A normalized tweet. `author.userName` is the same-author key `reconstruct_chain` matches on,
    and it is populated identically by `x_graphql._normalize` and by the twitterapi.io shape this
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
    monkeypatch.setattr(core, "read_x_cookies", lambda profile=None: {"auth_token": "t", "ct0": "c"})
    monkeypatch.setattr(core, "auth_headers", lambda *a, **k: {})
    from pipeline.ingestion import x_graphql as xg
    monkeypatch.setattr(xg, "read_x_cookies", lambda profile=None: {"auth_token": "t", "ct0": "c"})
    return ingest_x._ConvoFetcher(profile=None, checked=checked if checked is not None else set())


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
    assert f.chain("1") == []
    assert "1" not in f.checked        # unchecked → retried next run
    assert f.enabled is False          # and the whole run stops asking


def test_fetcher_returns_the_chain(monkeypatch):
    monkeypatch.setattr(core, "fetch_conversation",
                        lambda tid, c, h: [_tw("1", "a", "root"), _tw("2", "a", "cont")])
    f = _fetcher(monkeypatch)
    assert [t["id"] for t in f.chain("1")] == ["1", "2"]


def test_a_rate_limit_disables_the_fetcher_for_the_rest_of_the_run(monkeypatch):
    """`TweetDetail` is 150 per 15 minutes. Once it is spent every remaining bookmark would fail
    identically, so the run degrades to solo renders instead of burning through the queue."""
    def _limited(tid, c, h):
        raise RuntimeError("TweetDetail rate-limited (429) by x.com")

    monkeypatch.setattr(core, "fetch_conversation", _limited)
    f = _fetcher(monkeypatch)
    assert f.chain("1") == []
    assert f.enabled is False
    assert "1" not in f.checked        # never resolved → retried next run


# ── Step 6: sound thread-skip + ledger consistency (drives sync_bookmarks end-to-end) ──

class _FakeConvo:
    """Stand-in for _ConvoFetcher that RECORDS which tids the loop asked it to fetch, and mirrors
    the real 'resolved → mark checked' bookkeeping + the funnel counters. Returns no thread ([]) so
    rendering stays solo."""
    backend = "twitterapi"

    def __init__(self, profile, checked):
        self.checked = checked
        self.calls: list[str] = []
        self.n_calls = self.n_failed = self.n_chains = 0

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
    (the ledger bug the gate would otherwise introduce). A tweet WITH replies is still fetched."""
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
    monkeypatch.setattr(xg, "iterate_bookmarks", lambda limit=0, profile=None: iter(norms))
    monkeypatch.setattr(twapi_mod, "tweet_to_markdown",
                        lambda norm, article=None, thread_tweets=None, source=None,
                        footer_label=None: f"body {norm['id']}")
    monkeypatch.setattr(vision, "enrich_tweet_media", lambda norm, cache, *, describe_all: 0)
    monkeypatch.setattr(derive, "derive_x", _fake_derive)

    captured: dict = {}
    monkeypatch.setattr(ingest_x, "_ConvoFetcher",
                        lambda profile, checked: captured.setdefault("convo", _FakeConvo(profile, checked)))

    conn = schema.connect()
    summary = ingest_x.sync_bookmarks(conn, fake_embedder, fetch_threads=True)

    # The standalone tweet was NOT fetched; the replied-to tweet WAS.
    assert captured["convo"].calls == ["200"]
    # But BOTH are in the persisted ledger — the standalone via the skip-path add, so it fast-skips next run.
    ledger = load_state(opyt_home() / "x_convo_checked.json")
    assert "100" in ledger and "200" in ledger

    assert summary["added"] == 2
    # Step 1: per-stage timings surfaced for Phase 2 (measurement, not the estimated budget).
    assert "stage_seconds" in summary and {"vlm", "render"} <= set(summary["stage_seconds"])
    # Phase-2 instrumentation: per-call latency SHAPE + the stage FUNNEL + rate-limit ceilings.
    assert set(summary["stage_latency"]["render"]) == {"count", "mean", "p50", "p95", "max"}
    tf = summary["funnel"]["thread"]
    assert tf["calls"] == 1 and tf["skipped_standalone"] == 1   # only the replied-to tweet was fetched
    assert summary["funnel"]["vlm"]["describe_calls"] == 0      # enrich stubbed → no VLM calls
    assert set(summary["rate_limits"]) == {"embed", "vision_llm"}
    conn.close()
