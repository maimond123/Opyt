"""Conversation resolution: `reconstruct_chain` and `_ConvoFetcher`'s resolved-ledger semantics.

Network is monkeypatched — this pins the WIRING, not the live scrape.

`_ConvoFetcher` has one backend, the free `TweetDetail`, so the tests are about the LEDGER, which is where the real semantics live: a resolved id must
never be re-fetched, and a transient failure must never be recorded as resolved."""
from __future__ import annotations

from pipeline.curation import bookmarks

import pytest

from pipeline.platforms.x import conversation as x_conversation
from pipeline.platforms.x import local as x_local
import pipeline.platforms.x.rate as x_rate
from pipeline.ingestion.utils import SyncAuthError
from pipeline.kb import ingest_x


def _tw(tid, user, text="", reply_to=None):
    """A normalized tweet, as `x.records.normalize_tweet` shapes it. The user name doubles as the
    author id; `reply_to` is the (tweet id, user) it replies to."""
    replying = ({"isReply": True, "inReplyToStatusId": reply_to[0],
                 "inReplyToUserId": reply_to[1]} if reply_to else {})
    return {"id": tid, "text": text, "author": {"id": user, "userName": user}, **replying}


# ── reconstruct_chain (the shared reconstruction) ────────────────────────────────

def test_reconstruct_self_thread_root_keeps_author_continuation():
    # bookmark the ROOT: focal first, the author's own continuation, strangers' replies.
    tweets = [
        _tw("1", "author", "root take"),
        _tw("2", "author", "my continuation", ("1", "author")),
        _tw("3", "rando", "stranger reply", ("2", "author")),          # dropped — not the author
        _tw("4", "author", "more of my thread", ("2", "author")),
    ]
    chain = x_conversation.reconstruct_chain(tweets, "1")
    assert [t["id"] for t in chain] == ["1", "2", "4"]   # root + author's self-thread, no stranger


def test_reconstruct_drops_the_authors_replies_to_others():
    # A single post the author answered commenters under: no thread at all.
    tweets = [
        _tw("1", "author", "single take"),
        _tw("2", "rando", "so true", ("1", "author")),
        _tw("3", "author", "@rando thanks!", ("2", "rando")),
        _tw("4", "author", "@rando and see my earlier post", ("3", "author")),  # under a dropped one
    ]
    assert x_conversation.reconstruct_chain(tweets, "1") == []


def test_reconstruct_keeps_ancestors_and_a_self_thread_beside_replies_to_others():
    tweets = [
        _tw("1", "alice", "question"),
        _tw("2", "author", "@alice part 1", ("1", "alice")),
        _tw("3", "author", "part 2", ("2", "author")),
        _tw("4", "rando", "nice", ("3", "author")),
        _tw("5", "author", "@rando thanks", ("4", "rando")),
        _tw("6", "author", "part 3", ("3", "author")),
    ]
    chain = x_conversation.reconstruct_chain(tweets, "2")
    assert [t["id"] for t in chain] == ["1", "2", "3", "6"]


def test_reconstruct_reply_keeps_ancestors():
    # bookmark a REPLY: ancestor (the debate) comes before the focal.
    tweets = [_tw("1", "alice", "question"), _tw("2", "bob", "@alice answer"),
              _tw("3", "rando", "noise")]
    chain = x_conversation.reconstruct_chain(tweets, "2")
    assert [t["id"] for t in chain] == ["1", "2"]


def test_reconstruct_no_context_returns_empty():
    tweets = [_tw("1", "bob", "solo"), _tw("2", "rando", "unrelated reply")]
    assert x_conversation.reconstruct_chain(tweets, "1") == []     # only strangers reply → no thread


# ── _ConvoFetcher: the resolved-ledger semantics ─────────────────────────────────

def _fetcher(monkeypatch, checked=None):
    monkeypatch.setattr(x_local, "read_x_cookies", lambda: {"auth_token": "t", "ct0": "c"})
    monkeypatch.setattr(x_local, "auth_headers", lambda *a, **k: {})
    return ingest_x._ConvoFetcher(checked=checked if checked is not None else set())


def test_fetcher_marks_checked_on_success_even_when_no_thread(monkeypatch):
    """A genuine no-thread is a RESOLVED answer, not a miss. 91% of bookmarks have no conversation,
    so treating that as unresolved would re-fetch nearly the whole corpus every single run."""
    monkeypatch.setattr(x_conversation, "fetch_conversation", lambda tid, c: [])
    f = _fetcher(monkeypatch)
    assert f.chain("1") == []
    assert "1" in f.checked


def test_fetcher_does_not_mark_checked_on_transient_failure(monkeypatch):
    """The opposite direction, and the fail-safe one: a session that was rejected knows nothing
    about tweet 1, so recording it as resolved would lose that conversation permanently."""
    from pipeline.ingestion.utils import SyncAuthError

    def _dead(tid, c):
        raise SyncAuthError("session rejected")

    monkeypatch.setattr(x_conversation, "fetch_conversation", _dead)
    f = _fetcher(monkeypatch)
    assert f.chain("1") is None
    assert "1" not in f.checked        # unchecked → retried next run
    assert f.enabled is False          # and the whole run stops asking


def test_fetcher_returns_the_chain(monkeypatch):
    monkeypatch.setattr(x_conversation, "fetch_conversation",
                        lambda tid, c: [_tw("1", "a", "root"), _tw("2", "a", "cont")])
    f = _fetcher(monkeypatch)
    assert [t["id"] for t in f.chain("1")] == ["1", "2"]


def test_a_rate_limit_disables_the_fetcher_for_the_rest_of_the_run(monkeypatch):
    """`TweetDetail` is 150 per 15 minutes. Once it is spent every remaining bookmark would fail
    identically, so unread saves remain retryable without further conversation requests."""
    def _limited(tid, c):
        raise x_rate.XRateLimited("request budget spent")

    monkeypatch.setattr(x_conversation, "fetch_conversation", _limited)
    f = _fetcher(monkeypatch)
    assert f.chain("1") is None
    assert f.enabled is False
    assert "1" not in f.checked        # never resolved → retried next run


# ── sound thread-skip + ledger consistency (drives SavedXPosts end-to-end) ──

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


def test_standalone_bookmark_skips_thread_fetch_but_marks_ledger(kb_home, monkeypatch):
    """A provably-standalone tweet (no replies AND not a reply) is written to the resolved-ledger
    with its atom — so it is never owed to the conversation lane — while a tweet WITH replies is
    owed to it. The import itself reads no conversation at all."""
    from pipeline.kb import derive, schema
    from pipeline.oracles.sources.threads import state
    import pipeline.ingestion.x_render as twapi_mod
    import pipeline.kb.vision as vision
    from pipeline.ingestion.utils import load_state
    from opyt_core.paths import opyt_home
    from tests.kb.saved_x_support import save

    norms = [
        {"id": "100", "isReply": False, "replyCount": 0, "text": "solo",
         "url": "https://x.com/u/100", "entities": {"urls": []}, "extendedEntities": {"media": []}},
        {"id": "200", "isReply": False, "replyCount": 5, "text": "popular",
         "url": "https://x.com/u/200", "entities": {"urls": []}, "extendedEntities": {"media": []}},
    ]
    monkeypatch.setattr(twapi_mod, "tweet_to_markdown",
                        lambda norm, article=None, thread_tweets=None, source=None,
                        footer_label=None: f"body {norm['id']}")
    monkeypatch.setattr(vision, "enrich_tweet_media", lambda norm, cache: 0)
    monkeypatch.setattr(derive, "derive_x", _fake_derive)
    monkeypatch.setattr(x_conversation, "fetch_conversation",
                        lambda *a: pytest.fail("the import read a conversation"))

    conn = schema.connect()
    assert save(conn, *norms) == ["saved"]          # one unit saves the batch

    ledger = load_state(opyt_home() / "x_convo_checked.json")
    assert "100" in ledger and "200" not in ledger
    assert [w.atom_id for w in state.owed(conn, "bookmarks")] == ["x:200"]
    conn.close()


@pytest.fixture()
def saved_post(kb_home, monkeypatch):
    from unittest.mock import Mock
    from pipeline.kb import derive, schema, vision

    norm = {"id": "100", "text": "A saved post", "replyCount": 1, "isReply": False,
            "author": {"id": "100", "name": "U", "userName": "u"},
            "url": "https://x.com/u/status/100", "createdAt": "2026-08-01T12:00:00Z",
            "entities": {"urls": []}, "extendedEntities": {"media": []}}
    monkeypatch.setattr(x_local, "read_x_cookies", lambda: {"auth_token": "t", "ct0": "c"})
    monkeypatch.setattr(x_local, "auth_headers", lambda *a, **k: {})
    monkeypatch.setattr(x_conversation, "fetch_tweets_by_ids", lambda *a, **k: [norm])
    monkeypatch.setattr(derive, "derive_x", _fake_derive)
    images = Mock(return_value=0)
    monkeypatch.setattr(vision, "enrich_tweet_media", images)
    snapshot = Mock(wraps=ingest_x.changed_hash)
    monkeypatch.setattr(ingest_x, "changed_hash", snapshot)
    monkeypatch.setattr(bookmarks, "changed_hash", snapshot)
    conn = schema.connect()
    yield conn, norm, images, snapshot
    conn.close()


def _save(path, conn, norm=None):
    """1 when the post's atom was written. The bookmark arm is the saved-post import, which reads
    no conversation; the url arm is Hopper's single post."""
    if path == "bookmark":
        from tests.kb.saved_x_support import save
        return int("saved" in save(conn, norm))
    status, _ = ingest_x.x_atom_from_url(conn, "https://x.com/u/status/100")
    return int(status == "saved")


@pytest.mark.parametrize("path", ["url"])
@pytest.mark.parametrize("failure", [TimeoutError("timed out"), RuntimeError("unavailable"),
                                      x_rate.XRateLimited("request budget spent"),
                                      SyncAuthError("session rejected")])
def test_failed_context_still_writes_the_atom_bare_and_retries(saved_post, monkeypatch, kb_home,
                                                               path, failure):
    """A bookmark whose conversation read failed is still written, bare. Withholding the atom would
    cost it permanently, not temporarily: most bookmarks want `TweetDetail`, `_ConvoFetcher`
    disables for the whole run on the first refusal, and the re-run needs a resident worker.

    The fail-safe rule is intact. It says a FAILED external call must skip — no write, no
    mark-processed. The BOOKMARKS call succeeded, and everything written here is derived from it;
    the `TweetDetail` call that failed writes nothing and marks nothing, so `tid` stays out of the
    ledger and the next Enrichment run re-reads exactly this one. Same argument as for the `save`
    signal, extended to the body.

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

    monkeypatch.setattr(x_conversation, "fetch_conversation", fetch)

    assert _save(path, conn) == (1 if bookmark else 0)
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
    assert _save(path, conn) == (0 if bookmark else 1)
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
    assert _save(path, conn) == 0
    assert len(calls) == 2


@pytest.mark.parametrize("path", ["bookmark", "url"])
def test_focal_article_teaser_is_written_then_re_minted(saved_post, monkeypatch, kb_home,
                                                        path):
    """Same reasoning as the failed-context case above: a teaser is not a failed read, it is a
    smaller successful one (X ships the node with a 200, cover image, title and preview, ~1.1 KB)
    and many bookmarks are X-Articles.

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
    monkeypatch.setattr(x_conversation, "fetch_conversation", lambda *a: [])

    assert _save(path, conn, norm) == (1 if bookmark else 0)
    assert schema.count_atoms(conn, "x") == (1 if bookmark else 0)
    assert (conn.execute("SELECT COUNT(*) FROM curation_signals").fetchone()[0]
            == (1 if bookmark else 0))
    if bookmark:
        assert "The full article body." not in snapshot.call_args.args[1]  # the teaser, not the body
    # Unmarked even though this tweet wants no conversation read at all — the teaser branch skips
    # the ledger mark precisely so the next walk comes back for the body.
    assert "100" not in load_state(kb_home / "x_convo_checked.json")

    node["content_state"] = {"blocks": [{"type": "unstyled", "text": "The full article body."}]}
    assert _save(path, conn, norm) == 1
    assert schema.count_atoms(conn, "x") == 1        # RE-MINTED, not a second atom
    assert "The full article body." in snapshot.call_args.args[1]
    if bookmark:
        assert conn.execute("SELECT version FROM atoms WHERE atom_id='x:100'").fetchone()[0] == 2


# ── the truncated backfill ─────────────────────────────────────────
#
# The bug this pins: `_ConvoFetcher` disables itself for the whole run on the FIRST rate
# refusal, so on a real backlog every bookmark after that one took the `chain is None` branch.
# That branch returned None, `run_concurrent` feeds only non-None results to the consumer, and the
# consumer is where the author entity and the `save` signal are written — so the walk silently
# dropped the corroboration evidence for the entire tail. `funnel.thread.failed` reported 1,
# because the fetcher short-circuits before its own counter once disabled, and `producer_failed`
# counts RAISES, not legitimate Nones. A 500-bookmark run therefore looked exactly like a clean one.
#
# Signals are the scoring input (`ranking.Candidate.sort_key` ranks on distinct signal/platform pairs), so
# losing them is not a thin corpus — it is a WRONG candidate list, with no symptom.
#
# The signal half was fixed first; the ATOM half followed on the same day (R3). The refusal now
# costs the tail its THREAD CONTEXT, which Enrichment repays, instead of its existence.

def test_a_spent_tweetdetail_meter_never_costs_a_save(kb_home, monkeypatch):
    """Ten replies, every one owed a conversation read, against a spent TweetDetail meter: the
    import reads no conversation at all, so all ten atoms and all ten saves land, and all ten are
    owed to the thread lane. A spent meter costs fidelity later, never coverage now."""
    from pipeline.kb import derive, schema
    from pipeline.oracles.sources.threads import state
    import pipeline.ingestion.x_render as twapi_mod
    import pipeline.kb.vision as vision
    from tests.kb.saved_x_support import save

    norms = [{"id": str(i), "isReply": True, "replyCount": 0, "text": f"t{i}",
              "url": f"https://x.com/u/{i}", "entities": {"urls": []},
              "extendedEntities": {"media": []}} for i in range(10)]
    monkeypatch.setattr(twapi_mod, "tweet_to_markdown",
                        lambda norm, article=None, thread_tweets=None, source=None,
                        footer_label=None: f"body {norm['id']}")
    monkeypatch.setattr(vision, "enrich_tweet_media", lambda norm, cache: 0)
    monkeypatch.setattr(derive, "derive_x", _fake_derive)
    monkeypatch.setattr(x_conversation, "fetch_conversation",
                        lambda *a: (_ for _ in ()).throw(x_rate.XRateLimited("budget spent")))

    conn = schema.connect()
    assert save(conn, *norms) == ["saved"]          # one unit saves the batch

    assert schema.count_atoms(conn, "x") == 10
    assert conn.execute(
        "SELECT COUNT(*) FROM curation_signals WHERE signal_type='save' AND platform='x'"
    ).fetchone()[0] == 10
    assert len(state.owed(conn, "bookmarks")) == 10

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

    monkeypatch.setattr(x_conversation, "fetch_conversation", fetch)
    fetcher = _fetcher(monkeypatch)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(fetcher.chain, ["1", "2", "3", "4"])) == [[], [], [], []]
    assert peak == 1
    assert fetcher.n_calls == 4
