"""The BOOKMARK lookback — a SPEND filter, not a relevance filter.

Bookmarks are the third selector shape in onboarding, and they match neither of the other two: the
walk is a free cookie-scrape, but every surviving bookmark costs a twitterapi thread fetch and
often an image read (measured 2026-07-22: 791 bookmarks → 790 thread fetches + 254 VLM calls,
~4.1 s each). So the only thing this window buys is money, and it buys it ONLY if the drop happens
upstream of the paid work — a filter applied after the fetch saves nothing at all.

Two properties, both load-bearing:
  • an out-of-window bookmark costs NO thread fetch, NO image read, NO embed, NO atom;
  • the default (no window) is behaviourally identical to before the selector existed, so turning
    it on can never silently shrink an existing corpus.

And one semantic the wording depends on: the cutoff is the tweet's WRITE date, because X exposes
no bookmark timestamp. Hence SKIP-and-keep-walking, never break — the walk is ordered by SAVE
time, so an old post saved yesterday sits near the TOP of it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pipeline.kb import ingest_x, schema

_NOW = datetime.now(timezone.utc)


def _norm(tid: str, written: datetime):
    return {"id": tid, "isReply": False, "replyCount": 3, "text": f"post {tid}",
            "createdAt": written.strftime("%a %b %d %H:%M:%S +0000 %Y"),
            "url": f"https://x.com/u/{tid}", "entities": {"urls": []},
            "extendedEntities": {"media": []}}


class _RecordingConvo:
    """Records every thread fetch — the paid call the filter is supposed to prevent."""
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
    return {"who_id": f"x:user:{norm['id']}", "who_name": "U", "who_handle": "u", "who_site": None,
            "when_ts": "2024-01-01T00:00:00Z", "when_precision": "second",
            "source_tags": [], "about_entities": [], "description": "d"}


@pytest.fixture()
def walk(monkeypatch):
    """A two-bookmark walk: one written ~1 month ago, one written ~3 years ago. Returns the
    recorders (thread fetches, image reads) the assertions read."""
    from pipeline.ingestion import x_graphql as xg
    import pipeline.ingestion.x_render as twapi_mod
    import pipeline.kb.derive as derive
    import pipeline.kb.vision as vision

    norms = [_norm("recent", _NOW - timedelta(days=30)),
             _norm("ancient", _NOW - timedelta(days=1100))]
    monkeypatch.setattr(xg, "iterate_bookmarks", lambda limit=0, profile=None: iter(norms))
    monkeypatch.setattr(twapi_mod, "tweet_to_markdown",
                        lambda norm, article=None, thread_tweets=None, source=None,
                        footer_label=None: f"body {norm['id']}")
    monkeypatch.setattr(derive, "derive_x", _fake_derive)

    seen: dict = {}
    images: list[str] = []
    monkeypatch.setattr(vision, "enrich_tweet_media",
                        lambda norm, cache, *, describe_all: images.append(norm["id"]) or 0)
    monkeypatch.setattr(ingest_x, "_ConvoFetcher",
                        lambda profile, checked: seen.setdefault(
                            "convo", _RecordingConvo(profile, checked)))
    return seen, images


def _atom_ids(conn):
    return {r["atom_id"] for r in conn.execute("SELECT atom_id FROM atoms")}


def test_out_of_window_bookmark_costs_nothing_paid(kb_home, fake_embedder, walk):
    """`since` = 1 year ago drops the 3-year-old post — and drops it BEFORE the thread fetch, the
    image read, and the embed. Anything less than that is a filter that saves no money."""
    seen, images = walk
    conn = schema.connect()
    summary = ingest_x.sync_bookmarks(conn, fake_embedder, fetch_threads=True,
                                      since=_NOW - timedelta(days=365))

    assert seen["convo"].calls == ["recent"]        # NO thread fetch for the dropped bookmark
    assert images == ["recent"]                     # NO image read either
    assert _atom_ids(conn) == {"x:recent"}          # …and no atom, so no embed
    assert summary["added"] == 1 and summary["out_of_window"] == 1
    # `out_of_window` is its OWN counter: `skipped` means "already had it, unchanged" (free), this
    # means "you chose not to pay for it". Collapsing them makes a narrow window read as good dedup.
    assert summary["skipped"] == 0
    conn.close()


def test_default_window_is_identical_to_no_selector(kb_home, fake_embedder, walk):
    """The default must not shrink anyone's corpus: `since=None` ingests exactly what the code did
    before this parameter existed — including the three-year-old post."""
    seen, images = walk
    conn = schema.connect()
    summary = ingest_x.sync_bookmarks(conn, fake_embedder, fetch_threads=True)

    assert sorted(seen["convo"].calls) == ["ancient", "recent"]
    assert _atom_ids(conn) == {"x:recent", "x:ancient"}
    assert summary["added"] == 2 and summary["out_of_window"] == 0
    assert summary["since"] is None
    conn.close()


def test_an_unparseable_date_is_kept_not_dropped(kb_home, fake_embedder, monkeypatch):
    """A timestamp we can't read is not evidence the post is old. Dropping it would spend the
    user's window on a parser bug — so the fail-safe direction is KEEP."""
    from pipeline.ingestion import x_graphql as xg
    import pipeline.ingestion.x_render as twapi_mod
    import pipeline.kb.derive as derive
    import pipeline.kb.vision as vision

    bad = _norm("undated", _NOW)
    bad["createdAt"] = "not a date"
    monkeypatch.setattr(xg, "iterate_bookmarks", lambda limit=0, profile=None: iter([bad]))
    monkeypatch.setattr(twapi_mod, "tweet_to_markdown",
                        lambda norm, article=None, thread_tweets=None, source=None,
                        footer_label=None: "body")
    monkeypatch.setattr(derive, "derive_x", _fake_derive)
    monkeypatch.setattr(vision, "enrich_tweet_media", lambda norm, cache, *, describe_all: 0)

    conn = schema.connect()
    summary = ingest_x.sync_bookmarks(conn, fake_embedder, fetch_threads=False,
                                      since=_NOW - timedelta(days=1))
    assert summary["added"] == 1 and summary["out_of_window"] == 0
    conn.close()


def test_the_filter_skips_rather_than_stopping_the_walk(kb_home, fake_embedder, monkeypatch):
    """The walk is ordered by SAVE time and the cutoff is on WRITE time, so an out-of-window post
    can sit anywhere in it — including first. A `break` would have silently truncated everything
    saved after it; this asserts the row after the drop still lands."""
    from pipeline.ingestion import x_graphql as xg
    import pipeline.ingestion.x_render as twapi_mod
    import pipeline.kb.derive as derive
    import pipeline.kb.vision as vision

    norms = [_norm("ancient", _NOW - timedelta(days=1100)),   # saved most recently, written 2023
             _norm("recent", _NOW - timedelta(days=30))]
    monkeypatch.setattr(xg, "iterate_bookmarks", lambda limit=0, profile=None: iter(norms))
    monkeypatch.setattr(twapi_mod, "tweet_to_markdown",
                        lambda norm, article=None, thread_tweets=None, source=None,
                        footer_label=None: f"body {norm['id']}")
    monkeypatch.setattr(derive, "derive_x", _fake_derive)
    monkeypatch.setattr(vision, "enrich_tweet_media", lambda norm, cache, *, describe_all: 0)

    conn = schema.connect()
    summary = ingest_x.sync_bookmarks(conn, fake_embedder, fetch_threads=False,
                                      since=_NOW - timedelta(days=365))
    assert _atom_ids(conn) == {"x:recent"}          # the walk continued PAST the dropped row
    assert summary["out_of_window"] == 1
    conn.close()


def test_curation_pull_threads_the_window_to_the_bookmark_arm_only(kb_home, monkeypatch):
    """`bookmark_since` is the bookmark arm's knob and nothing else's — the other five sources
    carry no window at all, and must not silently acquire one."""
    from pipeline.kb import ingest_curation

    got: dict = {}
    monkeypatch.setattr(ingest_x, "sync_bookmarks",
                        lambda conn, emb, **kw: got.setdefault("bookmarks", kw) or {"source": "x"})
    for name in ("sync_lists_signals", "sync_substack_subs", "sync_following_signals",
                 "sync_likes_signals"):
        monkeypatch.setattr(ingest_curation, name, lambda conn, **kw: {"ok": True})
    monkeypatch.setattr(ingest_curation, "sync_substack_saved",
                        lambda conn, emb, **kw: {"ok": True})

    cutoff = _NOW - timedelta(days=365)
    conn = schema.connect()
    out = ingest_curation.curation_pull(conn, object(), bookmark_since=cutoff)
    assert got["bookmarks"]["since"] == cutoff
    # And the pull's own clock covers all six labels — the only timing the signal-only arms have.
    assert set(out["stage_seconds"]) == {"x-bookmarks", "x-lists", "substack-subs",
                                         "substack-saved", "x-following", "x-likes"}
    conn.close()
