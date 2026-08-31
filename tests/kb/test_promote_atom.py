"""Implicit one-way promotion — a human-initiated ingest that hits an existing frontier atom
flips it to the human lane (RULED 2026-08-25, docs/plans/2026-08-24-era-reads-claims-carry.md).

The claims here are the ones that fail silently in production if they break:

  • THE FLIP IS ONE-WAY. `entry_mode` is what Frontier stage 1 selects its input from, so a
    demotion — a nightly frontier refresh overwriting a human-attested row — shrinks the query
    generator's only input every night, invisibly. The `WHERE entry_mode = 'frontier'` clause is
    the whole guard; these tests drive the machine-lane callers straight at human-attested rows.
  • THE TARGET IS THE CALLER'S MODE. `entry_mode` records how an atom was FOUND, so a footprint
    crawl that collides with a frontier atom must land on `author_referenced`, not on a hardcoded
    `user-saved`, or provenance becomes a lie the moment two lanes meet.
  • PROMOTION OPENS THE WALLET. The re-read trigger keys on arrival, and a promoted atom's
    `first_seen` is the CRAWL date — months stale. Without `promoted_at` the engagement is
    invisible to the scheduler and the promotion buys nothing.
  • NO LANE VOCABULARY ESCAPES. A promotion answers exactly like a fresh save. Explaining the flip
    means teaching the taxonomy, which is the confusion the abstraction boundary exists to prevent.
"""
from __future__ import annotations

import pytest

from pipeline.kb import (hopper, ingest_blog, ingest_common, ingest_papers, ingest_x,
                         link_router, schema)


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _seed(conn, atom_id, entry_mode, source_type="blog"):
    schema.upsert_atom(conn, {"atom_id": atom_id, "source_type": source_type,
                              "entry_mode": entry_mode, "raw_hash": "h0"})
    return atom_id


def _mode(conn, atom_id):
    return conn.execute("SELECT entry_mode, promoted_at FROM atoms WHERE atom_id = ?",
                        (atom_id,)).fetchone()


# ── the primitive ───────────────────────────────────────────────────────────────
def test_a_frontier_atom_takes_the_callers_mode(conn):
    """Not a hardcoded 'user-saved'. A crawl of a tracked person's own site that collides with an
    atom Frontier found first is `author_referenced` — that is how it was found."""
    _seed(conn, "blog:a.com/p", "frontier")
    ingest_common.promote_atom(conn, "blog:a.com/p", "author_referenced")
    row = _mode(conn, "blog:a.com/p")
    assert row["entry_mode"] == "author_referenced" and row["promoted_at"] is not None


def test_a_human_attested_atom_is_never_touched(conn):
    """Idempotence in the direction that matters: a second deposit of the same URL must not
    re-stamp `promoted_at` and make an old engagement look new to the re-read trigger."""
    _seed(conn, "blog:a.com/p", "user-saved")
    ingest_common.promote_atom(conn, "blog:a.com/p", "user-saved")
    assert _mode(conn, "blog:a.com/p")["promoted_at"] is None


def test_a_machine_lane_ingest_promotes_nothing(conn):
    """The no-op that lets every presence-hit site call this unconditionally. `atomize_paper` runs
    under both lanes; passing the frontier mode through must not flip anything, in either
    direction — this is what keeps the primitive from becoming a demotion primitive."""
    _seed(conn, "blog:a.com/p", "frontier")
    _seed(conn, "blog:b.com/p", "user-saved")
    ingest_common.promote_atom(conn, "blog:a.com/p", "frontier")
    ingest_common.promote_atom(conn, "blog:b.com/p", "frontier")
    assert _mode(conn, "blog:a.com/p")["entry_mode"] == "frontier"
    assert _mode(conn, "blog:b.com/p")["entry_mode"] == "user-saved"


# ── the call sites ──────────────────────────────────────────────────────────────
_ARTICLE_URL = "https://www.theverge.com/2026/8/1/some-cool-article"
_ARTICLE_ATOM = "blog:theverge.com/2026/8/1/some-cool-article"


def test_a_hopper_deposit_makes_its_own_entry_mode_claim_true(conn):
    """The measured defect: this branch REPORTED `entry_mode: "user-saved"` while writing nothing,
    so a deposit of a URL the frontier already held left the row in the machine lane forever."""
    _seed(conn, _ARTICLE_ATOM, "frontier")
    out = hopper.save(conn, None, _ARTICLE_URL, confirm=True)

    assert out["status"] == "already_present" and out["entry_mode"] == "user-saved"
    assert _mode(conn, _ARTICLE_ATOM)["entry_mode"] == "user-saved"
    # Capability, not affordance: nothing in the answer says a lane changed.
    assert "frontier" not in repr(out).lower() and "promot" not in repr(out).lower()


def test_the_blog_adapter_promotes_on_its_own_presence_skip(conn, fake_embedder, monkeypatch):
    """Hopper's pre-check is a shortcut, not the guarantee — the adapters re-check, and the
    adapter's own skip is the path a caller that bypasses hopper takes."""
    monkeypatch.setattr(ingest_blog, "_feed_date_map", lambda base: {})
    _seed(conn, _ARTICLE_ATOM, "frontier")
    status, aid = ingest_blog.article_atom_from_url(conn, fake_embedder, _ARTICLE_URL,
                                                    entry_mode="user-saved")
    assert (status, aid) == ("present", _ARTICLE_ATOM)
    assert _mode(conn, _ARTICLE_ATOM)["entry_mode"] == "user-saved"


_TWEET_URL = "https://x.com/karpathy/status/1750000000000000000"
_TWEET_ATOM = "x:1750000000000000000"


def test_the_x_adapter_promotes_on_its_presence_skip(conn, fake_embedder):
    """A hand-dumped post is the same act as a bookmark; the store already holding it via the
    frontier changes nothing about that."""
    _seed(conn, _TWEET_ATOM, "frontier", source_type="x")
    status, aid = ingest_x.x_atom_from_url(conn, fake_embedder, _TWEET_URL,
                                           entry_mode="user-saved")
    assert (status, aid) == ("present", _TWEET_ATOM)
    assert _mode(conn, _TWEET_ATOM)["entry_mode"] == "user-saved"


def test_the_paper_policy_b_skip_no_longer_discards_the_save_signal(conn):
    """Policy B skips an immutable paper BEFORE the paid fetch, and until 2026-08-25 that threw
    away the user's save entirely. The skip stays — only the attestation is recorded."""
    paper = {"paperId": "S2ABC", "externalIds": {"ArXiv": "2501.00002"}, "title": "T"}
    aid = ingest_papers.paper_atom_id(paper)
    _seed(conn, aid, "frontier", source_type="paper")
    assert ingest_papers.atomize_paper(conn, None, paper, entry_mode="user-saved") is None
    assert _mode(conn, aid)["entry_mode"] == "user-saved"


def test_a_frontier_refresh_of_that_same_paper_never_demotes_it(conn):
    """The hazard from the other side, on the exact call the admit rail makes nightly."""
    paper = {"paperId": "S2ABC", "externalIds": {"ArXiv": "2501.00002"}, "title": "T"}
    aid = ingest_papers.paper_atom_id(paper)
    _seed(conn, aid, "user-saved", source_type="paper")
    assert ingest_papers.atomize_paper(conn, None, paper, entry_mode="frontier") is None
    assert _mode(conn, aid)["entry_mode"] == "user-saved"


def test_the_router_promotes_on_its_pre_check(conn):
    """`mint_artifact` is where a github/paper deposit actually lands, so its own present path
    needs the call — hopper's free pre-check answering first is not something to rely on."""
    _seed(conn, "github:owner/name", "frontier", source_type="github")
    res = link_router.mint_artifact(conn, None, "https://github.com/owner/name", "github",
                                    entry_mode="user-saved")
    assert res["status"] == "present"
    assert _mode(conn, "github:owner/name")["entry_mode"] == "user-saved"


# ── the X bookmark sweep ────────────────────────────────────────────────────────
def test_the_bookmark_sweep_promotes_what_it_skips(kb_home, fake_embedder, monkeypatch):
    """The spec's named case, and the one whose wiring is not obvious: `_work` runs on the producer
    pool, where the one rule is that nothing but the consumer touches `conn`. So the skips are
    collected and promoted after the walk, on the writer thread.

    Driven by running the sweep TWICE with the atom demoted in between, because that is the real
    shape — the second pass reaches the hash-unchanged skip, which is where a frontier atom the
    user has bookmarked would sit.
    """
    from datetime import datetime, timedelta, timezone
    from pipeline.ingestion import x_graphql as xg
    import pipeline.ingestion.x_render as twapi_mod
    import pipeline.kb.derive as derive
    import pipeline.kb.vision as vision

    now = datetime.now(timezone.utc)
    norm = {"id": "recent", "isReply": False, "replyCount": 0, "text": "post",
            "createdAt": (now - timedelta(days=30)).strftime("%a %b %d %H:%M:%S +0000 %Y"),
            "url": "https://x.com/u/recent", "entities": {"urls": []},
            "extendedEntities": {"media": []}}
    monkeypatch.setattr(xg, "iterate_bookmarks", lambda limit=0, profile=None: iter([dict(norm)]))
    monkeypatch.setattr(twapi_mod, "tweet_to_markdown",
                        lambda n, article=None, thread_tweets=None, source=None,
                        footer_label=None: "body recent")
    monkeypatch.setattr(derive, "derive_x", lambda n: {
        "who_id": "x:user:1", "who_name": "U", "who_handle": "u", "who_site": None,
        "when_ts": "2024-01-01T00:00:00Z", "when_precision": "second",
        "source_tags": [], "about_entities": [], "description": "d"})
    monkeypatch.setattr(vision, "enrich_tweet_media",
                        lambda n, cache, *, describe_all: 0)

    c = schema.connect()
    assert ingest_x.sync_bookmarks(c, fake_embedder, fetch_threads=False)["added"] == 1
    c.execute("UPDATE atoms SET entry_mode = 'frontier' WHERE atom_id = 'x:recent'")
    c.commit()

    summary = ingest_x.sync_bookmarks(c, fake_embedder, fetch_threads=False)
    assert summary["added"] == 0 and summary["skipped"] == 1     # still a skip: nothing re-embedded
    assert _mode(c, "x:recent")["entry_mode"] == "user-saved"
    c.close()
