"""The free bookmark pass and the metered one — `sync_bookmarks(enrich=False|True)`.

RULED 2026-09-13 (R2/R3): background ⟺ blocked by a rate meter; everything else blocks until
complete, and an atom never waits on a metered fetch. The Bookmarks page is free (11 requests of a
500/15-min bucket for 1,001 bookmarks), so every atom is written from it; `TweetDetail` is exactly
150/15 min against 966 bookmarks that want it, so thread context and image descriptions are the
upgrade that arrives later and re-mints the atom.

These drive the walk end to end with the REAL renderer and the real vision path (the `ocr` fixture
fakes only the image model), because the thing being pinned is that two passes over one corpus
converge — a wrongly-shaped second render would delete context from an atom that already had it,
and a stubbed renderer cannot show that.
"""
from __future__ import annotations

import pytest

from pipeline.ingestion import x_graphql_core as core
from pipeline.kb import ingest_x, schema


def _fake_derive(norm):
    tid = norm["id"]
    return {"who_id": f"x:user:{tid}", "who_name": "U", "who_handle": "u", "who_site": None,
            "when_ts": "2024-01-01T00:00:00Z", "when_precision": "second",
            "source_tags": [], "about_entities": [], "description": "d"}


def _norm(tid="100", *, photos=0, reply_count=0):
    media = [{"type": "photo", "media_url_https": f"https://pbs/{tid}-{i}.jpg"}
             for i in range(photos)]
    return {"id": tid, "text": f"the saved take {tid}", "replyCount": reply_count, "isReply": False,
            "author": {"id": tid, "name": "U", "userName": "u"},
            "url": f"https://x.com/u/status/{tid}", "createdAt": "2026-08-01T12:00:00Z",
            "entities": {"urls": []}, "extendedEntities": {"media": media}}


@pytest.fixture()
def walk(kb_home, monkeypatch):
    """One bookmark with a photo AND replies, so BOTH metered upgrades are owed. `chain` is a
    mutable box the test fills when it wants the conversation to answer."""
    from pipeline.ingestion import x_graphql
    from pipeline.kb import derive

    norm = _norm(photos=1, reply_count=3)
    chain: list = []
    monkeypatch.setattr(core, "read_x_cookies", lambda: {"auth_token": "t", "ct0": "c"})
    monkeypatch.setattr(core, "auth_headers", lambda *a, **k: {})
    monkeypatch.setattr(core, "fetch_conversation", lambda tid, c, h: list(chain))
    monkeypatch.setattr(x_graphql, "iterate_bookmarks", lambda **k: iter([norm]))
    monkeypatch.setattr(derive, "derive_x", _fake_derive)
    conn = schema.connect()
    yield conn, norm, chain
    conn.close()


def _md(conn):
    return conn.execute("SELECT text FROM chunks WHERE atom_id='x:100'").fetchone()["text"]


def _atom(conn):
    return conn.execute("SELECT * FROM atoms WHERE atom_id='x:100'").fetchone()


# ── the free pass ──────────────────────────────────────────────────────────────

def test_the_free_pass_writes_the_atom_and_spends_nothing_metered(walk, fake_embedder, ocr):
    conn, _norm_, _chain = walk
    out = ingest_x.sync_bookmarks(conn, fake_embedder)

    assert out["added"] == 1 and out["enrich"] is False
    assert schema.count_atoms(conn, "x") == 1
    # Neither meter was touched: no TweetDetail call, no image read.
    assert out["funnel"]["thread"]["calls"] == 0
    assert ocr.calls == []
    # …and the run says what it still owes, which is what the Enrichment loop terminates on.
    assert out["deferred"] == 1


# ── the metered pass: one re-mint, then nothing ────────────────────────────────

def test_enrichment_re_mints_the_bare_atom_with_its_chain_and_its_image(walk, fake_embedder, ocr):
    """The whole point of the split. The atom exists from the free pass; Enrichment upgrades it in
    place — same atom_id, bumped version, original `first_seen` — and both the chunk table and its
    FTS mirror carry the new text, with no orphan left behind."""
    conn, norm, chain = walk
    ingest_x.sync_bookmarks(conn, fake_embedder)
    bare = dict(_atom(conn))
    bare_md = _md(conn)
    assert "Thread" not in bare_md and "*Image:*" not in bare_md

    chain.extend([norm, {"id": "101", "text": "my continuation",
                         "author": {"userName": "u"}, "entities": {"urls": []}}])
    out = ingest_x.sync_bookmarks(conn, fake_embedder, enrich=True)

    assert out["added"] == 1 and out["threads"] == 1 and out["enrich"] is True
    assert schema.count_atoms(conn, "x") == 1              # RE-MINTED, not a twin
    now = dict(_atom(conn))
    assert now["version"] == bare["version"] + 1
    assert now["first_seen"] == bare["first_seen"]         # the save's own date survives the upgrade
    assert now["raw_hash"] != bare["raw_hash"]

    md = _md(conn)
    assert "my continuation" in md and "*Image:*" in md
    # `replace_chunks` replaces BOTH tables — an FTS row still holding the bare text would answer
    # keyword searches with content the atom no longer has.
    rows = conn.execute("SELECT COUNT(*) c FROM chunks WHERE atom_id='x:100'").fetchone()["c"]
    fts = conn.execute("SELECT COUNT(*) c FROM chunks_fts WHERE atom_id='x:100'").fetchone()["c"]
    assert rows == fts == 1
    assert "my continuation" in conn.execute(
        "SELECT text FROM chunks_fts WHERE atom_id='x:100'").fetchone()["text"]
    # Nothing is owed any more.
    assert out["deferred"] == 0


def test_a_second_enrichment_run_over_a_settled_atom_is_a_no_op(walk, fake_embedder, ocr):
    """Idempotence, and the reason Enrichment can loop until `deferred` hits 0 without re-embedding
    the corpus each window: the skip gate fires before any fetch once all three terms are true."""
    conn, norm, chain = walk
    ingest_x.sync_bookmarks(conn, fake_embedder)
    chain.append(norm)
    ingest_x.sync_bookmarks(conn, fake_embedder, enrich=True)
    version = _atom(conn)["version"]
    reads = len(ocr.calls)

    out = ingest_x.sync_bookmarks(conn, fake_embedder, enrich=True)

    assert out["added"] == 0 and out["skipped"] == 1 and out["deferred"] == 0
    assert out["funnel"]["thread"]["calls"] == 0            # skipped BEFORE the fetch
    assert len(ocr.calls) == reads                          # and before the image read
    assert _atom(conn)["version"] == version


# ── the three-term gate (1.3) ──────────────────────────────────────────────────
#
# ⚠️ THE REGRESSION THIS PINS, which would otherwise be silent on all 258 photo-bearing bookmarks.
# `_ConvoFetcher.chain` marks the tid resolved on ANY successful read — an EMPTY chain included —
# so after one Enrichment run `atom in seen` and `tid in convo_checked` are both true forever. A
# two-term gate would skip the bookmark before the image read, permanently.

def test_an_undescribed_photo_survives_a_resolved_conversation(walk, fake_embedder, ocr):
    conn, _norm_, _chain = walk
    ingest_x.sync_bookmarks(conn, fake_embedder)

    # Enrichment run 1: the conversation answers EMPTY (32% of them do) and the image read FAILS,
    # so the tid is marked resolved while the photo is not. Both gate terms are now true.
    ocr.respond(lambda url, context: None)
    ingest_x.sync_bookmarks(conn, fake_embedder, enrich=True)
    from pipeline.ingestion.utils import load_state
    from opyt_core.paths import opyt_home
    assert "100" in load_state(opyt_home() / "x_convo_checked.json")
    assert "*Image:*" not in _md(conn)

    # Enrichment run 2: the image model is back. The bookmark must NOT be skipped.
    from pipeline.ocr_cascade import MediaRead
    ocr.respond(lambda url, context: MediaRead("a chart of CPI", "chart", True))
    out = ingest_x.sync_bookmarks(conn, fake_embedder, enrich=True)

    assert out["skipped"] == 0, "the two-term gate skipped a bookmark with an undescribed photo"
    assert out["added"] == 1
    assert "a chart of CPI" in _md(conn)
