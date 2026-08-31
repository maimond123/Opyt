"""ingest_common — the shared write path: hash-skip idempotency, body-only chunking,
and the kb_meta identity record. Uses the fake embedder + $OPYT_HOME sandbox (no API)."""
from __future__ import annotations

from pipeline.kb import schema
from pipeline.kb.ingest_common import snapshot_and_hash, store_atom
from pipeline.kb.raw_store import write_snapshot


def test_snapshot_and_hash_is_idempotent(kb_home):
    seen: dict[str, str] = {}
    decided = snapshot_and_hash("x", "x:1", "hello", seen)
    assert decided is not None
    seen["x:1"] = decided[1]                                   # record the hash (as an ingester would)
    assert snapshot_and_hash("x", "x:1", "hello", seen) is None    # unchanged → skip
    assert snapshot_and_hash("x", "x:1", "hello world", seen) is not None  # changed → re-store


def test_store_atom_chunks_body_only_and_records_identity(kb_home, fake_embedder):
    conn = schema.connect()
    md = '---\nsource: github\nauthor: "@o"\n---\n\nan agent framework body'
    raw_ref, raw_hash = write_snapshot("github", "github:o/r", md)
    atom = dict(atom_id="github:o/r", source_type="github", what_kind="artifact",
                who_id="github:o", when_ts="2024-01-01", when_precision="push",
                about_entities=[], source_url="u",
                raw_ref=raw_ref, raw_hash=raw_hash, description="d", payload={},
                entry_mode="user-saved")
    store_atom(conn, fake_embedder, atom=atom, snapshot_text=md,
               )

    # Chunk text is BODY-only — frontmatter never reaches the FTS/vector router surface.
    txt = conn.execute("SELECT text FROM chunks WHERE atom_id='github:o/r'").fetchone()[0]
    assert "source: github" not in txt
    assert "an agent framework body" in txt

    # The embedding identity was recorded on first write (the kb_meta guard).
    meta = conn.execute("SELECT embed_model, embed_dim, provider FROM kb_meta WHERE id=1").fetchone()
    assert meta["embed_model"] == "fake-bow" and meta["embed_dim"] == fake_embedder.dim

    conn.close()


def test_stage_timer_distribution_reports_percentiles():
    """StageTimer records per-entry samples so a run reports the SHAPE of each stage's calls, not
    just the total — the Phase-2 worker-sizing input. Percentile math is nearest-rank."""
    from pipeline.kb.ingest_common import StageTimer, _percentile

    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(xs, 0) == 1.0
    assert _percentile(xs, 50) == 3.0
    assert _percentile(xs, 100) == 5.0
    assert _percentile([], 50) == 0.0                 # empty → 0.0, never IndexError

    t = StageTimer()
    for _ in range(4):
        with t.stage("thread_fetch"):
            pass
    d = t.distribution()["thread_fetch"]
    assert d["count"] == 4                             # one sample per stage() entry
    assert set(d) == {"count", "mean", "p50", "p95", "max"}
    assert "thread_fetch" in t.totals                  # totals still accumulate alongside samples


def test_looks_like_image_url_matches_extension_and_cdn():
    """The VLM enricher's include-filter. It was shared with `outbound_links`, the reference-edge
    extractor, which went with the `edges` table on 2026-08-23 — both arms are needed because
    Substack's CDN URLs carry no extension at all."""
    from pipeline.kb.ingest_common import looks_like_image_url

    assert looks_like_image_url("https://cdn.example.com/a.png")
    assert looks_like_image_url("https://cdn.example.com/a.JPEG?w=100")
    assert looks_like_image_url("https://substackcdn.com/image/fetch/x")   # no extension
    assert not looks_like_image_url("https://arxiv.org/abs/1")
