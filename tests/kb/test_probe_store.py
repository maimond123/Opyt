"""The probe store — candidate content held OUTSIDE the trusted KB.

The first three tests ARE the trust boundary. They are not documentation of it and they are not a
convention: writing candidate content into `atoms` would make an unvetted stranger's posts
indistinguishable from Oracle knowledge in every existing search, and the only thing that can
prevent that regression is an assertion that fails when someone reintroduces it.

The rest pin the mechanics the pull depends on: hash-skip idempotency, per-candidate scoping, and
the promotion drop.
"""
from __future__ import annotations

import pytest

from pipeline.kb import probe_store, retrieve, schema
from pipeline.kb.ingest_common import AtomSink


def _probe_atom(tid: str, who: str = "x:user:11", url: str = "https://x.com/a/status/1") -> dict:
    return {"atom_id": f"xprobe:{tid}", "source_type": "x", "who_id": who,
            "when_ts": "2026-08-01", "when_precision": "day", "source_url": url,
            "raw_ref": f"kb_raw/probe/{tid}.md", "raw_hash": f"h{tid}",
            "description": f"post {tid}", "payload": {"like_count": 3}}


def _sink(conn, embedder) -> AtomSink:
    """A sink pointed at the PROBE store — one keyword, no table name anywhere."""
    return AtomSink(conn, embedder, writer=probe_store.write_probe_atom)


# ── the trust boundary ────────────────────────────────────────────────────────

def test_probe_content_never_lands_in_atoms(kb_home, fake_embedder):
    conn = schema.connect()
    sink = _sink(conn, fake_embedder)
    sink.submit(_probe_atom("1"), "an agent framework for autonomous tools")
    sink.close()

    assert probe_store.count_probe_atoms(conn) == 1
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 0
    conn.close()


def test_existing_search_cannot_see_probe_rows(kb_home, fake_embedder):
    """The regression that matters: a probe post about agents must not answer an agent query on the
    trusted rail. Both arms are exercised — BM25 (literal tokens) and semantic."""
    conn = schema.connect()
    sink = _sink(conn, fake_embedder)
    sink.submit(_probe_atom("1"), "an agent framework for autonomous tools")
    sink.close()

    for mode in ("hybrid", "bm25", "semantic"):
        run = retrieve.search_atoms(conn, "agent framework", fake_embedder, mode=mode)
        assert run.hits == [], f"probe row leaked into search_atoms(mode={mode})"
    conn.close()


def test_probe_writer_signature_matches_the_trusted_writer(kb_home, fake_embedder):
    """The seam IS the signature match: `AtomSink` takes a `writer=` function, so the two writers
    must stay call-compatible or a probe write silently routes to the wrong store. This replaced an
    edges-refusal test when the `edges` table was deleted (2026-08-23)."""
    import inspect

    from pipeline.kb import ingest_common
    assert (list(inspect.signature(probe_store.write_probe_atom).parameters)
            == list(inspect.signature(ingest_common._write_atom).parameters))


# ── the trusted rail still works with the sink seam in place ──────────────────

def test_default_sink_still_writes_trusted_atoms(kb_home, fake_embedder):
    """`writer=` defaults to the trusted writer, so every existing adapter is untouched."""
    conn = schema.connect()
    sink = AtomSink(conn, fake_embedder)
    sink.submit({"atom_id": "x:1", "source_type": "x", "who_id": "x:user:11",
                 "description": "d", "raw_hash": "h"}, "an agent framework")
    sink.close()
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 1
    assert probe_store.count_probe_atoms(conn) == 0
    conn.close()


def test_both_stores_coexist_in_one_db(kb_home, fake_embedder):
    conn = schema.connect()
    trusted = AtomSink(conn, fake_embedder)
    trusted.submit({"atom_id": "x:1", "source_type": "x", "who_id": "x:user:99",
                    "description": "d", "raw_hash": "h"}, "a crypto rollup proof")
    trusted.close()
    probe = _sink(conn, fake_embedder)
    probe.submit(_probe_atom("2"), "a crypto rollup proof")
    probe.close()

    run = retrieve.search_atoms(conn, "crypto rollup", fake_embedder)
    assert [h.atom_id for h in run.hits] == ["x:1"]     # the trusted one, and ONLY it
    assert probe_store.count_probe_atoms(conn) == 1
    conn.close()


# ── the embed surface reaches the probe path ──────────────────────────────────

def test_probe_chunks_are_embedded_from_the_stripped_surface(kb_home, fake_embedder):
    """The probe renderer is the most scaffolding-heavy surface in the store — measured on the
    25-account live run, a short probe atom is 61% chrome and the strip removes 71% of a one-line
    post. So `embed_text` must be STORED, not just used in flight: it is the only record of what a
    vector was actually built from, and the probe store gets no benefit from the trusted side's
    column."""
    conn = schema.connect()
    md = ("# Dougie — 2026-06-23\n\nnew article soon\n\n---\n"
          "*Candidate probe · [Original post](https://x.com/DougieDeLuca/status/2069426566430445755)*\n")
    sink = _sink(conn, fake_embedder)
    sink.submit(_probe_atom("1"), md)
    sink.close()

    row = conn.execute("SELECT text, embed_text FROM probe_chunks WHERE atom_id='xprobe:1'"
                       ).fetchone()
    assert row["embed_text"] is not None
    assert len(row["embed_text"]) < len(row["text"])          # scaffolding actually came off
    assert "new article soon" in row["embed_text"]            # the author's words survive
    assert "Candidate probe" not in row["embed_text"]         # the footer LABEL goes
    assert "@DougieDeLuca" in row["embed_text"]               # ...but its handle is recovered
    assert "Candidate probe" in row["text"]                   # the STORED text is untouched
    conn.close()


def test_an_existing_probe_store_gains_the_column(kb_home, fake_embedder):
    """`CREATE TABLE IF NOT EXISTS` is a no-op on a table that already exists, so a column added to
    `_DDL` never reaches a store written before it. The additive path is what covers the 340 atoms
    already on disk from the first live run."""
    conn = schema.connect()
    probe_store.init_probe_schema(conn)
    conn.execute("DROP TABLE probe_chunks")
    conn.execute("CREATE TABLE probe_chunks (chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "atom_id TEXT NOT NULL, seq INTEGER NOT NULL, char_start INTEGER, "
                 "char_end INTEGER, text TEXT NOT NULL, vector BLOB, UNIQUE(atom_id, seq))")
    conn.commit()
    probe_store.init_probe_schema(conn)                       # must ADD the column, not fail
    cols = {r[1] for r in conn.execute("PRAGMA table_info(probe_chunks)")}
    assert "embed_text" in cols
    conn.close()


# ── mechanics the pull depends on ─────────────────────────────────────────────

def test_hash_ledger_scopes_to_one_candidate(kb_home, fake_embedder):
    conn = schema.connect()
    sink = _sink(conn, fake_embedder)
    sink.submit(_probe_atom("1", who="x:user:11"), "alpha")
    sink.submit(_probe_atom("2", who="x:user:22"), "beta")
    sink.close()

    assert probe_store.load_probe_hashes(conn) == {"xprobe:1": "h1", "xprobe:2": "h2"}
    assert probe_store.load_probe_hashes(conn, "x:user:11") == {"xprobe:1": "h1"}
    assert probe_store.probed_who_ids(conn) == {"x:user:11", "x:user:22"}
    conn.close()


def test_hash_ledger_is_empty_before_the_first_pull(kb_home):
    """Fail-safe: a store that has never probed answers "nothing seen", not "no such table"."""
    conn = schema.connect()
    assert probe_store.load_probe_hashes(conn) == {}
    assert probe_store.count_probe_atoms(conn) == 0
    conn.close()


def test_rewriting_an_atom_replaces_its_chunks_and_bumps_version(kb_home, fake_embedder):
    conn = schema.connect()
    sink = _sink(conn, fake_embedder)
    sink.submit(_probe_atom("1"), "an agent framework")
    sink.close()
    first = conn.execute("SELECT COUNT(*) FROM probe_chunks").fetchone()[0]

    grown = {**_probe_atom("1"), "raw_hash": "h1-v2"}
    sink = _sink(conn, fake_embedder)
    sink.submit(grown, "an agent framework, now with a much longer body about crypto rollups")
    sink.close()

    row = conn.execute("SELECT version, raw_hash FROM probe_atoms WHERE atom_id='xprobe:1'"
                       ).fetchone()
    assert row["version"] == 2 and row["raw_hash"] == "h1-v2"
    assert probe_store.count_probe_atoms(conn) == 1          # replaced in place, not duplicated
    # Boundaries shift on a rewrite, so the OLD chunks must be gone rather than merged.
    n_fts = conn.execute("SELECT COUNT(*) FROM probe_chunks_fts WHERE atom_id='xprobe:1'"
                         ).fetchone()[0]
    assert n_fts == conn.execute("SELECT COUNT(*) FROM probe_chunks").fetchone()[0] >= first
    conn.close()


# ── the probe TTL is jittered per candidate ─────────────────────────────────────
#
# All 188 live candidates carry near-identical `pulled_at` stamps from one backfill burst, so a
# flat cutoff makes them all fall due on the same day. Worse, the cluster does not decay: a batch
# re-probed together is re-stamped together, so the burst re-forms every cycle. Each probe is a
# PACED X request against a shared budget, which is what makes a burst expensive rather than untidy.

def test_the_flat_cutoff_became_a_per_candidate_one(kb_home):
    """The behavioural proof. Two hundred candidates stamped in one burst, aged to exactly the flat
    TTL: under one cutoff this set is all-or-nothing. Under per-candidate TTLs it splits."""
    conn = schema.connect()
    try:
        for i in range(200):
            probe_store.record_pull(conn, f"x:user:{i}", probe_store.STATUS_OK)
        conn.execute("UPDATE probe_pulls SET pulled_at = datetime('now', '-30 days')")
        conn.commit()

        fresh = probe_store.fresh_who_ids(conn, ttl_days=30)

        assert 0 < len(fresh) < 200                 # a flat cutoff gives exactly 0 or exactly 200
        assert 0.3 < len(fresh) / 200 < 0.7         # ...and the split sits mid-band
    finally:
        conn.close()


def test_two_hundred_candidates_fan_out_across_a_measurable_window(kb_home):
    from collections import Counter

    ttls = sorted(probe_store.candidate_ttl_days(f"x:user:{i}", 30.0) for i in range(200))

    assert ttls[0] >= 30.0 * (1 - probe_store.PROBE_TTL_JITTER)     # 22.5d, the eager end
    assert ttls[-1] <= 30.0 * (1 + probe_store.PROBE_TTL_JITTER)    # 37.5d, the patient end
    assert ttls[-1] - ttls[0] > 13                                  # a window, not a point
    # Spread, not merely widened: no single day of the window holds a big share of the roster.
    assert max(Counter(int(t) for t in ttls).values()) <= 40


def test_the_ttl_is_stable_across_calls_and_across_processes(kb_home):
    """⚠️ NEVER `random()`, and never Python's built-in `hash()` — which is salted per process, so
    a TTL built on it would change on every restart. Staleness has to stay a pure function of
    stored state, which is the property the repeat-run harness checks."""
    import subprocess
    import sys
    from pathlib import Path

    mine = probe_store.candidate_ttl_days("x:user:33836629", 30.0)
    assert all(probe_store.candidate_ttl_days("x:user:33836629", 30.0) == mine for _ in range(5))

    out = subprocess.run(
        [sys.executable, "-c",
         "from pipeline.kb import probe_store as p; "
         "print(repr(p.candidate_ttl_days('x:user:33836629', 30.0)))"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[2]))
    assert out.returncode == 0, out.stderr
    assert float(out.stdout.strip()) == mine


def test_two_candidates_get_different_ttls(kb_home):
    assert (probe_store.candidate_ttl_days("x:user:1", 30.0)
            != probe_store.candidate_ttl_days("x:user:2", 30.0))


def test_a_fresh_pull_is_still_fresh_and_a_zero_ttl_still_re_pulls_everyone(kb_home):
    """The jitter must not move the two ends. A pull made seconds ago is inside even the eagerest
    TTL, and `ttl_days=0` still means a full re-pull."""
    conn = schema.connect()
    try:
        probe_store.record_pull(conn, "x:user:1", probe_store.STATUS_OK)
        assert probe_store.fresh_who_ids(conn, ttl_days=30) == {"x:user:1"}
        assert probe_store.fresh_who_ids(conn, ttl_days=0) == set()
    finally:
        conn.close()


def test_a_failed_row_is_never_fresh_however_recent(kb_home):
    """Unchanged by the jitter, and pinned again because it now runs through Python rather than a
    SQL `status IN (...)` — a `failed` row is an observation about US, not about the candidate."""
    conn = schema.connect()
    try:
        probe_store.record_pull(conn, "x:user:1", probe_store.STATUS_FAILED, detail="boom")
        assert probe_store.fresh_who_ids(conn, ttl_days=30) == set()
    finally:
        conn.close()


def test_an_undateable_row_reads_as_due_not_as_permanently_fresh(kb_home):
    """Fail-safe direction: a candidate whose stamp we cannot parse gets re-observed. The opposite
    default would freeze them out of every future run, silently."""
    conn = schema.connect()
    try:
        probe_store.record_pull(conn, "x:user:1", probe_store.STATUS_OK)
        conn.execute("UPDATE probe_pulls SET pulled_at='not-a-timestamp'")
        conn.commit()
        assert probe_store.fresh_who_ids(conn, ttl_days=30) == set()
    finally:
        conn.close()
