"""chunks.vector STORAGE WIDTH — the write/read/guard triangle.

Half precision was chosen on measurement (100.0% recall@10 and 100.0% top-1 identical against the
float32 baseline on the real corpus), so what these tests pin is not "is f16 good enough" — that
was answered by the eval — but the three mechanical properties that make a per-store width SAFE:

  1. what `_attach` writes is what `kb_meta.storage_dtype` says, at the byte level;
  2. the reader takes the width from kb_meta rather than assuming, so a store written at either
     width returns the SAME ranking;
  3. a build whose width disagrees with the store RAISES, because the vector arm joins every
     candidate blob into one buffer and reshapes once — mixed widths there produce no exception,
     just wrong answers. That is the failure this whole column exists to prevent.

`convert_chunk_storage_dtype` is the escape hatch that lets (3) be absolute: a width change is a
local re-cast, not a re-embed, so the guard costs a user no money to satisfy.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.kb import schema
from pipeline.kb import embed as embed_mod
from pipeline.kb.embed import (
    CHUNK_STORAGE_DTYPE,
    SubspaceError,
    assert_model,
    convert_chunk_storage_dtype,
    ensure_kb_meta,
    stored_dtype,
)
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.raw_store import write_snapshot
from pipeline.kb.retrieve import search_atoms

CORPUS = [
    ("github:root/agentkit",
     "an autonomous agent framework with tools for building agents"),
    ("github:stranger/agents", "an agent framework library"),
    ("x:1", "thoughts on rollup and proof systems"),
    ("github:root/dash", "a react web dashboard"),
]


def _add(conn, emb, atom_id, snapshot):
    raw_ref, raw_hash = write_snapshot("github", atom_id, snapshot)
    store_atom(conn, emb, atom=dict(
        atom_id=atom_id, source_type="github", what_kind="artifact", who_id="github:root",
        when_ts="2024-05-01", when_precision="day", about_entities=[],
        source_url=f"https://example/{atom_id}", raw_ref=raw_ref, raw_hash=raw_hash,
        description=f"{atom_id} card", payload={}, entry_mode="user-saved",
    ), snapshot_text=snapshot)


def _corpus(conn, emb):
    for atom_id, snapshot in CORPUS:
        _add(conn, emb, atom_id, snapshot)


# ── 1. the write side: blob width IS what kb_meta records ───────────────────────

def test_chunks_are_stored_at_the_declared_width(kb_home, fake_embedder):
    conn = schema.connect()
    _corpus(conn, fake_embedder)

    assert stored_dtype(conn) == CHUNK_STORAGE_DTYPE == "float16"
    itemsize = np.dtype(CHUNK_STORAGE_DTYPE).itemsize
    rows = conn.execute("SELECT LENGTH(vector) n FROM chunks WHERE vector IS NOT NULL").fetchall()
    assert rows, "expected chunks"
    # dim × itemsize exactly — the live-run analogue of this is 4096 × 2 = 8192 bytes/chunk,
    # half of the 16,384 the float32 store cost.
    assert {r["n"] for r in rows} == {fake_embedder.dim * itemsize}
    conn.close()


def test_stored_vector_round_trips_to_the_embedding_it_came_from(kb_home, fake_embedder):
    """Narrowing is a STORAGE choice: decoded at the recorded width, a blob is still the vector the
    embedder produced, to half-precision. (The bag-of-words fake emits exact 1/sqrt(k) components,
    so this is a real round-trip, not a comparison of two zeros.)"""
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    dt = np.dtype(stored_dtype(conn))
    for r in conn.execute("SELECT text, vector FROM chunks WHERE vector IS NOT NULL"):
        want = fake_embedder.embed([r["text"]])[0]
        got = np.frombuffer(r["vector"], dtype=dt).astype(np.float32)
        assert np.allclose(got, want, rtol=1e-3, atol=1e-3)
    conn.close()


# ── 2. the read side: width comes from the store, not from an assumption ────────

def test_float32_and_float16_stores_return_the_same_ranking(kb_home, fake_embedder,
                                                            monkeypatch, tmp_path):
    """The reader must follow `kb_meta`, so a store written at EITHER width searches correctly.

    Same input, two stores, one query: identical atom order. A reader that hardcoded float32 would
    read the f16 store's blobs as half as many, twice-as-wide floats — reshape into garbage and
    return a different (silently wrong) order."""
    def _build(home, dtype):
        monkeypatch.setenv("OPYT_HOME", str(home))
        monkeypatch.setattr(embed_mod, "CHUNK_STORAGE_DTYPE", dtype)
        conn = schema.connect()
        _corpus(conn, fake_embedder)
        assert stored_dtype(conn) == dtype
        run = search_atoms(conn, "agent framework tools", fake_embedder, k=8, mode="semantic")
        out = [(h.atom_id, round(h.score, 3)) for h in run.hits]
        conn.close()
        return out

    wide = _build(tmp_path / "f32", "float32")
    narrow = _build(tmp_path / "f16", "float16")
    assert [a for a, _ in wide] == [a for a, _ in narrow]
    for (_, s32), (_, s16) in zip(wide, narrow):
        assert abs(s32 - s16) < 1e-3          # same cosines, to half-precision


# ── 3. the guard: a width disagreement fails LOUD, before any spend ─────────────

def test_assert_model_raises_on_a_store_of_the_other_width(kb_home, fake_embedder):
    """A float16 build pointed at a float32 store must RAISE. This is the whole point of the
    column: writing into it would leave the store holding blobs of two widths, and the vector arm
    would reshape them into a wrong answer with no exception anywhere."""
    conn = schema.connect()
    ensure_kb_meta(conn, fake_embedder.model, fake_embedder.dim, fake_embedder.provider,
                   "", storage_dtype="float32")

    assert_model(conn, fake_embedder, storage_dtype="float32")     # agreeing build: fine
    with pytest.raises(SubspaceError) as e:
        assert_model(conn, fake_embedder, storage_dtype="float16")
    assert "float16" in str(e.value) and "float32" in str(e.value)
    conn.close()


def test_ensure_kb_meta_raises_on_width_drift(kb_home, fake_embedder):
    """The same guard on the WRITE path — `_write_atom` calls this per atom, so it is the last
    line before a mixed-width blob actually lands."""
    conn = schema.connect()
    ensure_kb_meta(conn, "m", 4, "p", "", storage_dtype="float32")
    with pytest.raises(SubspaceError):
        ensure_kb_meta(conn, "m", 4, "p", "", storage_dtype="float16")
    conn.close()


def test_ingest_into_a_float32_store_refuses_rather_than_mixing(kb_home, fake_embedder,
                                                                monkeypatch):
    """End to end: a store built by a float32 build, then ingested by this (float16) build."""
    monkeypatch.setattr(embed_mod, "CHUNK_STORAGE_DTYPE", "float32")
    conn = schema.connect()
    _add(conn, fake_embedder, "github:root/agentkit", "an agent framework")

    monkeypatch.setattr(embed_mod, "CHUNK_STORAGE_DTYPE", "float16")
    with pytest.raises(SubspaceError):
        _add(conn, fake_embedder, "github:root/dash", "a react web dashboard")
    # Nothing of the second atom landed — the guard fires before the write, not after.
    widths = {r["n"] for r in
              conn.execute("SELECT DISTINCT LENGTH(vector) n FROM chunks WHERE vector IS NOT NULL")}
    assert len(widths) == 1, f"store went mixed-width: {widths}"
    conn.close()


# ── the escape hatch: converting is local and free, so the guard can be absolute ─

def test_convert_rewrites_blobs_and_restamps_meta(kb_home, fake_embedder, monkeypatch):
    monkeypatch.setattr(embed_mod, "CHUNK_STORAGE_DTYPE", "float32")
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    before = search_atoms(conn, "agent framework tools", fake_embedder, k=8, mode="semantic")
    n_chunks = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]

    monkeypatch.setattr(embed_mod, "CHUNK_STORAGE_DTYPE", "float16")
    assert convert_chunk_storage_dtype(conn, "float16") == n_chunks
    assert stored_dtype(conn) == "float16"
    assert convert_chunk_storage_dtype(conn, "float16") == 0          # idempotent

    assert_model(conn, fake_embedder, storage_dtype="float16")        # the guard now passes
    after = search_atoms(conn, "agent framework tools", fake_embedder, k=8, mode="semantic")
    assert [h.atom_id for h in after.hits] == [h.atom_id for h in before.hits]
    # And the blobs really are half as wide now — the conversion touched bytes, not just metadata.
    assert {r["n"] for r in conn.execute(
        "SELECT DISTINCT LENGTH(vector) n FROM chunks WHERE vector IS NOT NULL")} == \
        {fake_embedder.dim * 2}
    conn.close()
