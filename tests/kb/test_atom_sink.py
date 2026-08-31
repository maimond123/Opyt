"""AtomSink — cross-atom embed batching (ARC-1 Phase 1, Step 2).

The sink defers embedding until ~batch_size chunks accrue, then embeds them in ONE call and
writes each atom back with its own vector slice. These tests pin the properties that make that
safe: positional alignment across a flat vector list (the silent-corruption risk), poison-chunk
isolation (Fail-safe), flush-at-threshold + close-flushes-remainder, and store_atom wrapper parity.

A deterministic text→vector fake lets us assert the STRONG alignment property directly — every
stored vector equals the embedding of ITS OWN chunk text — so a mis-slice can't hide behind a
green count. No network, no spend.
"""
from __future__ import annotations

import hashlib

import numpy as np

from pipeline.kb import schema
from pipeline.kb.embed import EmbedError, stored_dtype
from pipeline.kb.ingest_common import AtomSink, store_atom


class RecordingEmbedder:
    """Deterministic text→vector embedder that RECORDS each batch it was handed. A text maps to a
    fixed float32 vector via sha256, so distinct texts get distinct vectors and the SAME text always
    embeds to the SAME vector — which is what lets a test assert `stored_vector == embed(its_text)`.
    `poison`: if any text in a batch contains this marker, the batch raises EmbedError (all-or-nothing,
    like the real hosted embedder) so the sink's per-atom isolation path is exercised."""

    provider = "local"
    model = "fake-rec"
    query_instruction = ""

    def __init__(self, dim: int = 6, batch_size: int = 64, poison: str | None = None):
        self.dim = dim
        self.batch_size = batch_size
        self.poison = poison
        self.calls: list[list[str]] = []   # one entry per embed() call = the batch of texts it saw

    def _vec(self, t: str) -> np.ndarray:
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h[: self.dim], dtype=np.uint8).astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def embed(self, texts, *, role: str = "document"):
        self.calls.append(list(texts))
        if self.poison and any(self.poison in t for t in texts):
            raise EmbedError("poison chunk in batch", retryable=False)
        return [self._vec(t) for t in texts]


def _mk_atom(aid: str) -> dict:
    return dict(atom_id=aid, source_type="x", what_kind="opinion", who_id="x:u",
                when_ts="2024-01-01", when_precision="day",
                about_entities=[], source_url="u", raw_ref="ref", raw_hash="h",
                description="d", payload={}, entry_mode="user-saved")


def _body(marker: str, approx_chars: int) -> str:
    """A body of DISTINCT position-tagged tokens ~approx_chars long. Distinct tokens → distinct
    (overlapping) chunk windows → distinct vectors, so alignment is actually observable."""
    toks, total, i = [], 0, 0
    while total < approx_chars:
        t = f"{marker}{i:05d}"
        toks.append(t)
        total += len(t) + 1
        i += 1
    return " ".join(toks)


def _chunks(conn):
    return conn.execute("SELECT atom_id, text, vector FROM chunks").fetchall()


def _assert_every_vector_matches_its_text(conn, emb):
    """The strong alignment check: the vector stored against a chunk IS the embedding of that
    chunk's own text. A mis-slice (atom A's vector on atom B's text) fails this.

    Decoded at the store's OWN recorded width (never a hardcoded float32) — so this doubles as the
    storage round-trip check. The tolerance is half-precision-sized: f16 has ~3 decimal digits, so
    a mis-slice (an unrelated vector) still fails by orders of magnitude."""
    rows = _chunks(conn)
    assert rows, "expected chunks to be written"
    dt = np.dtype(stored_dtype(conn))
    for r in rows:
        stored = np.frombuffer(r["vector"], dtype=dt).astype(np.float32)
        assert np.allclose(stored, emb._vec(r["text"]), rtol=1e-3, atol=1e-3), \
            f"vector/text misaligned on {r['atom_id']}"


# ── batching + alignment: many atoms, varied chunk counts, ONE embed call ─────────

def test_batches_across_atoms_and_aligns_vectors(kb_home):
    conn = schema.connect()
    emb = RecordingEmbedder()
    written: list[str] = []
    sink = AtomSink(conn, emb)   # default flush threshold (64) — far above our chunk total

    # Three atoms with 1 / 2 / 3 chunks → the flat vector list has non-trivial per-atom spans.
    for aid, chars in [("x:a", 800), ("x:b", 2500), ("x:c", 4000)]:
        sink.submit(_mk_atom(aid), _body(aid, chars),
                    on_written=(lambda a=aid: written.append(a)))
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0   # nothing flushed yet
    sink.close()

    assert len(emb.calls) == 1                                    # ONE embed call for all three atoms
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert len(emb.calls[0]) == total_chunks                      # every chunk went in that one call
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 3
    assert set(written) == {"x:a", "x:b", "x:c"}                  # on_written fired once per durable atom
    _assert_every_vector_matches_its_text(conn, emb)             # <-- the alignment guarantee
    conn.close()


# ── flush-at-threshold: submit crosses the chunk threshold → auto-flush before close ──

def test_auto_flushes_at_chunk_threshold(kb_home):
    conn = schema.connect()
    emb = RecordingEmbedder()
    sink = AtomSink(conn, emb, flush_chunks=2)

    sink.submit(_mk_atom("x:1"), _body("x:1", 200))              # 1 chunk → pending 1 < 2, no flush
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0
    assert emb.calls == []

    sink.submit(_mk_atom("x:2"), _body("x:2", 200))              # 1 chunk → pending 2 ≥ 2 → flush
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 2
    assert len(emb.calls) == 1
    conn.close()


def test_default_flush_threshold_is_four_batches(kb_home):
    """Default flush_chunks = 4×batch_size, NOT batch_size. Flushing AT batch_size would spill a
    near-empty tail HTTP call on every flush (~2× the ideal call count); a 4× buffer amortizes it.
    Proven behaviorally: the buffer sails PAST batch_size without flushing, then trips at 4×."""
    conn = schema.connect()
    emb = RecordingEmbedder(batch_size=4)                        # default threshold = 4×4 = 16
    sink = AtomSink(conn, emb)

    for i in range(15):                                          # 15 one-chunk atoms: past 4, under 16
        sink.submit(_mk_atom(f"x:{i}"), _body(f"x:{i}", 120))
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0   # did NOT flush at batch_size
    assert emb.calls == []

    sink.submit(_mk_atom("x:15"), _body("x:15", 120))           # 16th chunk → crosses 4×batch_size
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 16  # flushed at 4×batch_size
    assert len(emb.calls) == 1
    conn.close()


def test_close_flushes_remainder(kb_home):
    conn = schema.connect()
    emb = RecordingEmbedder()
    sink = AtomSink(conn, emb)                                    # threshold 64, never reached

    sink.submit(_mk_atom("x:1"), _body("x:1", 200))
    sink.submit(_mk_atom("x:2"), _body("x:2", 200))
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0   # buffered, not written
    sink.close()
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 2
    assert len(emb.calls) == 1
    conn.close()


# ── poison-chunk isolation: one bad atom skips, every good atom still writes ───────

def test_poison_chunk_isolates_to_its_own_atom(kb_home):
    conn = schema.connect()
    emb = RecordingEmbedder(poison="POISON")
    written: list[str] = []
    sink = AtomSink(conn, emb)

    sink.submit(_mk_atom("x:good1"), _body("x:good1", 300),
                on_written=(lambda: written.append("x:good1")))
    sink.submit(_mk_atom("x:bad"), "this body has a POISON token in it",
                on_written=(lambda: written.append("x:bad")))
    sink.submit(_mk_atom("x:good2"), _body("x:good2", 300),
                on_written=(lambda: written.append("x:good2")))
    sink.close()

    ids = {r[0] for r in conn.execute("SELECT atom_id FROM atoms").fetchall()}
    assert ids == {"x:good1", "x:good2"}          # the poisoned atom skipped; good atoms wrote
    assert written == ["x:good1", "x:good2"]       # on_written never fired for the skipped atom
    # Batched call raised (poison present) → then one isolated re-embed per atom = 1 + 3 calls.
    assert len(emb.calls) == 4
    _assert_every_vector_matches_its_text(conn, emb)
    conn.close()


# ── store_atom wrapper parity: the 8 non-X call sites still get synchronous, durable writes ──

def test_store_atom_wrapper_writes_synchronously(kb_home):
    conn = schema.connect()
    emb = RecordingEmbedder()
    store_atom(conn, emb, atom=_mk_atom("x:sa"), snapshot_text=_body("x:sa", 500))

    # Written and durable the moment store_atom returns (submit + immediate flush).
    assert conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_id='x:sa'").fetchone()[0] == 1
    _assert_every_vector_matches_its_text(conn, emb)   # chunks carry real (aligned) vectors
    conn.close()
