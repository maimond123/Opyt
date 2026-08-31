"""The two properties the vector arm's two-pass streaming scan rests on.

Prerequisite 0 of docs/plans/2026-08-26-foreign-kb-service-phase3.md changed HOW
`atom_semantic_search` reads, not WHAT it computes. It now (a) projects the two columns cosine
reads and pays the wide card projection once per WINNER instead of once per chunk in the store,
and (b) max-pools in `VEC_BATCH` batches instead of materializing every chunk vector at once.
Measured on the real 2,805-atom store: 460 MB · 542 ms → 84 MB · 229 ms, hits bitwise identical.

Each half can break in exactly one way, and this file asserts against each:

  • BATCHING could change the answer — it is legal only because a chunk's score depends on that
    chunk and the query alone, and a maximum taken in pieces is the maximum. A softmax or any
    normalization across the candidate set would break both. `VEC_BATCH=1` and a single batch
    spanning the whole store must agree with 512.
  • THE SPLIT could render the wrong row — pass 1 knows which chunk won and pass 2 fetches cards
    by `chunk_id`, so a mis-keyed join would attach one atom's snippet and span to another's
    score, silently and plausibly. So every hit is checked against a recomputed argmax.

Derivation of the two fixes and their numbers: docs/lessons/query-cost-is-residency-not-arithmetic.md.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline.kb import retrieve, schema
from pipeline.kb.embed import stored_dtype
from tests.kb.test_export import _QUERIES, _corpus


@pytest.fixture()
def store(kb_home, fake_embedder):
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    yield conn
    conn.close()


def _hit_tuple(h):
    """Everything the arm decides, at FULL precision — `repr` on the score rather than a round,
    since a rounding that hides a drift is the one thing this file must not do."""
    return (h.atom_id, h.chunk_seq, h.chunk_span, h.snippet, h.who_name, h.description,
            h.body_state, h.entry_mode, h.sem_rank, repr(h.score))


@pytest.mark.parametrize("batch", [1, 2, 7, 100_000])
def test_the_answer_does_not_depend_on_where_a_batch_ends(store, fake_embedder, monkeypatch,
                                                          batch):
    """`batch=1` puts every chunk in its own batch; `batch=100_000` is one batch for the whole
    store, which is the all-at-once computation the streaming version replaced. Both must agree
    with the shipped 512 — if either did not, some cross-candidate dependency has entered the
    arm and batching it is no longer sound."""
    shipped = retrieve.VEC_BATCH   # read ONCE, before anything is patched
    for spec in _QUERIES:
        k = spec.get("k", 8)
        monkeypatch.setattr(retrieve, "VEC_BATCH", shipped)
        want = retrieve.atom_semantic_search(store, spec["query"], fake_embedder, None, k)
        monkeypatch.setattr(retrieve, "VEC_BATCH", batch)
        got = retrieve.atom_semantic_search(store, spec["query"], fake_embedder, None, k)
        assert [_hit_tuple(h) for h in got] == [_hit_tuple(h) for h in want], (spec, batch)


@pytest.mark.parametrize("spec", _QUERIES, ids=lambda s: json.dumps(s, sort_keys=True))
def test_the_card_rendered_is_the_chunk_that_won(store, fake_embedder, spec):
    """Pass 2 must render the chunk pass 1 ranked, for the atom it ranked it under.

    Recomputed here from the store rather than compared against a frozen copy of the old
    function: the max-pool is the arm's DEFINITION, so an independent recomputation of it is a
    stronger oracle than yesterday's implementation, and it does not rot when a filter is added.
    """
    q = np.asarray(fake_embedder.embed([spec["query"]], role="query")[0], dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-9)
    dt = np.dtype(stored_dtype(store))

    hits = retrieve.atom_semantic_search(store, spec["query"], fake_embedder, None,
                                         spec.get("k", 8))
    for h in hits:
        rows = store.execute(
            "SELECT seq, char_start, char_end, text, vector FROM chunks "
            "WHERE atom_id = ? AND vector IS NOT NULL ORDER BY seq", (h.atom_id,)).fetchall()
        assert rows, h.atom_id
        mat = np.frombuffer(b"".join(r["vector"] for r in rows), dtype=dt).reshape(len(rows), -1)
        mat = mat.astype(np.float32)
        mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sims = mat @ qn
        win = rows[int(np.argmax(sims))]
        assert (h.chunk_seq, h.chunk_span, h.snippet) == \
               (win["seq"], (win["char_start"], win["char_end"]), win["text"]), h.atom_id
        assert h.score == pytest.approx(float(sims.max()), abs=1e-6)


def test_the_scan_never_holds_more_than_one_batch_of_vectors(store, fake_embedder, monkeypatch):
    """The RAM claim, asserted at the seam it depends on: the arm decodes at most `VEC_BATCH`
    vectors per call to numpy, so peak residency is one batch plus one float per ATOM — flat in
    corpus size. Measuring RSS here would be measuring the machine; measuring the widest buffer
    the arm ever asks for measures the code."""
    widest = 0
    real = np.frombuffer

    def spy(buf, *a, **kw):
        nonlocal widest
        widest = max(widest, len(buf))
        return real(buf, *a, **kw)

    monkeypatch.setattr(retrieve, "VEC_BATCH", 2)
    monkeypatch.setattr(retrieve.np, "frombuffer", spy)
    retrieve.atom_semantic_search(store, "agent framework", fake_embedder, None, 8)

    total = store.execute("SELECT COUNT(*) FROM chunks WHERE vector IS NOT NULL").fetchone()[0]
    one_vector = len(store.execute(
        "SELECT vector FROM chunks WHERE vector IS NOT NULL LIMIT 1").fetchone()[0])
    assert total > 2, "the corpus must exceed one batch or this asserts nothing"
    assert widest <= 2 * one_vector
