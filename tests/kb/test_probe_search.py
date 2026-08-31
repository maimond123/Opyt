"""probe_search — candidate search answers with PEOPLE, and never crosses the trust boundary.

Two failure classes have completely different costs, so they get different weight here.

A ranking bug is cheap: the wrong candidate appears third instead of first, a human reads the
evidence, and nothing downstream is corrupted. A BOUNDARY bug is not recoverable by reading — an
unvetted stranger's take enters an Oracle-grade answer with no marking, and the reader has no way
to tell. So the boundary tests assert on structure (which tables were touched, which ids came back)
rather than on scores, and they are the ones written to fail loudly.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.kb import probe_search, probe_store, schema
from pipeline.kb.ingest_common import AtomSink


def _seed(conn, embedder, *, trusted=(), probe=()):
    """Trusted and probe atoms with distinguishable text, through the real write paths."""
    if trusted:
        t = AtomSink(conn, embedder)
        for i, (who, text) in enumerate(trusted):
            t.submit({"atom_id": f"x:{i}", "source_type": "x", "who_id": who,
                      "description": "d", "raw_hash": f"h{i}"}, text)
        t.close()
    if probe:
        p = AtomSink(conn, embedder, writer=probe_store.write_probe_atom)
        for i, (who, text) in enumerate(probe):
            p.submit({"atom_id": f"xprobe:{i}", "source_type": "x", "who_id": who,
                      "description": "d", "raw_hash": f"p{i}"}, text)
        p.close()


# ── the boundary: the tests that must never go quiet ──────────────────────────

def test_search_never_returns_a_trusted_atom(kb_home, fake_embedder):
    """The load-bearing one. Trusted and probe rows both contain the query's words; only the probe
    author may come back. A regression here is trust laundering, not a ranking miss."""
    conn = schema.connect()
    _seed(conn, fake_embedder,
          trusted=[("x:user:trusted", "an agent framework with autonomous tools")],
          probe=[("x:user:cand", "an agent framework with autonomous tools")])

    hits = probe_search.search_candidates(conn, "agent framework autonomous", fake_embedder)

    assert [h["who_id"] for h in hits] == ["x:user:cand"]
    assert all(h["who_id"] != "x:user:trusted" for h in hits)
    conn.close()


def test_probe_search_reads_no_trusted_table(kb_home, fake_embedder):
    """The guard cannot catch this one. Naming `atoms`/`chunks` is legal in every module, so a join
    into the trusted store from here would pass every static check and silently defeat the point.

    Asserted by watching the connection: every statement this path executes is captured, and none
    of them may mention a trusted table. Structural, so it survives a rewrite of the ranking."""
    conn = schema.connect()
    _seed(conn, fake_embedder,
          trusted=[("x:user:trusted", "an agent framework")],
          probe=[("x:user:cand", "a crypto rollup proof")])

    seen: list[str] = []
    conn.set_trace_callback(seen.append)
    probe_search.search_candidates(conn, "crypto rollup", fake_embedder)
    conn.set_trace_callback(None)

    assert seen, "no SQL captured — the trace callback did not attach, so this proves nothing"
    for sql in seen:
        flat = " ".join(sql.split()).lower()
        for table in (" atoms", " chunks", " chunks_fts"):
            assert f"from{table}" not in flat and f"join{table}" not in flat, (
                f"probe_search read a TRUSTED table — this is the boundary, not a style rule:\n"
                f"  {sql}")
    conn.close()


# ── the shape of the answer: accounts, with evidence ──────────────────────────

def test_the_unit_of_the_answer_is_an_account_not_a_post(kb_home, fake_embedder):
    """The whole reason this is its own function rather than a `scope=` flag on trusted search.
    One candidate with four matching atoms is ONE row, carrying passages — not four hits."""
    conn = schema.connect()
    _seed(conn, fake_embedder,
          probe=[("x:user:cand", f"a crypto rollup proof number {i}") for i in range(4)])

    hits = probe_search.search_candidates(conn, "crypto rollup proof", fake_embedder)

    assert len(hits) == 1, "four matching atoms from one person must collapse to one account"
    assert hits[0]["atoms"] == 4
    assert isinstance(hits[0]["evidence"], list)


def test_evidence_is_capped_and_deduplicated(kb_home, fake_embedder):
    """Both arms routinely surface the SAME passage, so an uncapped, undeduplicated payload shows
    the reader one post three times and calls it three pieces of evidence."""
    conn = schema.connect()
    _seed(conn, fake_embedder,
          probe=[("x:user:cand", f"a crypto rollup proof number {i}") for i in range(8)])

    hits = probe_search.search_candidates(conn, "crypto rollup proof", fake_embedder, evidence=2)
    ev = hits[0]["evidence"]
    assert len(ev) <= 2
    assert len({e["snippet"] for e in ev}) == len(ev), "the same passage was listed twice"


def test_an_empty_query_lists_who_has_been_probed(kb_home, fake_embedder):
    """The natural opening call from a screen. Making it raise would force every caller to
    special-case its own first move — and it needs NO embedder, which is why the tool defers
    constructing one."""
    conn = schema.connect()
    _seed(conn, fake_embedder, probe=[("x:user:a", "one"), ("x:user:b", "two")])

    hits = probe_search.search_candidates(conn, "", None)      # embedder deliberately None
    assert {h["who_id"] for h in hits} == {"x:user:a", "x:user:b"}
    assert all(h["evidence"] == [] for h in hits)


# ── fail-safe: a missing input degrades to empty, never to a crash ────────────

def test_an_unprobed_store_returns_empty_rather_than_raising(kb_home, fake_embedder):
    """CLAUDE.md's fail-safe invariant. A store nobody has probed is the DEFAULT state, not an
    error — and `probe_tables_exist` means asking must not create the store either."""
    conn = schema.connect()
    assert probe_search.search_candidates(conn, "anything", fake_embedder) == []

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not {t for t in tables if t.startswith("probe_")}, (
        "a candidate SEARCH created the probe store — reads must not run DDL")
    conn.close()


def test_a_query_matching_nothing_returns_no_accounts(kb_home, fake_embedder):
    """The floor has to actually cut. Without it every probed account comes back on every query,
    which makes the ranking decorative."""
    conn = schema.connect()
    _seed(conn, fake_embedder, probe=[("x:user:cand", "a crypto rollup proof")])

    # The query shares NO vocabulary with the stored text, so under FakeEmbedder its cosine is
    # exactly 0.0 — comfortably under the floor. No semantic evidence may survive.
    hits = probe_search.search_candidates(conn, "dashboard react web", fake_embedder)
    for h in hits:
        assert all(e["arm"] != "semantic" for e in h["evidence"]), (
            f"an unrelated passage cleared EVIDENCE_FLOOR={probe_search.EVIDENCE_FLOOR} — "
            f"the floor is not cutting")
    conn.close()


# ── the embed_text reader, and the fallback it made mandatory ─────────────────

def test_evidence_renders_the_stripped_surface_not_the_raw_render(kb_home, fake_embedder):
    """Snippets come from `embed_text`, so OPYT's own render template does not eat the budget.

    Measured on the real store, a raw snippet opens with `# Name — 2026-04-14` and can be almost
    entirely `**Quoting** [@handle](https://x.com/…/status/…)`. Both facts are already fields on
    the row being rendered, so spending characters on them buys the reader nothing."""
    conn = schema.connect()
    _seed(conn, fake_embedder,
          probe=[("x:user:cand",
                  "# Some Person — 2026-04-14\n\na crypto rollup proof of the framework\n\n"
                  "---\n*Candidate probe · [Original post](https://x.com/p/status/1)*\n")])

    hits = probe_search.search_candidates(conn, "crypto rollup proof", fake_embedder)
    snip = hits[0]["evidence"][0]["snippet"]

    assert "a crypto rollup proof of the framework" in snip
    assert "# Some Person" not in snip and "2026-04-14" not in snip
    assert "Candidate probe" not in snip and "Original post" not in snip
    conn.close()


def test_a_null_embed_text_falls_back_to_text_rather_than_rendering_empty(kb_home, fake_embedder):
    """The fallback `probe_store`'s schema comment said would become mandatory the moment a reader
    appeared. A NULL means the vector came from `text` verbatim — it is a legal state, not a
    corrupt one, and it must render the passage rather than an empty string."""
    conn = schema.connect()
    _seed(conn, fake_embedder, probe=[("x:user:cand", "a crypto rollup proof")])
    conn.execute("UPDATE probe_chunks SET embed_text = NULL")
    conn.commit()

    hits = probe_search.search_candidates(conn, "crypto rollup proof", fake_embedder)
    snip = hits[0]["evidence"][0]["snippet"]
    assert snip, "a NULL embed_text rendered an EMPTY snippet — evidence that says nothing"
    assert "crypto rollup proof" in snip
    conn.close()


def test_only_a_continuation_chunk_is_marked_as_cut(kb_home, fake_embedder):
    """A leading `…` means "this passage starts mid-thought", and it has to be TRUE to be useful.

    The signal is `seq`, not `char_start`. `char_start` indexes the raw source document, whose
    frontmatter occupies the first ~165 characters — measured on the real store it is never 0, so a
    `char_start > 0` test marks EVERY passage and degrades to decoration. This pins the distinction
    that made the first version wrong."""
    conn = schema.connect()
    _seed(conn, fake_embedder, probe=[("x:user:cand", "a crypto rollup proof of the framework")])

    # seq 0 is the atom's own opening — never marked
    hits = probe_search.search_candidates(conn, "crypto rollup proof", fake_embedder)
    assert not hits[0]["evidence"][0]["snippet"].startswith("…")

    # force the same row to look like a continuation; now it must be marked
    conn.execute("UPDATE probe_chunks SET seq = 1")
    conn.commit()
    hits = probe_search.search_candidates(conn, "crypto rollup proof", fake_embedder)
    assert hits[0]["evidence"][0]["snippet"].startswith("…"), (
        "a continuation passage was not marked — the reader cannot tell it starts mid-word")
    conn.close()


# ── the calibrated-constant trap ──────────────────────────────────────────────

def test_the_evidence_floor_is_its_own_number(kb_home):
    """`EVIDENCE_FLOOR` must NOT be `sitting_builder.FLOOR_DEFAULT`. That number is calibrated as
    the 99th percentile of random-pair cosine over the TRUSTED corpus plus a margin — a different
    population answering a different question. Borrowing a calibrated constant outside the data it
    was calibrated on is how a number silently stops meaning anything, and this pins that they are
    allowed to drift apart."""
    from pipeline.kb.sitting_builder import FLOOR_DEFAULT

    assert probe_search.EVIDENCE_FLOOR != FLOOR_DEFAULT
    assert 0.0 < probe_search.EVIDENCE_FLOOR < 1.0
