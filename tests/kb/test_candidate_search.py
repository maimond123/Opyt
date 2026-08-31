"""candidate_search — the promotion answer, composed from BOTH stores, with every row's basis named.

The costly failures here are not ranking failures. Two are:

  1. **Silent provenance.** The payload is read by a MODEL. Probe passages and trusted atoms now
     arrive in ONE list, so a per-row label is the only thing standing between "unvetted stranger"
     and "something I saved". A missing label is unrecoverable by reading.
  2. **A summed score.** The probe arm carries ~25 posts per account and the saved arm usually 1.
     Add them and the saved-only candidates — the ~506 people this module exists to surface —
     vanish to the bottom of every list while LOOKING correctly ranked.

Both are asserted structurally rather than on scores.
"""
from __future__ import annotations

import pytest

from pipeline.kb import candidate_search, probe_store, schema
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


class _Cand:
    """A ranked candidate, as `screen.rank_candidates` hands them over."""

    def __init__(self, uid: str, sig: int = 3):
        self.canonical_id, self.members = f"p:{uid}", [f"x:user:{uid}"]
        self.name = self.handle = f"c{uid}"
        self.distinct_signals = sig
        self.retired = False


@pytest.fixture()
def population(monkeypatch):
    """Rank N candidates, none of them Oracles — the eligible population the counts measure
    themselves against."""
    from pipeline.kb import screen

    def _set(n: int, sigs: dict | None = None):
        cands = [_Cand(str(i), (sigs or {}).get(i, 3)) for i in range(n)]
        monkeypatch.setattr(screen, "rank_candidates", lambda c, **kw: cands)
        monkeypatch.setattr(schema, "is_oracle", lambda c, cid: False)
    return _set


# ── the boundary: provenance travels per ROW, not per payload ─────────────────

def test_each_row_names_the_store_its_evidence_came_from(kb_home, fake_embedder, population):
    """The reason the old single `provenance` line on the payload had to go. It said UNVETTED about
    everything; with trusted atoms in the same list that sentence would tell the host not to cite
    its own knowledge base."""
    conn = schema.connect()
    _seed(conn, fake_embedder,
          probe=[("x:user:0", "a crypto rollup proof")],
          trusted=[("x:user:1", "a crypto rollup proof")])
    population(2)

    rows = candidate_search.candidates_payload(conn, "crypto rollup", fake_embedder)["candidates"]
    by_who = {r["who_id"]: r for r in rows}
    assert by_who["x:user:0"]["basis"] == "probed"
    assert by_who["x:user:1"]["basis"] == "saved"
    for who, word in (("x:user:0", "unvetted"), ("x:user:1", "trusted")):
        for ev in by_who[who]["evidence"]:
            assert word in ev["provenance"].lower()
    conn.close()


def test_a_saved_only_candidate_is_reachable_at_all(kb_home, fake_embedder, population):
    """The whole feature in one assertion. Before the atom arm this person was invisible to every
    query — searchable by curation signal only — despite their writing being chunked, embedded and
    indexed the entire time."""
    conn = schema.connect()
    _seed(conn, fake_embedder, trusted=[("x:user:0", "a crypto rollup proof")])
    population(1)

    out = candidate_search.candidates_payload(conn, "crypto rollup", fake_embedder)
    assert [r["who_id"] for r in out["candidates"]] == ["x:user:0"]
    assert out["searchable_by_saved_post"] == 1
    conn.close()


def test_one_saved_post_is_not_buried_under_a_probed_accounts_volume(kb_home, fake_embedder,
                                                                     population):
    """The summed-score trap, pinned. `probe_search` scores an account by an UNCAPPED RRF sum over
    its passages, so an account with 12 matching posts accumulates ~12x the score of one with a
    single matching atom. Summed across arms it would win every slot on volume alone. The arms are
    interleaved by WITHIN-arm rank instead, so the saved-only candidate takes slot two."""
    conn = schema.connect()
    _seed(conn, fake_embedder,
          probe=[("x:user:0", f"a crypto rollup proof number {i}") for i in range(12)],
          trusted=[("x:user:1", "a crypto rollup proof")])
    population(2)

    rows = candidate_search.candidates_payload(conn, "crypto rollup", fake_embedder)["candidates"]
    assert [r["who_id"] for r in rows] == ["x:user:0", "x:user:1"]
    conn.close()


def test_a_candidate_in_both_stores_appears_once_on_the_deeper_evidence(kb_home, fake_embedder,
                                                                        population):
    """A sampled timeline is strictly deeper evidence than the one post that was saved, and showing
    the person twice would be them competing with themselves for a slot."""
    conn = schema.connect()
    _seed(conn, fake_embedder,
          probe=[("x:user:0", "a crypto rollup proof")],
          trusted=[("x:user:0", "a crypto rollup proof")])
    population(1)

    rows = candidate_search.candidates_payload(conn, "crypto rollup", fake_embedder)["candidates"]
    assert [r["who_id"] for r in rows] == ["x:user:0"]
    assert rows[0]["basis"] == "probed"
    conn.close()


# ── who is eligible at all ────────────────────────────────────────────────────

def test_confirmed_oracles_are_excluded(kb_home, fake_embedder, monkeypatch):
    """An Oracle's real footprint is already in `atoms`. Ranking them as a candidate would ask the
    user to promote someone they have already promoted."""
    from pipeline.kb import screen

    conn = schema.connect()
    _seed(conn, fake_embedder,
          probe=[("x:user:1", "a crypto rollup proof"), ("x:user:2", "a crypto rollup proof")])
    monkeypatch.setattr(screen, "rank_candidates", lambda c, **kw: [_Cand("1"), _Cand("2")])
    monkeypatch.setattr(schema, "is_oracle", lambda c, cid: cid == "p:2")

    out = candidate_search.candidates_payload(conn, "crypto rollup", fake_embedder)
    assert [r["who_id"] for r in out["candidates"]] == ["x:user:1"]
    conn.close()


def test_min_signals_filters_before_either_arm_runs(kb_home, fake_embedder, population):
    conn = schema.connect()
    _seed(conn, fake_embedder,
          probe=[("x:user:0", "a crypto rollup proof")],
          trusted=[("x:user:1", "a crypto rollup proof")])
    population(2, sigs={1: 1})

    out = candidate_search.candidates_payload(conn, "crypto rollup", fake_embedder, min_signals=2)
    assert [r["who_id"] for r in out["candidates"]] == ["x:user:0"], (
        "the saved-arm candidate is below the signal floor and must not be searched")
    conn.close()


# ── the counts, and the residue that must stay visible ────────────────────────

def test_the_three_populations_are_reported_separately(kb_home, fake_embedder, population):
    """A candidate absent because nobody sampled them looks IDENTICAL, in a ranked list, to one
    whose writing did not match. Those need opposite actions, so the counts that separate them
    travel in the payload — and 'has a saved post' is now its own population, not part of
    'not probed'."""
    conn = schema.connect()
    _seed(conn, fake_embedder,
          probe=[("x:user:0", "a crypto rollup proof")],
          trusted=[("x:user:1", "a crypto rollup proof")])
    population(3)

    out = candidate_search.candidates_payload(conn, "crypto rollup", fake_embedder)
    assert out["probed"] == 1
    assert out["searchable_by_saved_post"] == 1
    assert out["no_local_material"] == 1, "one candidate has neither a timeline nor a saved post"
    conn.close()


def test_thin_coverage_is_said_in_words_not_just_counted(kb_home, fake_embedder, population):
    """The counts alone were NOT enough, which is the whole reason this exists. A reader holding a
    number and an empty candidate list still has to work out that the second is caused by the
    first — and the reader is a model that will otherwise report the absence as a finding
    ("nobody you follow writes about this"). The live store spent 2026-08-11 to 2026-08-16 in
    exactly that state."""
    conn = schema.connect()
    _seed(conn, fake_embedder, probe=[("x:user:0", "a crypto rollup proof")])
    population(10)

    note = candidate_search.candidates_payload(conn, "crypto rollup", fake_embedder)["coverage_note"]
    assert "9 of 10" in note
    low = note.lower()
    assert "coverage" in low and "not that nobody matches" in low
    conn.close()


def test_a_saved_post_takes_someone_OUT_of_the_coverage_gap(kb_home, fake_embedder, population):
    """The denominator change, pinned. This note used to count 'not probed'; a candidate whose
    saved post is searchable is no longer dark, and counting them as missing would understate
    coverage and train the reader to discount a note that is usually wrong."""
    conn = schema.connect()
    _seed(conn, fake_embedder,
          trusted=[(f"x:user:{i}", "a crypto rollup proof") for i in range(19)])
    population(20)

    out = candidate_search.candidates_payload(conn, "crypto rollup", fake_embedder)
    # Under the OLD denominator this is 20 of 20 "not probed" — a full-throated coverage warning
    # on a store that can actually answer the question for 19 of them.
    assert out["probed"] == 0
    assert out["searchable_by_saved_post"] == 19
    assert out["no_local_material"] == 1            # …only ONE candidate is genuinely dark
    assert "coverage_note" not in out
    conn.close()


def test_the_notice_goes_quiet_once_coverage_is_near_complete(kb_home, fake_embedder, population):
    """Same rule as `signal_reconcile` and `list_freshness`: a line printed on every call stops
    being read. Here the threshold also protects the notice's own truth — with 1 of 12 dark, a
    thin result really is a thin MATCH, and a coverage caveat would mislead the other way."""
    conn = schema.connect()
    _seed(conn, fake_embedder,
          probe=[(f"x:user:{i}", "a crypto rollup proof") for i in range(11)])
    population(12)

    out = candidate_search.candidates_payload(conn, "crypto rollup", fake_embedder)
    assert out["no_local_material"] == 1            # still REPORTED as a number
    assert "coverage_note" not in out               # just not narrated
    conn.close()


# ── fail-safe ─────────────────────────────────────────────────────────────────

def test_an_empty_store_returns_empty_rather_than_raising(kb_home, fake_embedder):
    """CLAUDE.md's fail-safe invariant. A store nobody has probed and nobody has saved into is the
    DEFAULT state, not an error — and asking must not create the probe store either."""
    conn = schema.connect()
    out = candidate_search.candidates_payload(conn, "anything", fake_embedder)
    assert out["candidates"] == []
    assert out["probed"] == 0 and out["searchable_by_saved_post"] == 0

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not {t for t in tables if t.startswith("probe_")}, (
        "a candidate SEARCH created the probe store — reads must not run DDL")
    conn.close()


def test_an_empty_query_lists_who_is_there_from_both_arms(kb_home, fake_embedder, population):
    """The natural opening call. No query means no embedder is built, so neither arm may need one."""
    conn = schema.connect()
    _seed(conn, fake_embedder,
          probe=[("x:user:0", "a crypto rollup proof")],
          trusted=[("x:user:1", "a crypto rollup proof")])
    population(2)

    rows = candidate_search.candidates_payload(conn, "", None)["candidates"]
    assert {r["who_id"] for r in rows} == {"x:user:0", "x:user:1"}
    conn.close()


def test_the_saved_arm_never_reads_the_probe_store(kb_home, fake_embedder, population):
    """The arms answer from different stores and must stay that way: `search_saved_atoms` is the
    trusted half, and a probe row leaking into it would put unvetted content behind a `saved`
    label — the one mislabelling this module's whole design is meant to prevent."""
    conn = schema.connect()
    _seed(conn, fake_embedder,
          probe=[("x:user:0", "a crypto rollup proof")],
          trusted=[("x:user:1", "a crypto rollup proof")])

    hits = candidate_search.search_saved_atoms(
        conn, "crypto rollup", fake_embedder, who_ids={"x:user:0", "x:user:1"})
    assert [h["who_id"] for h in hits] == ["x:user:1"]
    for ev in hits[0]["evidence"]:
        assert ev["atom_id"].startswith("x:"), "a probe atom id reached the trusted arm"
    conn.close()
