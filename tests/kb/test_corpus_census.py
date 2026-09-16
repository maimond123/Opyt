"""The census sample — and every way a spread of a corpus could quietly misrepresent it.

This is the replacement for `top_topics`, so it inherits that surface's failure mode and has to
not repeat it: a PARTIAL VIEW THAT READS LIKE A FINDING. Five hashtags off 5 of 1,801 atoms were
wrong because nothing downstream could tell they were five. A sample is wrong the same way the
moment it over-represents one corner and nothing says so.

  • BREADTH BEATS VOLUME. The densest part of a store is its most repetitive part — the reason
    `densest_unread` was deleted. A sample that walks authors in rounds cannot be captured by
    one prolific voice; `LIMIT`/random both can.
  • THE SAMPLE DECLARES ITSELF. `sampled`/`of` and `authors`/`of_authors` ride every result,
    because "300 of 1801 from 300 authors" and "300 of 1801 from 4 authors" are the same size
    and completely different evidence.
  • IT IS OFF UNLESS ASKED FOR. Every existing `aggregate` caller's payload is unchanged.
  • NOTHING IS STORED. The store is byte-identical after a census; the naming is the host's,
    in its response, and this module never invents a label.
"""
from __future__ import annotations

import pytest

from opyt_core.kb import kb_aggregate
from pipeline.kb import corpus_census as cc
from pipeline.kb import schema


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _seed(conn, rows):
    """rows: (who_id, when_ts) — one atom each, described by its own index."""
    for i, (who, when) in enumerate(rows):
        conn.execute(
            "INSERT INTO atoms (atom_id, source_type, who_id, when_ts, entry_mode, what_kind, "
            "description, payload) VALUES (?,?,?,?,'user-saved','post',?,'{}')",
            (f"a:{i}", "substack", who, when, f"{who} post {i}"))
    conn.commit()


# ── breadth ─────────────────────────────────────────────────────────────────────
def test_one_prolific_author_cannot_capture_the_sample(conn):
    """⚠️ THE WHOLE REASON THIS IS A ROUND-ROBIN. 200 atoms from one voice and 1 each from ten
    others is a store ABOUT eleven things; a plain LIMIT or a random draw returns ~95% one
    voice and the host names that voice's preoccupation as the corpus."""
    _seed(conn, [("x:loud", f"2026-01-{1 + i % 28:02d}") for i in range(200)]
                + [(f"x:q{i}", "2026-02-01") for i in range(10)])

    got = cc.corpus_sample(conn, 11)

    assert got["authors"] == 11, "every author should appear before anyone repeats"
    assert sum(1 for d in got["descriptions"] if d.startswith("x:loud ")) == 1


def test_a_budget_larger_than_the_author_count_goes_round_again(conn):
    """The other side of it: breadth first does not mean breadth ONLY. Once every author has
    been heard from, the rounds continue, so a narrow store still fills its budget."""
    _seed(conn, [(f"x:w{i % 4}", f"2026-03-{1 + i:02d}") for i in range(20)])

    got = cc.corpus_sample(conn, 12)

    assert got["sampled"] == 12 and got["authors"] == 4
    assert all(sum(1 for d in got["descriptions"] if d.startswith(f"x:w{i} ")) == 3
               for i in range(4))


def test_a_single_author_store_is_sampled_without_complaint(conn):
    """There is no breadth to find and that is not a failure — the counts say so plainly, which
    is what lets a reader discount whatever the host names from it."""
    _seed(conn, [("x:solo", f"2026-04-{1 + i:02d}") for i in range(20)])

    got = cc.corpus_sample(conn, 5)

    assert (got["sampled"], got["authors"], got["of_authors"], got["of"]) == (5, 1, 1, 20)


def test_the_same_store_returns_the_same_spread_twice(conn):
    """A census that reshuffles on every call cannot be checked against itself — and a host
    asked to justify a subject it named would be quoting a sample that no longer exists."""
    _seed(conn, [(f"x:w{i % 7}", f"2026-05-{1 + i % 28:02d}") for i in range(60)])

    first = cc.corpus_sample(conn, 20)["descriptions"]
    assert first == cc.corpus_sample(conn, 20)["descriptions"]


# ── declaring itself ────────────────────────────────────────────────────────────
def test_the_sample_reports_what_it_is_a_sample_of(conn):
    """A spread read as a complete listing is the failure `top_topics` was deleted for."""
    _seed(conn, [(f"x:w{i % 9}", "2026-06-01") for i in range(50)])

    got = cc.corpus_sample(conn, 10)

    assert (got["sampled"], got["of"]) == (10, 50)
    assert (got["authors"], got["of_authors"]) == (9, 9)
    assert "SAMPLE" in got["host_note"] and "search(query=" in got["host_note"]
    # STRINGS, not rows: the three id fields were dropped 2026-09-16 after the first live call
    # overflowed the host's tool-result cap and its own `jq` recovery read only `.description`.
    assert all(isinstance(d, str) for d in got["descriptions"])
    assert "atoms" not in got


def test_the_note_sends_the_host_to_verify_before_naming(conn):
    """The one instruction that separates this from the retired Stage 6: a name is a guess
    until the index confirms it. Without this line the host reads a spread and asserts."""
    _seed(conn, [(f"x:w{i % 9}", "2026-06-01") for i in range(50)])

    note = cc.corpus_sample(conn, 10)["host_note"]
    assert "CHECK each one" in note
    assert "do not store the names" in note


# ── the empty and the absent ────────────────────────────────────────────────────
@pytest.mark.parametrize("n", [0, -5, None])
def test_no_ask_means_no_key_rather_than_an_empty_census(conn, n):
    """Fail-safe: an absent key is a smaller lie than a present-but-empty one, which reads as
    'your corpus is about nothing'."""
    _seed(conn, [("x:a", "2026-07-01")])
    assert cc.corpus_sample(conn, n) is None


def test_an_empty_store_yields_no_census(conn):
    assert cc.corpus_sample(conn, 10) is None


def test_an_over_large_ask_is_clamped_not_refused(conn):
    """A host misjudging its budget is not an error worth failing a read-only call over — but
    it must not be allowed to drown its own context either."""
    _seed(conn, [(f"x:w{i}", "2026-08-01") for i in range(20)])

    assert cc.corpus_sample(conn, 10_000)["sampled"] == 20      # clamped to SAMPLE_MAX, then store
    assert cc.SAMPLE_MAX == 500


# ── through the aggregate surface ───────────────────────────────────────────────
def test_aggregate_omits_the_census_unless_it_is_asked_for(conn):
    """OFF BY DEFAULT: this is the one costly key here, and every caller that existed before it
    must get a byte-identical payload."""
    _seed(conn, [(f"x:w{i % 5}", "2026-09-01") for i in range(30)])
    conn.close()

    assert "corpus_sample" not in kb_aggregate()
    assert kb_aggregate(sample=8)["corpus_sample"]["sampled"] == 8


def test_a_scoped_census_describes_the_scope_and_not_the_store(conn):
    """`of` must count the SCOPE. Reporting the whole store's total beside a filtered spread
    would say "8 of 30" about a slice that only ever held 10 — plausible, unfalsifiable."""
    _seed(conn, [(f"x:w{i % 5}", "2026-09-01") for i in range(30)])
    conn.execute("UPDATE atoms SET source_type = 'github' WHERE atom_id LIKE 'a:1%'")
    conn.commit()
    n_github = conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE source_type = 'github'").fetchone()[0]
    conn.close()

    got = kb_aggregate({"source_type": "github"}, sample=50)["corpus_sample"]
    assert got["of"] == n_github < 30
    assert got["sampled"] == n_github


def test_a_census_writes_nothing(conn):
    """The invariant this whole design exists to keep: interpretations are query-time and
    disposable. A census that left a row behind would be Stage 6 wearing a different name."""
    _seed(conn, [(f"x:w{i % 5}", "2026-09-01") for i in range(30)])
    before = conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()

    kb_aggregate(sample=30)

    c = schema.connect()
    try:
        assert c.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == before
        assert {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")} == tables
    finally:
        c.close()
