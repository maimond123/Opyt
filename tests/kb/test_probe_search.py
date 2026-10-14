"""probe_search — candidate search answers with people, and never crosses the trust boundary.

Two failure classes have completely different costs, so they get different weight here.

A ranking bug is cheap: the wrong candidate appears third instead of first, a human reads the
evidence, and nothing downstream is corrupted. A boundary bug is not recoverable by reading — an
unvetted stranger's take enters an Oracle-grade answer with no marking, and the reader has no way
to tell. So the boundary tests assert on structure (which tables were touched, which ids came back)
rather than on scores, and they are the ones written to fail loudly.
"""
from __future__ import annotations

from pipeline.kb import schema
from pipeline.kb.ingest_common import store_atom
from pipeline.oracles import probe_search, probe_store


def _seed(conn, *, trusted=(), probe=()):
    """Trusted and probe atoms with distinguishable text, through the real write paths."""
    for i, (who, text) in enumerate(trusted):
        store_atom(conn, {"atom_id": f"x:{i}", "source_type": "x", "who_id": who,
                          "description": "d", "raw_hash": f"h{i}", "payload": {}}, text)
    for i, (who, text) in enumerate(probe):
        probe_store.store_probe_atom(conn, {"atom_id": f"xprobe:{i}", "source_type": "x",
                                            "who_id": who, "description": "d",
                                            "raw_hash": f"p{i}"}, text)


# ── the boundary: the tests that must never go quiet ──────────────────────────

def test_search_never_returns_a_trusted_atom(kb_home):
    """The load-bearing one. Trusted and probe rows both contain the query's words; only the probe
    author may come back. A regression here is trust laundering, not a ranking miss."""
    conn = schema.connect()
    _seed(conn, trusted=[("x:user:trusted", "an agent framework with autonomous tools")],
          probe=[("x:user:cand", "an agent framework with autonomous tools")])

    hits = probe_search.search_candidates(conn, "agent framework autonomous")

    assert [h["who_id"] for h in hits] == ["x:user:cand"]
    assert all(h["who_id"] != "x:user:trusted" for h in hits)
    conn.close()


def test_probe_search_reads_no_trusted_table(kb_home):
    """The guard cannot catch this one. Naming `atoms`/`bodies` is legal in every module, so a join
    into the trusted store from here would pass every static check and silently defeat the point.

    Asserted by watching the connection: every statement this path executes is captured, and none
    of them may mention a trusted table. Structural, so it survives a rewrite of the ranking."""
    conn = schema.connect()
    _seed(conn, trusted=[("x:user:trusted", "an agent framework")],
          probe=[("x:user:cand", "a crypto rollup proof")])

    seen: list[str] = []
    conn.set_trace_callback(seen.append)
    probe_search.search_candidates(conn, "crypto rollup")
    conn.set_trace_callback(None)

    assert seen, "no SQL captured — the trace callback did not attach, so this proves nothing"
    for sql in seen:
        flat = " ".join(sql.split()).lower()
        for table in (" atoms", " bodies", " bodies_fts"):
            assert f"from{table}" not in flat and f"join{table}" not in flat, (
                f"probe_search read a TRUSTED table — this is the boundary, not a style rule:\n"
                f"  {sql}")
    conn.close()


# ── the shape of the answer: accounts, with evidence ──────────────────────────

def test_the_unit_of_the_answer_is_an_account_not_a_post(kb_home):
    """The whole reason this is its own function rather than a `scope=` flag on trusted search.
    One candidate with four matching atoms is one row, carrying passages — not four hits."""
    conn = schema.connect()
    _seed(conn, probe=[("x:user:cand", f"a crypto rollup proof number {i}") for i in range(4)])

    hits = probe_search.search_candidates(conn, "crypto rollup proof")

    assert len(hits) == 1, "four matching atoms from one person must collapse to one account"
    assert hits[0]["atoms"] == 4
    assert isinstance(hits[0]["evidence"], list)


def test_evidence_is_capped_and_deduplicated(kb_home):
    """An account's items can repeat a passage (a thread re-quoted, a post re-shared), so an
    uncapped, undeduplicated payload shows the reader one post three times and calls it three
    pieces of evidence."""
    conn = schema.connect()
    _seed(conn, probe=[("x:user:cand", f"a crypto rollup proof number {i}") for i in range(8)])

    hits = probe_search.search_candidates(conn, "crypto rollup proof", evidence=2)
    ev = hits[0]["evidence"]
    assert len(ev) <= 2
    assert len({e["snippet"] for e in ev}) == len(ev), "the same passage was listed twice"


def test_an_empty_query_lists_who_has_been_probed(kb_home):
    """The natural opening call from a screen. Making it raise would force every caller to
    special-case its own first move."""
    conn = schema.connect()
    _seed(conn, probe=[("x:user:a", "one"), ("x:user:b", "two")])

    hits = probe_search.search_candidates(conn, "")
    assert {h["who_id"] for h in hits} == {"x:user:a", "x:user:b"}
    assert all(h["evidence"] == [] for h in hits)


# ── fail-safe: a missing input degrades to empty, never to a crash ────────────

def test_an_unprobed_store_returns_empty_rather_than_raising(kb_home):
    """Fail-safe: a store nobody has probed is the default state, not an
    error — and `probe_tables_exist` means asking must not create the store either."""
    conn = schema.connect()
    assert probe_search.search_candidates(conn, "anything") == []

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not {t for t in tables if t.startswith("probe_")}, (
        "a candidate SEARCH created the probe store — reads must not run DDL")
    conn.close()


def test_a_query_matching_nothing_returns_no_accounts(kb_home):
    """The cut has to actually cut. If every probed account came back on every query, the
    ranking would be decorative."""
    conn = schema.connect()
    _seed(conn, probe=[("x:user:cand", "a crypto rollup proof")])

    assert probe_search.search_candidates(conn, "dashboard react web") == []
    conn.close()


# ── the evidence: a window of the whole item, placed on its matches ───────────

def test_evidence_is_the_window_around_the_matches_not_the_items_opening(kb_home):
    """The item is stored whole, so its opening is often nothing to do with the query: a long
    post whose only match sits far past the first `EVIDENCE_CHARS` characters must show that
    passage, or the evidence says nothing about why the account matched."""
    conn = schema.connect()
    opening = "an unrelated opening about gardening and the weather. " * 12   # ~660 chars
    body = opening + "Here is the point: a crypto rollup proof settles on the base layer. " + (
        "and then some closing remarks about lunch. " * 10)
    assert body.index("rollup") > 2 * probe_search.EVIDENCE_CHARS
    _seed(conn, probe=[("x:user:cand", body)])

    hits = probe_search.search_candidates(conn, "crypto rollup proof")
    snip = hits[0]["evidence"][0]["snippet"]
    assert "crypto rollup proof" in snip
    assert not snip.startswith("an unrelated opening"), "the snippet is the item's opening"
    assert len(snip) <= probe_search.EVIDENCE_CHARS + 1          # + the ellipsis
    assert chr(1) not in snip and chr(2) not in snip, "highlight markers leaked into the payload"
    conn.close()


def test_a_short_item_is_its_own_evidence_whole(kb_home):
    conn = schema.connect()
    _seed(conn, probe=[("x:user:cand", "a crypto rollup  proof\nof the framework")])

    hits = probe_search.search_candidates(conn, "crypto rollup proof")
    assert hits[0]["evidence"][0]["snippet"] == "a crypto rollup proof of the framework"
    conn.close()


def test_a_passage_with_no_early_space_is_cut_at_the_limit():
    """With no space in the first 240 characters (CJK, a long URL), `rfind` gave -1 and the
    snippet kept the whole passage."""
    s = probe_search.snippet("字" * 600)
    assert len(s) <= 241 and s.endswith("…")
