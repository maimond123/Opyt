"""`date_from` / `date_to` on the atom rail — the boundary behaviour, not the plumbing.

Before this, asking "what did X post after 5/11/2026" silently dropped the date, or matched it as
CONTENT when a host put it in `query`. Either way results came back looking filtered.

The one decision worth pinning is what a COARSE date means at a boundary. `when_ts` is always
`YYYY-MM-DD` shaped, so comparison is uniform; imprecision lives in `when_precision`, and a
`year`-precision atom is a Jan-1 FLOOR standing for its whole year. Compared strictly it would
vanish from every "after mid-year" query, invisibly. Compared as a RANGE it can be over-included,
visibly — the host reads `when_precision` on the hit and discards it. The host can filter down; it
cannot filter up. That asymmetry is what these tests hold in place.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from opyt_core import kb as kb_entry
from pipeline.kb import schema
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.raw_store import write_snapshot
from pipeline.kb.retrieve import _PRECISION_SPAN, _date_bound

# One shared word ("agent") so a single bm25 query reaches every atom and the DATE is the only
# thing separating them.
OLD = "x:old"            # day precision, 2024-03-04
MID = "x:mid"            # day precision, 2026-05-11 — the boundary date itself
NEW = "x:new"            # day precision, 2026-09-20
YEAR = "paper:2026.001"  # YEAR precision, stored at 2026-01-01 (a floor, not a real day)
BLANK = "x:blank"        # when_ts = ""   (the undated shape the derivers write)
NULL = "x:null"          # when_ts = NULL (the undated shape the column allows)


def _add(conn, emb, atom_id, *, when_ts, when_precision, who_id="x:user:1", topics=("ai-agents",),
         source_type="x", text="an agent framework post"):
    raw_ref, raw_hash = write_snapshot(source_type, atom_id, text)
    store_atom(conn, emb, atom=dict(
        atom_id=atom_id, source_type=source_type, what_kind="opinion", who_id=who_id,
        when_ts=when_ts, when_precision=when_precision,
        about_entities=[], source_url=f"https://example/{atom_id}", raw_ref=raw_ref,
        raw_hash=raw_hash, description=f"{atom_id} card",
        payload={"source_tags": list(topics)}, entry_mode="user-saved",
    ), snapshot_text=text)


def _corpus(conn, emb):
    _add(conn, emb, OLD, when_ts="2024-03-04", when_precision="day")
    _add(conn, emb, MID, when_ts="2026-05-11", when_precision="day")
    _add(conn, emb, NEW, when_ts="2026-09-20", when_precision="day")
    _add(conn, emb, YEAR, when_ts="2026-01-01", when_precision="year",
         who_id="scholar:1", source_type="paper", text="an agent framework paper")
    _add(conn, emb, BLANK, when_ts="", when_precision="unknown")
    _add(conn, emb, NULL, when_ts=None, when_precision=None)


def _ids(conn=None, **kw):
    return {h["atom_id"] for h in kb_entry.run_kb_search("agent", mode="bm25", k=20, **kw)["hits"]}


@pytest.fixture()
def corpus(kb_home, fake_embedder):
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    conn.close()
    return fake_embedder


# ── exact dates, both bounds, inclusive ──────────────────────────────────────────

def test_a_lower_bound_keeps_the_boundary_day_itself(corpus):
    """Inclusive at the edge. "after May 11" asked with `date_from="2026-05-11"` must include
    the 11th — an off-by-one here silently loses a whole day and looks like nothing."""
    assert _ids(date_from="2026-05-11") == {MID, NEW, YEAR}


def test_an_upper_bound_keeps_the_boundary_day_itself(corpus):
    assert _ids(date_to="2026-05-11") == {OLD, MID, YEAR}


def test_the_two_bounds_compose_into_a_window(corpus):
    assert _ids(date_from="2026-02-01", date_to="2026-06-01") == {MID, YEAR}


# ── THE decision: a year-precision atom spans its year ───────────────────────────

def test_a_year_only_atom_survives_a_lower_bound_later_in_that_same_year(corpus):
    """The central decision, asserted directly. `YEAR` is stored at 2026-01-01 because only the
    year is known — a FLOOR, not a real day. Compared strictly it drops out of a May query and
    nothing anywhere says so; the paper may well have been published in June."""
    assert YEAR in _ids(date_from="2026-05-11")
    assert YEAR in _ids(date_from="2026-12-31")     # even the last day of its own year


def test_a_year_only_atom_does_not_leak_into_the_next_year(corpus):
    """The other half — range-aware is not the same as unfiltered. Its range ENDS at 2026-12-31,
    so a 2027 lower bound excludes it."""
    assert YEAR not in _ids(date_from="2027-01-01")


def test_a_year_only_atom_is_still_bounded_from_above_at_its_floor(corpus):
    """`date_to` compares the range START. 2025 is before anything the atom could be."""
    assert YEAR not in _ids(date_to="2025-12-31")


def test_a_day_precision_atom_is_not_widened_by_the_year_rule(corpus):
    """The rule keys on `when_precision`, NOT on the value's shape. A genuine Jan-1 post is an
    exact day and must filter exactly — this is where the vault rail's `date LIKE '%-01-01'`
    sniffing would produce a false positive, and why the precision column is not optional."""
    conn = schema.connect()
    _add(conn, corpus, "x:jan1", when_ts="2026-01-01", when_precision="day")
    conn.close()
    assert "x:jan1" not in _ids(date_from="2026-05-11")


# ── undated atoms: excluded by BOTH bounds, and said out loud ────────────────────

def test_an_undated_atom_is_excluded_by_a_lower_bound(corpus):
    got = _ids(date_from="2024-01-01")
    assert BLANK not in got and NULL not in got


def test_an_undated_atom_is_excluded_by_an_upper_bound_too(corpus):
    """The direction that used to disagree with itself. `'' >= '2026-05-11'` is FALSE while
    `'' <= '2026-05-11'` is TRUE, so an undated row was dropped by a lower bound and KEPT by an
    upper one — an inconsistency nobody chose and nothing reported."""
    got = _ids(date_to="2026-12-31")
    assert BLANK not in got and NULL not in got


def test_undated_atoms_come_back_when_the_date_filter_is_dropped(corpus):
    """They are excluded BY THE FILTER, not missing from the store — the fact that makes the
    notice below actionable."""
    assert {BLANK, NULL} <= _ids()


def test_an_excluded_undated_atom_produces_a_notice_naming_the_count(corpus):
    """The disclaimer IS the deliverable. A `year` atom has partial evidence and earns the
    benefit of the doubt; an undated atom has none, so it is dropped — and dropping evidence the
    filter could not even EVALUATE has to be said, not inferred from a smaller result."""
    out = kb_entry.run_kb_search("agent", mode="bm25", k=20, date_from="2024-01-01")
    note = next(n for n in out["notices"] if n["code"] == "undated_excluded")

    assert note["count"] == 2                                # BLANK and NULL
    assert "2 atoms" in note["message"]
    assert "no recorded date" in note["message"]
    assert "excluded" in note["message"]


def test_no_undated_notice_when_no_date_filter_was_asked_for(corpus):
    """`notices` being empty on a healthy query is a contract. An undated atom is only news
    when something tried to filter on dates."""
    assert kb_entry.run_kb_search("agent", mode="bm25", k=20)["notices"] == []


def test_the_notice_still_fires_when_the_date_filter_empties_everything(corpus):
    """The case where the count matters MOST: zero hits, and the reason is partly that two atoms
    had no date to compare. Computed before the empty-candidate early return, deliberately."""
    out = kb_entry.run_kb_search("agent", mode="bm25", k=20, date_from="2030-01-01")
    assert out["hits"] == []
    assert next(n for n in out["notices"] if n["code"] == "undated_excluded")["count"] == 2


# ── partial dates widen to their natural edge ────────────────────────────────────

def test_a_bare_year_lower_bound_means_january_first(corpus):
    assert _ids(date_from="2026") == _ids(date_from="2026-01-01") == {MID, NEW, YEAR}


def test_a_bare_year_upper_bound_means_december_thirty_first(corpus):
    """The asymmetry that makes widening correct rather than convenient. Floored, `date_to=
    "2026"` would quietly mean "before January 2nd" and lose almost the whole year."""
    assert _ids(date_to="2026") == {OLD, MID, NEW, YEAR}


def test_a_year_month_bound_widens_to_the_month_edges(corpus):
    assert _date_bound("2026-02", end=False) == "2026-02-01"
    assert _date_bound("2026-02", end=True) == "2026-02-28"      # real month length, not 31
    assert _date_bound("2024-02", end=True) == "2024-02-29"      # leap year


# ── a malformed bound is an ERROR, never a dropped filter ────────────────────────

def test_the_shape_that_motivated_this_feature_raises(corpus):
    """`"5/11/2026"` is what a host actually sends. Ignoring it returns unfiltered results that
    LOOK filtered — the exact silent-wrong this feature exists to remove. An error is
    correctable; a filter that never ran is invisible."""
    with pytest.raises(ValueError, match="not a date"):
        kb_entry.run_kb_search("agent", mode="bm25", date_from="5/11/2026")


@pytest.mark.parametrize("bad", ["yesterday", "2026-13-01", "2026-02-30", "26-05-11", "20260511"])
def test_other_unparseable_bounds_raise_too(bad):
    with pytest.raises(ValueError):
        _date_bound(bad, end=False)


def test_an_absent_bound_is_not_a_filter(corpus):
    """None and "" both mean "no bound" — a host filling an unused optional slot with an empty
    string must not get an empty-set filter, which is how `who_id` distinguishes them too."""
    assert _date_bound(None, end=False) is None and _date_bound("", end=True) is None
    assert _ids(date_from="", date_to=None) == _ids()


# ── composition with the other filters ───────────────────────────────────────────

def test_a_date_bound_narrows_the_other_filters_rather_than_replacing_them(corpus):
    """Every clause is ANDed. A date filter that dropped the tag/author restriction would return
    a stranger's atoms under a query the caller believes is scoped to one person."""
    assert _ids(who_id="x:user:1", date_from="2026-01-01") == {MID, NEW}   # YEAR is scholar:1
    assert _ids(source_type="paper", date_from="2026-01-01") == {YEAR}
    assert _ids(tags=["ai-agents"], date_to="2024-12-31") == {OLD}
    assert _ids(tags=["nothing-tagged-this"], date_from="2020-01-01") == set()


def test_the_applied_bounds_are_reported_normalized(corpus):
    """`trace.filters` shows filters AS APPLIED. `date_to="2026"` was applied as 2026-12-31, and
    reporting what the caller typed would hide the widening the engine chose for them."""
    out = kb_entry.run_kb_search("agent", mode="bm25", k=20, date_from="2026", date_to="2026")
    assert out["trace"]["filters"]["date_from"] == "2026-01-01"
    assert out["trace"]["filters"]["date_to"] == "2026-12-31"


# ── the counterfactual that PRICES the filter ────────────────────────────────────

def test_filter_cost_prices_a_date_bound(corpus):
    """The assertion that catches the `filter_costs` trap. The date clause has to reach THREE
    places — `_filter_clauses`, `_count_matching`'s forwarding, and `filter_costs`' args dict.
    Wire only the first two and this reads `{}`: 3 atoms excluded, and the one field whose job
    is exposing silent narrowing says nothing."""
    out = kb_entry.run_kb_search("agent", mode="bm25", k=20, date_from="2026-05-11")
    # Dropping it would restore OLD (dated, outside the window) plus BLANK and NULL (undated).
    assert out["insights"]["filter_cost"]["date_from"] == 3


def test_the_priced_cost_and_the_undated_count_are_different_numbers(corpus):
    """They answer different questions and must not be derived from each other. `filter_cost` is
    "how many more would I get dropping this bound" — outside-the-window PLUS undated. The
    notice is only the undated subset: what the filter could not EVALUATE, not what it evaluated
    and rejected."""
    out = kb_entry.run_kb_search("agent", mode="bm25", k=20, date_from="2026-05-11")
    note = next(n for n in out["notices"] if n["code"] == "undated_excluded")
    assert note["count"] == 2 and out["insights"]["filter_cost"]["date_from"] == 3


def test_a_date_bound_that_costs_nothing_is_not_reported(corpus):
    """`filter_cost` is self-pruning by construction, and a date bound is not an exception."""
    out = kb_entry.run_kb_search("agent", mode="bm25", k=20, date_from="2020-01-01",
                                 source_type="paper")
    assert "date_from" not in out["insights"]["filter_cost"]


# ── aggregate takes the same bounds ──────────────────────────────────────────────

def test_aggregate_scopes_by_date_with_the_same_semantics(corpus):
    """One concept, one spelling, one meaning across both tools — including the two edges that
    make the meaning: the year-only atom counts, the undated ones do not."""
    scoped = kb_entry.kb_aggregate({"date_from": "2026-05-11"})
    assert scoped["total"] == 3                                  # MID, NEW, YEAR
    assert kb_entry.kb_aggregate()["total"] == 6                 # everything, undated included


# ── the structural guard ─────────────────────────────────────────────────────────

_KB = Path(__file__).resolve().parents[2] / "pipeline" / "kb"


def _assigned_when_precision_literals(path: Path) -> set[str]:
    """Every string literal the module can put in `when_precision`.

    TWO shapes, because the derivers use both: a dict entry (`"when_precision": "day"`) and a
    tuple assignment (`when_ts, when_precision = f"{year}-01-01", "year"`). Collecting only the
    first would miss `year`, which is the single most important value here."""
    out: set[str] = set()

    def literals(node) -> set[str]:
        """String constants in VALUE position under `node`. A key LOOKUP is not a value —
        `derive_blog` reads `article.get("when_precision")` to pass an ingester's precision
        through, and counting that string would have the guard checking its own key name."""
        looked_up = {id(a) for n in ast.walk(node) if isinstance(n, ast.Call)
                     for a in n.args}
        looked_up |= {id(n.slice) for n in ast.walk(node) if isinstance(n, ast.Subscript)}
        return {n.value for n in ast.walk(node)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and id(n) not in looked_up}

    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "when_precision":
                    out |= literals(value)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names = target.elts if isinstance(target, ast.Tuple) else [target]
                for i, name in enumerate(names):
                    if not (isinstance(name, ast.Name) and name.id.endswith("when_precision")):
                        continue
                    value = node.value
                    if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
                        value = value.elts[i]        # match by position, not the whole RHS
                    out |= literals(value)
    return out


def test_every_precision_the_derivers_write_has_a_declared_range():
    """The guard that stops a new `when_precision` inheriting a wrong default silently.

    `_PRECISION_SPAN` decides what a value means at a date boundary, and an unlisted one falls
    through to exact-day. The case that motivated this: `approx`, drafted in `ingest_blog` as a
    Wayback UPPER bound, for which exact-day is WRONG on the `date_to` side — and nothing at
    runtime would say so, the filter would just quietly return fewer atoms. (That rung was never
    wired and was deleted 2026-08-28, along with its `_PRECISION_SPAN` row; the guard stays,
    because the next precision someone adds will arrive the same way.) Same shape as the AST guard in
    `test_payload_key_names.py`, and for the same reason — there is no runtime moment where
    "a precision fell through" is observable.
    """
    written = _assigned_when_precision_literals(_KB / "derive.py")
    # `derive_blog` passes an ingester-supplied precision straight through, so the blog module's
    # `_WHEN_*` constants are writers too even though they never appear in `derive.py`.
    written |= {node.value.value
                for node in ast.walk(ast.parse((_KB / "ingest_blog.py").read_text()))
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and any(isinstance(t, ast.Name) and t.id.startswith("_WHEN_")
                        for t in node.targets)}

    assert "year" in written                              # the guard is reading the right thing
    missing = sorted(written - set(_PRECISION_SPAN))
    assert not missing, (
        f"when_precision values with no entry in `_PRECISION_SPAN`: {missing}. Decide the date "
        f"RANGE each one means — BOTH ends — and add it. An unlisted value silently filters as "
        f"an exact day, which is wrong for any precision that is a bound rather than a date.")


def test_the_range_map_widens_exactly_one_precision_today():
    """The positive half. The guard above is satisfied trivially by a map where every entry is
    exact-day, which is the state this feature exists to fix."""
    from pipeline.kb.retrieve import _WHEN
    assert [p for p, expr in _PRECISION_SPAN.items() if expr != _WHEN] == ["year"]
