"""Amendment 3's real enforcement (Job 7, D14) — a sitting is never time-bounded.

    Time may SELECT a sitting. Time may ORDER a sitting. Time must never BOUND a sitting's
    membership — and material must never be dropped by AGE to fit a budget.
    (docs/plans/2026-08-13-sitting-seed-channels-and-the-re-read-trigger.md, §11, ratified by
    David 2026-08-14.)

⚠️ NARROWED 2026-08-24 (docs/plans/2026-08-24-era-reads-claims-carry.md), and the three checks
below are UNCHANGED by it — they passed on the new admission loop before this paragraph was
written, which is the evidence that the narrowing is real rather than an exemption. What moved:

    POOL CONSTRUCTION stays time-blind. Which atoms are in a region is cosine against the seed at
    the floor, and no clock touches it. That half is the whole invariant and it is untouched.

    THE BUDGET CUT MAY NOW SLICE BY TIME. A region bigger than one read is cut into contiguous
    chronological PARTS, oldest first, and every part is read — nothing is discarded. The old
    partitioner was MMR rank, which is anti-clustering by construction: it spreads similar items,
    so as a partitioner it systematically severed a claim from its rebuttal (the rebuttal scores
    redundant against its target and lands in the next part).

    DROP-BY-AGE REMAINS FORBIDDEN, and that is the line. The deleted `bookmark_reader` kept "the
    NEWEST saves" and threw the rest away — material gone, never read, nobody told. A part cut
    defers; a recency window deletes. Check 3 still catches the deleting shape, because the
    forbidden thing is a date sort SLICED to a budget with the tail discarded, and the admission
    loop keeps every atom it does not admit — they are the next part's pool.

The `.guards.py` rule this test is cross-referenced from (`sitting-membership-never-time-bounded`)
is line-based text and cannot tell a date predicate in a WHERE clause from the file's own LEGAL uses
of the same column names — `_relevance`'s membership query legally has no date filter at all, the
region-chain query legally SELECTs `when_ts`, and the fracture-listing query legally orders by
`built_at`. This test is the STRONGER check the rule's message points to, matching the precedent
`tests/test_local_auth.py::test_never_binds_a_routable_address` set on main 2026-08-15: it parses the
real Python AST rather than grepping text, so a comparison split across lines or a renamed local
cannot slip past it, and it can tell a SELECTed or ORDERed column apart from an actual bound.

⚠️ SCOPED TO `pipeline/kb/sitting_builder.py` AND `pipeline/kb/sitting_render.py` ONLY — the two files
Amendment 3 governs (it is a MEMBERSHIP invariant about which atoms join a region, plus the render
layer's chronological ORDERING of them, which is explicitly legal and is exactly the shape check 3
exists to keep legal — see D14's "order, never bound"). `sitting_scheduler.py`'s `built_at`-ordered
continuation queue decides which SITTING to act on next, a different question the amendment does not
reach, so it stays out of scope. Split out of `sitting_builder.py` 2026-08-16 (docs/plans/2026-08-16-
refactoring-and-composability-audit.md, step B2): the membership code (`_relevance`, `build_sitting`)
stayed in `sitting_builder.py`; the chronological-order code (`_chronological`, `chronological_order`,
`_atom_bodies`) moved to `sitting_render.py`, and this guard moved with it — `sitting_store.py` and
`sitting_zoom.py` touch no atom-level date field at all, so neither is in `TARGETS` below.

⚠️ `TIME_FIELDS` DELIBERATELY EXCLUDES `built_at`. That column is the SITTING record's own build
timestamp, not a candidate ATOM's date — ordering by it (the fracture-listing query, `:1446`) is not
a membership decision and Amendment 3 has nothing to say about it. Folding it in would make this
test fail on the file's OWN clean code.

⚠️ THREE CHECKS, and each one's blind spot is named rather than papered over:
  1. Python-level date comparisons (`if row["when_ts"] < cutoff`) — a real `ast.Compare` node.
  2. SQL-embedded date predicates (`WHERE ... when_ts < ?`) — a string literal, invisible to check 1.
  3. sort-then-slice by a time field (`sorted(pool, key=lambda a: a["when_ts"])[:budget]`) — the
     BUDGET-TRIMMING shape D14's table calls out by name, using `bookmark_reader.MAX_INPUT_CHARS`'s
     deleted "keeps the newest" rule as its historical example.
None of the three catches the others' failure mode, which is why there are three rather than one.
Check 3 only catches the DIRECT chained shape (`sorted(...)[...]`) — a sort assigned to a variable
and sliced later, elsewhere in the function, needs a human reading the diff. That gap is real and
recorded rather than claimed away: a fully general data-flow check was rejected as more likely to
produce a false positive against the file's own legitimate chronological sort (`:1584`, no slice, kept
in full) than to catch a regression a reviewer would not also catch.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGETS = (REPO / "pipeline" / "kb" / "sitting_builder.py",
          REPO / "pipeline" / "kb" / "sitting_render.py")

# An atom's own date. NOT `built_at` — see the module docstring for why.
TIME_FIELDS = {"when_ts", "first_seen", "ingested_at"}
ORDERING_OPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def _trees() -> list[tuple[Path, ast.AST]]:
    return [(t, ast.parse(t.read_text(encoding="utf-8"), filename=str(t))) for t in TARGETS]


def _names_in(node: ast.AST) -> set[str]:
    """Every identifier an operand could plausibly be keyed on: a bare Name, an Attribute's
    `.attr`, or a string Constant (`row["when_ts"]`'s subscript key)."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.add(n.value)
    return out


def _literal_text(node: ast.AST) -> str:
    """The literal text of a string-building expression — a plain Constant, an f-string's literal
    segments (the interpolated parts of every f-string in this file are helper calls like
    `_human_clause()`, never a raw predicate, so skipping them cannot hide one), or `+` concatenation."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(_literal_text(v) for v in node.values if isinstance(v, ast.Constant))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_text(node.left) + _literal_text(node.right)
    return ""


def _sql_argument(node: ast.AST) -> str | None:
    """The literal SQL text of a `<conn/cur>.execute(...)` call's first argument, or None."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "execute" and node.args):
        return None
    return _literal_text(node.args[0])


def test_no_python_level_date_comparison():
    """No `ast.Compare` anywhere in either module orders one of the atom-level time fields against
    anything with <, >, <=, or >=. A SORT KEY (`sorted(..., key=lambda a: a["when_ts"])`, in
    `sitting_render._chronological`) uses no explicit operator and is untouched by this — ordering
    is allowed, bounding is not."""
    violations = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            fields = set()
            for operand in (node.left, *node.comparators):
                fields |= _names_in(operand)
            if fields & TIME_FIELDS and any(isinstance(op, ORDERING_OPS) for op in node.ops):
                violations.append((path.name, node.lineno))
    assert not violations, (
        f"a date-field ordering comparison exists at {violations} — Amendment 3 forbids time "
        f"BOUNDING a sitting's membership; SELECT the column or sort by it, never compare it with "
        f"<, >, <=, >=")


def test_no_sql_predicate_bounds_a_time_field():
    """Every `conn.execute(...)` call's SQL argument, scanned for a WHERE/AND clause comparing one
    of the atom-level time fields with an ordering operator. A bare SELECT of the column (in
    `sitting_render._atom_bodies`), or an ORDER BY on `built_at` (in `sitting_zoom.zoomed_from` —
    out of `TARGETS`, but also not even a member of `TIME_FIELDS`), has no adjacent operator and
    does not trip this."""
    violations = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            sql = _sql_argument(node)
            if not sql:
                continue
            for field in TIME_FIELDS:
                for op in ("<", ">", "<=", ">="):
                    if f"{field} {op}" in sql or f"{op} {field}" in sql:
                        violations.append((path.name, node.lineno, f"{field} {op}"))
    assert not violations, (
        f"a SQL date predicate exists at {violations} — Amendment 3 forbids a WHERE/AND clause "
        f"bounding sitting membership by an atom's date")


def test_no_sort_by_date_is_sliced_to_a_budget():
    """The BUDGET-TRIMMING shape: `sorted(pool, key=<touches a time field>)[:n]`. The deleted
    `bookmark_reader.MAX_INPUT_CHARS` did exactly this — 'trimming keeps the newest saves' — and
    D14 names it as the loophole that makes the membership rule enforceable rather than decorative:
    an implementation could satisfy 'no date filter' by admitting everything and then trimming the
    old ones to fit, which is the same violation wearing a different hat."""
    violations = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice)):
                continue
            call = node.value
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == "sorted"):
                continue
            key_kw = next((kw for kw in call.keywords if kw.arg == "key"), None)
            touched = _names_in(key_kw.value) if key_kw else set()
            if touched & TIME_FIELDS:
                violations.append((path.name, node.lineno))
    assert not violations, (
        f"a sort keyed on a time field is sliced to a budget at {violations} — Amendment 3 forbids "
        f"dropping material by AGE to fit a budget; cut by MMR redundancy, the same way "
        f"`build_sitting`'s own budget-stop loop already does")
