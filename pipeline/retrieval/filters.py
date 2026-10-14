"""
pipeline/retrieval/filters.py — the pre-filter: which atoms a read may consider at all.

One spelling of every filter a read takes (`tags`, `what_kind`, `source_type`, `who_id`, the date
bounds and `entry_mode`), shared by the filter itself (`candidate_atom_ids`), the counterfactual
that prices each filter (`filter_costs`), the undated count behind the `undated_excluded` notice,
and the store's date span shown before sharing. Search, aggregate and feed all filter through here,
so the surfaces cannot drift.
"""

from __future__ import annotations

import calendar
import re
from datetime import date as _date

from pipeline.kb.derive import slugify


def slug_tags(tags):
    """A caller's `tags` → the slug list the pre-filter takes. No tags asked for → None.
    Tags that all slugify away return `[]` ("match nothing"), never None ("no filter")."""
    if not tags:
        return None
    return [s for s in (slugify(t) for t in tags) if s]


def in_clause(candidates: set[str] | None, alias: str = "a") -> tuple[str, list]:
    """A candidate-restriction fragment. None → no restriction (empty fragment)."""
    if candidates is None:
        return "", []
    ids = list(candidates)
    ph = ",".join("?" for _ in ids)
    return f" AND {alias}.atom_id IN ({ph})", ids


# ── date bounds ───────────────────────────────────────────────────────────────
# Precision comes from the `when_precision` column, not from the date string's shape.
# `when_ts` is always `YYYY-MM-DD` shaped or empty.
_WHEN = "a.when_ts"
_PREC = "a.when_precision"
# An atom with a date. Both spellings of "undated" are in the corpus, NULL and ''.
_DATED = f"({_WHEN} IS NOT NULL AND {_WHEN} != '')"

# How far an atom's true date range extends past `when_ts`, keyed by `when_precision`. A `year`
# atom stores Jan 1 as a floor, so its range runs to Dec 31; everything else is the single day it
# names. `tests/kb/test_date_filter.py` asserts every deriver-written precision has a key here.
_PRECISION_SPAN: dict[str, str] = {
    "day": _WHEN,       # a real publication date
    "push": _WHEN,      # GitHub's last-push day — activity, not publication; filtered exactly
    "year": f"substr({_WHEN},1,4) || '-12-31'",   # Jan 1 is a floor — the atom spans its year
    "unknown": _WHEN,   # always paired with an empty `when_ts` — the undated clause drops it first
    "": _WHEN,
}

_DATE_INPUT = re.compile(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?$")


def _range_end_sql() -> str:
    """The atom's range END, as a SQL expression over the `atoms a` row.

    Generated from `_PRECISION_SPAN`; with no widening precision the CASE disappears and the
    clause stays sargable."""
    wide = [(p, expr) for p, expr in _PRECISION_SPAN.items() if expr != _WHEN]
    if not wide:
        return _WHEN
    arms = " ".join(f"WHEN '{p}' THEN {expr}" for p, expr in wide)
    return f"(CASE {_PREC} {arms} ELSE {_WHEN} END)"


def date_bound(value: str | None, *, end: bool) -> str | None:
    """A caller's date → `YYYY-MM-DD`. None/empty → None, meaning no bound was asked for.

    Accepts `YYYY`, `YYYY-MM`, `YYYY-MM-DD` and widens a partial to its natural edge:
    `date_from="2026"` is `2026-01-01`, `date_to="2026"` is `2026-12-31`.

    Raises `ValueError` on anything else (e.g. `"5/11/2026"`), so a malformed filter is never
    silently ignored."""
    raw = (value or "").strip()
    if not raw:
        return None
    m = _DATE_INPUT.match(raw)
    if m:
        try:
            year = int(m.group(1))
            month = int(m.group(2)) if m.group(2) else (12 if end else 1)
            if m.group(3):
                day = int(m.group(3))
            else:
                day = calendar.monthrange(year, month)[1] if end else 1
            return _date(year, month, day).isoformat()
        except (ValueError, calendar.IllegalMonthError):
            pass          # well-shaped but impossible ("2026-13", "2026-02-30")
    raise ValueError(
        f"date bound {value!r} is not a date. Use YYYY, YYYY-MM or YYYY-MM-DD; a partial widens "
        f"to its natural edge (date_from='2026' is 2026-01-01, date_to='2026' is 2026-12-31)."
    )


def candidate_atom_ids(conn, tags: list[str] | None, what_kind: str | None,
                       source_type: str | None, who_id: str | list[str] | None = None,
                       date_from: str | None = None, date_to: str | None = None,
                       *, entry_mode: str | list[str] | None = None,
                       ) -> set[str] | None:
    """The pre-filter (step 1). Returns the atom_id set matching all given constraints, or None
    when no filter is requested (search the whole store).

    `who_id` is exact id matching, one id or a list, never a name (handle → ids is
    `who.resolve_who`). `tags` overlap `payload.source_tags`; `what_kind`/`source_type` are exact.
    `date_from`/`date_to` are normalized `YYYY-MM-DD` bounds (see `date_bound`), inclusive.
    `entry_mode` is an allow-list of how the atom arrived; the human-attested pile is
    `list(schema.HUMAN_ATTESTED)`.
    """
    built = _filter_clauses(tags, what_kind, source_type, who_id, date_from, date_to, entry_mode)
    if built is None:
        return set()                    # an explicitly empty author/tag set → match nothing
    clauses, params = built
    if not clauses:
        return None                     # nothing was asked for → search the whole store
    sql = "SELECT a.atom_id FROM atoms a WHERE " + " AND ".join(clauses)
    return {r["atom_id"] for r in conn.execute(sql, params)}


def entry_mode_clause(modes: list[str]) -> str:
    """`a.entry_mode IN (?, …)` for these modes — the allow-list's one SQL spelling, shared by the
    pre-filter and the feed. The caller passes `modes` as the params."""
    return f"a.entry_mode IN ({','.join('?' * len(modes))})"


def _filter_clauses(tags: list[str] | None, what_kind: str | None, source_type: str | None,
                    who_id: str | list[str] | None, date_from: str | None = None,
                    date_to: str | None = None,
                    entry_mode: str | list[str] | None = None,
                    ) -> tuple[list[str], list] | None:
    """The pre-filter's WHERE clauses, shared by the filter and the counterfactual that prices it
    (`filter_costs`), so a reported cost always describes the query that ran.

    Returns `(clauses, params)`; an empty clause list means no filter was asked for. Returns None
    for match nothing (an explicitly empty author, tag or entry-mode set)."""
    clauses: list[str] = []
    params: list = []
    if what_kind:
        clauses.append("a.what_kind = ?")
        params.append(what_kind)
    if source_type:
        clauses.append("a.source_type = ?")
        params.append(source_type)
    if entry_mode is not None:
        # Allow-list: a mode added later is excluded by default. Empty list matches nothing.
        modes = [entry_mode] if isinstance(entry_mode, str) else list(entry_mode)
        modes = [m for m in modes if m]
        if not modes and not isinstance(entry_mode, str):
            return None
        if modes:
            clauses.append(entry_mode_clause(modes))
            params.extend(modes)
    if who_id is not None:
        # An empty list matches nothing (a failed `resolve_who` must not widen to the whole
        # store); a bare "" means no filter.
        ids = [who_id] if isinstance(who_id, str) else list(who_id)
        ids = [i for i in ids if i]
        if not ids and not isinstance(who_id, str):
            return None
        if ids:
            wph = ",".join("?" for _ in ids)
            clauses.append(f"a.who_id IN ({wph})")   # served by idx_atoms_who_id
            params.extend(ids)
    if tags is not None:
        # Empty list matches nothing; None is the only "no filter" input.
        if not tags:
            return None
        ph = ",".join("?" for _ in tags)
        clauses.append(
            f"a.atom_id IN ("
            f"SELECT a3.atom_id FROM atoms a3, "
            f"json_each(json_extract(a3.payload, '$.source_tags')) js "
            f"WHERE js.value IN ({ph}))"
        )
        params.extend(tags)
    if date_from or date_to:
        # An undated atom satisfies no date claim in either direction (explicit because SQL's
        # own `''` comparisons disagree with each other across `>=` and `<=`).
        clauses.append(_DATED)
    if date_from:
        # The atom's range end must reach the window, so a `year` atom matches a later
        # `date_from` in the same year.
        clauses.append(f"{_range_end_sql()} >= ?")
        params.append(date_from)
    if date_to:
        # The atom's range start must not pass the window. Sargable, unlike the date_from side.
        clauses.append(f"{_WHEN} <= ?")
        params.append(date_to)
    return clauses, params


def _count_matching(conn, tags, what_kind, source_type, who_id,
                    date_from=None, date_to=None, entry_mode=None) -> int:
    """How many atoms clear this exact filter combination. Match-nothing counts as 0."""
    built = _filter_clauses(tags, what_kind, source_type, who_id, date_from, date_to, entry_mode)
    if built is None:
        return 0
    clauses, params = built
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return conn.execute(f"SELECT COUNT(*) FROM atoms a{where}", params).fetchone()[0]


def count_undated(conn, tags, what_kind, source_type, who_id, entry_mode=None) -> int:
    """How many atoms clear every non-date filter but carry no date: the number behind the
    `undated_excluded` notice (atoms the date filter could not evaluate, not ones it rejected)."""
    built = _filter_clauses(tags, what_kind, source_type, who_id, entry_mode=entry_mode)
    if built is None:
        return 0
    clauses, params = built
    clauses = [*clauses, f"({_WHEN} IS NULL OR {_WHEN} = '')"]
    return conn.execute("SELECT COUNT(*) FROM atoms a WHERE " + " AND ".join(clauses),
                        params).fetchone()[0]


def date_span(conn) -> tuple[str | None, str | None]:
    """`(oldest, newest)` publication date over every dated atom, or `(None, None)` for a store
    with none. The span a person is shown before sharing their knowledge base."""
    row = conn.execute(f"SELECT MIN({_WHEN}), MAX({_WHEN}) FROM atoms a WHERE {_DATED}").fetchone()
    return (row[0], row[1]) if row else (None, None)


def filter_costs(conn, tags: list[str] | None = None, what_kind: str | None = None,
                 source_type: str | None = None, who_id: str | list[str] | None = None,
                 date_from: str | None = None, date_to: str | None = None,
                 entry_mode: str | list[str] | None = None) -> dict[str, int]:
    """For each active filter, how many more atoms the query would have reached with that one
    filter dropped. `{}` when no filter costs anything. Every filter `_filter_clauses` knows must
    appear in `args` below, or every reported cost comes out wrong.
"""
    args = {"tags": tags, "what_kind": what_kind, "source_type": source_type, "who_id": who_id,
            "date_from": date_from, "date_to": date_to, "entry_mode": entry_mode}
    # `""` is an unused slot, not a filter; `[]` is one (pricing it shows `who=` resolved to nobody).
    active = [name for name, v in args.items() if v is not None and v != ""]
    if not active:
        return {}
    baseline = _count_matching(conn, **args)
    out: dict[str, int] = {}
    for name in active:
        gain = _count_matching(conn, **{**args, name: None}) - baseline
        if gain > 0:
            out[name] = gain
    return out


def applied_filters(slugs, what_kind, source_type, who_id, date_from, date_to,
                    entry_mode) -> dict:
    """The filters as applied (slugified tags, expanded who_ids, normalized dates), for a search's
    trace. Only an explicit `entry_mode` is listed; the default scope is not a filter."""
    applied = {}
    if slugs is not None:
        applied["tags"] = slugs
    if what_kind:
        applied["what_kind"] = what_kind
    if source_type:
        applied["source_type"] = source_type
    if who_id is not None and who_id != "":
        applied["who_id"] = who_id
    if date_from:
        applied["date_from"] = date_from
    if date_to:
        applied["date_to"] = date_to
    if entry_mode is not None:
        applied["entry_mode"] = entry_mode
    return applied
