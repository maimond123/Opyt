"""The first time a KIND of material lands in a store, say what is there — once, per kind.

⚠️ THE PRODUCT WENT QUIET AT THE EXACT MOMENT IT HAD SOMETHING TO SAY. Measured on the first
real hosted onboarding, 2026-09-15: setup finished, 1,004 atoms were in the store, and the last
thing OPYT said was "Anything else you want to add right now, or good to leave it here?" The user
had to ask "do I have anything in my KB as of now?" to find out — and the answer was excellent
(1,004 items, 707 people, February 2021 to September 2026, who is most represented, what the
recent material skews toward). None of it was volunteered.

The tour that would have said it already existed (`oracle_tools._tour` → `opyt_core.suggest`), and
it was not broken. It was attached in ONE place — the ingest presentation, gated on `atoms_added`
in that same call's results — and two things had moved out from under that gate:

- since the 60-second wall, an ingest RETURNS BEFORE THE ATOMS LAND, so at the moment the tour is
  computed `atoms_added` is zero or single digits; and
- 1,001 of those 1,004 atoms were BOOKMARKS, which arrive on the saved-content thread and never
  pass through an ingest presentation at all.

So the fix cannot live in the ingest result. It is the same shape as `pull_runs.completion_notice`
— OPYT speaks only when called, so a thing that happened while nobody was looking has to ride back
on whatever the user does next — and this module is deliberately its neighbour.

WHY "PER KIND" AND NOT "PER CALL". A notice that rides every call becomes furniture and the reader
learns to skip the one field that says something happened; that is `completion_notice`'s own
argument, and `screen`'s for omitting `omitted: 0`. Per kind is bounded — in practice three to five
notices over a store's whole life — and each one is genuinely new material rather than the same
material counted again. Growth WITHIN a kind is deliberately silent: a background pull streaming
atoms in would otherwise fire on every call for as long as it ran.

IT DOES NOT CLAIM THE MATERIAL JUST ARRIVED, and the wording is careful about it. The first call
after this ships will fire on a store that has been full for days, which is correct — the user was
never told — but "your saved posts just landed" would be a lie to that reader. It says what IS
there.
"""
from __future__ import annotations

import json

from opyt_core.paths import opyt_path

# Stamped as it is handed over, exactly like `pull_runs.completion_notice`. Un-stamped this would
# ride on every call forever, which is the failure mode described above.
_STAMP = "material_announced.json"

# (source_type, entry_mode) → what the material IS, in the user's own terms. Keyed on BOTH because
# `entry_mode` is how it entered and `source_type` is what it is, and the user experiences those
# as different material: posts they saved themselves are not the same event as the archives of
# people they chose to track, even when both are X.
_LABELS = {
    ("x", "user-saved"): "the X posts you saved yourself",
    ("substack", "user-saved"): "the Substack posts you saved yourself",
    ("x", "oracle-footprint"): "posts from the people you track on X",
    ("substack", "oracle-footprint"): "the Substack archives of the people you track",
    ("blog", "oracle-footprint"): "the blog archives of the people you track",
    ("scholar", "oracle-footprint"): "the published research of the people you track",
    ("github", "oracle-footprint"): "the code of the people you track",
}


def _label(source_type: str, entry_mode: str) -> str:
    """A phrase for one kind of material. Falls back rather than dropping the kind: an unlabelled
    source is still material the user has and still deserves to be mentioned, and a silent drop
    here is how a new `source_type` would go unannounced forever without anything saying so."""
    if (known := _LABELS.get((source_type, entry_mode))) is not None:
        return known
    if entry_mode == "oracle-footprint":
        return f"{source_type} material from the people you track"
    if entry_mode == "user-saved":
        return f"the {source_type} posts you saved yourself"
    return f"{source_type} material"


def _announced() -> set[str]:
    """Kinds already handed over. An unreadable stamp reads as EMPTY, which risks repeating a
    notice once — the opposite direction from silently swallowing one, and the right way round:
    the cost of this file being wrong is a repeated sentence, not a user who never learns."""
    try:
        return set(json.loads(opyt_path(_STAMP).read_text()))
    except Exception:
        return set()


def _mark(keys: set[str]) -> None:
    """Fail-safe: a stamp that cannot be written must not break the call that earned it. The
    notice has already been handed over by then, so the cost is a repeat, not a loss."""
    try:
        path = opyt_path(_STAMP)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(_announced() | keys)))
    except Exception:
        pass


def _kinds(conn) -> list[tuple[str, str, int]]:
    """Every (source_type, entry_mode) that has atoms, with its count. `[]` on any failure —
    a store too young to have the table is the normal case on a first call, not an error."""
    try:
        rows = conn.execute(
            "SELECT source_type, COALESCE(entry_mode, ''), COUNT(*) FROM atoms "
            "GROUP BY source_type, entry_mode").fetchall()
    except Exception:
        return []
    return [(str(s), str(m), int(n)) for s, m, n in rows if n]


def new_material_notice(conn) -> dict | None:
    """Material in this store the user has never been told about, or None.

    Returns the kinds, the measured shape of the store (`opyt_core.suggest.suggestions`, the same
    tour the ingest presentation carries), and the instruction that makes the host USE it rather
    than file it. Stamps as it hands over.

    `tour` may be absent: `suggestions` returns `{}` for a store too thin to say anything useful
    about, and a notice about what landed is still worth carrying without it.
    """
    kinds = _kinds(conn)
    if not kinds:
        return None
    already = _announced()
    fresh = [(f"{s}:{m}", _label(s, m), n) for s, m, n in kinds if f"{s}:{m}" not in already]
    if not fresh:
        return None

    notice: dict = {"in_your_library": [{"material": label, "items": n}
                                        for _k, label, n in sorted(fresh, key=lambda f: -f[2])]}
    try:
        from opyt_core.kb import kb_aggregate
        from opyt_core.suggest import suggestions
        if tour := suggestions(conn, kb_aggregate()):
            notice["tour"] = tour
    except Exception:
        pass      # Fail-safe: the tour is the most droppable half — same rule as `_tour`.

    notice["host_note"] = (
        "⚠️ LEAD WITH THIS — there is real material in their knowledge base and nobody has told "
        "them what is in it. Say what is there before anything else, and before asking what they "
        "want to do next: the counts, who is most represented, what it covers, what surprised "
        "you. `aggregate()` and `search()` will give you the detail to say it with. Do NOT ask "
        "permission to describe it and do NOT offer a tour — they asked for a library and it is "
        "here, so talking about it IS the answer. Say what IS in the store, never that it just "
        "arrived: some of this may have landed days ago and only be reaching them now.")
    _mark({k for k, _l, _n in fresh})
    return notice
