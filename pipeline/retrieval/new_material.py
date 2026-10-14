"""The first time a kind of material lands in a store, say what is there — once, per kind.

OPYT speaks only when called, so the notice rides back on whatever the user does next, like
`told.completion_notice`. Growth within a kind stays silent; the one exception is a saved-posts
import still running when its kind first speaks, which speaks once more when it has landed. The
wording says what is there, never that it just arrived.
"""
from __future__ import annotations

import json
from collections.abc import Callable

from opyt_core.paths import opyt_path

# Stamped as it is handed over, like `told.completion_notice`, so each kind speaks once.
_STAMP = "material_announced.json"

# (source_type, entry_mode) → what the material is, in the user's terms. Keyed on both because
# saved posts and tracked people's archives are different material even on the same platform.
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
    """A phrase for one kind of material, with a generic fallback so a new `source_type` is still
    announced."""
    if (known := _LABELS.get((source_type, entry_mode))) is not None:
        return known
    if entry_mode == "oracle-footprint":
        return f"{source_type} material from the people you track"
    if entry_mode == "user-saved":
        return f"the {source_type} posts you saved yourself"
    return f"{source_type} material"


def _announced() -> set[str]:
    """Kinds already handed over. An unreadable stamp reads as empty: a repeat beats a loss."""
    try:
        return set(json.loads(opyt_path(_STAMP).read_text()))
    except Exception:
        return set()


def _mark(keys: set[str]) -> None:
    """Record kinds as announced. Fail-safe: a failed write costs a repeat, not the call."""
    try:
        path = opyt_path(_STAMP)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(_announced() | keys)))
    except Exception:
        pass


def _kinds(conn) -> list[tuple[str, str, int]]:
    """Every (source_type, entry_mode) that has atoms, with its count. `[]` on any failure
    (a store too young to have the table is normal on a first call)."""
    try:
        rows = conn.execute(
            "SELECT source_type, COALESCE(entry_mode, ''), COUNT(*) FROM atoms "
            "GROUP BY source_type, entry_mode").fetchall()
    except Exception:
        return []
    return [(str(s), str(m), int(n)) for s, m, n in rows if n]


def _stamp(source_type: str, entry_mode: str, importing: dict) -> str:
    """The stamp a kind is announced under. A saved-posts kind whose import is still running is
    announced once as partial, and once more, under its own key, when the import has landed."""
    partial = entry_mode == "user-saved" and source_type in importing
    return f"{source_type}:{entry_mode}" + (":partial" if partial else "")


def _entry(label: str, n: int, progress: tuple[int, int] | None) -> dict:
    entry = {"material": label, "items": n}
    if progress:
        entry["still_importing"] = {"saved": progress[0], "offered_so_far": progress[1]}
    return entry


def new_material_notice(conn, tour: Callable[[], dict] | None = None,
                        importing: dict[str, tuple[int, int]] | None = None) -> dict | None:
    """Material in this store the user has never been told about, or None.

    Returns the kinds, the store's shape from `tour` (called only when there is news; may be
    `{}`), and the host instruction. Stamps as it hands over. `importing` maps a platform to its
    running saved-posts import as (posts saved, posts offered so far).
    """
    importing = importing or {}
    kinds = _kinds(conn)
    if not kinds:
        return None
    already = _announced()
    fresh = [(_stamp(s, m, importing), _label(s, m), n, importing.get(s) if m == "user-saved"
              else None) for s, m, n in kinds]
    fresh = [f for f in fresh if f[0] not in already]
    if not fresh:
        return None

    notice: dict = {"in_your_library": [_entry(label, n, progress) for _k, label, n, progress
                                        in sorted(fresh, key=lambda f: -f[2])]}
    try:
        if tour is not None and (shape := tour()):
            notice["tour"] = shape
    except Exception:
        pass      # fail-safe: the tour is the most droppable half of the notice

    notice["host_note"] = (
        "⚠️ LEAD WITH THIS — there is real material in their knowledge base and nobody has told "
        "them what is in it. Say what is there before anything else, and before asking what they "
        "want to do next: the counts, who is most represented, what it covers, what surprised "
        "you. `aggregate()` and `search()` will give you the detail to say it with. Do NOT ask "
        "permission to describe it and do NOT offer a tour — they asked for a library and it is "
        "here, so talking about it IS the answer. Say what IS in the store, never that it just "
        "arrived: some of this may have landed days ago and only be reaching them now.")
    if any(progress for *_rest, progress in fresh):
        notice["host_note"] += (
            " ⚠️ STILL IMPORTING: material marked `still_importing` is only partly in. Say so "
            "plainly, with its numbers (saved so far, of how many found so far), say the rest is "
            "still arriving in the background, and describe it as a first look, never as the "
            "whole library. OPYT says so again when the import has finished.")
    _mark({k for k, *_rest in fresh})
    return notice
