"""
pipeline/arrivals/provenance.py — why an atom is in the user's store, as a short clause, from how
it entered (`entry_mode`). A decision: the flow gathers the names and the watches it needs.

  | entry_mode          | why                                                        |
  |---------------------|------------------------------------------------------------|
  | `user-saved`        | you saved it                                               |
  | `oracle-footprint`  | your Oracle <who_id's name> posted it                      |
  | `author_referenced` | <each referrer in `payload.referenced_by`> linked it       |

A watch's find is never listed here: the host showed it when it checked it.
"""
from __future__ import annotations

# How many names a clause spells out before it counts the rest.
_NAMED = 3


def why(atom: dict, names: dict[str, str]) -> str:
    """`atom` holds `entry_mode`, `who_id` and its parsed `payload`; `names` maps an entity id to
    its display name."""
    match atom["entry_mode"]:
        case "user-saved":
            return "you saved it"
        case "oracle-footprint":
            return f"your Oracle {_name(names, atom['who_id'])} posted it"
        case "author_referenced":
            people = list(dict.fromkeys(_name(names, r["by"], r.get("handle"))
                                        for r in atom["payload"].get("referenced_by") or []))
            # An archive walk stores a repo an Oracle's account points at without a referrer.
            return f"{_join(people)} linked it" if people else "one of your Oracles linked it"
    return f"it arrived as {atom['entry_mode']}"


def _name(names: dict[str, str], entity_id: str | None, handle: str | None = None) -> str:
    return names.get(entity_id) or (f"@{handle}" if handle else entity_id)


def _join(items: list[str]) -> str:
    """"a", "a and b", "a, b and c", "a, b, c and 1 other"."""
    if (rest := len(items) - _NAMED) > 0:
        items = [*items[:_NAMED], f"{rest} other{'s' if rest > 1 else ''}"]
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"
