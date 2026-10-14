"""One collection response and the evidence needed to continue or finish its walk."""

from dataclasses import dataclass


class Incomplete(RuntimeError):
    """The response cannot establish a complete collection; keep the walk owed."""


@dataclass(frozen=True)
class Page:
    """Accepted records and unfiltered observations are deliberately separate.

    A deleted user or an unreadable tweet still occupies a place in a timeline. Filtering it
    cannot prove the collection ended. Only endpoints with measured repeating-item termination
    set `repeat_is_end`; a cursor cycle by itself never establishes completion.
    """
    records: tuple[tuple[str, dict], ...]
    cursor: str | None
    observed: tuple[str, ...]
    ended: bool = False
    repeat_is_end: bool = False


def continuation(page: Page, seen: set[str], cursors: set[str]) -> str | None:
    """A repeated X item page can end infinite scroll; a cycling cursor cannot prove an end."""
    if page.ended:
        return None
    if page.cursor in cursors:
        raise Incomplete("collection cursor repeated before an established end")
    if page.cursor is None:
        return None
    if page.observed and set(page.observed) <= seen:
        if page.repeat_is_end:
            return None
        raise Incomplete("collection returned another cursor but no new items")
    return page.cursor


def error_inside_entry(error, timeline_path: tuple[str, ...]) -> bool:
    """Whether one GraphQL error sits inside a single timeline entry, so it cannot hide an entry.

    `timeline_path` is the envelope as `timeline()` takes it, leading "data" included; error
    paths omit that "data". The check exists for completeness, so only a path that continues
    past `<timeline>, "instructions", <i>, "entries", <j>` into that entry is tolerated:

    | error path                                               | inside an entry |
    |----------------------------------------------------------|-----------------|
    | <timeline>.instructions.<i>.entries.<j>.content...       | yes             |
    | <timeline>.instructions.<i>.entries.<j> (the entry)      | no              |
    | <timeline>.instructions... / entries list / above        | no              |
    | elsewhere in data, missing, or not a list                | no              |
    """
    path = error.get("path") if isinstance(error, dict) else None
    prefix = [*timeline_path[1:], "instructions"]
    if not isinstance(path, list) or len(path) <= len(prefix) + 3:
        return False
    i, entries, j = path[len(prefix):len(prefix) + 3]
    return (path[:len(prefix)] == prefix and type(i) is int
            and entries == "entries" and type(j) is int)


def timeline(data: dict, *path: str) -> dict:
    """Validate the collection envelope before an absent field can look like an empty list.

    `path` starts at the response root ("data", ...). A GraphQL error refuses the page unless
    it sits inside one entry (`error_inside_entry`), e.g. a list's undecodable banner."""
    value = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise Incomplete(f"collection response lacks {'.'.join(path)}")
        value = value[key]
    if not isinstance(value, dict) or not isinstance(value.get("instructions"), list):
        raise Incomplete("collection response lacks timeline instructions")
    if not all(error_inside_entry(error, path) for error in data.get("errors") or ()):
        raise Incomplete("collection response has unresolved GraphQL errors")
    return value


def entries(value: dict) -> list[dict]:
    """Flatten validated timeline instructions without filtering away pagination evidence."""
    out = []
    for instruction in value["instructions"]:
        if not isinstance(instruction, dict):
            raise Incomplete("invalid timeline instruction")
        if "entries" in instruction:
            rows = instruction["entries"]
        elif "entry" in instruction:
            rows = [instruction["entry"]]
        elif instruction.get("type") in ("TimelineTerminateTimeline", "TimelineClearCache"):
            continue  # carry no entries
        else:
            raise Incomplete("unrecognized timeline instruction")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise Incomplete("invalid timeline entries")
        if any(not isinstance(row.get("content"), dict) for row in rows):
            raise Incomplete("invalid timeline entry content")
        out.extend(rows)
    return out


def from_timeline(value: dict, records: tuple[tuple[str, dict], ...], *,
                  repeat_prefix: str = "", observed: tuple[str, ...] | None = None,
                  ended: bool = False) -> Page:
    """Keep pagination evidence before normalization drops unavailable or non-target rows."""
    rows = entries(value)
    cursors = [row["content"].get("value") for row in rows
               if row["content"].get("cursorType") == "Bottom"]
    if len(cursors) > 1 or any(not isinstance(c, str) or not c for c in cursors):
        raise Incomplete("invalid bottom cursor")
    items = [row for row in rows if not row["content"].get("cursorType")]
    if any(not isinstance(row.get("entryId"), str) or not row["entryId"] for row in items):
        raise Incomplete("timeline item lacks an identity")
    if observed is None:
        observed = tuple(row["entryId"] for row in items)
    terminated = any(ins.get("type") == "TimelineTerminateTimeline"
                     and ins.get("direction") == "Bottom" for ins in value["instructions"])
    if (value["instructions"] and not terminated
            and not any("entries" in ins or "entry" in ins for ins in value["instructions"])):
        raise Incomplete("timeline has no item page or bottom termination")
    return Page(records, cursors[0] if cursors else None, observed,
                ended=ended or terminated or not items,
                repeat_is_end=bool(repeat_prefix) and bool(observed)
                and all(key.startswith(repeat_prefix) for key in observed))
