"""
pipeline/arrivals/grouping.py — the new atoms by topic: one group per topic whose words or people
match any of them, then the atoms no topic matches, and how many of each one answer lists.

A topic's posts are matched by `pipeline/topics/posts.py`, the same reading its check gets, so What's
New and the topic cannot disagree on what belongs to it. `groups` is the one read; `arrange` and
`cut` decide from values.
"""
from __future__ import annotations

from pipeline.topics import posts


def groups(conn, new_ids: list[str], topics: list[dict], *, unfiled: bool) -> list[dict]:
    """`arrange` over each topic's matches among `new_ids`."""
    among = set(new_ids)
    return arrange(new_ids, [(t, posts.matching(conn, t, among)) for t in topics],
                   unfiled=unfiled)


def arrange(new_ids: list[str], matches: list[tuple[dict, set[str]]], *,
            unfiled: bool) -> list[dict]:
    """`{topic, atom_ids}` for each topic that matched anything, in `matches`' order, then, when
    `unfiled`, `{topic: None, atom_ids}` for the ids no topic matched. Ids keep `new_ids`' order,
    and an atom two topics match is in both groups."""
    out, filed = [], set()
    for topic, matched in matches:
        if matched:
            out.append({"topic": topic, "atom_ids": [a for a in new_ids if a in matched]})
            filed |= matched
    rest = [a for a in new_ids if a not in filed]
    if unfiled and rest:
        out.append({"topic": None, "atom_ids": rest})
    return out


def cut(grouped: list[dict], limit: int) -> list[dict]:
    """Each group with the ids it lists (`listed`): `limit` entries in all, handed one per group in
    turn, so one busy topic cannot take every line from the others; newest first within each."""
    shares, left = [0] * len(grouped), max(0, limit)
    while left and any(n < len(g["atom_ids"]) for n, g in zip(shares, grouped)):
        for i, group in enumerate(grouped):
            if left and shares[i] < len(group["atom_ids"]):
                shares[i] += 1
                left -= 1
    return [{**group, "listed": group["atom_ids"][:n]} for group, n in zip(grouped, shares)]
