"""
pipeline/oracles/screen.py — the Oracle candidate screen, `oracle(action='screen')`.

Ranks (`ranking`) and returns the pick-your-Oracles payload: the groups the host prints — one per
source the user's people reach them through, then the people the Oracles' posts name who are in
none of them — each in rank order and bounded so it fits, plus the two questions to ask. The MCP
`oracle` tool hands this to the host, which prints it; the user picks. The host lists plain
organisations, publications and products in their own last group (`_NOTE`): no model call of
OPYT's own decides what an account is.

Invariants:
  • The payload has to fit the host's token limit: each card carries a sentence, not signal
    rows; each group is cut to `limit` with the rest counted (`more`); the total is bounded
    (`SCREEN_LIMIT_MAX`).
  • A person on two sources is listed once, under the stronger (`ranking.stronger_source`), with
    the other named on their line.
  • Nothing is selected by default; the user picks.
"""

from __future__ import annotations

from pipeline.oracles import freshness
from pipeline.oracles import mentions
from pipeline.oracles import windows
from pipeline.oracles.ranking import (PAPERS, Candidate, known_platforms, rank_candidates,
                                      source_of, source_rank, stronger_source)
from pipeline.platforms import connections

# The last group: people the Oracles' posts name who are in none of the user's lists. Also what
# `source=` takes to list it further.
NAMED = "named"

# One group per source, the top `limit` of each in rank order with a count of the rest.
_TITLES = {"x": "Your top people on X", "substack": "Your top people on Substack",
           PAPERS: "Your top people from papers", NAMED: "Named by your Oracles"}
_ALSO = {"x": "on X", "substack": "on Substack", PAPERS: "in your papers"}

# How many cards each group prints before "+ N more".
GROUP_LIMIT = 10
# The most cards a `screen` returns across all its groups. A total, not a per-group bound, so a
# raised `limit` is clamped to this divided by the number of groups and no exemption can overrun
# the host's limit.
SCREEN_LIMIT_MAX = 80

# The literal first question the host asks. Nothing is selected by default.
PICK_ASK = ("Pick the people you want as Oracles — I'd start with about 5, so you'll see their "
            "content within a few minutes. You can add more any time.")

_NOTE = ("Print the groups exactly as given, in order: each `title`, then one line per "
         "candidate, `name — reason`, then `+ N more` when `more` is above 0. If a listed account "
         "is plainly an organisation, publication, product or bot rather than one person, move it "
         "to one last group titled \"Organisations, media and projects\"; when unsure, leave it "
         "where it is. Nothing else: no import summary, no counts, no other groups, no "
         "explanation of how the list was made, no commentary on any group. Then ask `ask`, and "
         "when `lookback_options.x` is present ask its `ask` as a separate question, and stop. Nobody is selected until the user names "
         "them. Pass the picks' `canonical_id`s to oracle(action='confirm', canonical_ids=[...]) "
         "with the X answer as `x_lookback`; a pick with no `canonical_id` (\"Named by your "
         "Oracles\") and someone not listed go in `add_handles` by handle. If the "
         "user asks for more of a group, call screen again with that group's `source` and a "
         "larger `limit`.")


def reflect(signals: list[dict]) -> str:
    """The user's own signals as one short plain reason — "you follow, bookmarked 12". Endorsements
    first, then content, then the two acts the user did not make. Degrades honestly: "paid
    subscriber" only when is_paid is known True; a None/unknown is_paid falls to "you subscribe"
    (`curation/substack_people.is_paid` holds the mapping)."""
    parts: list[tuple[int, str]] = []
    for s in signals:
        st, pf, c, extra = s["signal_type"], s["platform"], s["count"], (s["extra"] or {})
        if st == "follow":
            parts.append((0, "you follow"))
        elif st == "subscribe":
            parts.append((1, "paid subscriber" if extra.get("is_paid") is True
                          else "you subscribe"))
        elif st == "list":
            names = extra.get("list_names") or []
            parts.append((2, f"in your {', '.join(names)} list{'s' if len(names) > 1 else ''}"
                          if names else f"in {_n(c, 'of your lists', 'of your lists')}"))
        elif st == "save":
            # Keyed on the platform, so a saved paper is never called a post.
            parts.append((3, f"bookmarked {c}" if pf == "x"
                          else f"saved {_n(c, 'paper', 'papers')}" if source_of(pf) == PAPERS
                          else f"saved {_n(c, 'post', 'posts')}"))
        elif st == "like":
            parts.append((4, f"liked {c}"))
        elif st == "recommended":
            # One of the two signals the user did not create: name whose act this was, so the
            # candidate is never mistaken for something the user chose. `c` is how many Oracles
            # recommend them, not how many times.
            parts.append((5, "recommended by one of your Oracles" if c <= 1
                          else f"recommended by {c} of your Oracles"))
        elif st == "coauthor":
            # The weakest signal in the system, and the phrasing says so: the user did nothing.
            # An Oracle put their name next to this person's, on `c` papers.
            parts.append((6, f"co-wrote {_n(c, 'paper', 'papers')} with one of your Oracles"))
    return ", ".join(text for _, text in sorted(parts, key=lambda p: p[0]))


def _n(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def group_candidates(ranked: list[Candidate], *,
                     source: str | None = None) -> list[tuple[str, list[tuple[Candidate, str]]]]:
    """Ranked candidates → `[(group, [(candidate, home source)])]` in print order, keeping the rank
    order inside every group, so a group's order is the rank among its own people. A group with
    nobody in it is left out. Each candidate is listed under `stronger_source` (a paper-only
    author's is PAPERS), or under `source` when given.

    `source` lists everyone on that source under it, whatever their stronger source is — the
    bounded answer to "show me my Substack people" — and leaves out everyone not on it."""
    groups: dict[str, list[tuple[Candidate, str]]] = {}
    for c in ranked:
        if source and source not in {source_of(s["platform"]) for s in c.signals}:
            continue
        home = source or stronger_source(c.signals)
        groups.setdefault(home, []).append((c, home))
    return sorted(groups.items(), key=lambda g: (source_rank(g[0]), g[0]))


def split_named(named: list[dict], ranked: list[Candidate]) -> tuple[dict[str, int], list[dict]]:
    """The people the Oracles' posts name (`mentions.named_by_oracles`, in rank order) → how many
    Oracles named each candidate, by canonical id, and everyone named who is not a candidate, in
    rank order. A candidate stays in their own group; the rest are "Named by your Oracles"."""
    person = {m: c.canonical_id for c in ranked for m in c.members}
    times: dict[str, int] = {}
    newcomers = []
    for row in named:
        if (cid := person.get(row["who_id"])) is not None:
            times.setdefault(cid, len(row["oracles"]))
        else:
            newcomers.append(row)
    return times, newcomers


def _named_reason(n_oracles: int) -> str:
    return ("named by one of your Oracles" if n_oracles == 1
            else f"named by {n_oracles} of your Oracles")


def _card(cand: Candidate, home: str, named: int = 0) -> dict:
    """One candidate as the host prints them: a name and one short reason, from their signals on
    the source they are listed under, how many Oracles name them, and any other source they reach
    the user through.

    Signal rows and links are left out on purpose: they overran the host's
    token limit, and each was something the host felt bound to explain. A `screen` is a list of
    people to say out loud.
    """
    others = sorted({source_of(s["platform"]) for s in cand.signals} - {home}, key=source_rank)
    reason = reflect([s for s in cand.signals if source_of(s["platform"]) == home])
    if named:
        reason += ", " + _named_reason(named)
    if others:
        reason += ", also " + " and ".join(_ALSO.get(o, f"on {o}") for o in others)
    return {"canonical_id": cand.canonical_id, "name": cand.name, "handle": cand.handle,
            "reason": reason}


def _newcomer_card(row: dict) -> dict:
    """Someone the Oracles name who is in none of the user's lists: no `canonical_id`, so a pick
    goes to `add_handles`."""
    return {"canonical_id": None, "name": row["name"], "handle": row["handle"],
            "reason": _named_reason(len(row["oracles"]))}


def build_screen(conn, *, limit: int = GROUP_LIMIT, source: str | None = None) -> dict:
    """The full screen payload the `oracle` tool hands the host: ranks, then lists the candidates
    in `groups`, in print order — each with its `title`, its top `limit` `candidates` in rank
    order, and `more`, the count of the rest. The host prints them as given and asks `ask`.

    The rank runs over everyone — the cap is on what is returned, never on what is considered, so
    no rank moves when `limit` or `source` does."""
    # Refuse an unknown `source`: zero candidates would read like a connected platform nobody
    # arrives through.
    if source and source != NAMED:
        known = known_platforms(conn)
        # In source terms, so "papers" names both registries and "openalex" still works.
        if source_of(source) not in {source_of(p) for p in known}:
            return {"error": f"source={source!r} is not a platform this store holds a signal "
                             f"from. Known: {', '.join(known) or '(none yet)'}. Nothing was "
                             f"filtered; omit `source` for the whole roster.",
                    "known_platforms": known}

    ranked = rank_candidates(conn)
    times_named, newcomers = split_named(mentions.named_by_oracles(conn), ranked)
    grouped = [] if source == NAMED else group_candidates(
        ranked, source=source_of(source) if source else None)
    with_named = bool(newcomers) and source in (None, NAMED)
    applied = max(0, min(limit, SCREEN_LIMIT_MAX // max(1, len(grouped) + with_named)))
    lists = [(key, [_card(c, home, times_named.get(c.canonical_id, 0))
                    for c, home in members[:applied]], len(members))
             for key, members in grouped]
    if with_named:
        lists.append((NAMED, [_newcomer_card(r) for r in newcomers[:applied]], len(newcomers)))
    groups = [{"title": _TITLES.get(key, f"Your top people on {key}"),
               "candidates": cards,
               "more": total - len(cards),
               "source": key}      # what `source=` takes to list this group further
              for key, cards, total in lists]
    out = {
        # With `source=` this is the matched total, not the store's. `source` rides beside it so a
        # reader can never mistake a filtered count for the whole roster.
        "total_candidates": sum(total for _, _, total in lists),
        "groups": groups,
        "ask": PICK_ASK,
        "note": _NOTE,
    }
    if source:
        out["source"] = source
    if limit > applied:
        # Said plainly rather than silently honored. The ask behind an oversized `limit` is almost
        # always "show me everyone on platform X" — which `source=` answers without the payload.
        out["limit_clamped"] = {
            "asked": limit, "applied": applied,
            "note": (f"`limit={limit}` was clamped to {applied} per group: a screen returns at "
                     f"most {SCREEN_LIMIT_MAX} people, because the full list of a real store runs "
                     f"past the host's token limit. To go further into one group, pass its "
                     f"`source=`; to find someone specific, `oracle(action='candidates', "
                     f"query='...')` searches by what they write."),
        }
    return out


def screen(conn, *, limit: int, source: str | None = None) -> dict:
    """action='screen' — the grouped candidate payload plus `oracle_freshness`. See `oracle()`.

    Routing, the provider notice, "still scoring", completion and new material ride this answer
    through the server's rider flow (`mcp_server/riders.py`)."""

    out = build_screen(conn, limit=limit, source=source)
    # The question rides with the candidates rather than in a docstring the host may paraphrase,
    # and only when X is connected: a Substack/blog-only user has no X window to choose.
    out["lookback_options"] = {"x": windows.x_question()} if connections._connected_x() else {}

    # Rides unconditionally (unlike `candidates`' `list_freshness`) — `screen` is the deliberate,
    # occasional "what is my people situation" call, and a roster with no last-pulled column is the
    # blind spot that let Oracles sit frozen. Fail-safe: a registry read failure degrades to
    # candidates without freshness, never to an error.
    try:
        out["oracle_freshness"] = freshness.status_summary(conn)
    except Exception as e:
        out["oracle_freshness"] = {"error": f"{type(e).__name__}: {e}"}

    return out
