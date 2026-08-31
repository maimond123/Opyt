"""
mcp_server/frontier_tools.py — the MCP surface for Frontier stage 4 (SURFACE).

ONE pull tool, `frontier(...)`: the ranked list of what stage 2 staged, each row carrying its own
state, plus the two acts the user can perform on it (look, and say stop). The ranking itself lives
in `pipeline/kb/frontier_surface.py` — importable and testable without an MCP server, the
radar_tools idiom.

Re-founded on v1's artifact-frontier shape (push hint + pull payload + seen-tracking + pagination)
but on the v2 rail's SQLite tables instead of a vault JSON seen-set. Admission into the KB is
autonomous stage 3's job (`pipeline/kb/frontier_admit.py`), not this tool's — this module only
informs, it never writes. `save_repo` (the old GitHub write path) was deleted 2026-08-13; it never
ran.
"""
from __future__ import annotations

# How many candidates one `frontier()` call hands back. The rest are REPORTED, not dropped: the
# response carries `remaining`, so a large queue is paginated across calls. v1's discipline, kept
# verbatim — the host cannot rescue what it is never told exists.
DELIVER_CAP = 20

_SURFACE = "frontier"


def deliver(limit: int = DELIVER_CAP, dismiss=None, include_dismissed: bool = True,
            conn=None) -> dict:
    """Assemble the ranked payload, record what it showed, and record what was dismissed.

    Order matters. Dismissals are written FIRST, so anything dismissed in this call comes back in
    this same response labelled `dismissed` rather than silently disappearing between the request
    and the answer. That is constraint 6 — show what was decided — applied to the one call where
    it is easiest to get wrong.
    """
    from pipeline.kb import frontier_surface as fs
    from pipeline.kb import schema

    own = conn is None
    conn = conn or schema.connect()
    try:
        dismissed_n = fs.record_dismissed(conn, list(dismiss or []), surface=_SURFACE)

        # Rank EVERYTHING once, then narrow in this layer. One pass gives both the delivered set
        # and the count of what the opt-out held back, and there is no way for the two numbers to
        # disagree about the same call.
        staged = fs.rank_candidates(conn)
        ranked = staged if include_dismissed else [c for c in staged if not c["dismissed"]]
        delivered = ranked[:max(0, int(limit or 0))]
        fs.record_shown(conn, [c["candidate_id"] for c in delivered], surface=_SURFACE)

        out = {
            "status": "ok",
            "candidates": [_card(c) for c in delivered],
            "showing": len(delivered),
            "total": len(ranked),
            "remaining": max(0, len(ranked) - len(delivered)),
        }
        if dismissed_n:
            out["dismissed"] = dismissed_n
        if len(staged) != len(ranked):
            # Even the explicit opt-out reports its own cost. A count the caller can see is a
            # different thing from a silent filter.
            out["hidden_by_include_dismissed"] = len(staged) - len(ranked)
        if not ranked:
            out["note"] = ("NO CANDIDATES STAGED. Frontier stage 2 pulls artifacts on a schedule "
                           "from the user's standing queries; an empty store means it has not run "
                           "yet, or has run and found nothing new. This is not an error and there "
                           "is nothing for the user to fix.")
        return out
    finally:
        if own:
            conn.close()


def _card(c: dict) -> dict:
    """One candidate, as the host sees it. `score` and `why` ride along because a host that has to
    explain an ordering to a human should not have to guess at it — and they are recomputed every
    call, never read from the store."""
    return {
        "candidate_id": c["candidate_id"],
        "source": c["source"],
        "title": c["title"],
        "url": c["url"],
        "published": c["published"],
        "summary": c["summary"],
        "payload": c["payload"],
        "state": c["state"],
        "shown_before": c["shown_n"],
        "queries": c["n_queries"],
        "score": c["score"],
        "why": c["why"],
        # Present ONLY when this card absorbed others — the ids it stands for. Omitted rather than
        # sent as [] so a host never has to distinguish "no duplicates" from "not checked".
        **({"duplicate_of": c["duplicate_of"]} if c.get("duplicate_of") else {}),
    }


def notice() -> dict | None:
    """The push half: a count plus the strongest unseen candidate, or None.

    None means the field is ABSENT from whatever response carries this, so a quiet frontier costs
    an unrelated conversation nothing. v1's `followup()` contract, kept — it is the part of the v1
    design that most deserved to survive.
    """
    from pipeline.kb import frontier_surface as fs
    from pipeline.kb import schema

    conn = None
    try:
        conn = schema.connect()
        return fs.notice(conn)
    except Exception:
        return None                      # a notice must never break the tool it rides on
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def register_frontier_tools(mcp) -> None:
    @mcp.tool()
    def frontier(limit: int = DELIVER_CAP, dismiss: list[str] | None = None,
                 include_dismissed: bool = True) -> dict:
        """The FRONTIER queue: recent artifacts (papers and repos — arXiv preprints, GitHub
        repos, and published literature across every discipline via OpenAlex) that the user's
        own standing queries pulled from the outside world, RANKED by how much they deserve
        attention right now. Read-only and free. Use it when the user asks what is new, what the
        frontier found, or what they have not looked at yet.

        What the ranking is. Recomputed on every call from checkable facts, never stored: how many
        standing queries found the same artifact (the strongest signal — independent convergence),
        how many regions of the user's KB asked for those queries, how recent it is, how
        substantial it is for its kind (stars for a repo, abstract length for a paper — never
        compared across sources), minus how often it has already been shown.

        One artifact is ONE card even when several sources staged it under different ids (the same
        preprint reached by DOI and by its /abs/ page, a paper deposited twice). Such a card
        carries `duplicate_of` naming the ids it stands for — it is a merge, so no signal is lost
        and `total` counts artifacts rather than rows.

        Nothing is ever filtered out. Every term demotes; none excludes. Each candidate carries a
        `state` — "new", "seen" (shown before), "dismissed" (the user said stop), or stage 3's
        verdict — and a dismissed one still comes back, ranked last and labelled. Report a
        dismissed item as dismissed; do not hide it from the user and do not re-pitch it.

        Calling it again advances the queue. This tool records what it showed you, and being shown
        demotes — so a second call surfaces the NEXT batch rather than re-pitching the same head.
        There is no cursor to pass. `remaining` > 0 means there are more below the cut; call again,
        or raise `limit`. Nothing gets stranded: an unseen candidate carries no penalty at all, so
        it outranks everything already shown.

        There is no save step here, and you should not invent one. Admission into the knowledge
        base is Frontier stage 3's job and it is AUTONOMOUS — it runs on its own schedule, with no
        approval step and nothing for you to call. A candidate's `state` tells you what stage 3 has
        already decided: "materialized" means it is in the knowledge base, "rejected" means the
        fetch mechanically failed. "rejected" is NEVER a quality judgement — stage 3 has no
        judge. Report it as a fetch failure, never as "not good enough".
        Do not tell the user YOU added or kept anything, and do not offer to: nothing you do here
        admits an artifact, and as of 2026-08-13 there is no tool anywhere that admits one on
        request — the `save_paper` / `save_repo` vault writers were deleted. Stage 3 is the only
        admission path and it runs on its own.

        Params: `limit` (default 20), `dismiss` (list of candidate_ids the user explicitly wants
        stopped — pass ONLY on an explicit request, never on inference), `include_dismissed`
        (default True; passing False hides them and reports how many it hid).
        """
        return deliver(limit=limit, dismiss=dismiss, include_dismissed=include_dismissed)
