"""
mcp_server/atoms_tools.py

The atom-KB MCP tools — the "trusted router" query surface. All LLM-FREE (the HOST model
reasons over what they return; no metered API here beyond a tiny query-embedding for the
vector arm). Registered onto the client FastMCP by `register_atoms_tools` below.

  search    — enforced-hybrid routing: which atoms + which chunk-span, with the pointer.
  open      — follow an atom's pointer → the REAL raw snapshot text (the trust invariant).
  aggregate — pure-SQL state-of-play skeleton for a dossier (then open() to ground claims).

All three take `kb=`, which picks WHICH knowledge base to read: omitted is the user's own, any
other name is a registered peer's (pipeline/kb/peers.py). Read-only, and the reader's own store is
never written to from a foreign read.

Named `search`, not `opyt_search`: the server owns only this segment of the name, and each
namespacing client adds its own prefix (Claude Code renders `mcp__Opyt__search`).

The contract for you (the host): search returns ROUTING, never a content claim. Before
asserting what a source SAYS, call `open(atom_id)` and read its raw. A description/snippet is
a signpost, not a citation.
"""
from __future__ import annotations

from collections import deque

# ── session state ────────────────────────────────────────────────────────────────
# Session-scoped: one process per client session (server.py::main, STDIO), so module
# globals need no session id. Would need a real session key if ever served over HTTP/SSE
# with a shared process. Lives here (not opyt_core/kb.py) so opyt_core stays session-free.
_SEARCHES: int = 0
_OPENED: set[str] = set()
_RECENT: deque[set[str]] = deque(maxlen=5)
# Frontier's push notice rides `search`'s envelope, at most once per session (avoid nagging).
_FRONTIER_NOTICED: bool = False


def _reset_session() -> None:
    """Clear the per-session counters. For tests — module globals otherwise leak between them."""
    global _SEARCHES, _FRONTIER_NOTICED
    _SEARCHES = 0
    _FRONTIER_NOTICED = False
    _OPENED.clear()
    _RECENT.clear()


def _attach_frontier_notice(out: dict) -> None:
    """Attach Frontier's `notice` to a search response, once per session, only when there's
    something to say. Wrapped bare so a broken notice never breaks search (fail-safe). The
    latch trips on emit, not on check, since Stage 2 can stage candidates mid-session.
    """
    global _FRONTIER_NOTICED
    if _FRONTIER_NOTICED:
        return
    try:
        from mcp_server.frontier_tools import notice
        n = notice()
    except Exception:
        return
    if n:
        out["frontier"] = n
        _FRONTIER_NOTICED = True


def register_atoms_tools(mcp) -> None:
    from opyt_core.kb import kb_aggregate, kb_open, run_kb_search

    @mcp.tool()
    def search(query: str, tags: list[str] | None = None, what_kind: str | None = None,
               source_type: str | None = None, who: str | None = None,
               who_id: str | list[str] | None = None, date_from: str | None = None,
               date_to: str | None = None, entry_mode: str | list[str] | None = None,
               k: int = 8, mode: str = "hybrid", kb: str | None = None) -> dict:
        """Route to the most relevant ATOMS in the trusted knowledge base (David's saved
        opinions + tracked artifacts). This is a ROUTER, not an answer: each hit is a thin
        card — matched-chunk snippet + a pointer (source_url / raw_ref / atom_id) + ranks —
        NOT a statement of what the source says. To assert what a source actually claims,
        call `open(atom_id)` and read its raw text. That split is the whole trust model.

        RETURNS {hits, notices, insights, trace, frontier_atoms} — and the non-hit keys are
        read DIFFERENT ways:
          • `notices` — finished sentences about what your QUERY did (a filter that matched
            nothing, a handle that resolved to nobody or to two people, results truncated).
            Surface these to the user when the list is non-empty; they are written to be
            repeated as-is. `[]` on a healthy query, which is the normal case.
          • `insights` — VALUES about the evidence: `authors`/`sources`/`topics` counts,
            `date_span`, `body_state`, `saved_vs_crawled`, `corpus_newest`, `filter_cost`
            (what each filter cost you), and `resolved_who` when you passed `who=`. These
            describe `hits` ONLY. On a default call `saved_vs_crawled` therefore lists no
            crawled atoms — they are in `frontier_atoms`, not missing.
          • `trace` — VALUES about what the ENGINE did: `ran` (which arms actually ran),
            `score_scale` (the units `score` is in), `candidates`/`ranked`/`showing`,
            `cutoff`, `fts_query`, `filters` as applied.
          • `frontier_atoms` — a SECOND, separately ranked list: atoms found by the user's
            standing keyword queries rather than saved or written by anyone they follow. Present
            only on a default call (see `entry_mode`). Same card shape as `hits`, capped at 8 and
            floored at a fraction of its own top score, with `floor.dropped` saying how many the
            floor removed. Offer it as "and from the frontier crawl…", never merged into `hits`:
            its scores are ranked against other frontier atoms and mean nothing next to theirs.

        `frontier_atoms` is NOT the `frontier` key you may also see here. That one is the
        Frontier QUEUE's push notice — staged candidates not yet in the KB — and it appears at
        most once a session. These are atoms already in the KB that matched THIS query.

        `insights` and `trace` are for your reasoning — do not recite them to the user.
        They are bare values on purpose. Use them to decide what to do next (open something,
        re-query, drop a filter, warn about a lopsided result); say the CONCLUSION in your own
        words, never the fields. Only `notices` is written to be read out.

        `score` is not comparable across calls unless `trace.score_scale` matches. It is a raw
        cosine under `semantic` (and under a `hybrid` run that dropped its keyword arm), a
        reciprocal rank under `bm25`, and a fused rank sum under a true `hybrid` run. 0.03 in one
        scale can outrank 0.7 in another. A known bias rides the cosine: a longer document
        max-pools higher for having more chunks to draw from, so weigh a long hit's lead over a
        short one as smaller than it looks.

        Retrieval: an optional tag/kind/source/author pre-filter, then a BM25 arm and a
        semantic arm, fused by rank. Ranking is pure relevance — there is no trust re-rank.
        `mode="hybrid"` often runs only the semantic arm: a conceptual query with no literal
        token (most natural-language questions over three words) gives BM25 a weight of 0 and
        the keyword arm is skipped. `trace.ran` says which arms really ran; do not assume both.

        Read `body_state` before quoting a snippet. It says how much of the source we actually
        stored: "complete" (the whole body), "partial" (knowingly short of it — a paywall teaser,
        a truncated feed entry), "absent" (no body, the card is all there is), or "pending" (not
        yet determined). On "partial" or "absent", do not present the text as the full thing —
        say what you have, and follow `source_url` for the rest. `body_basis` says how that was
        decided: "observed" (we saw the boundary), "stated" (the source declared it), "assumed".

        Read `when_precision` before reporting a date, especially under `date_from`/`date_to`.
        `when_ts` always LOOKS like a day, and for two values it is not one:
          • "year" — only the YEAR is known (common for papers); `when_ts` is that Jan 1 as a
            FLOOR, not a real day. Such a hit is included whenever its year OVERLAPS your
            window, deliberately — a wrongly-included atom you can see and discard, a wrongly
            excluded one you cannot. Caveat it to the user ("published sometime in 2025"),
            never as a confirmed date match.
          • "push" — GitHub's LAST-PUSH date, NOT a publication date. A repo matching "after
            May" was ACTIVE then and may have been created years earlier. A different KIND of
            date, not a coarser one, and the easiest thing here to misreport.
        Atoms with NO recorded date are excluded by either bound, and a notice says how many.

        Each hit also carries `payload` — whatever extras that atom's SOURCE had, returned
        verbatim. It is NOT a fixed schema and it is NOT filterable: GitHub atoms carry
        stars/code_language, X atoms like_count/is_thread, papers citationCount/venue. Read the
        keys that are there; never assume a key exists because another hit had it.

        Args:
            query: Natural-language query. Rare literal tokens (a lib/symbol name) engage the
                keyword arm; conceptual phrasing leans on the semantic arm — both fire in hybrid.
            tags: Restrict to atoms tagged with ANY of these topics (slugs, e.g. "ai-agents").
                Matched as slugs: a value that normalizes to nothing matches NOTHING (never
                "no filter"), and `notices` tells you when a value was dropped or rewritten.
            what_kind: Restrict to a kind: "opinion" (saved posts) or "artifact" (repos).
            source_type: Restrict to ONE source. Live values: "x", "github", "substack",
                "blog", "paper".
            who: Restrict to one author by HANDLE — "@karpathy", "karpathy", a Substack/blog
                URL, or an id. THIS is how you answer "what did <person> say about X". Putting
                their name in `query` instead matches by CONTENT, so posts merely MENTIONING
                them rank alongside posts they WROTE (measured: 5 hits, 1 of them theirs).
                Resolved LOCALLY against people already in the store — free, no network, and it
                never invents anyone. A handle nobody has matches nothing, never everything, and
                `insights.resolved_who` + a `who_unresolved` notice tell you which case you hit:
                untracked person, or tracked person with nothing on this topic.
            who_id: Restrict by EXACT entity id, one or several ("x:user:33836629",
                ["github:karpathy", "x:user:33836629"]). Use when you already have ids — from a
                prior hit's `who_id`, or `insights.resolved_who[].who_ids`. Prefer `who` when
                all you have is a handle; a person's atoms are spread across a PER-PLATFORM id
                each, so one id alone returns one platform's worth of them.
            date_from: Earliest atom date to include, INCLUSIVE. "2026", "2026-05" or
                "2026-05-11" — a partial widens to its natural edge, so date_from="2026" means
                2026-01-01. ANY other shape is an ERROR, not a dropped filter: "5/11/2026"
                raises. THIS is how you answer "what did they post after <date>" — putting a
                date in `query` matches it as CONTENT, which is not a filter at all.
            date_to: Latest atom date to include, INCLUSIVE. Same formats; a partial widens the
                OTHER way, so date_to="2026" means 2026-12-31.
            entry_mode: How the atom ARRIVED. Leave it OFF for the normal case: the answer then
                comes SECTIONED — `hits` is the full k over what the user saved, their Oracles
                published, or those Oracles cited, and `frontier_atoms` carries the keyword-crawl
                finds separately. Set it to scope the whole answer to one population instead:
                "frontier" with a larger `k` is how you dig into the crawl ("show me more of
                what the crawl found"), and a list like ["user-saved"] narrows to one arrival
                path. Scoping returns ONE list and no `frontier_atoms` key.
            k: Max atoms to return (default 8). `trace.cutoff` shows the score at the boundary,
                so you can tell whether raising it would have helped.
            mode: "hybrid" (default), "semantic", or "bm25". See `trace.ran` for what ran.
            kb: Read SOMEONE ELSE'S knowledge base instead of your own. Omit for yours (the
                normal case). A name here must be one this install has registered; an unknown
                one returns no hits and a notice naming the ones that exist. Every hit carries
                the `kb` it came from — "me" for your own — and an atom id means nothing outside
                its own store, so pass that same value back to `open(atom_id, kb=...)`. Attribute
                anything you repeat from a foreign hit to that knowledge base, not to the user.
        """
        global _SEARCHES
        out = run_kb_search(query, tags=tags, what_kind=what_kind, source_type=source_type,
                            who=who, who_id=who_id, date_from=date_from, date_to=date_to,
                            entry_mode=entry_mode, k=k, mode=mode, kb=kb)
        _SEARCHES += 1
        ids = {h["atom_id"] for h in out["hits"]}
        # Checked against the last 5 searches, not just the prior one (A, B, A circling is common).
        overlap = max((len(ids & prev) for prev in _RECENT), default=0)
        _RECENT.append(ids)
        out["trace"]["session"] = {
            "search_n": _SEARCHES,
            "overlap_with_recent": overlap if overlap >= 2 else None,
            "opened_so_far": len(_OPENED),
        }
        if _SEARCHES >= 3 and not _OPENED:
            # A process fact, not a judgement about the evidence: searching alone never
            # grounds a claim, and this is the moment the pattern is visible. A static
            # docstring line dilutes across a long context; this arrives when it applies.
            out["notices"].append({
                "code": "nothing_grounded", "searches": _SEARCHES,
                "message": f"You have searched {_SEARCHES} times this session without calling "
                           f"open() once. A snippet is a signpost, not a citation — open an "
                           f"atom and read its raw text before asserting what a source says."})
        # Frontier's queue is the READER's own staged artifacts. Riding it on a foreign result
        # would tell them their own backlog grew because they looked at somebody else's KB.
        if out["trace"].get("kb", "me") == "me":
            _attach_frontier_notice(out)
        return out

    @mcp.tool()
    def open(atom_id: str, kb: str | None = None) -> dict:
        """Follow an atom's pointer and return its REAL raw snapshot text + live source_url.
        Call this before citing or asserting anything an atom "says" — `search` only
        routes you to the atom; THIS gives you the ground truth to reason from. Returns
        {atom_id, source_url, raw, description, body_state, body_basis, payload, …}; an unknown
        id returns {error: "not found"}.

        The pre-citation check: `raw` is only as complete as `body_state` says. "complete" means
        you have the whole body. "partial" means you have a knowing fragment — a paywall teaser
        or a truncated entry — so quoting it as the full article invents a citation; attribute
        what you have, and send the reader to `source_url`. "absent" means there is no body at
        all. `body_basis` says how that was determined (observed / stated / assumed). `payload`
        holds that source's own extras verbatim, and its keys differ per source.

        `kb` must be whatever the hit card carried. An atom id is scoped to ONE knowledge base —
        the same tweet in two people's stores is one id in each — so opening a foreign id without
        its `kb` either finds nothing or hands you your own copy of the same source.
        """
        out = kb_open(atom_id, kb=kb)
        if "error" not in out:
            _OPENED.add(atom_id)      # what clears `search`'s nothing_grounded notice
        return out

    @mcp.tool()
    def aggregate(tags: list[str] | None = None, what_kind: str | None = None,
                  source_type: str | None = None, who_id: str | list[str] | None = None,
                  date_from: str | None = None, date_to: str | None = None,
                  kb: str | None = None) -> dict:
        """A state-of-play skeleton over a SCOPE of the KB — counts by kind/source, trust
        coverage, topic/entity distribution, and the most-recent atom DESCRIPTIONS (mechanical,
        so safe to read without opening). Use it to draft a dossier or "what do I have on X",
        THEN `open()` the pivotal atoms to ground each claim in raw text. Scope is optional:
        omit everything for the whole store, or filter by tags / what_kind / source_type /
        who_id / date_from / date_to.

        This takes IDs, not handles. To scope to a person, call `search(who="@handle")`
        first and pass its `insights.resolved_who[].who_ids` here.

        Args:
            date_from: Earliest atom date, INCLUSIVE — "2026", "2026-05" or "2026-05-11"; a
                partial widens to its natural edge (date_from="2026" is 2026-01-01). Any other
                shape RAISES rather than being ignored.
            date_to: Latest atom date, INCLUSIVE. Same formats, widening the other way
                (date_to="2026" is 2026-12-31).
            kb: Summarize SOMEONE ELSE'S knowledge base instead of your own. Omit for yours.
                `trusted_atoms` then counts atoms whose author THAT owner confirmed, not you.

        Counts here are a plain filter result: an undated atom is excluded by either bound, and
        a year-only atom counts if its year overlaps the window. Unlike `search`, this tool
        does NOT report how many undated atoms that dropped — use search when that matters.
        """
        return kb_aggregate({"tags": tags, "what_kind": what_kind, "source_type": source_type,
                             "who_id": who_id, "date_from": date_from, "date_to": date_to},
                            kb=kb)
