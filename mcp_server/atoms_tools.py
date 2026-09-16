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

Two notices ride `search`'s envelope from here rather than from `opyt_core`, because both are
about the SESSION or the install and `opyt_core` stays free of both: Frontier's queue push
(session-latched, local reads only) and R2's reciprocal share-back offer (latched on disk, per
peer, foreign reads only). They are mutually exclusive by construction — one fires on your own
store, the other on somebody else's.
"""
from __future__ import annotations

import re
from collections import deque
from pathlib import Path

# ── session state ────────────────────────────────────────────────────────────────
# Session-scoped: one process per client session (server.py::main, STDIO), so module
# globals need no session id. Would need a real session key if ever served over HTTP/SSE
# with a shared process. Lives here (not opyt_core/kb.py) so opyt_core stays session-free.
_SEARCHES: int = 0
_OPENED: set[str] = set()
_RECENT: deque[set[str]] = deque(maxlen=5)
# Frontier's push notice rides `search`'s envelope, at most once per session (avoid nagging).
_FRONTIER_NOTICED: bool = False
# The thin-coverage enrichment offer, also at most once per session, for the same reason.
_THIN_OFFERED: bool = False


def _import_outstanding() -> list[str]:
    """Work the user was promised that has not delivered yet — QUEUED-and-never-started rails,
    plus Enrichment if it is in flight.

    ⚠️ THE DIFFERENCE BETWEEN "YOU ARE THIN ON THIS" AND "I HAVE NOT FINISHED READING". A thin
    result is the same arithmetic in both cases and the store cannot tell them apart without
    asking this, so `thin_coverage` diagnosed the first whenever it saw the second — telling a
    user their interests outrun their corpus while their bookmarks sat unimported. Confidently
    wrong, on the one surface where being wrong is expensive: the notice asks them to go enrich a
    subject they may already have material on.

    `started_at IS NULL` and nothing else on the rail half: a rail that has run is a rail whose
    absence of results is a real observation, and one mid-run is not a promise outstanding. Only
    the never-started ones are work the user was told about and has not received.

    ⚠️ ENRICHMENT IS NOT A RAIL ROW. It is an in-process thread (`pipeline/kb/enrichment.py`), so
    the rail read above is structurally blind to it — and that blindness would have restored
    exactly the defect this function exists to prevent, on the very flow that replaced it: bodies
    that used to sit unimported now land immediately and fill in their thread context over the
    next few of x.com's windows, which is a mid-import store by any other name. The in-process
    probe is what keeps `import_incomplete` firing instead of `thin_coverage`.

    Fail-safe in both halves: an unreadable jobs database or a probe that throws contributes
    nothing — the search says what it always said, rather than failing a query over a notice.
    """
    out: list[str] = []
    try:
        from pipeline.kb.rail_jobs import RailJobStore
        from pipeline.kb.rail_worker import RAILS
        out = sorted(j.rail for j in RailJobStore().list_jobs()
                     if j.started_at is None and j.rail in RAILS)
    except Exception:
        pass
    try:
        from pipeline.kb import enrichment
        if enrichment.is_running():
            out.append("enrichment")
    except Exception:
        pass
    return out


def _offer_marker(kb_name: str) -> Path:
    """Where the reciprocal offer for one normalized peer name is latched. Resolved at call time
    so it honors `$OPYT_HOME`; names that normalize alike intentionally share one latch, which
    suppresses the later offer because R2 says once and never repeated."""
    from opyt_core.paths import opyt_path
    slug = re.sub(r"[^a-z0-9]+", "-", kb_name.lower()).strip("-") or "peer"
    return opyt_path("share_offers", slug)


def _attach_reciprocal_offer(out: dict, kb_name: str) -> None:
    """Offer to share back, ONCE per peer, after their first read that actually returned
    something (R2).

    R2 rejected mutual-by-construction on a conversion argument: requiring someone to publish
    before they can read excludes anyone with an empty knowledge base, which is most new
    installs. So sharing is one-directional and the second direction is an OFFER, made at the one
    moment it is earned — the reader has just seen that this works, on their own question.

    THE LATCH IS ON DISK, not in the session. R2 says once, never repeated, and a session-scoped
    flag re-offers on every new session, which is a nag with extra steps. One marker per normalized
    peer name means a name collision intentionally suppresses the later offer.

    Empty hits do not count: an offer riding a search that found nothing is asking for a favour
    on the strength of a disappointment.

    Layered HERE rather than in `opyt_core/kb.py` for the same reason the session counters are —
    `opyt_core` stays session-free and knows nothing about markers. Wrapped bare: a marker that
    cannot be written must never break a search (fail-safe)."""
    try:
        marker = _offer_marker(kb_name)
        if marker.exists():
            return
        out["notices"].append({
            "code": "reciprocal_offer", "kb": kb_name,
            "message": f"That answer came from {kb_name}'s knowledge base. If the user has one of "
                       f"their own, ask whether they want to share it back — the `share` tool "
                       f"returns a link they can send. Ask once; do not raise it again."})
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except Exception:
        pass


def _attach_pull_notice(out: dict) -> None:
    """Carry back an Oracle pull that finished while nobody was looking.

    ⚠️ OPYT CANNOT PUSH. An MCP server speaks when it is called and never otherwise, so a user
    who closed the conversation mid-pull has exactly one way of ever finding out it landed: the
    next thing they do. That includes `search`, which is why a notice about Oracles rides on a
    tool that has nothing to do with them.

    ONCE. `completion_notice` stamps the run as it hands it over, so this fires on the first call
    after a pull finishes and never again — an un-stamped notice would ride on every search
    forever and the reader would learn to skip it.

    Fail-safe and silent: no store, no run, or an unreadable one contributes nothing. This is a
    courtesy on somebody else's answer and must never be able to break it."""
    try:
        from pipeline.kb import pull_runs

        conn = pull_runs.connect()
        try:
            if (notice := pull_runs.completion_notice(conn)) is not None:
                out.setdefault("notices", []).append({"code": "pull_finished", **notice})
        finally:
            conn.close()
    except Exception:
        pass


def _attach_allowance_notice(out: dict) -> None:
    """Why HALF this search ran, when the cause is the model provider.

    GATED ON THE DEGRADE, never on the call. `_attach_pull_notice` above rides every search and
    must therefore stamp itself; this one rides only a search that actually lost its vector arm,
    so it repeats — which is the point, because a spent allowance is a STATE that persists until
    the user acts, not an event that is over once mentioned. See `allowance_notice`.

    It sits BESIDE `vector_arm_unavailable` rather than replacing it: that notice says what the
    reader got (keyword ranking), this one says why and what ends it. Only a provider-level cause
    produces one — a foreign store's subspace mismatch leaves the first notice alone.

    Fail-safe and silent, like every other rider here."""
    try:
        if not any(n.get("code") == "vector_arm_unavailable" for n in out.get("notices", [])):
            return
        from pipeline.kb.allowance_notice import allowance_notice
        if (notice := allowance_notice()) is None:
            return
        # THE CAVEAT NEEDS AN INSTRUCTION, NOT JUST A FIELD. `vector_arm_unavailable` carries no
        # `host_note`, so what half ran is AVAILABLE to the host and never asked for — and the
        # reader most hurt by that is the one whose conceptual question now finds nothing and
        # concludes their library does not have it. Said here rather than in the shared notice
        # because this is the surface where it is true, and stripped of vocabulary: "BM25" and
        # "vector arm" describe our machinery, not their problem.
        notice["host_note"] += (
            " ON THIS ANSWER: only the keyword half of the search ran, so a question phrased "
            "conceptually may miss things it would normally find — say that plainly, and do not "
            "use the words 'BM25', 'vector' or 'embedding'. If they were looking for something "
            "they expect to be there, suggest they retry with the words the source itself would "
            "have used.")
        out.setdefault("notices", []).append(notice)
    except Exception:
        pass


def _attach_suggestions(out: dict, kb: str | None) -> None:
    """Add `suggested` — what this store can support — to an aggregate of the user's OWN store.

    ⚠️ OWN STORE ONLY, and that is a correctness rule rather than a courtesy. `aggregate(kb=...)`
    summarizes a peer's knowledge base, and every suggestion here is phrased as something the
    READER should do next. "14 items on agentic payments — read them in order" is true of the
    peer's store and useless as advice, because the reader cannot sit on atoms they do not hold.
    A foreign aggregate therefore gets counts and nothing else.

    This is also the repeatable home for the measurement the ingest tour shows once. A one-shot
    message at the end of a first ingest cannot serve a user three weeks later against a bigger
    store, and the same function answers both because it is a function OF the store: `sprouts`
    and `frontier` become the right answer only once there is unread material and standing
    queries, neither of which exists on day one.

    Fail-safe, and silent about it: no `suggested` key at all is a smaller lie than an empty one.
    """
    if kb not in (None, "me") or not out.get("total"):
        return
    try:
        from opyt_core.suggest import suggestions
        from pipeline.kb import schema
        conn = schema.connect()
        try:
            block = suggestions(conn, out)
        finally:
            conn.close()
        if block:
            out["suggested"] = block
    except Exception:
        pass


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

        EVERY HIT YOU MENTION MUST CARRY ITS LINK. `cite` is a ready-made markdown link — name
        the hit inside it, so the title you were already going to write IS the link:

            •  **[Earth Day 2025](https://x.com/NASA/status/…)** — satellite naming tool

        In that shape a link costs no extra width, which matters because the link that costs
        width is the one that gets dropped from a short list. A bullet with no link in it is
        never acceptable. A hit with no stored URL has `cite: null` — say so plainly rather than
        inventing one. `source_url` is the same pointer unwrapped, when you need the bare string.

        This is not decoration: a result the user cannot click back to is a claim they cannot
        check, and being checkable is the entire reason this is a router rather than an answer.
        Never make them ask where something came from. `source_url` is the same pointer unwrapped,
        for when you need the bare string. The rule covers `frontier_atoms` identically, and
        `open()` returns `cite` too, for the moment you actually assert what a source says.

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
        global _SEARCHES, _THIN_OFFERED
        out = run_kb_search(query, tags=tags, what_kind=what_kind, source_type=source_type,
                            who=who, who_id=who_id, date_from=date_from, date_to=date_to,
                            entry_mode=entry_mode, k=k, mode=mode, kb=kb)
        _SEARCHES += 1
        _attach_pull_notice(out)
        _attach_allowance_notice(out)
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
        answered_by = out["trace"].get("kb", "me")
        if answered_by == "me":
            _attach_frontier_notice(out)
        elif out["hits"]:
            _attach_reciprocal_offer(out, answered_by)

        # ⚠️ THE HIGHEST-INTENT ENRICHMENT MOMENT THERE IS. A user who searches for a subject
        # their store barely covers has just said, by asking, that they care about something
        # OPYT is thin on — better signal than any tag count, and root-agnostic. The trigger is
        # coverage ARITHMETIC, not a quality judgement: the whole store ranked fewer atoms than
        # the call asked for, with no filter narrowing it (a filtered search is thin on
        # purpose). Own store only — enriching is advice to the reader about THEIR material,
        # same rule as the frontier notice above. Once per session, like that notice: `search`
        # is high-frequency, and an offer on every thin call trains the reader to skip it.
        ranked = out["trace"].get("ranked")
        if (not _THIN_OFFERED and answered_by == "me"
                and not any(v for v in (tags, what_kind, source_type, who, who_id,
                                        date_from, date_to, entry_mode))
                and isinstance(ranked, int) and ranked < k):
            _THIN_OFFERED = True
            # An unfinished import produces this exact arithmetic, so RULE IT OUT before calling
            # the store thin. Checked here rather than at the top of `search` because the gate
            # above is rare and this reads a second database. See `_import_outstanding`.
            waiting = _import_outstanding()
            if waiting:
                out["notices"].append({
                    "code": "import_incomplete", "ranked": ranked, "asked": k, "waiting": waiting,
                    "message": (
                        f"This unfiltered search matched {ranked} atoms in the whole store — "
                        f"fewer than the {k} you asked for. Do NOT read that as the user being "
                        f"thin on this subject: an import they were promised is still outstanding "
                        f"({', '.join(waiting)}), so material that would answer this may simply "
                        f"not be in the store yet"
                        + (" — enrichment is still running, so some saved posts are in the store "
                           "without their thread context or their image descriptions"
                           if "enrichment" in waiting else "")
                        + f". Say that plainly, answer from what is here, and "
                        f"offer nothing else — an enrichment offer here asks them to go find "
                        f"what they already gave OPYT.")})
                return out
            out["notices"].append({
                "code": "thin_coverage", "ranked": ranked, "asked": k,
                "message": (
                    f"This unfiltered search matched {ranked} atoms in the whole store — fewer "
                    f"than the {k} you asked for. The user has just named a subject their store "
                    f"is thin on, which is the best enrichment signal there is. Offer, in one "
                    f"line: to web-search the current landscape on it for them, and to put it "
                    f"on standing watch. If they want the watch, agree the exact wording with "
                    f"them first — it runs forever in those words — then "
                    f"sitting(action='watchlist', add=[...]); the first pull runs in the "
                    f"background. If they decline, drop it; this offer is made once per "
                    f"session.")})
        return out

    @mcp.tool()
    def open(atom_id: str, kb: str | None = None) -> dict:
        """Follow an atom's pointer and return its REAL raw snapshot text + live source_url.
        Call this before citing or asserting anything an atom "says" — `search` only
        routes you to the atom; THIS gives you the ground truth to reason from. Returns
        {atom_id, source_url, raw, description, body_state, body_basis, payload, …}; an unknown
        id returns {error: "not found"}.

        `source_url` here obeys the same rule it does on a search hit: whenever you tell the user
        what this atom says, give them the link to it in the same breath, without being asked.

        The pre-citation check: `raw` is only as complete as `body_state` says. "complete" means
        you have the whole body. "partial" means you have a knowing fragment — a paywall teaser
        or a truncated entry — so quoting it as the full article invents a citation; attribute
        what you have, and send the reader to `source_url`. "absent" means there is no body at
        all. `body_basis` says how that was determined (observed / stated / assumed). `payload`
        holds that source's own extras verbatim, and its keys differ per source.

        PAPERS DEEPEN ON OPEN. A researcher's back catalogue is stored abstract-only — hundreds of
        papers nobody has read yet — so opening one is what triggers the PDF fetch, and `raw` comes
        back as the whole document with `body_state: "complete"`. That costs a download, so it is
        worth one call per paper you actually need in depth, not a sweep of every hit. A paper that
        stays "partial" after an open has no reachable open PDF (paywalled, or scanned without a
        text layer) — that is the real answer, not a retry prompt.

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
                  kb: str | None = None, sample: int = 0) -> dict:
        """A state-of-play skeleton over a SCOPE of the KB — counts by kind/source, trust
        coverage, author distribution, and the most-recent atom DESCRIPTIONS (mechanical, so
        safe to read without opening). Use it to draft a dossier or "what do I have on X",
        THEN `open()` the pivotal atoms to ground each claim in raw text. Scope is optional:
        omit everything for the whole store, or filter by tags / what_kind / source_type /
        who_id / date_from / date_to.

        This takes IDs, not handles. To scope to a person, call `search(who="@handle")`
        first and pass its `insights.resolved_who[].who_ids` here.

        ── "WHAT TOPICS DOES MY CORPUS COVER?" ──────────────────────────────────────────
        There is no topic list in this store to hand you, and that is on purpose: the only
        tag space here is `source_tags`, the AUTHOR's own hashtags, which covers a fraction
        of atoms and was never a taxonomy. Answer it in four steps instead — this is the
        supported way, not a workaround:

          1. `aggregate(sample=200)` — `corpus_sample` comes back: `descriptions`, a spread
             walking AUTHORS in rounds, so it shows the store's breadth rather than its
             busiest corner.
          2. READ it and name the recurring subjects in your own words. These are guesses.
          3. `search(query=<subject>)` on each one. A subject that is really here returns
             strong hits from several authors; a guess that is not returns weak scattered
             ones. Drop those silently — a wrong subject stated confidently is the exact
             failure this replaced.
          4. Report only what survived, each subject carrying its atoms as `cite` links, and
             say how many you sampled of the total.

        Do not skip step 3, do not present the raw sample as the answer, and do not save the
        names — they are your reading of this corpus today, made fresh each time it is asked.

        Args:
            sample: How many atom descriptions to include as `corpus_sample` (0 = none, the
                default; capped at 500). 200 is a good census of a few-thousand-atom store —
                it was 300 until a live run at that size overflowed the tool-result cap and
                had to be recovered with a shell, which not every MCP client has. Each
                description is ~140 characters, so this is the one costly thing here — ask
                for it when the question is about the corpus as a whole, not every call.
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
        out = kb_aggregate({"tags": tags, "what_kind": what_kind, "source_type": source_type,
                            "who_id": who_id, "date_from": date_from, "date_to": date_to},
                           kb=kb, sample=sample)
        _attach_suggestions(out, kb)
        _attach_pull_notice(out)
        return out
