"""
mcp_server/sitting_tools.py — the `sitting` tool: read one topical region of the KB, end to end.

ONE tool with four actions, the `frontier` idiom. The logic lives in `pipeline/kb/` —
`sitting_builder` grows the region, `sitting_surface` reports its scope, `sitting_reader` reads the
`queries` lens, `sitting_claims` reads the `claims` lens (Job N), `sitting_lenses` holds the four
host-side lenses plus `sprouts` (Job L) — all importable and testable with no MCP server. This file
is a delegate and should stay one.

Why this file exists: the seeded-region rail was fully built (builder, reader, fracture,
continuations, a coverage ledger) with no tool surface — every sitting ever created was typed at a
CLI by the module's author. This is the way to ask for one.

What a sitting is, in one breath: pick a point in the corpus, grow the region around it with NO time
bound at all, read the whole thing in publication order. That last part is what a recency window
cannot do — a window shows you the last 90 days of everything, a sitting shows you one conversation
from its beginning, and reading it in order is what makes "what changed" visible.

Four actions, rather than four tools. The surface is small on purpose (tool-selection recall
degrades past ~30) and these four are one workflow: see what is there, read it (or look at it a
different way), look at it again. `frontier` already set the precedent with four.

No `zoom` action. Fracture is automatic — a region too big for one read decides for itself
whether it is one long conversation (continue it in parts) or several wearing one label (split it).
"Fracture" is not a word a user has, so a capability behind that name would sit here unpressed and
the rail would appear to have a feature it did not have.
"""
from __future__ import annotations

_ACTIONS = ("preview", "read", "render", "lens", "watchlist")


def register_sitting_tools(mcp) -> None:

    @mcp.tool()
    def sitting(action: str = "preview", query: str | None = None,
                sitting_id: str | None = None, atom_ids: list[str] | None = None,
                floor: float | None = None, budget_tokens: int | None = None,
                lens: str | None = None, claim: str | None = None,
                add: list[str] | None = None, drop: list[str] | None = None) -> dict:
        """Read one topic of the user's knowledge base end to end, in publication order — as
        standing research queries, a briefing, a trajectory, or a search for what contradicts or
        is missing from it.

        Reach for this when the user wants to know what their saved material actually SAYS about a
        subject over time — "what have I been collecting on agentic payments", "how has the
        thinking on X moved", "read everything I've saved about Y", "what's in my blind spots". It
        is the opposite of the search tool: search finds the few best-matching items, this
        assembles EVERY item on a topic and reads the whole set in date order. Search answers
        "where is it"; this answers "what happened".

        Do not reach for it to look something up, to answer a factual question, or to find a
        specific post — that is the knowledge-base search tool, and it is free and instant. This
        assembles a large context and, on `read`, makes a model call (the other actions do not).

        The five actions:

          • `preview` (FREE, the default) — name a topic and find out what is actually there. It
            grows the region and reports its size, the stretch of time it covers, how many people
            wrote it, a few of the items by name, and anything about its shape that would make a
            read disappointing. It calls no model. Costs one metered embedding for a typed phrase
            (a fraction of a cent) and nothing else.
            This is not a permission step. It is here because a phrase can resolve to four items or
            to two hundred and the user cannot tell which in advance — the size is a property of
            what they have saved, not of the words they typed. Skip it whenever the user has
            already said to go ahead.
            A preview alone queues nothing — consumption subscribes a region, construction does
            not. Nothing reads it, and nothing spends, until `read` or `lens` actually consumes it.

          • `read` (SPENDS) — reads the assembled region with a model. A topic too big for one
            sitting is read in PARTS, oldest stretch first; each read carries forward the claims
            every earlier part established and is asked to confirm, revise or refute them. Two lenses, pick with `lens`:
              - `queries` (default) — emits standing research queries from the material. Those then
                run on a schedule against papers, repos and datasets, catching what gets published
                NEXT in that thread.
              - `claims` — extracts 8-15 falsifiable claims the material actually makes, each one
                naming specifics (systems, numbers, dates), citing every atom that supports it, and
                stating what observation would prove it WRONG. Use this when the user wants to know
                what their saved material actually establishes, not what to watch next.
            Pass a `sitting_id` from a preview, or pass `query` to build and read in one step. The
            two lenses spend and read INDEPENDENTLY — reading a region for `claims` does not use up
            or block its `queries` read, and the reverse holds too.

          • `render` (FREE) — hand back a region that was already built, as the document a reader
            would see. Nothing is re-grown and nothing is re-read.

          • `watchlist` (FREE) — the standing questions currently being watched on the user's
            behalf, with how often each runs, how many times it has come up, and whether the user
            typed it or a read of their material proposed it. Pass a `sitting_id` or `query` to see
            one region's; pass neither for everything.
            SHOW THIS ONLY WHEN ASKED — "what am I watching", "show my watchlist", "did anything
            change". Never volunteer it at the start of a session or alongside unrelated work.
            `add` puts questions the user names onto the list; those never decay and are removed
            only by `drop`. `drop` retires a question EVERYWHERE — a question two regions both
            watch is retired for both, because the list is one list of questions, not a copy per
            region. Say so before dropping something the user did not name precisely.

          • `lens` (SPENDS only on material never lensed before) — hand back an `instruction` plus
            a `document`, and read them YOURSELF, right here in this conversation, to answer the
            user directly. This is how you answer a question ABOUT the material rather than
            generating queries against it.
            The `document` is NOT the region's raw text. A topic read across several sittings is
            summarised one stretch at a time, and what comes back is those summaries labelled with
            the dates they cover — so you are joining stretches, not re-reading everything. Each
            stretch is summarised once ever, so asking the same lens again is free, and asking a
            DIFFERENT question of the same lens is free too. Only material that has never been
            lensed this way costs anything. Pass `lens` to pick which reading:
              - `briefing` — what this material actually says, as knowledge, not a table of contents.
              - `trajectory` — how the thinking on this topic MOVED over time: what changed,
                reversed, or got abandoned.
              - `disconfirmation` — what in this material would UNDERMINE a belief. Pass `claim`
                with the belief being tested; without one, it red-teams the material's own apparent
                thesis.
              - `gaps` — answer a question using only this material, and if nothing here answers
                it, the CLOSEST it comes and why that falls short (never a bare "nothing here").
                Pass `claim` as the question.
              - `sprouts` — everything no sitting has ever read: true orphans, unread regions,
                fracture leftovers. This is what "what's in my blind spots" means. Needs no
                `sitting_id`, `query`, or `atom_ids` — it is not about one topic.
            Like `read`, pass a `sitting_id` from a preview or a `query`/`atom_ids` to build one in
            the same call (`sprouts` needs neither). The answer YOU give is never written anywhere
            — no queries, no table, no record of it — because it is about the topic as it stands
            today and would be wrong the moment anything is added. This is a conversation, not a
            rail.

        What comes back from a `read`, lens `queries`: a `consensus` — how the conversation moved,
        what reversed, what is unresolved — plus the queries it emitted. Show the user the
        consensus. It is the part written for a human.

        What comes back from a `read`, lens `claims`: a `claims` list, each one `{claim,
        falsified_by, atom_ids}`. Show the user the claims themselves — `falsified_by` is what lets
        them decide whether to believe one, so surface it alongside the claim rather than dropping
        it.

        What comes back from a `lens`: `instruction` and `document`. Read `document` following
        `instruction` and write the answer yourself — there is no second call to make, and nothing
        here reads the document for you.

        Read the `warnings` in a preview back to the user. They say when a region is a poor fit
        for the question — a region spanning three days has no arc to find, a region that is 85%
        one author generates queries pointing back at that author's own work. Pass the `lens` you
        intend to use to `preview` and the warnings are specific to it (a short span breaks
        `trajectory` and barely touches `briefing`). They are advisory: nothing here refuses to
        read, because whether the region is right depends on what the user is asking and only they
        know that.

        A region is read once per lens (by `read`). A second `read` of the same `sitting_id` with
        the SAME lens is refused as already read — it would be the same input for the same money.
        The two `read` lenses do not share this guard: a region read for `queries` can still be
        read for `claims`, and the reverse. A region is re-read (same lens) when it has GAINED
        enough new material to be worth redoing, and the rail decides that on its own; to force it,
        preview the same phrase again, which grows a fresh region over the corpus as it is NOW.
        `lens` (the action) has no such limit — each stretch is summarised once and reused after
        that, so reading the same region's `briefing` twice costs nothing the second time.

        Args:
            action: "preview" (default, free) | "read" (spends) | "render" (free)
                | "lens" (spends only on never-lensed material) | "watchlist" (free).
            query: the topic, in the user's own words. It does NOT have to be wording that appears
                in their saved material — the match is by meaning, so "prediction markets" finds a
                thread nobody in the corpus ever called that. If nothing in the corpus is close
                enough, it says so rather than assembling something plausible out of near-misses.
            sitting_id: an id from an earlier preview. Required for `render`; on `read`/`lens` it
                is the alternative to `query` (not needed for `lens="sprouts"`).
            atom_ids: seed from specific items instead of a phrase — the region around THESE.
                Free (no embedding). Use when the user points at something they just saw.
            floor: how tightly related an item must be to join the region, 0-1. Leave unset. Higher
                is narrower, and a value below the corpus' measured noise ceiling is raised to it —
                down there an admission is not distinguishable from unrelated text. Two builds of
                one topic at different floors are two different regions with separate read
                histories.
            budget_tokens: cap on how much is read in one sitting. Leave unset.
            lens: for `preview`, which lens's warnings to compute (default "queries"). For action
                `read`, which API lens to spend on: "queries" (default) | "claims". For action
                `lens`, which reading to return: "briefing" | "trajectory" | "disconfirmation" |
                "gaps" | "sprouts".
            claim: the belief or question being tested, for `lens="disconfirmation"` or
                `lens="gaps"` only. Optional on both.
            add: for `watchlist` only — questions to start watching, in the user's own words.
            drop: for `watchlist` only — questions to stop watching, matched on their text.
                Retires them for every region, not just this one.

        Returns a dict whose `status` is one of "preview", "ok" (a read or lens landed), "skipped"
        (with a `reason` — most often already read), "failed", or "error". A "skipped" or "failed"
        `read` result wrote no queries and left the region unread, so it can be retried once the
        reason is fixed.

        A `scheduler` key appears ONLY when the rail that drains the read queue needs a human — it
        has never run, or it stopped after repeated failures. Read its `note` back to the user when
        it is there; its absence is the normal case and means nothing needs saying.
        """
        if action not in _ACTIONS:
            return {"status": "error",
                    "reason": f"action must be one of {', '.join(_ACTIONS)} (got {action!r})"}

        from pipeline.kb import schema, sitting_builder as sb, sitting_render as sre, sitting_surface as ss

        conn = schema.connect()
        try:
            if action == "render":
                if not sitting_id:
                    return {"status": "error", "reason": "render needs a sitting_id"}
                try:
                    return {"status": "ok", "sitting_id": sitting_id,
                            "document": sre.render_sitting(conn, sitting_id)}
                except KeyError:
                    return {"status": "error", "reason": f"no sitting {sitting_id!r}"}

            if action == "watchlist":
                return _watchlist(conn, sitting_id=sitting_id, query=query,
                                  add=add, drop=drop)

            if action == "lens":
                return _lens(conn, sb, lens=lens, sitting_id=sitting_id, claim=claim, query=query,
                            atom_ids=atom_ids, floor=floor, budget_tokens=budget_tokens)

            if action == "read" and sitting_id:
                return {**_read(conn, sitting_id, lens=lens), **_dispose(conn)}

            if not (query or atom_ids):
                return {"status": "error",
                        "reason": "preview needs a query (a topic) or atom_ids to seed from"}

            try:
                built = _build(conn, sb, query=query, atom_ids=atom_ids, floor=floor,
                               budget_tokens=budget_tokens)
            except sb.SeedError as e:
                # The honest failure, and the one worth passing through verbatim: "nothing matched
                # your phrase" and "this corner of the corpus is empty" are different facts, and
                # an empty region would render them the same.
                return {"status": "error", "reason": str(e)}

            report = ss.scope(conn, built, lens=lens or "queries")
            if action == "preview":
                return {"status": "preview", **report, **_dispose(conn)}
            # `read` with a fresh phrase: the region was just built above, so this reads THAT.
            return {**_read(conn, built["sitting_id"], lens=lens), "scope": report,
                    **_dispose(conn)}
        finally:
            conn.close()


def _build(conn, sb, *, query, atom_ids, floor, budget_tokens) -> dict:
    """Resolve the seed and grow the region. Free except for one embedding on a typed phrase.

    The embedder is built ONLY for the phrase arm — `atom_ids` seeds are entirely local, and
    constructing an embedder loads a model client. Same shape as `hopper_tools` skipping it on a
    preview and `opyt_core/kb.py` skipping it for a bm25-only search.

    PERSISTS. Building is free and a sitting is an EVENT, so a duplicate build appends a row rather
    than clobbering one — which is what keeps a region that was read from silently becoming unread
    again. The row is also what makes the region addressable by id for a later `read` or `render`.
    """
    embedder = None
    if query and not atom_ids:
        from pipeline.kb.embed import get_kb_embedder
        embedder = get_kb_embedder()
    seed = sb.resolve_seed(conn, query=query, atom_ids=atom_ids, embedder=embedder)
    dials = {}
    if floor is not None:
        dials["floor"] = floor
    if budget_tokens is not None:
        dials["budget_tokens"] = budget_tokens
    return sb.build_sitting(conn, seed, **dials)


def _lens(conn, sb, *, lens, sitting_id, claim, query, atom_ids, floor, budget_tokens) -> dict:
    """Job L. Resolve or build the region (unless `sprouts`, which is not about one region) and
    hand back `{instruction, document}` for the host to reduce in-session.

    ⚠️ THIS SPENDS, on cache misses only. `sitting_lenses.read_lens` is MAP-REDUCE since
    2026-08-24: one model call per part that has never been mapped under this lens, and nothing at
    all for a part that has. A closed part is frozen, so it is mapped once per lens EVER and steady
    state on any region is one call for the open tail. `sprouts` still calls nothing — it has no
    chain and no part to cache against. The REDUCE is always free: the host reading this tool's
    result is the only reader the joined document ever gets.

    Building here reuses `_build`, so a bare `lens` call with a fresh `query` behaves exactly like
    `read` with a fresh `query`: one call builds AND reads. `sprouts` skips this whole branch — it
    has no seed, so `sitting_id`/`query`/`atom_ids` are ignored for it rather than required.
    """
    from pipeline.kb import sitting_lenses as sl
    if not lens:
        return {"status": "error", "reason": f"lens needs a lens name: {', '.join(sl.LENSES)}"}
    try:
        sid = sitting_id
        if lens != "sprouts" and not sid:
            if not (query or atom_ids):
                return {"status": "error",
                        "reason": f"lens={lens!r} needs a sitting_id, or a query/atom_ids to "
                                 f"build one (lens='sprouts' needs neither)"}
            built = _build(conn, sb, query=query, atom_ids=atom_ids, floor=floor,
                           budget_tokens=budget_tokens)
            sid = built["sitting_id"]
        res = sl.read_lens(conn, lens, sitting_id=sid, claim=claim)
        return {**res, **_dispose(conn)}
    except (sl.LensError, sb.SeedError) as e:
        # Caller mistakes — an unknown lens, a missing sitting_id, or an unresolvable phrase — not
        # runtime failures. Nothing was attempted, so this is "error", never "failed".
        return {"status": "error", "reason": str(e)}
    except KeyError:
        return {"status": "error", "reason": f"no sitting {sitting_id!r}"}


def _dispose(conn) -> dict:
    """Fire the read-queue scheduler NOW if it has work, and surface its health only when it is bad.

    Not redundant with the session-open spawner: that one coalesces hourly, which would suppress
    the spawn right when it matters most (someone just pointed at a region). This only changes WHEN
    the read happens, never whether — the scheduler drains pointed regions unattended regardless.
    Silent on the happy path so a fresh spawn doesn't read as "the scheduler has never run."
    """
    try:
        from pipeline.kb import sitting_scheduler as sch
        h = sch.health(conn)
        # A tripped breaker means the child would exit without reading, so do not fork one.
        spawned = bool(h["claims_waiting"] and not h["breaker_open"]
                       and sch.spawn_sitting_scheduler(force=True))
        return {"scheduler": h} if (h["needs_attention"] and not spawned) else {}
    except Exception:
        # FAIL-SAFE: a scheduling hiccup never breaks a tool call. The read the caller asked for has
        # already happened by the time this runs.
        return {}


def _watchlist(conn, *, sitting_id, query, add, drop) -> dict:
    """The watchlist surface: list, add, drop. Calls no model and grows no region.

    PULL-ONLY (AMENDED 2026-08-25, David). This is reachable when the user asks for it and inside
    the result of a read they themselves triggered — never pushed at session open. Standing queries
    already run quietly; announcing them unprompted is the recital the frontier surface's own
    etiquette exists to prevent. Pull-only stays coherent because the `generator='user'` exemption,
    not the announcement, is what protects a question the user added.

    Scoping to a region resolves the region WITHOUT building one: an existing `sitting_id` names it,
    a phrase names it only if a region for that phrase already exists. A watchlist request must not
    quietly buy an embedding and mint a new region as a side effect of asking what is being watched.
    """
    from pipeline.kb import frontier_queries as fq, sitting_reader as sr, sitting_store as sst

    gen, scope = None, "everything"
    if sitting_id:
        s = sst.get_sitting(conn, sitting_id)
        if s is None:
            return {"status": "error", "reason": f"no sitting {sitting_id!r}"}
        gen, scope = sr.generator_for(s["seed_ref"]), s["seed_ref"]
    elif query:
        gen, scope = sr.generator_for(query), query
        if not fq.active_queries(conn, generator=gen):
            return {"status": "error",
                    "reason": f"nothing is being watched for {scope!r} — read that region first "
                              f"(action='read'), or ask for the whole watchlist with no query"}

    added = [t for t in (add or []) if fq.add_user_query(conn, t)]
    dropped = [t for t in (drop or []) if fq.retire_query(conn, t)]
    missed = [t for t in (drop or []) if t not in dropped]

    out = {"status": "ok", "scope": scope, "watching": fq.watchlist(conn, generator=gen)}
    if added:
        out["added"] = added
        out["added_note"] = ("these run until you drop them — nothing retires a question you "
                             "added yourself")
    if dropped:
        out["dropped"] = dropped
        out["dropped_note"] = "retired everywhere, not only for this region"
    if missed:
        # SHOW DECIDED, DON'T HIDE: a drop that matched nothing is a fact the user needs, not a
        # silent no-op that reads as success.
        out["not_found"] = missed
    return out


def _read(conn, sitting_id: str, *, lens: str | None = None) -> dict:
    """Delegate to the reader for `lens` (default `"queries"`), which never raises and records
    every outcome as a run.

    `read`'s two lenses are the two API lenses (D7, D20) — `queries` (Job 4) and `claims` (Job N).
    They are separate calls answering separate questions, each with its OWN once-only read guard, so
    reading one never blocks or unblocks the other on the same `sitting_id`. The other four lens
    names (`briefing`, `trajectory`, `disconfirmation`, `gaps`, `sprouts`) are the `lens` ACTION, not
    a value here — `read` never routes to them, because they call no model and store nothing (D17),
    which is a different contract than this action promises ("spends", records a run).

    Transport-agnostic by construction — both readers go through
    `reader_core.call(resolve_backend(), ...)` and nothing here may assume which transport that
    resolves to (metered API by default, or an opted-in subscription transport). That's why the
    docstring above says "spends" and never "free" or "a few cents".
    """
    lens = lens or "queries"
    from pipeline.kb import sitting_claims as scl
    if lens == "claims":
        return scl.read_claims(conn, sitting_id)
    if lens != "queries":
        from pipeline.kb import sitting_lenses as sl
        return {"status": "error",
                "reason": f"read does not take lens={lens!r} — 'queries' or 'claims' spend and "
                         f"emit a record; {', '.join(l for l in sl.LENSES if l != 'claims')} are "
                         f"the `lens` action instead (spends only on never-mapped material)"}
    # The same ritual the scheduler runs, from the one place it is spelled: a tool-initiated read
    # closes a part exactly as a scheduled one does.
    res = scl.read_part(conn, sitting_id)
    if res.get("status") == "ok":
        # PULL-ONLY, and this is one of its two doors: a read the USER triggered ends at a natural
        # decision point, so the diff and the list it changed are shown here and nowhere else. The
        # scheduler's own reads record the same diff and surface nothing — no unprompted recitals.
        from pipeline.kb import frontier_queries as fq
        res["watching"] = fq.watchlist(conn, generator=res.get("generator"))
        res["watchlist_note"] = ("this is what is now being watched for this topic — say so to "
                                 "add or drop any of it")
    return res
