"""
mcp_server/atoms_tools.py — the atom-KB "trusted router" tools: `search`, `open`, `aggregate`.

LLM-free: the host model reasons over what they return. All three take `kb=` (omitted is the
user's own store, any other name a registered peer's) and are read-only. They are annotated
open-world all the same: a peer can be served over HTTPS, and `open` reads a paper's PDF from
the web. Notices about the install ride their answers through `mcp_server/riders.py`, not from
here.

Named `search`, not `opyt_search`: each namespacing client adds its own prefix.
"""
from __future__ import annotations

from pipeline.retrieval.reads import kb_aggregate, kb_open, run_kb_search


def register_atoms_tools(mcp) -> None:
    @mcp.tool(annotations={"title": "Search the knowledge base", "readOnlyHint": True,
                           "destructiveHint": False, "openWorldHint": True})
    def search(query: str, tags: list[str] | None = None, what_kind: str | None = None,
               source_type: str | None = None, who: str | None = None,
               who_id: str | list[str] | None = None, date_from: str | None = None,
               date_to: str | None = None, entry_mode: str | list[str] | None = None,
               k: int = 8, kb: str | None = None) -> dict:
        """Route to the most relevant ATOMS in the trusted knowledge base (the user's saved
        opinions + tracked artifacts). This is a ROUTER, not an answer: each hit is a thin
        card — an excerpt of the item + a pointer (source_url / atom_id) —
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
          • `trace` — VALUES about what the ENGINE did: `candidates`/`ranked`/`showing`,
            `cutoff`, `fts_query`, `filters` as applied.
          • `frontier_atoms` — a SECOND, separately ranked list: papers and repos the watches of
            the user's topics found and the host kept, rather than saved or written by anyone
            they follow. Present only on a default call (see `entry_mode`). Same card shape as
            `hits`, capped at 8. Offer it as "and from the topics
            you follow…", never merged into `hits`: its scores are ranked against other
            frontier atoms and mean nothing next to theirs.

        One more key may ride a search of your own store: `latest`, how many items arrived since
        the user last looked and which topics they follow have finds waiting for your check. It
        rides every such search until acted on; say it once per conversation, as its `host_note`
        says.

        `insights` and `trace` are for your reasoning — do not recite them to the user.
        They are bare values on purpose. Use them to decide what to do next (open something,
        re-query, drop a filter, warn about a lopsided result); say the CONCLUSION in your own
        words, never the fields. Only `notices` is written to be read out.

        `score` is the hit's reciprocal rank in this answer (1, 1/2, 1/3, …), not comparable
        across calls.

        Retrieval: an optional tag/kind/source/author pre-filter, then keyword ranking (BM25).
        Ranking is pure relevance — there is no trust re-rank. Send keywords and named things,
        not a sentence or a question: the names, terms of art and distinctive words a source
        would use. Each item is matched whole on its words (stemmed; common words dropped),
        never on meaning. Several short searches with different words find more than one long
        one. When a search comes back thin, try a synonym or the term of art before telling the
        user the store has nothing.

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
            query: Keywords, not a sentence: `tokenizer Karpathy`, `x402 micropayments`. A rare
                literal token (a library or symbol name, a term of art) ranks best.
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
                published, or what those link to, and `frontier_atoms` carries what the user's
                topics kept, separately. Set it to scope the whole answer to one population
                instead: "frontier" with a larger `k` is how you dig into what the topics kept
                ("show me more of what my topics found"), and a list like ["user-saved"] narrows to
                one arrival path. Scoping returns ONE list and no `frontier_atoms` key.
            k: Max atoms to return (default 8). Measured on a real library: on a well-covered
                topic the first 8 hold about a third of the relevant results, and results 9-25
                are still about half relevant; on a narrow topic, results past 25 are mostly
                unrelated. `trace.cutoff` shows the score at the boundary, so you can tell
                whether raising it would have helped.
            kb: Read SOMEONE ELSE'S knowledge base instead of your own. Omit for yours (the
                normal case). A name here must be one this install has registered; an unknown
                one returns no hits and a notice naming the ones that exist. Every hit carries
                the `kb` it came from — "me" for your own — and an atom id means nothing outside
                its own store, so pass that same value back to `open(atom_id, kb=...)`. Attribute
                anything you repeat from a foreign hit to that knowledge base, not to the user.
        """
        return run_kb_search(query, tags=tags, what_kind=what_kind, source_type=source_type,
                             who=who, who_id=who_id, date_from=date_from, date_to=date_to,
                             entry_mode=entry_mode, k=k, kb=kb)

    @mcp.tool(annotations={"title": "Open an item's source", "readOnlyHint": True,
                           "destructiveHint": False, "openWorldHint": True})
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

        PAPERS ARE READ ON OPEN. A paper is stored as its abstract, so opening one reads its full
        text from its open PDF into THIS response (`raw` gains a `## Full text` section and
        `body_state: "complete"`); nothing is added to the knowledge base. Opening a post that
        links papers returns them too, under `papers`: each with its `full_text` while one open's
        size budget lasts, and an `open_id` for the rest — `open(open_id)` reads that paper,
        stored or not (`stored: false`). A paper whose `full_text` stays null, or that stays
        "partial", has no reachable open PDF (paywalled, or scanned without a text layer) — that
        is the real answer, not a retry prompt. A clipped paper says how much of it you have.

        A PAPER'S FIGURES ARE NOT IN `raw`. Its body is the text extracted from the PDF, and its
        charts are drawings, so only their captions come through. When the answer depends on
        what a figure shows (a curve, a value read off a plot), say so instead of guessing from
        the caption, then read the PDF itself: the `[PDF]` or `[arXiv]` link at the end of
        `raw` (an arXiv paper's PDF is `https://arxiv.org/pdf/<id>`). A fetch that returns a
        page's TEXT will not show you the figure: download the file and open it as a document
        you can see page by page. If you cannot, ask the user to attach the PDF.

        `kb` must be whatever the hit card carried. An atom id is scoped to ONE knowledge base —
        the same tweet in two people's stores is one id in each — so opening a foreign id without
        its `kb` either finds nothing or hands you your own copy of the same source.
        """
        return kb_open(atom_id, kb=kb)

    @mcp.tool(annotations={"title": "Summarize the knowledge base", "readOnlyHint": True,
                           "destructiveHint": False, "openWorldHint": True})
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
        return kb_aggregate({"tags": tags, "what_kind": what_kind, "source_type": source_type,
                             "who_id": who_id, "date_from": date_from, "date_to": date_to},
                            kb=kb, sample=sample)
