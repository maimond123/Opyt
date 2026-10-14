"""
mcp_server/oracle_tools.py — the Oracle screen and `add_oracle`, the atom-KB roster write surface.

One action-dispatched tool, `oracle`; the whole screen happens in chat. Each action calls one
flow in `pipeline/oracles/`, where degrade-open and fail-safe are enforced:

  • screen     — `screen.screen`: the candidates in groups to print, each with one short
                 reason, and the two questions to ask. Nobody is selected by default.
  • candidates — `candidates.candidates`: the same people ranked by what they write. The other
                 half of the promotion decision.
  • confirm    — `admit.confirm_picks`: commit picks (ranked `canonical_ids` and/or raw
                 `add_handles`) and record what each is owed. It pulls
                 nothing: the background scrapers do.
  • ingest     — `admit.ask_again`: a wider window, subjects, a top-up, a rediscovery.
  • progress   — `progress.progress`: where each pick stands, by milestone.
  • map        — `people_map.people_map`: the people the user reads, grouped by what they write,
                 and which Oracles sit in each group.

`add_oracle` is `admit.add`; it reads the windows before it opens the store, so a bad preset
refuses without touching it.
"""
from __future__ import annotations

from pipeline.kb import schema
from pipeline.oracles import admit, candidates, people_map, progress, screen, windows
from pipeline.timeparse import utc_now


def register_oracle_tools(mcp) -> None:

    @mcp.tool(annotations={"title": "Choose Oracles", "readOnlyHint": False,
                           "destructiveHint": False, "openWorldHint": True})
    def oracle(action: str = "screen", canonical_ids: list[str] | None = None,
               add_handles: list[str] | None = None, top_n: int = 30,
               limit: int = 10, source: str | None = None,
               force: bool = False, x_lookback: str | None = None,
               web_lookback: str | None = None, scholar_lookback: str | None = None,
               scholar_topics: list[str] | None = None, query: str = "",
               min_signals: int = 1, oracle: str | None = None) -> dict:
        """Choose who to trust: turn the people you already curate (follows, Lists, bookmarks,
        subscriptions, likes) into your **Oracles** — the sources the KB deep-ingests and roots
        trust on. Runs entirely in chat.

        NOT the setup tool. `onboard` readies the machine (keys, the worker, the first curation
        pull); this decides WHO is in. Reach for this when the user asks who to trust, wants to see
        or change their people, or asks whether their sources are current.

        FLOW: call `action='screen'` → print its `groups` exactly as given and nothing else: each
        `title`, one line per candidate (`name — reason`), and `+ N more` when `more` is above 0
        → ask its `ask` (who to pick; nobody is selected until the user names them) and, only
        when `lookback_options.x` is present, its `ask` as a second, separate question → call
        `action='confirm'` with the picked `canonical_ids` and the X answer as `x_lookback`.
        No import summary, no counts, no sub-groups, no paragraphs about who was left out.
        Confirm records what each person is owed and returns at once; nothing is pulled inside
        the call. Its `message` says what to tell the user about what is still to come — it
        promises background work only when OPYT's background helper is actually running.

        Why the X question is not optional: the default is only 6 months, so a user who wanted
        their Oracle's longer history gets a fraction of it and is never told. Offer exactly the
        presets in `lookback_options.x`, don't invent your own.

        To add someone NOT in the list, pass them in `add_handles` — an X @handle, or ANY http
        URL: a Substack, a personal blog, a newsletter, a site. A non-Substack home becomes a
        `blog:{host}` entity and is ingested through the blog adapter. They're resolved and added
        on the spot. This is the whole path for a user who follows named writers rather than
        accounts on a platform — there is no list to screen, so nothing else surfaces them.

        `action`:
          • "screen"  (default) — the candidates in `groups`, in print order: one per source
                       (X, Substack, papers); you list plain organisations, media and projects in
                       one last group (the payload's `note` says how). Each group
                       holds its top `limit` people in rank order and `more`, the count of the
                       rest. Each candidate carries a `reason` ("you follow, bookmarked 12") and
                       its `canonical_id` (pass these to confirm). Someone on two sources is
                       listed once, under the stronger, and the reason says "also on" the other.
                       Once the user has Oracles, a group "Named by your Oracles" follows: people
                       their posts quote, mention or link who are in none of the user's lists.
                       They have no `canonical_id`, so a pick goes in `add_handles` by handle. A
                       listed candidate the Oracles name says "named by N of your Oracles".
          • "confirm" — commit Oracle picks and record how far back each is owed (`x_lookback`,
                       `web_lookback`, `scholar_lookback`, and `scholar_topics` for one pick).
                       `canonical_ids` = the ones the user kept (verbatim from screen);
                       `add_handles` = raw X handles or any http URL to add (resolved-at-confirm).
                       Idempotent, and it returns at once: nothing is pulled inside the call.
                       Returns {confirmed, unresolved, unknown, total_oracles, picks, message} —
                       surface `unresolved` (a lookup couldn't find it). Every pick the user
                       names is confirmed as given, company and publication accounts included:
                       never hold one back, question it or suggest an alternative. A
                       `needs_topics` entry is a research venue: read its `note`, because nothing
                       of it is pulled until the user picks its subjects. `picks` is where each
                       person stands (see `progress`); `message` is what to tell the user about
                       what is still to come.
          • "ingest"  — ask again for confirmed Oracles (all, or the `canonical_ids` you pass): a
                       wider window, subjects (`scholar_topics`), a top-up (`x_lookback=
                       'since_last'`: their newest posts, checked now) or a fresh discovery of
                       their other profiles (`force`). Windows only ever widen. Returns at once,
                       shaped like confirm's `picks` and `message`.
          • "progress" — WHERE EACH PERSON STANDS. One entry per Oracle (narrow with
                       `canonical_ids`):
                         · `newest_posts` — their newest X posts are in (null: no X timeline);
                         · `archives` — what they publish beyond X is in: Substack, blog, code,
                           papers (null: nothing was asked of it);
                         · `reach` — how far back their X goes (`covered_from`) against how far
                           it is owed (`asked_from`), `reached` once it gets there;
                         · `asked` — the windows on record;
                         · `stuck` — present only when the USER can do something: `cause`, and
                           the `step` that clears it. Offer the step; nothing else needs them.
                         · `other_sources` — present when OPYT found more of their writing:
                           tell the user its `message` as given, once. They are already being
                           pulled; nothing is asked.
                       `waiting_on` says when X will next move, when it is waiting. It waits up
                       to ~40 seconds for something to change and returns the moment it does, or
                       at once when nothing is left to wait for.

        ⚠️ THERE IS NO "COMPLETE", AND YOU NEVER HAVE TO CALL `progress` FOR THE WORK TO HAPPEN.
        Nothing is pulled inside any call, and nothing waits on `progress`: what lands while the
        user is away rides back on their next call, once. Call `progress` when they ask how it is
        going, or to tell them as people land: say who is in, in their terms. `false` means "not
        yet" — never that something is missing, and never anything about why.

        NEVER say a pull "timed out" because `progress` returned with nothing new. It answers on
        a clock so that YOU can speak; the work neither sees that clock nor is cut by it. A
        writer deep inside a blog archive produces no news for minutes at a time, and that is
        the system working.

        IF THIS CALL EVER RETURNS AN ERROR INSTEAD OF A RESULT — a timeout, a transport failure,
        a connection drop — three rules, and they are absolute:

          1. **Do not start another ingest.** Not the same one, and NOT a smaller one either:
             calls to this server run one at a time, so a "smaller" second call waits for the
             first to finish and then fails having done less. Splitting the work is the
             mitigation that GUARANTEES the failure it was meant to avoid — measured twice.
             An ask is recorded the moment it is made, and the work runs in the background. Call
             `oracle(action='progress')` — that is exactly what it is for.
          2. **Never say "timed out", "the request failed", or "error" to the user.** They did not
             make a request; they asked for a library. Say you are checking where it stands, and
             call `progress`. No mechanism, no apology, no offer to retry.
          3. **Never explain an absence you did not measure.** If a writer has fewer posts than
             expected, or something looks missing, do NOT attribute it to a timeout or a rate
             limit. Those are guesses, and they are usually wrong — the real causes are a
             content-quality gate, an archive the crawler has not reached, or a pull still in
             flight. Look before you say (`oracle(action='screen')`, `aggregate`), or say you
             don't know yet.
          • "candidates" — the OTHER half of the promotion decision. `screen` ranks people by how
                       hard the USER vouched for them (distinct curation signals); this ranks the
                       same people by what they actually write. Pass `query` to ask a topic
                       question ("who writes about agent memory"); omit it to list who is there.
                       Evidence comes from three arms and every row names which, in `basis`:
                         · "probed" — a light sample of their own posts (~25 X posts, or ~5
                           Substack titles for someone with no X).
                         · "saved"  — a post the USER saved, so it IS a knowledge-base atom and is
                           citable (open it first). Usually exactly ONE post, and it is a positive
                           example rather than a sample of their output: it matched partly BECAUSE
                           the user kept it. Good evidence for "worth reading more of", weak
                           evidence for "this is what they mostly write about".
                         · "mentioned" — people an ORACLE's own posts quote, @mention or link,
                           with the Oracle's words in `evidence[].words` and the way they named
                           them in `kind` (quote / mention / link). The text is ABOUT the person,
                           not by them. A quote can be agreement or pushback: judge the stance by
                           reading the Oracle's words, never from the kind. `canonical_id` null
                           means they are not a candidate yet (add_oracle takes the handle).
                           Pass `oracle` to ask about one Oracle, and this is the only basis
                           listed: "who did Soren disagree with about X", "who does Imas work
                           with".
                       Scores are comparable only WITHIN a basis; the two are interleaved by rank,
                       never added. Read back `no_local_material` — a candidate absent because
                       nothing of theirs is stored looks exactly like one whose writing did not
                       match, and those need opposite actions.
          • "map" — the people the user reads, grouped by what they write: each group's
                       `words`, its `central` people, and which of the user's `oracles` sit in
                       it. Name each group from its words and people, and leave out a group with
                       no clear subject (the payload's `note` says how). A group with no Oracle
                       is where a new Oracle would add the most. Takes no other argument; up to
                       ~15 seconds on a large store.

        Is my stuff still current? You do not ask — `screen` answers it unasked, under
        `oracle_freshness`: per-Oracle, per-source `last_pulled_at` / `hours_overdue` /
        `never_refreshed`, worst first, and for a source that keeps failing its `failures`,
        `given_up` (no longer tried until the user asks for that Oracle again), `retry_at` and
        `last_error`. If `needs_attention` is true, read its `note` out — that flag means
        refreshing has fallen past twice its target, a silent failure otherwise.

        A `model_routing` key, when present, lists models that are `dead` (no OpenRouter
        provider survives the deny-list — the stage using them cannot run) or `fragile` (one
        surviving provider — one withdrawal from dead). Read it out: it is absent whenever
        everything is routable, so its presence IS the news.

        Args:
            action: "screen" | "candidates" | "confirm" | "ingest" | "progress" | "map".
            query: candidates — a topic question to rank people by ("agent memory", "biotech
                funding"). Omit for "who has been sampled".
            min_signals: candidates — only people with at least this many distinct curation
                signals (default 1 = everyone in the list). Above 1 drops the "mentioned" arm,
                whose people usually have no signal of the user's.
            oracle: candidates — one Oracle's handle or name. THE ANSWER TO "who does this Oracle
                interact with": `candidates` lists only the people that Oracle's own posts quote,
                mention or link (add `query` for one subject), and `coauthors` lists who they
                co-wrote papers with (name, shared papers). Unknown or ambiguous → `error` plus
                the `oracles` to choose.
            canonical_ids: confirm — the kept candidates' canonical_ids; ingest — which confirmed
                Oracles to ask again for (omit = all confirmed — usually NOT what you want for a
                top-up, so name the person); progress — whose status to report (omit = all).
            add_handles: confirm — people to add beyond the ranked list: raw X @handles, or any
                http URL (Substack, blog, newsletter, personal site).
            top_n: candidates — how many people to return (default 30). NOT a screen knob —
                `screen` has its own, `limit`.
            limit: screen — how many people each group lists (default 10); the rest are counted
                in the group's `more`. A screen returns at most 80 people in all, so a bigger
                number is clamped and reported back as `limit_clamped`. To go further into one
                group use `source=`; to find ONE person use
                `oracle(action='candidates', query='...')`.
            source: screen — list only people who reach the user through ONE source (a group's
                `source`: 'x', 'substack', 'papers', or 'named' for "Named by your Oracles").
                THE ANSWER TO "show me more of my Substack writers". Membership, not exclusivity: someone you follow on X and subscribe to
                on Substack is listed under Substack here. Ranks do not move — this narrows who
                is listed, never how anyone scored.
            force: ingest — re-run DISCOVERY for these people. Reach for this when a person's
                sources look wrong or incomplete, e.g. they started a blog since they were added.
            x_lookback: confirm/ingest — when X is connected, how far back to pull each Oracle's
                X timeline: "6mo" (the default), "1yr", "2yr". Hard-capped at 2 years whatever
                you pass. Ask the user only after X is connected — not for cost (the pull is
                free), but because the default truncates.
                "since_last" is the top-up: check for what has appeared since this Oracle was
                last pulled, now, and nothing else. Use it whenever the user wants someone
                brought current rather than re-asked for.
            web_lookback: confirm/ingest — how far back to pull the Substack/blog archive:
                "1yr", "2yr", "5yr", "all" (the default). Omit unless the user asks.
            scholar_lookback: confirm/ingest — how far back to pull their papers: "2yr", "5yr",
                "10yr", "all" (the default). Free and abstract-only, so the whole corpus is the
                default — a researcher's foundational paper is usually not their most recent.
                Ask this one WITH NUMBERS: `add_oracle`'s preview reports `scholar_counts`.
            scholar_topics: confirm/ingest — narrow their papers to these OpenAlex topic ids,
                taken from `add_oracle`'s preview under `scholar_topics`. The SUBJECT filter, and
                usually the more useful of the two: a researcher's record runs to 174 subjects
                and a venue's to 200+, and the user wants a handful. It PERSISTS onto the source —
                every future refresh pulls only these subjects — so pass it only for the ONE
                Oracle named. Omit it to leave an existing choice untouched (what a top-up wants);
                pass `[]` to go back to a researcher's whole corpus. A research VENUE has no
                whole corpus to go back to: without subjects, none of its papers are pulled.
        """
        conn = schema.connect()
        try:
            if action == "screen":
                return screen.screen(conn, limit=limit, source=source)

            if action == "candidates":
                return candidates.candidates(conn, query=query, top_n=top_n,
                                             min_signals=min_signals, oracle=oracle)

            if action == "confirm":
                return admit.confirm_picks(conn, canonical_ids=canonical_ids,
                                           add_handles=add_handles, x_lookback=x_lookback,
                                           web_lookback=web_lookback,
                                           scholar_lookback=scholar_lookback,
                                           scholar_topics=scholar_topics)

            if action == "ingest":
                return admit.ask_again(conn, canonical_ids=canonical_ids, force=force,
                                       x_lookback=x_lookback, web_lookback=web_lookback,
                                       scholar_lookback=scholar_lookback,
                                       scholar_topics=scholar_topics)

            if action == "progress":
                return progress.progress(conn, canonical_ids=canonical_ids)

            if action == "map":
                return people_map.people_map(conn)

            return {"error": f"unknown action {action!r} — use 'screen', 'candidates', 'confirm', "
                             f"'ingest', 'progress', or 'map'."}
        finally:
            conn.close()

    @mcp.tool(annotations={"title": "Add an Oracle", "readOnlyHint": False,
                           "destructiveHint": False, "openWorldHint": True})
    def add_oracle(reference: str, confirm: bool = False,
                   x_lookback: str | None = None, web_lookback: str | None = None,
                   scholar_lookback: str | None = None,
                   scholar_topics: list[str] | None = None, force: bool = False) -> dict:
        """Add a person to your knowledge base as an **Oracle** — a trusted source OPYT
        deep-ingests from their verified footprint and roots trust on. X is included only when the
        user has connected X; discovering an X profile does not enable a pull. This
        is the atom-KB "add a person" — the only one: the old vault-era `add_person` tool, which
        wrote the legacy vault and couldn't admit a Substack/blog-rooted person, is retired.

        SAFE, TWO-PHASE — always preview before you add:
          • FIRST call with confirm=False (the default) → a PREVIEW. It resolves `reference` and
            returns who they are (name, bio, followers) — or, if already known, their roster entry
            — and writes NOTHING. Read it back to the user to confirm it's the right person. If the
            reference doesn't resolve you get `unresolved` and there's nothing to confirm.
          • THEN, once the user agrees, call again with confirm=True → adds them to the roster and
            records how far back to pull. It returns at once; nothing is pulled inside the call,
            and `oracle(action='progress')` shows where they stand. Never call confirm=True
            without showing the preview first — the guard is against adding the WRONG PERSON,
            whose writing then fills the library.
          • A company, team or publication account is a valid Oracle when the user wants it.
            Never push back on one or suggest an alternative; the preview only checks it is the
            account they meant.

        `reference` is polymorphic: an X @handle ("@karpathy"), a Substack/blog/site URL
        ("https://simonwillison.net"), or a canonical_id from `oracle(action='screen')` (to promote
        a below-the-cut candidate). To add someone by NAME, YOU resolve the name → their @handle or
        URL first (your own knowledge / a web search), then pass that — there is no name-search
        endpoint.

        TWO windows, asked separately, because they answer to different constraints:
          • `x_lookback` — when X is connected: "6mo" (default) / "1yr" / "2yr". Hard-capped at
            2 years whatever you pass. Ask only after X is connected; the ~6-month default silently
            leaves most of a prolific account's history out.
            For someone ALREADY on the roster, "since_last" checks for what is new since their
            last pull, now — the top-up. Reach for it when the user says "update" or "catch up",
            not "add".
          • `scholar_lookback` — "2yr" / "5yr" / "10yr" / "all" (default). Their papers. Free
            and abstract-only, so the whole corpus is the default. The ONE selector that can ask
            with real numbers — the preview reports `scholar_counts`.
          • `web_lookback` — "1yr" / "2yr" / "5yr" / "all" (default). A durable archive, so it
            already pulls everything. Pass it only to NARROW.
        A single shared window would be wrong for one of the two by construction — it either
        over-pulls X or truncates the archive. Windows only ever widen. The result's
        `picks[0].asked` holds the windows on record — TELL the user how far back you asked for.

        Reading the result: `message` is what to tell the user about what is still to come, and
        `picks[0]` is where they stand (see `oracle(action='progress')`).

        Their other sources (a blog, a newsletter) are found in the background from what their
        own profile declares. Do NOT web-search for them: OPYT neither asks for nor accepts
        URLs found that way.

        Args:
            reference: an X @handle, a Substack/blog URL, or a canonical_id.
            confirm: False (default) = preview only, no writes; True = add them and ask for
                their writing.
            x_lookback: window for the X timeline — "6mo" | "1yr" | "2yr" (default 6mo).
            web_lookback: window for the Substack/blog archive — "1yr" | "2yr" | "5yr" |
                "all" (default all). Narrows only.
            scholar_lookback: window for their papers — "2yr" | "5yr" | "10yr" | "all"
                (default all). Free and abstract-only, so the whole corpus is the default. The
                preview reports `scholar_counts` for anyone already in the roster with an OpenAlex
                id — ask this one with the real numbers, not by reciting presets.
            scholar_topics: OpenAlex topic ids to narrow their papers to, read off the preview's
                `scholar_topics.topics`. The SUBJECT filter, and the one that usually matters
                more than the window: a researcher's record spans ~170 subjects and a venue's
                200+, and only a few are what the user is after. It PERSISTS — every later
                refresh pulls only these subjects, which is the point, since a filter applied
                once would drift back to the whole corpus within a week. Omit to leave an
                existing choice alone; pass `[]` to clear it. A research VENUE without subjects
                gets none of its papers pulled: it publishes far more than one pull holds.
            force: re-run DISCOVERY for this person. Pass this when the user says their sources
                look wrong or incomplete, e.g. they started a blog since they were added. Costs a
                full probe walk in the background; do not pass it by default.
        """
        win = None
        if confirm:
            try:
                win = windows.parse(x_lookback, web_lookback, scholar_lookback, now=utc_now())
            except ValueError as e:
                return {"error": f"{e}. Nothing was written."}
        conn = schema.connect()
        try:
            return admit.add(conn, reference, win, confirm=confirm, x_lookback=x_lookback,
                             web_lookback=web_lookback, scholar_lookback=scholar_lookback,
                             scholar_topics=scholar_topics, force=force)
        finally:
            conn.close()
