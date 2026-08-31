"""
mcp_server/oracle_tools.py — Stage-4 Oracle SCREEN, the atom-KB onboarding write surface.

ONE tool, `oracle`, action-dispatched (keeps the surface small; the audit's Fork B). It is the
FIRST atom-KB MUTATION tool — the retired `add_person` tool wrote the legacy VAULT and keyed on an
X handle, so it could never confirm a canonical (possibly Substack-rooted) Oracle. Client-agnostic
+ MCP-first: the whole SCREEN happens in chat (the host reads the ranked candidates, the user says
keep/drop), no dashboard page.

  • screen  — FREE-ish (a cents-scale classify over every unclassified candidate, batched and
              cached, so the second call is free): the ranked candidate list + pre-tick flags +
              reflected-signal provenance, for the host to narrate.
  • confirm — commit picks (ranked `canonical_ids` and/or raw `add_handles`) into the `oracles`
              table; raw handles are resolved-at-confirm. This is the Stage-4→5 handoff.
  • ingest  — Stage-5/6: for each confirmed Oracle, run `discover_profile` (which now mines their
              trusted blog/Substack for their OTHER profiles) and route the trusted profiles into
              the atom-KB via `onboard_footprint` (gate → footprint adapter → resolve; org → an
              affiliation edge). This is the caller `.guards.py` reserved — it MUST route every
              blog/substack through `eligibility.gate`, which `onboard_footprint` does.
  • candidates — the same people ranked by what they write (a light timeline sample), not by
              how hard the user vouched. The other half of the promotion decision.

`refresh` and `status` came OFF this surface 2026-08-15 (both still run; only the tool entry
points went). `sync_follows` went with them and was DELETED 2026-08-26 — it had had no consumer
since its prospect partition was retired 2026-08-11, and no runner at all. Freshness is no longer
a report you request — it rides on `screen` as
`oracle_freshness`, because the frozen-Oracle defect proved a report nobody calls is not a
mechanism. See `docs/plans/2026-08-14-arc1-onboarding-context-brief.md` question 4.

Thin delegate: the logic lives in `pipeline/kb/screen.py` + `oracles.py` + `onboard_footprint.py`
(importable + testable without the MCP server). Degrade-open + fail-safe are enforced there.
"""
from __future__ import annotations


def _lookback_options() -> dict:
    """The window vocabulary the host asks with, read off the preset dicts themselves so a new
    preset shows up in the question without anyone remembering to update a prompt.

    Two selectors, two shapes, two defaults — they are NOT interchangeable, and each `why` says
    which constraint it answers to.

    ⚠️ A BOOKMARK SELECTOR IS DELIBERATELY ABSENT (deleted 2026-08-30). It was here, and it was a
    DEAD PROMPT: no MCP tool takes a bookmark lookback — `oracle` accepts `x_lookback` and
    `web_lookback` and nothing else — so a host that asked the question had nowhere to put the
    answer and silently discarded it. `--bookmark-lookback` exists only on
    `pipeline/kb/ingest_curation.py`'s CLI, for an operator, and defaults to `all`.

    It should not come back even if that parameter is added. It traded corpus completeness on an
    axis users misread: X exposes no bookmark timestamp, so the filter can only cut on when a post
    was WRITTEN, which drops the 2019 paper saved yesterday. The question's own note said so, and a
    question that has to warn you against answering it is a question to delete. What it bought was
    also gone by then — the thread fetch went free with the X cutover, leaving only the VLM image
    read, measured at $0.105 across 315 images on a ~1,080-bookmark backlog. Whether to import
    bookmarks at all is still asked, once, by `onboard`'s consent step. That is the right
    granularity."""
    from pipeline.kb.expand import WEB_LOOKBACK_PRESETS, X_LOOKBACK_PRESETS
    return {
        "x": {"presets": list(X_LOOKBACK_PRESETS), "default": "6mo", "ceiling": "2yr",
              "ask": "How far back should we pull their X timeline?",
              "why": "The window that silently truncates. Ephemeral stream, short default, "
                     "hard-capped at 2 years however far back you ask. Free since the X cutover "
                     "— it is bounded by wall-clock and by one logged-in session's rate limits, "
                     "not by money."},
        "web": {"presets": list(WEB_LOOKBACK_PRESETS), "default": "all", "ceiling": None,
                "ask": "How far back should we pull their Substack/blog archive?",
                "why": "FREE, durable corpus — pulls in FULL by default. Only ask if the user "
                       "wants it narrowed; a three-year-old essay is often still their best."},
    }


def _screen(conn, *, floor: int) -> dict:
    """action='screen' — the ranked candidate payload plus `oracle_freshness`. See `oracle()`."""
    from pipeline.kb import screen

    out = screen.build_screen(conn, floor=floor)
    # Lookback vocabulary rides with the candidates rather than a docstring the host may paraphrase.
    out["lookback_options"] = _lookback_options()

    # Rides UNCONDITIONALLY (unlike `candidates`' `list_freshness`) — `screen` is the deliberate,
    # occasional "what is my people situation" call, and a roster with no last-pulled column is the
    # blind spot that let Oracles sit frozen. Fail-safe: a registry read failure degrades to
    # candidates without freshness, never to an error.
    try:
        from pipeline.kb import oracle_refresh
        out["oracle_freshness"] = oracle_refresh.status_summary(conn)
    except Exception as e:
        out["oracle_freshness"] = {"error": f"{type(e).__name__}: {e}"}

    # Model routability rides `screen` the same way freshness does: a fragile model is invisible
    # everywhere else until it is dead. Cache-only (`fetch=False`) — a tool call must never pay
    # catalog round-trips; the rail preflight populates the cache. `unknown` is suppressed (it is
    # cache-miss noise here, not signal) and the key is omitted entirely when nothing is wrong.
    try:
        from pipeline import model_routing
        rep = model_routing.preflight(fetch=False)
        notice = {}
        if rep["dead"]:
            notice["dead"] = [f"{m} ({why}) — NO provider survives the deny-list"
                              for m, why in rep["dead"]]
        if rep["fragile"]:
            notice["fragile"] = [f"{m} ({why}) — only {orgs}; one withdrawal from dead"
                                 for m, why, orgs in rep["fragile"]]
        if notice:
            out["model_routing"] = notice
    except Exception:
        pass
    return out


def _candidates(conn, *, query: str, top_n: int, min_signals: int) -> dict:
    """action='candidates' — the same people ranked by what they write. See `oracle()`."""
    from pipeline.kb import candidate_search, ingest_curation
    from pipeline.kb.embed import get_kb_embedder

    # Reconcile the `save` signal from stored atoms BEFORE ranking (pure SQL, no network) so a
    # bookmark the catch-up landed is a candidate on the next read. Fail-safe: a reconcile failure
    # degrades to ranking whatever signals are already stored.
    try:
        reconciled = ingest_curation.reconcile_saved_signals(conn)
    except Exception as e:
        reconciled = {"error": f"{type(e).__name__}: {e}"}

    # Embedder built only when a query needs one — an empty query is pure SQL ("who's been probed").
    emb = get_kb_embedder() if (query or "").strip() else None
    out = candidate_search.candidates_payload(
        conn, query or "", emb, k=top_n, min_signals=min_signals)
    # Reported only when it changed something or failed — printed on every call, it trains the
    # reader to skip the field.
    if reconciled.get("inserted") or reconciled.get("orphans") or reconciled.get("error"):
        out["signal_reconcile"] = reconciled

    # The other 4 signals (list/follow/like/subscribe) leave no atom to reconcile from, so their
    # only freshness record is when `curation_catchup` last ran; this reports it, same
    # surface-only-when-stale rule as `signal_reconcile`. Fail-safe: a state-read failure degrades
    # to the payload without freshness.
    try:
        from pipeline.kb import curation_state
        freshness = curation_state.status_summary(
            conn, ingest_curation.COLLECTORS)
        if freshness.get("needs_attention"):
            out["list_freshness"] = freshness
    except Exception:
        pass
    return out


def _confirm(conn, *, canonical_ids, add_handles) -> dict:
    """action='confirm' — commit Oracle picks into the `oracles` table. See `oracle()`."""
    from pipeline.kb import oracles

    if not canonical_ids and not add_handles:
        return {"error": "confirm needs canonical_ids=[...] (kept picks) and/or "
                         "add_handles=[...] (raw handles to add)."}
    return oracles.confirm(conn, canonical_ids=canonical_ids, add_handles=add_handles)


def _ingest(conn, *, canonical_ids, force: bool, x_lookback, web_lookback) -> dict:
    """action='ingest' — deep-ingest confirmed Oracles into the atom-KB. See `oracle()`."""
    from pipeline.kb import oracles
    from pipeline.kb.embed import get_kb_embedder

    picks = oracles.confirmed_oracles(conn)
    if canonical_ids:
        want = set(canonical_ids)
        picks = [o for o in picks if o["canonical_id"] in want]
    if not picks:
        return {"error": "no confirmed Oracles to ingest — run action='confirm' first "
                         "(or pass canonical_ids of confirmed Oracles)."}

    # One shared per-Oracle engine (with the trust-root seed) for BOTH this SCREEN path
    # and `add_oracle` — see `oracles._ingest_oracle`. Before the shared engine, this
    # path never seeded trust roots; now it does.
    web_since = oracles._web_since(web_lookback)

    # `since_last` differs per Oracle (since THEIR last pull, not a fixed span), so resolve it up
    # front for all picks and REFUSE before spending anything if any pick has no derivable window —
    # a missing window silently falling through to the adapter's 183-day default would turn the
    # cheapest ask into the most expensive pull.
    if x_lookback == oracles.X_SINCE_LAST:
        per_oracle = {o["canonical_id"]: oracles.x_since_last(conn, o["canonical_id"])
                      for o in picks}
        blind = [o["name"] or o["canonical_id"]
                 for o in picks if per_oracle[o["canonical_id"]] is None]
        if blind:
            return {"error": f"{oracles.X_SINCE_LAST!r} needs a previous pull to "
                             f"measure from, and these have none: {', '.join(blind)}. "
                             f"Pass an explicit x_lookback ('6mo'/'1yr'/'2yr') for "
                             f"them — this is their first X pull, not a top-up."}
    else:
        shared = oracles._x_since(x_lookback)
        per_oracle = {o["canonical_id"]: shared for o in picks}

    embedder = get_kb_embedder()
    out = [oracles._ingest_oracle(conn, embedder, o, force=force,
                                  x_since=per_oracle[o["canonical_id"]],
                                  web_since=web_since)
           for o in picks]
    # One report per Oracle when the windows differ, so the cost-consent surface still
    # says what actually ran — a single date would be a lie under `since_last`.
    if x_lookback == oracles.X_SINCE_LAST:
        lookback = {"x": f"per-Oracle ({oracles.X_SINCE_LAST})",
                    "per_oracle": {cid: oracles._lookback_report(s, web_since)
                                   for cid, s in per_oracle.items()}}
    else:
        lookback = oracles._lookback_report(per_oracle[picks[0]["canonical_id"]],
                                            web_since)
    return {"ingested_oracles": len(out), "lookback": lookback, "results": out}


def register_oracle_tools(mcp) -> None:

    @mcp.tool()
    def oracle(action: str = "screen", canonical_ids: list[str] | None = None,
               add_handles: list[str] | None = None, top_n: int = 30, floor: int = 15,
               force: bool = False, x_lookback: str | None = None,
               web_lookback: str | None = None, query: str = "",
               min_signals: int = 1) -> dict:
        """Choose who to trust: turn the people you already curate (follows, Lists, bookmarks,
        subscriptions, likes) into your **Oracles** — the sources the KB deep-ingests and roots
        trust on. Runs entirely in chat.

        NOT the setup tool. `onboard` readies the machine (keys, consent, the first curation pull);
        this decides WHO is in. Reach for this when the user asks who to trust, wants to see or
        change their people, or asks whether their sources are current.

        FLOW: call `action='screen'` → read the ranked candidates to the user (the PRE-TICKED ones
        are people you've corroborated with ≥2 distinct signals — your default-yes set; the rest are
        shown unchecked; non-persons are demoted to the end, never hidden) → ask which to keep →
        call `action='confirm'` with the kept `canonical_ids` → then, before `action='ingest'`, ask
        the user how far back to pull and pass their answer as `x_lookback`.

        Why that question is not optional: the default is only ~6 months, so a user who wanted
        their Oracle's whole visible history gets a fraction of it and is never told. (It is no
        longer a COST question — X reads are free since the cutover — it is a completeness one,
        which is the same reason to ask and a different reason to give.) `action='screen'` returns the exact presets and defaults under `lookback_options` —
        offer those, don't invent your own. The web archive is FREE and already pulls in full, so
        ask about it (`web_lookback`) only if the user wants it NARROWED.

        To add someone NOT in the list, pass their X @handle or Substack URL in `add_handles` —
        they're resolved and added on the spot.

        `action`:
          • "screen"  (default) — the ranked candidate payload. Each candidate carries its
                       `reflected` signal ("you follow · subscribe · bookmarked 12×"), `pre_ticked`
                       / `shown_by_default` / `is_person` flags, `distinct_signals`, and its
                       `canonical_id` (pass these to confirm). `classify.ran=False` means the kind
                       classifier degraded open (LLM unavailable) — everyone stays person-eligible.
          • "confirm" — commit Oracle picks. `canonical_ids` = the ones the user kept (verbatim from
                       screen); `add_handles` = raw X handles / Substack URLs to add (resolved-at-
                       confirm). Idempotent. Returns {confirmed, unresolved, unknown, total_oracles}
                       — surface `unresolved` (handles a lookup couldn't find) to the user.
          • "ingest"  — deep-ingest confirmed Oracles into the atom-KB. For each (all confirmed, or
                       just the `canonical_ids` you pass) run discovery — which mines their trusted
                       blog/Substack for their OTHER profiles — then ingest each trusted personal
                       profile as atoms attributed to the Oracle; an org link becomes an affiliation;
                       an ambiguous one is left for review. Returns a per-source outcome report,
                       with the windows actually pulled under `lookback` — read those back.
          • "candidates" — the OTHER half of the promotion decision. `screen` ranks people by how
                       hard the USER vouched for them (distinct curation signals); this ranks the
                       same people by what they actually write. Pass `query` to ask a topic
                       question ("who writes about agent memory"); omit it to list who is there.
                       Evidence comes from TWO stores and every row names which, in `basis`:
                         · "probed" — a light sample of their own timeline (~25 posts). UNVETTED:
                           nobody vouched for this text, only for the person being worth a look.
                           Never cite it as knowledge-base content or quote it as fact.
                         · "saved"  — a post the USER saved, so it IS a knowledge-base atom and is
                           citable (open it first). Usually exactly ONE post, and it is a positive
                           example rather than a sample of their output: it matched partly BECAUSE
                           the user kept it. Good evidence for "worth reading more of", weak
                           evidence for "this is what they mostly write about".
                       Scores are comparable only WITHIN a basis; the two are interleaved by rank,
                       never added. Read back `no_local_material` — a candidate absent because
                       nothing of theirs is stored looks exactly like one whose writing did not
                       match, and those need opposite actions.

        Is my stuff still current? You do not ask — `screen` answers it unasked, under
        `oracle_freshness`: per-Oracle, per-source `last_pulled_at` / `hours_overdue` /
        `never_refreshed` / `breaker_open`, worst first. If `needs_attention` is true, read its
        `note` out — that flag means either nothing has ever opted in to refreshing, or the cycle
        has stretched past twice its target. Both are silent failures otherwise.

        A `model_routing` key, when present, lists models that are `dead` (no OpenRouter
        provider survives the deny-list — the stage using them cannot run) or `fragile` (one
        surviving provider — one withdrawal from dead). Read it out: it is absent whenever
        everything is routable, so its presence IS the news.

        Args:
            action: "screen" | "candidates" | "confirm" | "ingest".
            query: candidates — a topic question to rank people by ("agent memory", "biotech
                funding"). Omit for "who has been sampled".
            min_signals: candidates — only people with at least this many distinct curation
                signals (default 1 = everyone in the list).
            canonical_ids: confirm — the kept candidates' canonical_ids; ingest — which confirmed
                Oracles to ingest (omit = all confirmed — usually NOT what you want for a
                top-up, so name the person).
            add_handles: confirm — raw X @handles / Substack URLs to add beyond the ranked list.
            top_n: candidates — how many people to return (default 30). NOT a screen knob:
                `screen` classifies every candidate and returns all of them.
            floor: screen — minimum candidates shown by default before "see all" (default 15).
            force: ingest — the "ignore what we decided last time" override, and it now means TWO
                things. (1) Ingest a footprint source even when the single-author gate would skip
                or park it (for a solo publication the classifier mislabels). (2) Re-run DISCOVERY
                from scratch, ignoring the trust cache — reach for this when a person's sources
                look wrong or incomplete and their X profile has not changed, since the cache key
                only invalidates on a display-name or declared-link change. A source they created
                after the last run, or a fix on our side, leaves that key identical.
            x_lookback: ingest — how far back to pull each Oracle's X timeline: "6mo" (the
                default), "1yr", "2yr". Hard-capped at 2 years whatever you pass. Ask the user —
                not for cost (the pull is free), but because the default truncates.
                "since_last" is the cheap top-up: pull only what has appeared since this Oracle
                was last pulled. Use it whenever the user wants someone brought current rather
                than re-ingested — a 5-day gap costs one request instead of ~19. It errors rather
                than guessing if the Oracle has never been pulled.
            web_lookback: ingest — how far back to pull the Substack/blog archive: "1yr",
                "2yr", "5yr", "all" (the default). Only narrows; omit unless the user asks.
        """
        from pipeline.kb import schema

        conn = schema.connect()
        try:
            if action == "screen":
                return _screen(conn, floor=floor)

            if action == "candidates":
                return _candidates(conn, query=query, top_n=top_n, min_signals=min_signals)

            if action == "confirm":
                return _confirm(conn, canonical_ids=canonical_ids, add_handles=add_handles)

            if action == "ingest":
                return _ingest(conn, canonical_ids=canonical_ids, force=force,
                               x_lookback=x_lookback, web_lookback=web_lookback)

            # `refresh` and `status` left this surface 2026-08-15. Both still exist and still
            # run; only their tool entry points went. Named here because the error below is where
            # someone who remembers them will land:
            #   • refresh      — the loop runs itself from a session-open spawner and drains
            #     MAX_PAIRS_PER_RUN per run. Consent moved to `onboard`, which was the only thing
            #     the manual trigger was really for. Its wider-than-normal window survives as
            #     `python -m pipeline.kb.oracle_refresh --once --force`.
            #   • status       — split in two. Health became a NOTICE (a report you must know to
            #     ask for is the mechanism that failed); the roster half rides on `screen` as
            #     `oracle_freshness`.
            #   • sync_follows — DELETED 2026-08-26, writer and table both. No reader was ever
            #     built, and the CLI this text used to name had no entry point, so the command it
            #     advertised did nothing.
            return {"error": f"unknown action {action!r} — use 'screen', 'candidates', 'confirm' "
                             f"or 'ingest'.",
                    "moved": {
                        "refresh": "runs automatically now; to pull one Oracle call "
                                   "oracle(action='ingest', canonical_ids=[...], "
                                   "x_lookback='since_last')",
                        "status": "freshness rides on oracle(action='screen') as "
                                  "`oracle_freshness`",
                        "sync_follows": "DELETED 2026-08-26 — writer and table both. Nothing "
                                        "ever read the follow snapshot, and the CLI this used "
                                        "to name had no entry point.",
                    }}
        finally:
            conn.close()

    @mcp.tool()
    def add_oracle(reference: str, confirm: bool = False,
                   x_lookback: str | None = None, web_lookback: str | None = None,
                   extra_source_urls: list[str] | None = None, force: bool = False) -> dict:
        """Add a person to your knowledge base as an **Oracle** — a trusted source OPYT
        deep-ingests (their X timeline + Substack/blog archive + GitHub) and roots trust on. This
        is the atom-KB "add a person" — the only one: the old vault-era `add_person` tool, which
        wrote the legacy vault and couldn't admit a Substack/blog-rooted person, is retired.

        SAFE, TWO-PHASE — always preview before you ingest:
          • FIRST call with confirm=False (the default) → a PREVIEW. It resolves `reference` and
            returns who they are (name, bio, followers) — or, if already known, their roster entry
            — and writes NOTHING. Read it back to the user to confirm it's the right person. If the
            reference doesn't resolve you get `unresolved` and there's nothing to confirm.
          • THEN, once the user agrees, call again with confirm=True → runs the full ingest and
            writes to the store. Never call confirm=True without showing the preview first —
            the guard is against ingesting the WRONG PERSON, which no later call undoes.

        `reference` is polymorphic: an X @handle ("@karpathy"), a Substack/blog/site URL
        ("https://simonwillison.net"), or a canonical_id from `oracle(action='screen')` (to promote
        a below-the-cut candidate). To add someone by NAME, YOU resolve the name → their @handle or
        URL first (your own knowledge / a web search), then pass that — there is no name-search
        endpoint.

        TWO windows, asked separately, because they answer to different constraints:
          • `x_lookback` — "6mo" (default) / "1yr" / "2yr". Hard-capped at 2 years whatever you
            pass. ASK the user before a confirm=True; the ~6-month default silently leaves most of
            a prolific account's history out.
            For someone ALREADY on the roster, "since_last" pulls only what is new since their
            last pull — the cheap top-up, roughly one request for a few days' gap. Reach for it
            when the user says "update" or "catch up", not "add".
          • `web_lookback` — "1yr" / "2yr" / "5yr" / "all" (default). A durable archive, so it
            already pulls everything. Pass it only to NARROW.
        A single shared window would be wrong for one of the two by construction — it either
        over-pulls X or truncates the archive. The result echoes the windows that actually ran
        (including the X clamp) under `lookback` — TELL the user how far back you pulled.

        Reading the result — do not report `ingested` as if it were the whole story:
          • `ingested` counts sources that actually ingested; `blocked` counts sources where the
            host stopped us (Cloudflare, a truncated archive). A blocked source wrote nothing and
            is retried on the next run — say so plainly ("their Substack was blocked, nothing was
            saved, it'll retry") rather than implying it worked. `errors` is the "something is
            wrong, worth a look" bucket.
          • `atoms_added` vs `dispatched`: `lookback`/`limit` bound posts ATTEMPTED, not atoms
            saved, so these two diverge whenever posts are paywalled or fail the quality gate. If
            `atoms_added` is much smaller, tell the user the number they actually got.
          • `producer_failed` > 0 means posts vanished mid-run — mention it; nothing else records it.

        The result may ask you to do something — check for `followup`. A confirm=True result
        carries one, and acting on it is how this tool finds a person's blog / YouTube / podcasts
        at all. OPYT's four probes are deterministic (X bio, Substack convention, GitHub, Semantic
        Scholar); the open-web step is YOURS, because you have web search and OPYT would otherwise
        pay a second model for a worse version of it.
        So: run the search it describes, then call this tool AGAIN with the same `reference`,
        `confirm=True`, and the URLs in `extra_source_urls`. Send everything plausible — you do NOT
        need to verify ownership, because the trust graph re-checks every URL and rejects what it
        cannot corroborate. A URL you drop is invisible; a URL it rejects lands in `needs_review`.

        Args:
            reference: an X @handle, a Substack/blog URL, or a canonical_id.
            confirm: False (default) = preview only, no writes; True = run the ingest.
            x_lookback: window for the X timeline — "6mo" | "1yr" | "2yr" (default 6mo).
            web_lookback: window for the Substack/blog archive — "1yr" | "2yr" | "5yr" |
                "all" (default all). Narrows only.
            extra_source_urls: home/channel pages YOU found by web search (see `followup`). They
                enter as low-confidence candidates and are trust-checked like any other source —
                never trusted on your say-so. Individual posts/videos are dropped; send homes.
            force: re-run DISCOVERY from scratch, ignoring the cached result. Discovery is cached
                for a person whose X profile is unchanged, because identity is stable while their
                CONTENT is not — a normal re-add should not re-derive who they are. Pass this when
                the user says their sources look wrong or incomplete and re-adding did not help.
                The cache key is their display name plus declared links, so a source they created
                after the last run leaves it identical and only this gets past it.
                Costs a full probe walk; do not pass it by default.
        """
        from pipeline.kb import oracles, schema
        from pipeline.kb.embed import get_kb_embedder

        conn = schema.connect()
        try:
            # No embedder needed to PREVIEW — build it only for the ingest path.
            embedder = get_kb_embedder() if confirm else None
            return oracles.add_oracle(conn, embedder, reference, confirm=confirm,
                                      x_lookback=x_lookback, web_lookback=web_lookback,
                                      extra_source_urls=extra_source_urls, force=force)
        finally:
            conn.close()
