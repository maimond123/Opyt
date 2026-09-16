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

Thin delegate: the logic lives in `pipeline/kb/screen.py` + `oracles.py` + `onboard_footprint.py`
(importable + testable without the MCP server). Degrade-open + fail-safe are enforced there.
"""
from __future__ import annotations


def _lookback_options() -> dict:
    """The window vocabulary the host asks with, read off the preset dicts themselves so a new
    preset shows up in the question without anyone remembering to update a prompt.

    Three selectors, three shapes, three defaults — they are NOT interchangeable, and each `why`
    says which constraint it answers to.

    ⚠️ A BOOKMARK SELECTOR IS DELIBERATELY ABSENT (deleted 2026-08-30). It was here, and it was a
    DEAD PROMPT: no MCP tool takes a bookmark lookback — `oracle` accepts `x_lookback`,
    `web_lookback` and `scholar_lookback` and nothing else — so a host that asked the question had
    nowhere to put the answer and silently discarded it. `--bookmark-lookback` exists only on
    `pipeline/kb/ingest_curation.py`'s CLI, for an operator, and defaults to `all`.

    It should not come back even if that parameter is added. It traded corpus completeness on an
    axis users misread: X exposes no bookmark timestamp, so the filter can only cut on when a post
    was WRITTEN, which drops the 2019 paper saved yesterday. The question's own note said so, and a
    question that has to warn you against answering it is a question to delete. What it bought was
    also gone by then — the thread fetch went free with the X cutover, leaving only the VLM image
    read, measured at $0.105 across 315 images on a ~1,080-bookmark backlog. Whether to import
    bookmarks at all is still asked, once, by `onboard`'s consent step. That is the right
    granularity."""
    from pipeline.kb.expand import (SCHOLAR_LOOKBACK_PRESETS, WEB_LOOKBACK_PRESETS,
                                    X_LOOKBACK_PRESETS)
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
        "scholar": {"presets": list(SCHOLAR_LOOKBACK_PRESETS), "default": "all", "ceiling": None,
                    "ask": "How far back should we pull their papers?",
                    "why": "FREE and abstract-only, so the whole corpus is the default — a "
                           "researcher's foundational paper is usually not their most recent. "
                           "ASK THIS ONE WITH NUMBERS, and read them off the record in front of "
                           "you: `add_oracle`'s preview returns `lookback.scholar_counts` with "
                           "`total`, `years` and a `by_window` count per preset for THIS person. "
                           "Say those. Unlike the other two, this selector can know. There is a "
                           "second, sharper question next to it — `scholar_topics`, the subject "
                           "filter — and a long publication record usually wants that one more "
                           "than a shorter window."},
    }


def _screen(conn, *, floor: int, limit: int, source: str | None = None) -> dict:
    """action='screen' — the ranked candidate payload plus `oracle_freshness`. See `oracle()`."""
    from pipeline.kb import screen

    out = screen.build_screen(conn, floor=floor, limit=limit, source=source)
    # Lookback vocabulary rides with the candidates rather than a docstring the host may paraphrase.
    lookback_options = _lookback_options()
    from pipeline.ingestion.x_graphql import has_managed_x_session
    if not has_managed_x_session():
        # Do not ask a Substack/blog-only user to choose an X window they have not enabled.
        lookback_options.pop("x")
    out["lookback_options"] = lookback_options

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

    # WHY THIS SCREEN IS THIN. `classify` degrades OPEN by design — a batch of 100 that the
    # model refused leaves those people `kind=None` and writes a log line nobody reads — so the
    # user gets a shorter, less-labelled roster with no stated cause. That is the exact result
    # the 2026-09-15 user was looking at when they concluded their X login was broken.
    #
    # Gated on `ran is False`, which is a real degrade and not merely "nothing to do":
    # `classify_kinds` returns `ran: True` when every candidate was already classified. Unlike
    # the two riders below it this one does NOT stamp — it is a state that lasts until the user
    # acts, and every screen it thins deserves the same sentence. See `allowance_notice`.
    try:
        if out.get("classify", {}).get("ran") is False:
            from pipeline.kb.allowance_notice import allowance_notice
            if (blocked := allowance_notice()) is not None:
                out["model_provider"] = blocked
    except Exception:
        pass

    # A pull that finished while nobody was looking — see `pull_runs.completion_notice`. On
    # `screen` because it is the deliberate "what is my people situation" call, which is exactly
    # where somebody returning to a pull they left running will look, and because it is already
    # the home of every other rider that rides unconditionally. Once, then never again.
    try:
        from pipeline.kb import pull_runs
        if (finished := pull_runs.completion_notice(conn)) is not None:
            out["pull_finished"] = finished
    except Exception:
        pass

    # Material nobody has told them about, same rider rules. Separate `try` from the one above so
    # a failure in either still leaves the other's notice on the answer.
    try:
        from pipeline.kb import new_material
        if (landed := new_material.new_material_notice(conn)) is not None:
            out["new_material"] = landed
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


def _names(names: list[str]) -> str:
    """A list of writers joined the way a person says it — "Andrej", "Andrej and Martin",
    "Andrej, Martin and Soren". Not a comma-joined array: the envelope's whole job here is to hand
    the host a sentence about PEOPLE, and the one place it is allowed to name them should not read
    like a field dump."""
    if len(names) <= 1:
        return names[0] if names else "these writers"
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _is_filling_in(enrichment: dict | None) -> bool:
    """Is the background pass genuinely moving right now? The probe the copy rests on.

    Reads the start result the ingest just got, then falls back to the live `is_running()` probe —
    the same one `atoms_tools._import_outstanding` relies on — so a pass started by an EARLIER
    call (or by onboarding) still counts. `nothing_owed` is deliberately not `True`: it means
    there is no work, and a caller reaching this function has already decided there is.

    Fail-safe, and the direction matters: unreadable means NOT running, so the copy degrades to
    "tell me when you'd like me to pick it up" rather than to a promise nothing is keeping.
    """
    if (enrichment or {}).get("status") == "running":
        return True
    try:
        from pipeline.kb import footprint_enrichment
        return footprint_enrichment.is_running()
    except Exception:
        return False


def _ingest_presentation(results: list[dict], conn=None, enrichment=None) -> dict:
    """The user-facing outcome of an Oracle ingest; `results` remains diagnostic detail.

    The ingest engine reports each adapter factually so it can be debugged. That is not the
    vocabulary a person needs after onboarding: group completed work, work that will continue,
    and sources that were intentionally left out instead of exposing transport outcomes.

    `conn` grounds the closing `tour` in what actually landed. Optional, and its absence costs
    only the grounding: without a store to measure, the tour says nothing rather than falling
    back to a generic menu. That is the point of the change — see `_tour`.
    """
    completed, in_progress = [], set()
    review_oracles, available_x, reconnect_x = set(), set(), set()

    for result in results:
        name = result.get("name") or result.get("oracle_id") or "this writer"
        source_rows = result.get("results") or []
        ingested = sorted({r.get("type") for r in source_rows
                           if r.get("action") == "ingested" and r.get("type")})
        # A PARTIAL writer belongs in BOTH lists, and suppressing either half misreports them.
        # Real writing landed — they are readable now, and `completed` is where a reader who is
        # here is introduced — and more of it is still owed, which is what `in_progress` says.
        # Two facts, not an average. See `_merge_passes`: the breadth pass can complete while the
        # depth pass over the same person comes back short, so this reads across both.
        partial = any(r.get("action") == "ingested" and r.get("partial") for r in source_rows)
        if ingested:
            entry = {"oracle": name, "sources": ingested,
                     "atoms_added": int(result.get("atoms_added") or 0)}
            if partial:
                entry["partial"] = True
            completed.append(entry)
        if partial or any(r.get("action") == "deferred" for r in source_rows):
            in_progress.add(name)
        if any(r.get("action") == "needs-review" for r in source_rows):
            review_oracles.add(name)
        if any(r.get("action") == "needs_reconnect" and r.get("type") == "x"
               for r in source_rows):
            reconnect_x.add(name)
        if any(s.get("source_type") == "x" for s in result.get("available_sources") or []):
            available_x.add(name)

    # ⚠️ ORDER IS THE MESSAGE, so this dict is assembled in reading order rather than in whatever
    # order the checks happen to run. A host reads an envelope top-down and leads with what it
    # finds first, and until 2026-09-12 what it found first was WORK: an unverified profile to
    # adjudicate and two platforms to go connect, all of them ranked above the only key that says
    # what the user can now do. The user's first moment with a finished library was a chore list.
    #
    # So: what landed, then what they can do with it, then what genuinely blocks collection, and
    # LAST the two items that ask the user to go find or verify something. Nothing is dropped —
    # the review queue is durable and `oracle(action='review')` is its real home, which the
    # source-scoped onboarding plan already said: "the onboarding presentation is only a pointer
    # to it." A pointer does not have to be the first thing on the page.
    presentation = {"completed": completed}
    if any(c.get("partial") for c in completed):
        # Alongside the entry, not inside it: the flag is a fact about the writer, this is an
        # instruction about how to introduce them. The failure mode it heads off is the host
        # reading `partial` as "broken" and burying a good writer in caveats — half of somebody's
        # writing is still a library, and it is really here and really readable now.
        presentation["partial_note"] = ("A `partial` writer is really here and really readable — "
                                        "introduce them with the others, don't hold them back or "
                                        "hedge.")
    if sum(int(r.get("atoms_added") or 0) for r in results) > 0:
        tour = _tour(conn)
        if tour:
            presentation["tour"] = tour
    if in_progress:
        # ⚠️ THE PERSON NEVER LEARNS WHY, and the vocabulary this replaced is the reason the rule
        # is written down. Until 2026-09-14 this key said: "x.com's 15-minute request window ran
        # out … this install has no resident worker, and the rail that would resume them … Calling
        # `oracle(action='ingest')` again once the window refills collects them in the foreground."
        # Five pieces of jargon in three sentences, ending in a command the reader will never
        # type — beside neighbours in this same dict that already read like a person wrote them.
        #
        # They learn three things, in this order: what I have, in terms of their writers; what I
        # don't have yet, plainly and without an excuse; and that it is handled. The ONE permitted
        # gesture at cause is "X only lets us read so much at a time" — true, needs no vocabulary,
        # invites no follow-up. Banned everywhere here: rate limit, window, quota, meter,
        # requests, API, throttled, backfill, rail, worker, foreground, atoms, ingest.
        #
        # ⚠️ AND IT STAYS CONDITIONAL. "Filling in on its own" is a claim about a thread that may
        # not exist — consent, model routing or a held lease can each stop it. A user told an
        # abandoned Oracle was in hand stops waiting for a pull that cannot start (2026-09-13,
        # measured on a from-source install). What changed is not that lesson; it is that
        # `footprint_enrichment` makes the automatic case the normal one, so the honest sentence
        # is now the good one. See `_finish_in_background` for where the fact comes from.
        names = _names(sorted(in_progress))
        if _is_filling_in(enrichment):
            msg = (f"I have part of {names}'s writing so far — the recent posts, not the older "
                   f"ones yet. X only lets us read so much at a time, so the rest is filling in "
                   f"on its own while you work. Nothing for you to do.")
        else:
            # No time estimate, no mechanism, no command. If it is not running, the reason is
            # never the meter — it is consent, model routing or a held lease, and none of those
            # is fixed by a clock, so a countdown here would be a friendly-looking lie.
            msg = (f"I have part of {names}'s writing — the recent posts, not the older ones "
                   f"yet. The rest isn't moving on its own right now. Tell me when you'd like me "
                   f"to pick it up.")
        presentation["in_progress"] = {
            "oracles": sorted(in_progress),
            "resumes": _is_filling_in(enrichment),
            "message": msg,
            # ⚠️ THE LAST CLAUSE IS LOAD-BEARING. Handed a fact about a limit, a model will
            # helpfully explain the limit; handed unfinished work, it will helpfully offer to
            # finish it. Both are wrong here — the second re-invents the confirmation prompt this
            # revision removed, because offering to help reads as helpful.
            "guidance": ("Say this AFTER what landed, never before it — the person asked for a "
                         "library, not a status report. One or two sentences, your own words. "
                         "NEVER explain why X stopped us: no rate limits, no time windows, no "
                         "request counts, not even if they ask how it works. If they ask, say "
                         "it's still coming in and moving on its own. Do not name tools or "
                         "commands. Do NOT ask them to authorise finishing it — that is already "
                         "happening."),
        }
    if reconnect_x:
        # Above the two optional items and below the tour: a dead session is a real fault that
        # silently costs the user posts, unlike a profile nobody has adjudicated.
        presentation["needs_attention"] = {
            "source": "x",
            "oracles": sorted(reconnect_x),
            "message": "X needs to be reconnected before its posts can be collected.",
            "next_step": "onboard(source='x')",
        }
    if available_x:
        presentation["optional_connections"] = [{
            "source": "x",
            "oracles": sorted(available_x),
            "message": ("I found X profiles for these writers but left them out because X is not "
                        "connected. Connect X later if you want their posts included."),
            "next_step": "onboard(source='x')",
        }]
    if review_oracles:
        # Reworded as well as moved. "I found possible profiles but did not add them because I
        # could not verify that they belong to these writers" states a machine's difficulty and
        # hands the adjudication over — homework, delivered in the same breath as the library. It
        # now reports a decision already taken (left out, safely) and says the door stays open,
        # which is the same fact told from the user's side instead of the classifier's.
        presentation["possible_sources"] = {
            "oracles": sorted(review_oracles),
            "explanation": ("Some other profiles might belong to these writers. They were left "
                            "out rather than guessed at, and nothing is lost by leaving them "
                            "out — they keep."),
            "next_step": "oracle(action='review')",
            "mention": "ONLY if the user asks about these writers' other accounts, or after they "
                       "have answered the tour. Never lead with it: it is a decision waiting, "
                       "not a problem, and it is the one thing here that asks them to do work.",
        }
    return presentation


def _tour(conn) -> dict:
    """What this user can now do, measured against what just landed. `{}` when unmeasurable.

    ⚠️ THIS REPLACED A MENU, AND THE DIFFERENCE IS NOT COSMETIC. Until 2026-09-12 this returned
    three fixed labels — "ask a question across your writers", "get an overview of what they
    cover", "build a focused reading session" — for every possible library. A menu that cannot
    vary cannot be wrong about a particular store, but it also cannot be right about one: it
    offered a reading session to a user whose whole corpus was one author, and
    `sitting(action='preview')` would then warn that exact region is a poor fit. The product
    recommended a reading and then argued against it, one call apart, and the user paid the turn
    to find out.

    It also duplicated work the host had already done. The tool definitions the host holds
    describe what `search`, `aggregate` and `sitting` are for, better than three labels can. The
    one thing the host cannot get anywhere else is the shape of the store, so that is all this
    ships now — see `opyt_core/suggest`.

    The prompt is gone with the menu. "Want a quick tour?" asks permission to describe the
    product; a measured line about their own material needs no permission because it is already
    the answer.
    """
    if conn is None:
        return {}
    try:
        from opyt_core.kb import kb_aggregate
        from opyt_core.suggest import suggestions
        return suggestions(conn, kb_aggregate())
    except Exception:
        return {}      # Fail-safe: a tour is the most droppable thing in an ingest result.


def _ingest(conn, *, canonical_ids, force: bool, x_lookback, web_lookback,
            scholar_lookback=None, scholar_topics=None) -> dict:
    """action='ingest' — deep-ingest confirmed Oracles into the atom-KB. See `oracle()`."""
    from pipeline.kb import oracles, pull_runs
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
    scholar_since = oracles._scholar_since(scholar_lookback)

    from pipeline.ingestion.x_graphql import has_managed_x_session
    x_connected = has_managed_x_session()

    # `since_last` differs per Oracle (since THEIR last pull, not a fixed span), so resolve it up
    # front for all picks and REFUSE before spending anything if any pick has no derivable window —
    # a missing window silently falling through to the adapter's 183-day default would turn the
    # cheapest ask into the most expensive pull.
    if not x_connected:
        per_oracle = {o["canonical_id"]: None for o in picks}
    elif x_lookback == oracles.X_SINCE_LAST:
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

    # A topic selection names ONE person's subjects, so it may not be sprayed across a batch.
    # Refused rather than silently applied to the first pick: `canonical_ids` defaults to every
    # confirmed Oracle, and narrowing all of them to one researcher's topics is a change no
    # later call undoes for the ones it was not meant for.
    if scholar_topics is not None and len(picks) != 1:
        return {"error": f"scholar_topics narrows ONE Oracle's paper corpus, and this call "
                         f"covers {len(picks)}. Pass canonical_ids=[the one person or venue] "
                         f"alongside it."}

    embedder = get_kb_embedder()
    # RANKED ONCE, HERE, and handed down. `_breadth_then_depth` used to rank for itself, which was
    # fine while it was the only reader — but the run record's roster and the pass order have to
    # be the SAME list or "who is still waiting" answers about a different ordering than the one
    # being walked. One call, one ordering, same argument `_ordered_picks` itself makes for
    # reusing the screen's ranking instead of scoring a second time.
    ordered = _ordered_picks(conn, picks)
    for o in ordered:
        # Carried onto the roster row, because a pull this call may not live to finish has to be
        # able to ask for the window this call was given. See `pull_runs.open_run`.
        since = per_oracle.get(o["canonical_id"])
        o["window"] = since.isoformat() if since is not None else None
    lookback = _lookback_for(ordered, per_oracle, x_connected=x_connected, x_lookback=x_lookback,
                             web_since=web_since, scholar_since=scholar_since)

    # ONE PULL AT A TIME, and a second one JOINS rather than being refused. Two would split the
    # same two x.com buckets, which is the meter the whole breadth/depth ordering exists to
    # protect — but bouncing the call drops writers the user just asked for, and the likely
    # trigger is "oh, add one more person", not a banned retry. Refusing would also re-teach the
    # host the retry loop §C exists to stop.
    if (running := pull_runs.in_flight(conn)) is not None:
        return _join_running_pull(conn, running, ordered)

    # OPENED BEFORE THE FIRST REQUEST. Every pick gets a row now, not when its turn comes: this
    # is what lets a later call say who is still waiting, and an Oracle absent from the report
    # reads as an Oracle who failed. See `pipeline/kb/pull_runs.py`.
    run_id = pull_runs.open_run(conn, kind="ingest", picks=ordered, lookback=lookback)
    _spawn(lambda: _run_pull(run_id, ordered, force=force, per_oracle=per_oracle,
                             web_since=web_since, scholar_since=scholar_since,
                             scholar_topics=scholar_topics, x_connected=x_connected))

    # ALREADY DONE BY THE TIME WE GOT HERE. True under the test seam, where `_spawn` runs its
    # target inline, and true in production for a pull with nothing to fetch. Reporting the
    # finished run rather than announcing a start is not a special case — it is the same question
    # asked of the same record, and the answer happens to be complete.
    started = pull_runs.get_run(conn, run_id)
    if started is not None and started.finished_at is not None:
        return _report(conn, run_id)
    return _started_payload(run_id, ordered, lookback)


def _spawn(target):
    """Start `target` on a daemon thread and RETURN the thread.

    An indirection with exactly one job, and the same shape and reason as `onboard_tools._spawn`:
    tests replace it with a synchronous call, because a real thread outliving a test's
    monkeypatching would hit the actual network after the stub is gone."""
    import threading

    t = threading.Thread(target=target, name="opyt-oracle-pull", daemon=True)
    t.start()
    return t


def _run_pull(run_id: str, ordered: list[dict], *, force: bool, per_oracle: dict,
              web_since, scholar_since, scholar_topics, x_connected: bool) -> None:
    """The pull itself, off the call that asked for it.

    ⚠️ NOTHING HERE IS BOUNDED BY A CLOCK, AND NOTHING MAY BECOME SO. Archives run to completion
    (R4), timelines walk their whole window, and this takes as long as it takes — twelve minutes
    is a normal number. That is only safe because no tool call is waiting on it any more; the
    moment one is, somebody will be tempted to put a deadline here, and that is §A.

    ITS OWN CONNECTION AND ITS OWN EMBEDDER. SQLite connections are thread-bound and the caller's
    belongs to a request that has already returned — the same rule `footprint_enrichment.
    start_background` records. `get_kb_embedder` is called here rather than handed in for the
    same reason.

    A CRASH LEAVES THE RUN OPEN, DELIBERATELY. The heartbeat stops with the context manager, so
    within `LEASE_TTL` the record reads `stopped` — "4 of 7 are in" — which is exactly true.
    Closing the run in a `finally` would mark unfinished work done, which is the one thing the
    fail-safe invariant forbids."""
    from pipeline.kb import pull_runs
    from pipeline.kb.embed import get_kb_embedder
    from pipeline.ingestion.utils import log

    conn = None
    try:
        conn = pull_runs.connect()
        with pull_runs.Heartbeat(run_id):
            out = _breadth_then_depth(conn, get_kb_embedder(), ordered, run_id=run_id,
                                      force=force, per_oracle=per_oracle,
                                      web_since=web_since, scholar_since=scholar_since,
                                      scholar_topics=scholar_topics, x_connected=x_connected)
        pull_runs.close_run(conn, run_id)
        # R2: what the meter stopped is background work, not a question. Started AFTER the
        # foreground passes finish, never beside them — both walk the same two buckets, and
        # racing them would split a meter the whole design is organised around.
        _finish_in_background(_results_from(conn, run_id) or out)
    except Exception as e:
        log(f"[oracle-pull] {run_id[:8]} stopped: {type(e).__name__}: {e}")
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _started_payload(run_id: str, ordered: list[dict], lookback: dict) -> dict:
    """What a call that has STARTED a pull hands back, in under a second.

    ⚠️ IT NAMES `progress` EXPLICITLY, and that is the load-bearing part rather than the copy.
    §C's lesson from 2026-09-14 is that a tool description is instruction and not enforcement —
    a model read "do not call it again" and reasoned its way into a smaller retry regardless. The
    description tells the host to loop; this tells it, in the return value it is holding, what to
    call next. Two channels for the one instruction the whole design depends on.

    It promises no completion time. We do not know one: a writer with a large archive takes as
    long as their archive, and "about ten minutes" is a broken promise the first time somebody
    tracks a prolific one. It says nothing about what will be found either — explaining an
    absence nobody has measured yet is the second-order damage this plan was written to stop."""
    names = _names([o.get("name") or o["canonical_id"] for o in ordered])
    first = (ordered[0].get("name") or ordered[0]["canonical_id"]) if ordered else None
    return {
        "run_id": run_id, "status": "started", "lookback": lookback,
        "picks": [{"name": o.get("name") or o["canonical_id"],
                   "canonical_id": o["canonical_id"]} for o in ordered],
        "next_call": "oracle(action='progress')",
        "host_note": ("The pull is running now and this call did NOT wait for it. Call "
                      "`oracle(action='progress')` and keep calling it until `status` is no "
                      "longer 'running' — it blocks for up to ~40s and returns the moment a "
                      "writer lands, so it is how you follow the pull, not a poll. The full "
                      "report arrives in its `report` field. Do NOT start another ingest, and do "
                      "not tell the user anything has failed: nothing has."),
        "message": (f"I'm pulling {names} now"
                    + (f", starting with {first}" if first else "")
                    + ". Their archives run end to end, so this takes a few minutes — I'll tell "
                      "you as each one lands."),
    }


# ⚠️ NOT A BUDGET ON WORK, AND THE DISTINCTION IS THE WHOLE RULING. This bounds how long one
# QUESTION about a pull waits before it answers. Nothing is cut when it expires, no pull can see
# it, and no pull learns it happened — archives still run to completion (R4).
#
# `2026-09-14-sixty-second-wall.md` §A put a 20-second clock on the WORK instead. It shipped
# inert (the deadline never reached the engine), was fixed, was measured, and was reverted: a
# clock is useless against a unit of work that cannot be interrupted, and an archive should not
# be. If you are about to reuse this constant anywhere a pull can reach it, that is the same
# mistake with a new name.
#
# 40 against a measured client wall of 61.2 / 61.3 / 62.3s. Do not creep it upward: the headroom
# covers a slow first read on a cold store, not a longer wait.
PROGRESS_WAIT_SECONDS = 40.0
_PROGRESS_POLL_SECONDS = 0.5


def _progress(conn, *, run_id: str | None = None) -> dict:
    """action='progress' — where a running pull has got to. See `oracle()`.

    BLOCKS UNTIL SOMETHING MOVES, or `PROGRESS_WAIT_SECONDS`, whichever comes first. That is the
    ordinary shape of a long operation over a request/response transport, and it is the only
    shape that lets the assistant narrate a twelve-minute pull at all: each call sits well inside
    the client's wall, and ten of them cover seven minutes.

    A DB POLL, not a condition variable. The pull may be in another process — the ruling's D1 is
    an in-process thread today and is explicitly overrulable into a worker rail — and a poll
    survives that change untouched.

    Returns rather than raises on every absence: no run at all is `status: "none"`, not an error.
    A host handed an error reasons its way into a retry.
    """
    import time as _time

    from pipeline.kb import pull_runs

    deadline = _time.time() + PROGRESS_WAIT_SECONDS
    seen = None
    while True:
        run = (pull_runs.get_run(conn, run_id) if run_id
               else pull_runs.latest_run(conn, kind="ingest"))
        if run is None:
            return {"status": "none",
                    "message": ("No pull has been started on this store. `oracle(action='ingest')` "
                                "is what starts one.")}
        rows = pull_runs.oracles_for(conn, run.run_id)
        status = pull_runs.run_status(run, alive=pull_runs.is_alive(run))
        moved = sum(1 for o in rows if o.finished_at is not None)
        if status != "running" or (seen is not None and moved != seen) or _time.time() >= deadline:
            return _progress_payload(conn, run, rows, status)
        seen = moved
        _time.sleep(_PROGRESS_POLL_SECONDS)


def _progress_payload(conn, run, rows, status: str) -> dict:
    """What `progress` says, once it has something to say.

    ⚠️ EVERY WRITER IS NAMED, INCLUDING THE ONES NOTHING HAS TOUCHED. An Oracle missing from a
    report reads as an Oracle who failed — that is how *"Bryan Johnson's pull also timed out"* got
    said about a pull that was mid-flight and landing 57 files. `waiting` is the field that says
    "not yet" rather than letting silence say "nothing found".

    `in_flight` is a LIST, and the handoff plan that typed it as one person was wrong about the
    loop. Breadth runs over EVERYONE before anyone is deepened, so during that phase every writer
    genuinely is in hand at once; during depth it is one at a time. Reporting a single name would
    be false for the first phase of every pull."""
    import time as _time

    from pipeline.kb import pull_runs

    done = [{"name": o.name or o.canonical_id, "canonical_id": o.canonical_id,
             "atoms_added": (o.result or {}).get("atoms_added", 0)}
            for o in rows if o.finished_at is not None]
    out = {
        "run_id": run.run_id,
        "status": status,
        "done": done,
        "in_flight": [{"name": o.name or o.canonical_id, "canonical_id": o.canonical_id}
                      for o in rows if o.state == "in_flight"],
        "waiting": [{"name": o.name or o.canonical_id, "canonical_id": o.canonical_id}
                    for o in rows if o.state == "waiting"],
        "elapsed_seconds": round((run.finished_at or _time.time()) - run.started_at, 1),
    }
    if status != "running":
        # The full report, read off the record — the same one the call that started this pull
        # would have returned if it had waited. There is no in-memory copy to fall back on here,
        # which is the entire reason the record exists.
        out["report"] = _report(conn, run.run_id)
        # AND THE NOTICE IS SPENT. A host that has just been handed the whole report will tell
        # the user; letting the rider announce the same completion again on their next `search`
        # would have OPYT report one event twice.
        try:
            pull_runs.mark_reported(conn, run.run_id)
        except Exception:
            pass
    return out


def _join_running_pull(conn, running, picks: list[dict]) -> dict:
    """Add these writers to the pull already in flight, and say so plainly.

    ⚠️ THE ANSWER MUST NOT READ AS A FAILURE. A host handed "already running" as an error reasons
    its way into a retry — measured on 2026-09-14, where a rule saying "do not call it again" was
    read and then rationalised into a smaller call that timed out having done less. So this is a
    RESULT: the people asked for are on the roster, the pull will reach them, and the same
    `progress` call answers for all of them.

    Nobody new is not an error either. Re-asking for somebody already being pulled is what a host
    does when it is unsure, and the honest answer is that they are already in hand."""
    from pipeline.kb import pull_runs

    added = 0
    try:
        added = pull_runs.add_picks(conn, running.run_id, picks)
    except Exception:
        pass
    names = _names([p.get("name") or p["canonical_id"] for p in picks])
    return {"run_id": running.run_id, "status": "joined", "added": added,
            "message": (f"A pull is already running, and {names} "
                        + ("joined it." if added else "were already on it.")
                        + " Nothing was started twice and nothing was dropped — ask "
                          "`oracle(action='progress')` for where it has got to.")}


def _results_from(conn, run_id: str) -> list[dict]:
    """The per-Oracle results of a run, in roster order — READ BACK, not carried.

    ⚠️ THE LOOP'S RETURN VALUE IS NO LONGER WHAT ANYBODY REPORTS. It used to be, and that is why
    a tool call cut off at the client's 60-second wall took the entire report down with it: the
    only copy was a Python list on a stack frame nobody would ever reach again. The loop still
    returns it, because `add_oracle` drives the same engine with no run to read from.

    ONLY ROWS THAT WERE REACHED. An Oracle with no `result` was never visited, and a synthetic
    row for them would put a person in `results` that no pass produced — which is how *"Bryan
    Johnson's pull also timed out"* got said about a pull that was mid-flight. Naming the
    unreached is `progress`'s job (`waiting`), where the fact being reported is "not yet" rather
    than "nothing found". Today this can only be every row, since the call still blocks until the
    loop is done; it stops being true the moment `ingest` returns early."""
    from pipeline.kb import pull_runs

    return [o.result for o in pull_runs.oracles_for(conn, run_id) if o.result is not None]


def _report(conn, run_id: str, *, enrichment: dict | None = None) -> dict:
    """A run's report, assembled from the record. PURE — it starts nothing.

    That is load-bearing rather than tidy: this is read by the call that started the pull, by
    `progress`, and later by a call that merely notices a finished run nobody has mentioned. A
    reader that could start work would spend the X meter on somebody who typed `search`.

    ⚠️ THERE IS NO IN-MEMORY FALLBACK, AND THERE CANNOT BE ONE. A version of this took the loop's
    own return as a backstop for a store that refused writes. That died with the commit that
    stopped `ingest` waiting: the loop's return value now lives on a thread nobody is holding, so
    every reader that reaches here is reading the record or reading nothing.

    A store that will not record therefore under-reports — it says it reached nobody. That is the
    SAFE direction and the one this repo takes everywhere else: `seed_from_entities` makes the
    same call and says why — a dropped row "self-heals into a re-pull, which is the safe
    direction, not into a coverage claim". Dedup absorbs a re-pull; a coverage claim nothing
    backs is permanent."""
    from pipeline.kb import pull_runs

    run = pull_runs.get_run(conn, run_id)
    results = _results_from(conn, run_id)
    out = {"run_id": run_id, "ingested_oracles": len(results),
           "lookback": (run.lookback if run else None) or {}, "results": results,
           "footprint_enrichment": enrichment,
           "presentation": _ingest_presentation(results, conn, enrichment)}

    # A PULL THAT FETCHED EVERYTHING AND WROTE NOTHING. With the provider refused, every atom
    # reaches `ingest_common.AtomSink` and none of it survives the embed: the batch fails, the
    # per-atom retry fails, and each skips WITHOUT writing (`_flush_isolated` — correct, and
    # deliberately silent). What comes back is a pull that reached its writers and added zero
    # pieces, which reads as "there was nothing there" for people who post daily.
    #
    # Zero atoms is the gate because with the provider blocked it is no longer ambiguous: a pull
    # that added nothing while the model was refused added nothing BECAUSE it was refused.
    try:
        if results and not sum(int(r.get("atoms_added") or 0) for r in results):
            from pipeline.kb.allowance_notice import allowance_notice
            if (blocked := allowance_notice()) is not None:
                out["model_provider"] = blocked
    except Exception:
        pass
    return out


def _lookback_for(picks: list[dict], per_oracle: dict, *, x_connected: bool, x_lookback,
                  web_since, scholar_since) -> dict:
    """What windows this pull WILL use — computed before it runs, not after.

    It only ever depended on the windows, never on the outcome, and it was assembled after the
    pull purely because that is where the return value was built. Hoisting it is what lets the
    run record carry it from the first second, so the call that starts a pull can say what it is
    about to ask for — and so the report survives the call that started it.

    One report per Oracle when the windows differ, so the cost-consent surface still says what
    actually ran; a single date would be a lie under `since_last`."""
    from pipeline.kb import oracles

    if x_connected and x_lookback == oracles.X_SINCE_LAST:
        return {"x": f"per-Oracle ({oracles.X_SINCE_LAST})",
                "per_oracle": {cid: oracles._lookback_report(s, web_since, scholar_since)
                               for cid, s in per_oracle.items()}}
    lookback = oracles._lookback_report(per_oracle[picks[0]["canonical_id"]],
                                        web_since, scholar_since)
    if not x_connected:
        lookback.pop("x", None)
        lookback.pop("x_since", None)
    return lookback


def _finish_in_background(results: list[dict]) -> dict:
    """Hand whatever this ingest could not reach to the background pass, and report whether it
    actually started.

    ⚠️ THE RETURN VALUE IS A FACT, NOT A FORMALITY. `_ingest_presentation` chooses between two
    messages on it — "the rest is filling in on its own" and "the rest isn't moving right now" —
    and the first is a claim about a thread that may not exist. Consent, model routing and a held
    lease can each stop it, and a user told an abandoned Oracle was in hand stops waiting for a
    pull that cannot start. That lesson is 2026-09-13's and it survives this change; what changes
    is that the automatic case is now the normal one.

    Fail-safe: anything that goes wrong here reports `not_started`, which is the honest word for
    it, and costs only the more cautious of the two messages."""
    owed = any(r.get("action") == "deferred"
               or (r.get("action") == "ingested" and r.get("partial"))
               for result in results for r in (result.get("results") or []))
    if not owed:
        return {"status": "nothing_owed"}
    try:
        from pipeline.kb import footprint_enrichment
        return footprint_enrichment.start_background()
    except Exception as e:
        return {"status": "not_started", "error": f"{type(e).__name__}: {e}"}


def _ordered_picks(conn, picks: list[dict]) -> list[dict]:
    """The picks, MOST-VOUCHED-FOR FIRST — which is what makes a breadth pass mean anything.

    `schema.list_oracles` is `ORDER BY confirmed_at DESC`, and a single `confirm` stamps every pick
    in the same second, so today's order is effectively arbitrary. When the meter runs out
    part-way, arbitrary order decides who gets covered.

    Reuses the screen's own ranking rather than adding a second scoring function beside
    `Candidate.sort_key` — two orderings of the same people is how two surfaces end up disagreeing
    about who the user cares most about.

    Oracles added through `add_handles` carry no curation signals, so they are absent from `ranked`
    entirely. They sort AFTER every ranked pick, keeping their existing relative order (the sort is
    stable) — an explicitly named person is not less wanted, but there is no evidence to place them
    among the vouched-for, and inventing a position would be a scoring decision disguised as a
    fallback.

    Fail-safe: a ranking that cannot be read leaves the order alone."""
    try:
        from pipeline.kb import screen
        ranked = screen.interleave_tiers(screen.rank_candidates(conn))
        rank = {c.canonical_id: i for i, c in enumerate(ranked)}
    except Exception:
        return list(picks)
    return sorted(picks, key=lambda o: rank.get(o["canonical_id"], len(rank)))


def _merge_passes(breadth: dict, depth: dict) -> dict:
    """Two passes over one Oracle → ONE result, because that is what every reader expects.

    `_ingest_presentation` groups by name and reads `atoms_added` off the top level, so handing it
    two rows for one person would list them twice and split their count. The per-source rows are
    kept from BOTH passes and tagged with which one produced them: a breadth row saying `ingested`
    beside a depth row saying `deferred` is the honest description of an Oracle who is covered
    recently and still owed their older window. `_ingest_presentation` reads `type`/`action` only,
    so the extra `pass` key rides along as diagnostics.

    THE TWO PASSES NO LONGER RETURN THE SAME SHAPE, and this function is the one place that ever
    assumed they did. Since `x_only` (2026-09-14) the depth pass carries X rows and nothing else,
    where it used to re-emit every blog/GitHub row the breadth pass had already produced. That is
    a simplification, not a behaviour change: `_ingest_presentation` collapsed the duplicates
    through a set comprehension, so fewer rows in gives the same presentation out. What it does
    change is the counters, which SUM — a duplicate `ingested` row used to be counted twice."""
    from pipeline.kb import oracles

    merged = dict(depth)
    for rec in breadth.get("results") or []:
        rec.setdefault("pass", "breadth")
    for rec in merged.get("results") or []:
        rec.setdefault("pass", "depth")
    merged["results"] = (breadth.get("results") or []) + (merged.get("results") or [])
    for k in oracles._MERGED_COUNTERS:
        if k in breadth:
            merged[k] = merged.get(k, 0) + breadth[k]
    merged["discovery_ran_fresh"] = bool(breadth.get("discovery_ran_fresh")
                                         or depth.get("discovery_ran_fresh"))
    # The windows this Oracle actually got, both of them. `lookback` stays the DEEP one — it is
    # what the cost-consent surface reads — and the shallow one is named beside it rather than
    # folded in, because "30 days" and "183 days" are two facts, not an average.
    merged["breadth_lookback"] = breadth.get("lookback")
    if breadth.get("available_sources"):
        merged.setdefault("available_sources", [])
        have = {(a.get("source_type"), a.get("url")) for a in merged["available_sources"]}
        merged["available_sources"] += [a for a in breadth["available_sources"]
                                        if (a.get("source_type"), a.get("url")) not in have]
    return merged


def _breadth_window(conn, canonical_id: str, breadth_since):
    """The breadth floor for ONE Oracle — the SHALLOWER of the flat breadth window and what this
    person is actually missing. Never wider than `breadth_since`, so the breadth guarantee holds.

    ⚠️ A FLAT 30 DAYS FOR EVERYONE ON EVERY CALL IS AN INFINITE LOOP, and the second live run of
    2026-09-14 is the proof. Four Oracles, a fully refilled meter: hypersoren re-walked 204 tweets
    and VictorTaelin 400, both to rediscover a window their `covered_from` already said they held
    — and karpathy and martin_casado, who had never been pulled at all, were refused right after,
    exactly as they had been four hours earlier. 67.6s and ~25 requests bought 4 atoms and covered
    nobody new. `_ordered_picks` is a STABLE ranking, so this is deterministic rather than
    unlucky: every future call spends the bucket on the same two people and strands the same two.

    Requests are the scarce resource and `snapshot_and_hash` makes the re-walk produce almost
    nothing, so those requests are pure loss to the people who have none.

    The rail has had this right all along — `oracle_refresh.refresh_pair` takes `since_for(row)`
    first and reaches for `BREADTH_WINDOW_DAYS` only when `since is None`. 30 days is a FIRST
    window there, not a recurring one. This borrows that. A never-pulled Oracle has no
    `x_since_last`, so it is unaffected and still gets the full window; a recently-pulled one
    collapses to one or two requests, which is what frees the meter for whoever has nothing.

    Fail-safe: an unreadable window leaves the flat floor in place — wasting requests is the safe
    direction against missing posts.
    """
    from pipeline.kb import oracles

    if breadth_since is None:
        return None
    try:
        last = oracles.x_since_last(conn, canonical_id)
    except Exception:
        return breadth_since
    return breadth_since if last is None else max(breadth_since, last)


def _breadth_then_depth(conn, embedder, ordered: list[dict], *, run_id: str | None = None,
                        force: bool, per_oracle: dict,
                        web_since, scholar_since, scholar_topics, x_connected: bool) -> list[dict]:
    """Give EVERY Oracle recent coverage first, then deepen — as a loop over `_ingest_oracle`.

    `ordered` arrives ALREADY RANKED (`_ingest` calls `_ordered_picks`), because the run record's
    roster and the order this walks have to be one list rather than two computations of it.

    `run_id` is the durable report. When given, each Oracle is stamped in flight at their FIRST
    visit and finished after the depth pass — which is the point `_merge_passes` has made one
    result out of two, and a reader that saw the breadth row land first would count the person
    twice and split their atoms. When None (`add_oracle`, and every test that drives this loop
    directly) nothing is recorded and the loop is unchanged.

    ⚠️ A LOOP OVER THE ENGINE, NEVER A SECOND COPY OF IT. `.guards.py`'s `retired-expand-cli-engine`
    is explicit: "if a batch 'expand everyone now' surface is ever wanted again, build it as a loop
    OVER `_ingest_oracle`". Breadth is expressed as a WINDOW, which is the only thing that makes
    that possible — `oracle_refresh` has no `breadth_pass` to reuse (breadth is three lines inside
    `refresh_pair`), and its pair machinery does not apply here anyway, because the
    substack/blog/github entities do not EXIST until `onboard_footprint` routes them.

    THE PASS ORDER IS THE RESERVATION — and since 2026-09-14 it is the ONLY one. Breadth for
    every Oracle completes before any depth pull starts, so a depth walk cannot strand a later
    Oracle's breadth, which is the exact inversion breadth exists to prevent. There is no budget
    bookkeeping beside it: the predictive reserve that used to sit in `_ingest_oracle` was
    deleted with its twin on the refresh rail, leaving `x_graphql_core._refuse_if_spent` as the
    one place that decides a request cannot be made — and it decides on x.com's own evidence.

    An Oracle whose depth window is ALREADY inside the breadth window gets one pass, not two —
    `since_last` on a person pulled yesterday is narrower than 30 days, and a "breadth" pass
    deeper than the depth pass is just a more expensive depth pass.

    `force` applies to the breadth pass ONLY. Its one effect is `reverify` on discovery, and
    discovery has already run fresh by the time depth starts; re-running it could only return what
    breadth just got, at the cost of a second network sweep.
    """
    from datetime import timedelta

    from pipeline.kb import oracles, pull_runs
    from pipeline.kb.oracle_refresh import BREADTH_WINDOW_DAYS
    from pipeline.timeparse import utc_now

    # No X session → no metered timeline → nothing for a breadth window to protect. One pass.
    breadth_since = utc_now() - timedelta(days=BREADTH_WINDOW_DAYS) if x_connected else None

    def _mark(fn, *a):
        """Recording must never be able to sink a pull. A store that cannot be written loses the
        REPORT; raising here would lose the content too, which is the strictly worse direction —
        the same fail-safe rule `_finish_in_background` follows one level up."""
        if run_id is None:
            return
        try:
            fn(conn, run_id, *a)
        except Exception:
            pass

    def _run(o, since, *, forced, x_only=False):
        return oracles._ingest_oracle(conn, embedder, o, force=forced, x_since=since,
                                      web_since=web_since, scholar_since=scholar_since,
                                      scholar_topics=scholar_topics, x_only=x_only)

    # `x_since=None` is the adapter's own 183-day default, which is DEEPER than the breadth
    # window — so None wants breadth, exactly like an explicit older date.
    def _wants_breadth(o) -> bool:
        deep = per_oracle[o["canonical_id"]]
        return breadth_since is not None and (deep is None or deep < breadth_since)

    first: dict[str, dict] = {}
    for o in ordered:
        if _wants_breadth(o):
            _mark(pull_runs.start_oracle, o["canonical_id"])
            first[o["canonical_id"]] = _run(
                o, _breadth_window(conn, o["canonical_id"], breadth_since), forced=force)

    out = []
    for o in ordered:
        cid = o["canonical_id"]
        shallow = first.get(cid)
        _mark(pull_runs.start_oracle, cid)   # no-op for anyone breadth already stamped
        # ONLY the X window differs between the two passes. Breadth is a WINDOW and the window
        # only ever bounded X, so the papers, discovery and the whole blog/Substack/GitHub route
        # ran to completion already (R4) — `x_only` stops the deep pass paying for them twice.
        deep = _run(o, per_oracle[cid], forced=force and shallow is None,
                    x_only=shallow is not None)
        merged = _merge_passes(shallow, deep) if shallow else deep
        _mark(pull_runs.finish_oracle, cid, merged)
        out.append(merged)

    out.extend(_finish_late_arrivals(conn, run_id, walked={o["canonical_id"] for o in ordered},
                                     run=_run, force=force))
    return out


def _finish_late_arrivals(conn, run_id: str | None, *, walked: set[str], run, force: bool,
                          rounds: int = 8) -> list[dict]:
    """Pick up anyone a LATER `ingest` queued onto this run while it was walking.

    ⚠️ THIS IS WHAT MAKES "queued onto the running pull" TRUE. Without it a second call could add
    rows to the roster and nothing would ever pull them — the record would say three people were
    waiting while the loop iterated a fixed list of two, and the user would be told their newest
    writer was on the way. A promise nothing keeps is worse than a refusal.

    The likely trigger is not a banned retry, it is an ordinary "oh, add one more person"
    mid-pass, which is exactly why the alternative (bouncing the second call) is wrong: it drops
    a writer the user just asked for.

    ONE PASS EACH, not breadth-then-depth. The reservation breadth exists to make — everyone gets
    recent coverage before anyone gets deepened — was made when this run opened, and a latecomer
    cannot retroactively join it. They get their own window in one visit, which is what
    `add_oracle` does for a single person anyway.

    The window comes off the ROW. `x_lookback` was an argument to the call that queued them and
    that call is gone; re-deriving it here would silently give somebody the adapter's 183-day
    default when they asked for two years.

    `rounds` bounds a pathological ping-pong (each round can itself attract new arrivals) without
    bounding ordinary use: a round that finds nothing stops immediately, so the cap is only ever
    reached by somebody queueing faster than the pull drains, and stopping there leaves the rows
    for the next `ingest` rather than spinning."""
    from pipeline.kb import oracles, pull_runs
    from pipeline.timeparse import parse_ts

    if run_id is None:
        return []
    out: list[dict] = []
    for _ in range(rounds):
        try:
            late = [o for o in pull_runs.unfinished_for(conn, run_id)
                    if o.canonical_id not in walked]
        except Exception:
            return out                      # no record to read is not a reason to stop the pull
        if not late:
            return out
        try:
            by_id = {o["canonical_id"]: o for o in oracles.confirmed_oracles(conn)}
        except Exception:
            return out
        for row in late:
            walked.add(row.canonical_id)
            oracle = by_id.get(row.canonical_id)
            if oracle is None:              # unconfirmed since being queued — nothing to pull
                continue
            try:
                pull_runs.start_oracle(conn, run_id, row.canonical_id)
                since = parse_ts(row.window) if row.window else None
                result = run(oracle, since, forced=force)
                pull_runs.finish_oracle(conn, run_id, row.canonical_id, result)
                out.append(result)
            except Exception:
                continue                    # one latecomer must never sink the run
    return out


def _review(conn, *, review_action: str, review_id: int | None,
            verification_urls: list[str] | None, confirm: bool) -> dict:
    """action='review' — user decisions for sources discovery did not attribute automatically."""
    from pipeline.kb import oracles

    # Embeddings are needed only for the one path that can actually atomize a source. Listing,
    # checking evidence, and confirmation previews are database/discovery work only.
    action = (review_action or "list").strip().lower()
    embedder = None
    if action == "add" and confirm:
        from pipeline.kb.embed import get_kb_embedder
        embedder = get_kb_embedder()
    return oracles.review_sources(conn, embedder, action=action, review_id=review_id,
                                  verification_urls=verification_urls, confirm=confirm)


def register_oracle_tools(mcp) -> None:

    @mcp.tool()
    def oracle(action: str = "screen", canonical_ids: list[str] | None = None,
               add_handles: list[str] | None = None, top_n: int = 30, floor: int = 15,
               limit: int = 40, source: str | None = None,
               force: bool = False, x_lookback: str | None = None,
               web_lookback: str | None = None, scholar_lookback: str | None = None,
               scholar_topics: list[str] | None = None, query: str = "",
               min_signals: int = 1, review_action: str = "list",
               review_id: int | None = None, verification_urls: list[str] | None = None,
               confirm: bool = False, run_id: str | None = None) -> dict:
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
        the user how far back to pull and pass their answer as `x_lookback` when X is connected.

        Why that question is not optional: the default is only ~6 months, so a user who wanted
        their Oracle's whole visible history gets a fraction of it and is never told. (It is no
        longer a COST question — X reads are free since the cutover — it is a completeness one,
        which is the same reason to ask and a different reason to give.) `action='screen'` returns the exact presets and defaults under `lookback_options` —
        offer those, don't invent your own. The web archive is FREE and already pulls in full, so
        ask about it (`web_lookback`) only if the user wants it NARROWED.

        To add someone NOT in the list, pass them in `add_handles` — an X @handle, or ANY http
        URL: a Substack, a personal blog, a newsletter, a site. A non-Substack home becomes a
        `blog:{host}` entity and is ingested through the blog adapter. They're resolved and added
        on the spot. This is the whole path for a user who follows named writers rather than
        accounts on a platform — there is no list to screen, so nothing else surfaces them.

        `action`:
          • "screen"  (default) — the ranked candidate payload. Each candidate carries its
                       `reflected` signal ("you follow · subscribe · bookmarked 12×"), `pre_ticked`
                       / `shown_by_default` / `is_person` flags, `distinct_signals`, and its
                       `canonical_id` (pass these to confirm). `classify.ran=False` means the kind
                       classifier degraded open (LLM unavailable) — everyone stays person-eligible.
          • "confirm" — commit Oracle picks. `canonical_ids` = the ones the user kept (verbatim from
                       screen); `add_handles` = raw X handles or any http URL to add (resolved-at-
                       confirm). Idempotent. Returns {confirmed, unresolved, unknown, refused,
                       total_oracles} — surface BOTH `unresolved` (a lookup couldn't find it) and
                       `refused` (found it, declined it: a multi-author company/team/publication
                       site cannot be an Oracle root). Read each `refused` entry's `reason` back
                       to the user — it names the two things that DO work for that site.
          • "ingest"  — deep-ingest confirmed Oracles into the atom-KB. For each (all confirmed, or
                       just the `canonical_ids` you pass) run discovery — which mines their trusted
                       blog/Substack for their OTHER profiles — then ingest each trusted personal
                       profile as atoms attributed to the Oracle; an org link becomes an affiliation;
                       an ambiguous one is left out. X is collected only when X is connected.
                       Tell the user from `presentation`, which groups completed work, optional
                       sources, and a first-value tour. `results` is diagnostic detail, not prose
                       to repeat to the user.
          • "progress" — WHERE A PULL HAS GOT TO. Pass `run_id` for a specific one; omit it for
                       the most recent. It BLOCKS for up to ~40 seconds, returning the moment a
                       writer lands, so calling it repeatedly is how you follow a pull rather
                       than a way to poll at it.

        ⚠️ AFTER `ingest`, LOOP ON `progress` UNTIL `status` IS NOT "running". That is not
        optional and it is not polling — a first pull takes minutes, and this is the only way you
        will ever learn it finished. Between calls, tell the user who just landed (`done` carries
        names and counts). When `status` becomes "complete" or "stopped", `report` carries the
        full result and `report.presentation` is what you read to the user.

          · "running"  — still going. Call again. `waiting` names writers not yet reached; say
                         "not yet", never anything about why, and never that they are missing.
          · "complete" — every writer on the roster is done.
          · "stopped"  — the pull died (the app was quit, the machine slept). `done` is real and
                         durable; the rest simply has not happened. Offer to finish it — a fresh
                         `ingest` picks up exactly where this stopped. Do NOT call it a failure,
                         a crash, or an error to the user.

        NEVER say a pull "timed out" because `progress` returned with nothing new. It answers on
        a clock so that YOU can speak; the pull neither sees that clock nor is cut by it. A
        writer deep inside a blog archive produces no news for a minute at a time, and that is
        the system working.

        IF THIS CALL EVER RETURNS AN ERROR INSTEAD OF A RESULT — a timeout, a transport failure,
        a connection drop — three rules, and they are absolute:

          1. **Do not start another ingest.** Not the same one, and NOT a smaller one either:
             calls to this server run one at a time, so a "smaller" second call waits for the
             first to finish and then times out having done less. Splitting the work is the
             mitigation that GUARANTEES the failure it was meant to avoid — measured twice.
             The pull is running; you lost the report, not the work. Call
             `oracle(action='progress')` — that is exactly what it is for, and it will hand you
             the whole report once the pull lands.
          2. **Never say "timed out", "the request failed", or "error" to the user.** They did not
             make a request; they asked for a library, and it is filling. Say that: "OPYT is still
             pulling their writing in the background — I'll have more shortly." No mechanism, no
             apology, no offer to retry.
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
          • "review" — list possible sources discovery left out, or act on one `review_id`.
                       `review_action='add'` previews then needs `confirm=True` to include only
                       that source; `verify` re-checks supplied official `verification_urls` but
                       does not add anything; `dismiss` previews then needs `confirm=True` to
                       permanently leave the source out. `leave_out` is simply no action. Tell
                       the user from `items`; `diagnostics` preserves the discovery reason for
                       the host without making it onboarding prose. Items with no available
                       collector offer only dismiss or leave-out.

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
            action: "screen" | "candidates" | "confirm" | "ingest" | "review".
            query: candidates — a topic question to rank people by ("agent memory", "biotech
                funding"). Omit for "who has been sampled".
            min_signals: candidates — only people with at least this many distinct curation
                signals (default 1 = everyone in the list).
            canonical_ids: confirm — the kept candidates' canonical_ids; ingest — which confirmed
                Oracles to ingest (omit = all confirmed — usually NOT what you want for a
                top-up, so name the person).
            add_handles: confirm — people to add beyond the ranked list: raw X @handles, or any
                http URL (Substack, blog, newsletter, personal site).
            top_n: candidates — how many people to return (default 30). NOT a screen knob —
                `screen` has its own pair, `floor` and `limit`.
            floor: screen — minimum candidates shown by default before "see all" (default 15).
            limit: screen — the MOST candidate cards to return, in rank order (default 40). A
                TOTAL, with no exemption for pre-ticked people: a real store has hundreds of
                those and the full list runs past the token limit, which is what this prevents.
                `recommended_count` and `total_candidates` still report the true totals, and
                `omitted` appears whenever anything was cut. Raise it a page at a time if the
                user wants to see further down. CLAMPED at 80 — a bigger number is not
                honored, it is reported back as `limit_clamped`. To reach a whole group use
                `source=`; to find ONE person use `oracle(action='candidates', query='...')`.
            source: screen — show only people who reach the user through ONE platform ('x',
                'substack', …; the store's own vocabulary). THE ANSWER TO "show me my Substack
                writers", which otherwise has no bounded form and invites a `limit` big enough to
                blow the token budget. Membership, not exclusivity: someone you follow on X and
                subscribe to on Substack matches both. Ranks and pre-ticks do not move — this
                narrows who is listed, never how anyone scored.
            force: ingest — the "ignore what we decided last time" override, and it now means TWO
                things. (1) Ingest a footprint source even when the single-author gate would skip
                or leave it out (for a solo publication the classifier mislabels). (2) Re-run DISCOVERY
                from scratch, ignoring the trust cache — reach for this when a person's sources
                look wrong or incomplete and their X profile has not changed, since the cache key
                only invalidates on a display-name or declared-link change. A source they created
                after the last run, or a fix on our side, leaves that key identical.
            x_lookback: ingest — when X is connected, how far back to pull each Oracle's X
                timeline: "6mo" (the default), "1yr", "2yr". Hard-capped at 2 years whatever
                you pass. Ask the user only after X is connected — not for cost (the pull is
                free), but because the default truncates.
                "since_last" is the cheap top-up: pull only what has appeared since this Oracle
                was last pulled. Use it whenever the user wants someone brought current rather
                than re-ingested — a 5-day gap costs one request instead of ~19. It errors rather
                than guessing if the Oracle has never been pulled.
            web_lookback: ingest — how far back to pull the Substack/blog archive: "1yr",
                "2yr", "5yr", "all" (the default). Only narrows; omit unless the user asks.
            scholar_lookback: ingest — how far back to pull their papers: "2yr", "5yr", "10yr",
                "all" (the default). Free and abstract-only, so the whole corpus is the default —
                a researcher's foundational paper is usually not their most recent. Ask this one
                WITH NUMBERS: `add_oracle`'s preview reports `scholar_counts` for the person.
            scholar_topics: ingest — narrow their papers to these OpenAlex topic ids, taken from
                `add_oracle`'s preview under `scholar_topics`. The SUBJECT filter, and usually
                the more useful of the two: a researcher's record runs to 174 subjects and a
                venue's to 200+, and the user wants a handful. It PERSISTS onto the source — every
                future refresh pulls only these subjects — so pass it only for the ONE Oracle
                named in `canonical_ids`. Omit it to leave an existing choice untouched (what a
                top-up wants); pass `[]` to go back to the whole corpus.
            review_action: review — "list" (default), "add", "verify", or "dismiss".
            review_id: review — the item identifier returned by review_action="list".
            verification_urls: review/verify — official profile, site, or bio URLs that may
                connect the writer to this source.
            confirm: review/add and review/dismiss — set only after the preview the user approved.
        """
        from pipeline.kb import schema

        conn = schema.connect()
        try:
            if action == "screen":
                return _screen(conn, floor=floor, limit=limit, source=source)

            if action == "candidates":
                return _candidates(conn, query=query, top_n=top_n, min_signals=min_signals)

            if action == "confirm":
                return _confirm(conn, canonical_ids=canonical_ids, add_handles=add_handles)

            if action == "ingest":
                return _ingest(conn, canonical_ids=canonical_ids, force=force,
                               x_lookback=x_lookback, web_lookback=web_lookback,
                               scholar_lookback=scholar_lookback,
                               scholar_topics=scholar_topics)

            if action == "progress":
                return _progress(conn, run_id=run_id)

            if action == "review":
                return _review(conn, review_action=review_action, review_id=review_id,
                               verification_urls=verification_urls, confirm=confirm)

            return {"error": f"unknown action {action!r} — use 'screen', 'candidates', 'confirm', "
                             f"'ingest', 'progress', or 'review'."}
        finally:
            conn.close()

    @mcp.tool()
    def add_oracle(reference: str, confirm: bool = False,
                   x_lookback: str | None = None, web_lookback: str | None = None,
                   scholar_lookback: str | None = None,
                   scholar_topics: list[str] | None = None,
                   extra_source_urls: list[str] | None = None, force: bool = False) -> dict:
        """Add a person to your knowledge base as an **Oracle** — a trusted source OPYT
        deep-ingests from their verified footprint and roots trust on. X is included only when the
        user has connected X; discovering an X profile does not enable a pull. This
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
          • `x_lookback` — when X is connected: "6mo" (default) / "1yr" / "2yr". Hard-capped at
            2 years whatever you pass. Ask only after X is connected; the ~6-month default silently
            leaves most of a prolific account's history out.
            For someone ALREADY on the roster, "since_last" pulls only what is new since their
            last pull — the cheap top-up, roughly one request for a few days' gap. Reach for it
            when the user says "update" or "catch up", not "add".
          • `scholar_lookback` — "2yr" / "5yr" / "10yr" / "all" (default). Their papers. Free
            and abstract-only, so the whole corpus is the default. The ONE selector that can ask
            with real numbers — the preview reports `scholar_counts`.
          • `web_lookback` — "1yr" / "2yr" / "5yr" / "all" (default). A durable archive, so it
            already pulls everything. Pass it only to NARROW.
        A single shared window would be wrong for one of the two by construction — it either
        over-pulls X or truncates the archive. The result echoes the windows that actually ran
        (including the X clamp) under `lookback` — TELL the user how far back you pulled.

        Reading the result — use `presentation` to talk to the user. It groups completed work,
        possible profiles left out because ownership could not be verified, optional X profiles,
        and an opt-in first-value tour. `ingest` retains the raw per-source report for diagnostics;
        do not turn its action names or transport details into user-facing copy.

        The result may ask you to do something — check for `followup`. A confirm=True result
        carries one, and acting on it is how this tool finds a person's blog or newsletter
        at all. OPYT's three probes are deterministic (X bio, Substack convention,
        GitHub); the open-web step is YOURS, because you have web search and OPYT would otherwise
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
                existing choice alone; pass `[]` to clear it.
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
            out = oracles.add_oracle(conn, embedder, reference, confirm=confirm,
                                     x_lookback=x_lookback, web_lookback=web_lookback,
                                     scholar_lookback=scholar_lookback,
                                     scholar_topics=scholar_topics,
                                     extra_source_urls=extra_source_urls, force=force)
            if confirm and out.get("ingest"):
                out["presentation"] = _ingest_presentation([out["ingest"]], conn)
            return out
        finally:
            conn.close()
