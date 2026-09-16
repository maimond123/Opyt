"""
opyt_core/suggest.py — what this store can support right now, measured rather than listed.

⚠️ WHY THIS SHIPS NUMBERS AND NOT DESCRIPTIONS. The host already holds every tool's definition,
and those definitions already say when to reach for each one — `sitting`'s says it is for "what
have I been collecting on X" and explicitly not for looking something up. What the host cannot
know is whether THIS store can answer the question it is about to route: whether a sitting on a
topic would assemble four atoms or four hundred, and whether they came from one voice or twelve.
So a suggestion that re-describes the tools spends its budget on what the host already has, and a
suggestion that reports measurements spends it on the only thing missing.

That is also why this stays short without being vague. `_trial_prompt` records the rule these
strings live under: they are instructions to a model that expands them, so a five-item list
becomes five paragraphs. Measurements resist that — "14 items, 3 people, Feb–Sep" has one
reading, while "build a focused reading session" invites a paragraph explaining what that means.

⚠️ THE OPTIONS ARE INTENTIONS, NOT TOOLS, and that distinction is the whole discipline here.
The block this replaced listed `search`, `aggregate` and `sitting` — implementation restated, and
already described better by the tool definitions the host holds. What a user cannot be told by any
docstring is which DIRECTION they want to go next, because only they know:

  • WIDER  — more people writing about what they already read.
  • DEEPER — read what is already here on one subject, end to end.
  • RICHER — put a subject on standing watch so OPYT goes and gets more ON IT from outside.

Those are three different things to want, and choosing between them is real information the user
holds and nobody else does. A menu of tools is a feature list; a menu of directions is a question.

RICHER IS THE ONE THAT WAS BUILT AND NEVER OFFERED. `frontier_queries.add_user_query` puts a
user-typed subject on the watchlist against arXiv, GitHub and OpenAlex, it needs no sitting, no
read and no spend, and `_still_open` listed it as bullet four under ways to FEED OPYT. That framing
is backwards: naming a subject is the one move where OPYT does the work and the user does not, and
it is most valuable in exactly the store that can do least — a thin one, where DEEPER is blocked
and the honest answer used to be "collect more people", which is homework.

WHERE THE JUDGEMENT LIVES. `search`'s two channels set the line this follows: `insights` ships
VALUES and says "no thresholds live here; the host decides what is alarming", and `notices` ships
finished sentences only about what a CALL did. This module sits on the `insights` side — it picks
which facts to measure, never what the user should therefore do. The one judgement it does make,
`best_lens`, is delegated to `sitting_surface.lens_warnings` so the answer matches what `sitting`
itself will say one call later.

FAIL-SAFE, AND WHAT THAT MEANS HERE. An unreadable store returns `{}` — not an error, and never a
claim about data that is not there. An empty store returns `{}` too: "you have 0 items" is a
sentence about absence that a caller would have to suppress anyway, and the callers that matter
(`onboard`'s handoff, a fresh `aggregate`) already say it better in their own words.
"""

from __future__ import annotations

# WIDER's call depends on whether a collector already surfaced people. With candidates waiting,
# screening them is work OPYT already did; without, widening starts from the user naming
# somebody IN PLAIN WORDS — and stops there. Resolving a name to a URL, handle or ORCID is the
# host's job (web search), never the user's; asking a user to paste an identifier is the exact
# homework this rebalance exists to stop.
_WIDER_SCREEN = "oracle(action='screen')"
_WIDER_NAME = "oracle(action='confirm', add_handles=['https://…'])"

# Named in the text because it is the only brake that holds across a verbose model and a terse
# one. See `_trial_prompt` and `_what_opyt_can_do` for the two prior places this was learned.
_CAP = ("Put these to the user as ONE short question — a choice between directions, one line "
        "each, in your own words. Do NOT recite the list, do NOT name a tool that is not here, "
        "and do NOT explain what the tools do. If they pick one, make the `call`. This covers "
        "`choices` only: `orient`, when present, is work you do first and never read out.")


def _month(ts: str | None) -> str | None:
    """An ISO timestamp → "2026-02", or None. Trimmed rather than formatted: a month name would
    have to pick a locale, and every other date this surface reports is ISO."""
    return ts[:7] if ts and len(ts) >= 7 else None


def _have(conn, total: int) -> str:
    """The one-line inventory every suggestion hangs off — counts first, span only if dated.

    Written as a fragment rather than a sentence so the host has to put it in its own words. A
    finished sentence here would get repeated verbatim, and this belongs in the host's voice
    alongside whatever else it is saying about the call that carried it.

    Measured with its own one-row query rather than read off the aggregate or the topic shape,
    and both of those were tried first:

      • `len(agg["top_entities"])` UNDERSTATES. That list is `LIMIT 15`, so every store with more
        than fifteen authors reported exactly fifteen — a 23-author store said "15 people".
      • The top topic's shape describes the TOPIC. Using its span for the store's line said a
        1,240-atom store spanned one month, because the busiest tag happened to. (That seed is
        gone as of 2026-09-16 — the shape below is measured over the whole store — so this one
        can no longer fire. It stays as the record of why this query exists at all: the next
        person to reach for a convenient nearby number should see that both convenient numbers
        were wrong in ways nothing raised.)

    Both were silent wrongness — plausible numbers, no error — which is the failure mode this
    whole surface is most exposed to, since nothing downstream can check a count it was handed.
    """
    row = conn.execute("SELECT COUNT(DISTINCT who_id), MIN(when_ts), MAX(when_ts) "
                       "FROM atoms").fetchone()
    people = row[0] or 0
    parts = [f"{total} items", f"{people} {'person' if people == 1 else 'people'}"]
    first, last = _month(row[1]), _month(row[2])
    if first and last:
        parts.append(first if first == last else f"{first}–{last}")
    return " · ".join(parts)


def _frontier_waiting(conn) -> int:
    """Unseen Frontier candidates, or 0 if the queue is unreadable or absent.

    Wrapped bare because Frontier is optional: a store that never ran a frontier pass has no such
    table, and that is a missing input rather than a failure — the invariant is that an absent
    optional input produces fewer suggestions, not a crash.
    """
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM frontier_candidates WHERE status = 'new'").fetchone()[0]
    except Exception:
        return 0


def _watched(conn) -> set[str]:
    """The subjects already on standing watch, lowercased. Empty if the queue is unreadable.

    Read so RICHER never offers to start watching something already being watched — an offer the
    user accepts and that then reports "still_retired" or silently changes nothing is worse than
    no offer, because it spends their turn to tell them nothing happened.
    """
    try:
        from pipeline.kb import frontier_queries as fq
        return {(w.get("text") or "").strip().lower() for w in fq.watchlist(conn)}
    except Exception:
        return set()


def _candidates_waiting(conn) -> int:
    """People a collector already surfaced and nobody has screened. 0 if unreadable."""
    try:
        from pipeline.kb import screen
        return len(screen.rank_candidates(conn))
    except Exception:
        return 0


def suggestions(conn, agg: dict) -> dict:
    """Where this user could go next, as a choice between directions measured against the store.

    `agg` is a `kb_aggregate` result the caller already has — passed in rather than recomputed
    because both callers have just built one, and running the same pure-SQL pass twice to decorate
    its own output would be the kind of waste this module exists to avoid.

    ORDERING IS THE ARGUMENT. DEEPER leads when the store can support it, because reading what you
    already have is the thing the material is for. RICHER leads when it cannot, because a thin
    store's honest next move is to grow the SUBJECT, not the roster. WIDER is always last and
    never omitted: more people is a legitimate thing to want, and it was the only thing on offer
    for so long that demoting it is the point — removing it would just be the same mistake pointed
    the other way.

    Returns `{}` when there is nothing honest to say. Callers treat that as "add nothing", which
    is why every early return here is bare rather than a message about emptiness.
    """
    try:
        total = agg.get("total") or 0
        if not total:
            return {}

        from pipeline.kb import sitting_surface as ss

        # ⚠️ THE SUBJECT IS NOT IN THE STORE, AND PRETENDING OTHERWISE IS THE BUG THIS FIXES.
        # Until 2026-09-16 both offers below were seeded from `agg["top_topics"][0]` — the most
        # common `payload.source_tags` value. That field is author-declared hashtags, captured
        # faithfully and never meant as a taxonomy; on the live store it covered 5 atoms of
        # 1,801, so the "top topic" was a tag with count 1 and every offer built on it named a
        # subject the user had never chosen and the corpus was not about. RICHER's own note
        # already conceded the point — *"what they say they care about beats what their tags
        # happen to say"* — so the tag was known to be the weaker input while still being the
        # one wired in. Now there is no automatic subject: the store is measured WHOLE, and the
        # naming of what to read or watch belongs to the user, asked by the host.
        shape = ss.store_shape(conn)

        # ⚠️ DEEPER IS GATED ON TIER, NOT ON A CLEAN LENS, AND THE DIFFERENCE IS A FALSE SENTENCE.
        # `lens_warnings` describes a BUILT REGION — the thing a `sitting` actually assembles —
        # and every one of its rules can fire for a reason that has nothing to do with having
        # too little material. Gating on `best_lens(...) is not None` was survivable only while
        # the shape came from one tag, where the measured set roughly WAS the set to be read.
        # Measured over the whole store it broke immediately: on the live store (1,801 atoms)
        # exactly one atom carries no date, that single warning spoils every lens, and the
        # suggestion this produced was *"You have only 1801 items here, which is too few to read
        # end to end"* — a claim about the user's own material that is not close to true and
        # that nothing downstream could contradict.
        #
        # So thinness is asked directly, of the one property that means it: the reading tier.
        deep_enough = shape["tier"] == "standalone"
        lens, missing = ss.best_lens(shape)

        choices = []

        # ── DEEPER ──────────────────────────────────────────────────────────────
        # The numbers describe the WHOLE store, so they say a reading is supportable at all —
        # not that any one subject inside it is. The subject stays a placeholder on purpose:
        # `sitting` re-measures whatever query it is handed, so the narrower subject gets its
        # own honest shape one call later and nothing here has to guess which one is wanted.
        #
        # The lens rides along only when the store's shape picks one cleanly, and it is a HINT.
        # Naming one regardless would be the same overreach in a smaller place: a lens chosen
        # against the whole store is not chosen against the region the user's query will build.
        if deep_enough:
            choices.append({
                "direction": "deeper",
                "call": (f"sitting(query='…', lens='{lens}')" if lens else "sitting(query='…')"),
                "why": (f"{shape['atoms']} items here, "
                        f"{shape['authors']} {'author' if shape['authors'] == 1 else 'authors'}"
                        + (f", spanning {shape['days']} days" if shape.get("days") else "")
                        + " — enough to read a subject end to end, in order."),
                "note": "Ask the user which subject, in their own words, and put it in `query`. "
                        "The counts above are the whole store, not that subject — `sitting` "
                        "measures the region your query actually builds and says so if it is "
                        "too thin, or a poor fit for the lens.",
            })

        # ── RICHER ──────────────────────────────────────────────────────────────
        # Offered on its own merits, not as a consolation prize for a thin store: a user with a
        # deep corpus still has nothing coming in from OUTSIDE it until a subject is watched.
        #
        # ALWAYS OFFERED NOW, because the thing it used to be conditional on is gone. The guard
        # was `top_tag not in _watched(conn)` — do not offer to watch what is already watched,
        # an offer the user accepts that then changes nothing. That principle is intact; it
        # just cannot be enforced at offer time against a subject nobody has named yet. So the
        # watched list rides along and the host avoids re-proposing one, which is the same
        # brake applied one step later, where the subject actually exists.
        already = sorted(_watched(conn))
        thin = "" if deep_enough else (f" You have only {shape['atoms']} items here, which is "
                                       f"too few to read end to end — this is how that "
                                       f"changes.")
        choices.append({
            "direction": "richer",
            "call": "sitting(action='watchlist', add=['…'])",
            "why": ("Name a subject and OPYT pulls arXiv, GitHub and OpenAlex on it from here "
                    "on — the first pull starts in the background the moment it is added. "
                    "Nobody to name." + thin),
            "note": ("Get the subject from the user in their own words — the watch takes any "
                     "phrase they give. And offer the survey either way: a web-search over the "
                     "subject's current landscape enriches beyond what the watch pulls — its "
                     "papers, repos and recurring authors are candidates to save and confirm — "
                     "and when the subject is broad, the threads the user picks from it make "
                     "sharper watches."
                     + (f" Already on watch, so do not propose these again: "
                        f"{', '.join(already)}." if already else "")),
        })

        # ── WIDER ───────────────────────────────────────────────────────────────
        waiting = _candidates_waiting(conn)
        choices.append({
            "direction": "wider",
            "call": _WIDER_SCREEN if waiting else _WIDER_NAME,
            "why": (f"{waiting} people a collector already found are waiting to be screened."
                    if waiting else
                    "More people. The user gives names in plain words — then YOU do the "
                    "finding: web-search for each person's site, X handle, ORCID or OpenAlex "
                    "page and pass the URL. Never ask the user for a link or an id. If they "
                    "have no names, offer to find some: ask what subject, then web-search "
                    "who writes credibly on it — prefer people whose actual work you can "
                    "link, not follower counts — and bring the picks back as candidates to "
                    "confirm, never auto-added."),
        })

        # ── ORIENT ──────────────────────────────────────────────────────────────
        # NOT a fourth direction — it is what the host does BEFORE putting the question, so
        # that "which subject?" can be a recognition instead of a cold ask. Both DEEPER and
        # RICHER need a subject and neither can supply one: the tag seed that used to fake it
        # is gone, and nothing else in the store proposes subjects (`search` and `sitting` both
        # require a query — the index can verify a subject and can never suggest one).
        #
        # GATED ON THE ENVELOPE ALREADY BEING PARTIAL, derived rather than a threshold: the
        # aggregate carries `recent_descriptions`, so a store no larger than that list is one
        # the host can already see whole and a census would only re-read. The moment there is
        # more store than envelope, the host is looking at a corner and does not know it.
        #
        # ⚠️ BUILT FIRST, AND EMITTED FIRST, BECAUSE POSITION IS THE INSTRUCTION. It used to be
        # assigned into `out` after construction, so it serialized LAST — after `cap`, which
        # says "Put these to the user as ONE short question". A host reading top to bottom met
        # "ask now" before it met "do this first", and on the live run (2026-09-16) it did
        # exactly that: the census only happened because the user asked for it by hand. The key
        # order IS the ordering of the work, so a step that must precede the question has to
        # precede the instruction to ask it.
        seen = len(agg.get("recent_descriptions") or [])
        orient = {
            "call": "aggregate(sample=200)",
            "why": (f"You can see {seen} of {total} items from here, the most recent ones "
                    f"— not enough to know what this corpus is about."),
            "note": ("Do this BEFORE putting the question, and do not read it out: it is "
                     "how you find the subjects to offer. `corpus_sample` walks authors in "
                     "rounds, so it shows breadth rather than the busiest corner. Name the "
                     "subjects you see, CHECK each with `search(query=…)`, keep the ones "
                     "that come back strong from several authors, and offer those. A "
                     "subject you name without checking is a guess about the user's own "
                     "material that neither of you can catch."),
        } if seen and total > seen else None

        out = {
            **({"orient": orient} if orient else {}),
            "have": _have(conn, total),
            "cap": _CAP,
            "choices": choices[:3],
        }
        # Only when the store genuinely cannot support a reading. It used to key off `not lens`,
        # so a 1,801-atom store reported "sitting: not yet" because one atom lacked a date.
        if not deep_enough and missing:
            # Kept, but demoted from an option to a footnote, and it now sits BESIDE a door
            # rather than being the whole answer: RICHER above is what a user does about it.
            # It names the observed value and the threshold, both straight from `lens_warnings`,
            # so the number here is the number `sitting` quotes if they try anyway.
            out["not_yet"] = {"sitting": missing[0]}
        return out
    except Exception:
        # One bare guard for the whole body, not per-query: every input here is optional, and a
        # suggestion is the most droppable thing in any response that carries it.
        return {}
