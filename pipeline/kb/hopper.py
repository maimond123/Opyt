"""
pipeline/kb/hopper.py — ONE deposit surface. Hand it anything OPYT can ingest; it works out what
the thing is and routes it to the adapter that already handles it.

The name is the Minecraft hopper: you drop items in the top, the machinery underneath sorts them,
and the user never aims. Five kinds go in — a paper, a github repo, a Substack post, a plain
article, a single X post — and one atom comes out.

What this is not (do not re-propose either one):

  • It is not sleep-time REASONING. The synthesis layer that pre-derived insight over existing
    material was deleted 2026-08-06 with the lesson "structure without a forcing loop dies — the
    read side was dark", and the settled direction is find-then-read over machine-verifiable
    records. Hopper does not violate that, because routing is machine-verifiable: a url either is
    an arXiv link or it is not, and you can check. Hopper produces PLACEMENT, never opinion.
  • It is not a way into the trust graph. Hopper may create an ENTITY (atomizing anything does
    that — it is the substrate identity resolution runs on). It must NEVER create an ORACLE.
    `add_oracle` is the human gate on who becomes a tracked, trusted person, and routing around it
    with unvetted input is the one thing this surface could break.

PROVENANCE — every dumped atom is stamped `entry_mode='user-saved'`, same as a bookmark: the
Frontier query generator selects on that mode, so a hand-dump STEERS future standing queries
just like a bookmark does.

Why preview→confirm (borrowed from `add_oracle`): a wrong route is SILENT — a paper filed as a
blog post does not throw, it sits wrong forever. So the default call writes nothing and reports
where the thing is headed; the caller shows that to the user and calls again with confirm=True.
"""

from __future__ import annotations
from pipeline.kb.rail_runtime import rail_budget_exhausted

import sqlite3

from opyt_core.credentials_registry import SERVICES as _CRED_ENV
from pipeline import llm_client

from . import ingest_common
from . import link_router

# ── the resetting daily seatbelt ────────────────────────────────────────────────
# A runaway guard, not a considered budget: Hopper is cheap per call, so hitting $1.00/day means
# something is looping, not that the user saved a lot. Both preview and confirm refuse on trip,
# since a preview that works while confirm can't is a confusing half-state.
HOPPER_DAILY_USD = 1.00

# ONE constant, read by BOTH the `@llm_client.rail` decorator on `save` and by
# `_daily_budget_exhausted` — two matching string literals would drift, and a drifted pair fails
# silently (the meter fills under one name, the ceiling reads an empty meter under the other).
RAIL = "hopper"


def _daily_budget_exhausted() -> bool:
    """Has this rail's recorded spend today reached its ceiling? See
    `rail_runtime.rail_budget_exhausted` for why it is never the global total."""
    return rail_budget_exhausted(RAIL, HOPPER_DAILY_USD)

# What the paid stage will actually spend, per kind. Shown in the preview so a confirm is an
# informed one. Deliberately prose, not numbers-except-where-measured: the only hard price here is
# twitterapi.io's thread call. Everything else is "free fetch + metered LLM/embedding", and a fake
# precise figure would be worse than an honest shape.
#
# The GitHub variable name is read from the credential registry, not spelled here. This blurb is
# shown to a user deciding whether to confirm a spend, and "set this variable" is the action it
# implies — so it has to name the variable that actually exists today.
_COST = {
    "paper": "free metadata + PDF fetch, then a metered embed. Skips entirely if already stored.",
    "github": f"2 free GitHub API calls (60/hr unauthenticated, 5000 with "
              f"{_CRED_ENV['github']}), then a metered embed.",
    "substack": "free public fetch, then metered image reads + embed. Paywalled posts are skipped.",
    "article": "free page fetch, then a metered content-quality gate, image reads and embed. "
               "One extra free RSS fetch only if the page carries no date.",
    "x": "free reads on your own x.com session (the post itself, plus its conversation), "
         "then metered image reads + embed. Needs a logged-in X session in a local browser.",
}

_WHY_BASIS = {
    "sniffed": "the URL host says so — a fact, checked offline",
    "hint": "your kind hint; the URL host matched nothing known",
    "fallback": "nothing matched, so it is treated as a plain article",
}


def _thin_metadata_warning(conn: sqlite3.Connection, atom_id: str | None) -> str | None:
    """Did the atom land with a BODY but no identity — no title, no date, no real author?

    A throttled metadata fetch (e.g. Semantic Scholar) can leave a paper with real full text but
    "Untitled"/no date/placeholder author, and papers are immutable so that can never be
    repaired. Hopper must not answer "saved" and stop on that — it reports the degradation so
    the user isn't surprised later. Shape-based (any atom with no date+title, any adapter), read
    AFTER the write since Hopper passes no sink.
"""
    if not atom_id:
        return None
    row = conn.execute("SELECT when_ts, description, who_id FROM atoms WHERE atom_id=?",
                       (atom_id,)).fetchone()
    if row is None:
        return None
    when_ts, description, who_id = row[0] or "", row[1] or "", row[2] or ""
    untitled = "Untitled" in description
    placeholder_author = who_id.startswith("paper-authors:")
    if not (untitled and not when_ts):
        return None
    who = " and no resolvable author" if placeholder_author else ""
    return ("stored WITHOUT metadata — no title, no date" + who + ". The full text IS indexed and "
            "searchable, but this atom will never answer 'who wrote this, and when'. Cause is "
            "usually a throttled metadata lookup while the document fetch succeeded. Papers are "
            "immutable once written, so re-saving will NOT repair it.")


def _x_preview_card(url: str) -> tuple[str | None, str | None]:
    """For an X post: `(description, problem)` — a one-line "here is what this post is", or a
    reason we could not read it. FREE — it reads this machine's own X session; see
    `ingest_x.peek_tweet`.

    It exists because x.com serves a JS shell to unauthenticated fetchers, so unlike an
    article/paper/repo/Substack post (which the host model can fetch and describe itself), a bare
    status link tells the model nothing verifiable. It used to be the one place the preview SPENT
    (~$0.00015 through twitterapi.io); that is no longer a reason to skip it.

    Reuses `derive_x`'s mechanical description, so what you approve is literally what gets stored."""
    from . import derive, ingest_x, link_router

    tid = link_router.parse_tweet_id(url)
    if not tid:
        return None, None
    norm = ingest_x.peek_tweet(tid)
    if not norm:
        return None, ("could not read this post — it may be deleted, protected or suspended, or "
                      "the X session in your browser may have expired. A confirm would most "
                      "likely fail and store nothing.")
    try:
        return derive.derive_x(norm)["description"], None
    except Exception:                       # a malformed payload costs the card, never the preview
        return None, None


def preview(conn: sqlite3.Connection, reference: str, *, kind_hint: str | None = None,
            enrich: bool = True) -> dict:
    """Where would this go, and do we already have it? No writes, ever.

    Cheapness is a contract meaning "cheap enough that nobody skips it", not "literally zero":
    ZERO network for article/paper/github/substack (the host model already read those), ~$0.00015
    for an X post since a bare status link is otherwise unverifiable. `enrich=False` disables even
    that (the confirm path passes it so a save never fetches the same post twice).

    No TITLE for the four free kinds — a title costs a page fetch that would duplicate what the
    caller already has."""
    ref = (reference or "").strip()
    kind, basis = link_router.classify_reference(ref, hint=kind_hint)
    if kind is None:
        return {"routable": False, "reference": ref, "kind": None,
                "saw": ("not an http(s) URL" if ref else "empty reference"),
                "error": "cannot route this — Hopper takes a URL (an article, a paper, a github "
                         "repo, a Substack post, or an X post). Nothing was written."}

    atom_id = link_router.predicted_atom_id(ref, kind)
    present = bool(atom_id) and link_router.atom_present(conn, atom_id)
    out = {
        "routable": True, "reference": ref, "kind": kind, "why": _WHY_BASIS.get(basis, basis),
        "atom_id": atom_id, "already_present": present,
        "entry_mode": "user-saved", "cost": _COST.get(kind, "a metered embed."),
    }
    if kind == "substack":
        # The only kind whose id is not derivable offline — it keys on the post's numeric id.
        out["note"] = ("the atom id for a Substack post is only known after the fetch, so "
                       "'already present' cannot be answered here; the adapter dedups on it.")
    elif kind == "github":
        # The store keys on the API's canonical owner casing, which the URL may not match.
        out["note"] = ("the github atom id uses the API's canonical owner casing, so this id is a "
                       "best guess from the URL; a casing mismatch costs one re-fetch, not a twin.")
    if present:
        out["note"] = "already in the knowledge base — a confirm would be a no-op, no fetch, no spend."
        return out                          # nothing to verify, so nothing to spend verifying it
    if kind == "x" and enrich:
        card, problem = _x_preview_card(ref)
        if card:
            out["description"] = card       # what the atom will carry, verbatim
        elif problem:
            out["unreadable"] = problem
    return out


@llm_client.rail(RAIL)
def save(conn: sqlite3.Connection, embedder, reference: str, *, kind_hint: str | None = None,
         confirm: bool = False, profile: str | None = None) -> dict:
    """Route one reference into the atom KB. Two-phase: `confirm=False` (the default) PREVIEWS and
    writes nothing; `confirm=True` runs the paid ingest.

    The one in-process rail (the other four run as detached children). `llm_client.rail` restores
    the PREVIOUS label on exit rather than resetting, so an inner `hopper` call inside a
    long-lived MCP server can't un-attribute an outer rail's run.

    The label sits here (not on the `hopper` MCP tool) because this is the importable entry every
    caller goes through, covering the preview and the confirm alike. (It used to be worth naming
    the preview's own spend, ~$0.00015 for an X read through twitterapi.io; that read is free on
    the user's own session since 2026-08-30, and only the confirm costs anything now.)

    Returns the preview dict, or on a confirm `{status, kind, atom_id, entry_mode, …}` where status
    is one of:
      • "already_present" — the atom was there. No fetch, no spend, nothing written.
      • "saved"           — a new (or changed) atom is in the store.
      • "rejected"        — fetched fine, but the content gate found no substantive units. Not an
                            error: the page was nav/promo/boilerplate. Nothing stored, by design.
      • "blocked"         — the host stopped us (a bot-check or a challenge shell). Retryable.
      • "failed"          — the fetch failed, or the URL is not what its kind claims.
      • "unroutable"      — not a URL at all.
      • "budget_paused"   — the daily runaway guard tripped. Nothing fetched, nothing written.
                            Refuses the PREVIEW too, deliberately — see HOPPER_DAILY_USD.

    Fail-safe on every branch: a failure SKIPS. No partial atom, nothing marked processed, no
    guess. The next attempt starts clean.

    Does NOT run `resolve_entities`. The footprint callers re-resolve after a whole archive lands;
    one atom does not earn a graph pass, and the next real ingest picks it up.
    """
    from . import ingest_blog, ingest_x

    # Before the preview, since a preview isn't free either (X reads ~$0.00015). Refusal goes in
    # the returned payload, not a log — hopper has a human reading its output.
    if _daily_budget_exhausted():
        return {"status": "budget_paused", "reference": reference,
                "message": (f"hopper's ${HOPPER_DAILY_USD:.2f} daily runaway guard has tripped — "
                            f"that much spend in one day means something is looping, not that you "
                            f"saved a lot. Nothing was fetched and nothing was written. It resets "
                            f"at UTC midnight.")}

    # `enrich=not confirm`: on a confirm the adapter is about to fetch the post anyway, so paying
    # for a preview card here would buy the same tweet twice.
    pre = preview(conn, reference, kind_hint=kind_hint, enrich=not confirm)
    if not pre["routable"]:
        return {**pre, "status": "unroutable"}
    if not confirm:
        return {**pre, "status": "preview", "next": "call again with confirm=True to store it."}
    if pre["already_present"]:
        # The free pre-check already answered. Every adapter below re-checks too (they must —
        # substack has no offline id), so this is a shortcut, not the guarantee.
        #
        # The `entry_mode: "user-saved"` this returns used to be a claim with nothing behind it: a
        # deposit of a URL the frontier had already crawled reported the human mode while the row
        # kept the machine one. `promote_atom` makes the claim true. The output shape is unchanged
        # on purpose — a promotion answers exactly like any other save, because saying otherwise
        # means teaching the lane taxonomy at the interface.
        ingest_common.promote_atom(conn, pre["atom_id"], "user-saved")
        return {"status": "already_present", "kind": pre["kind"], "atom_id": pre["atom_id"],
                "entry_mode": "user-saved", "reference": pre["reference"]}

    ref, kind = pre["reference"], pre["kind"]
    if kind == "article":
        status, atom_id = ingest_blog.article_atom_from_url(
            conn, embedder, ref, entry_mode="user-saved")
    elif kind == "x":
        status, atom_id = ingest_x.x_atom_from_url(
            conn, embedder, ref, entry_mode="user-saved", profile=profile)
    else:
        res = link_router.mint_artifact(conn, embedder, ref, kind, entry_mode="user-saved")
        atom_id = res["atom_id"]
        # `mint_artifact` speaks the vouch path's vocabulary; translate it into this surface's.
        # "minted" covers both a fresh write and Substack's mint-or-present collapse — the honest
        # word for it here is "saved: the atom is in the store now".
        status = {"present": "present", "minted": "saved"}.get(res["status"], "failed")

    out = {"status": {"present": "already_present"}.get(status, status),
           "kind": kind, "atom_id": atom_id, "entry_mode": "user-saved", "reference": ref}
    if out["status"] == "rejected":
        out["detail"] = ("the content-quality gate found no substantive units on that page "
                         "(nav / promo / boilerplate). Nothing was stored.")
    elif out["status"] == "blocked":
        out["detail"] = ("the host served a bot-check or challenge page instead of the article. "
                         "Nothing was stored; it is worth retrying later.")
    elif out["status"] == "failed":
        out["detail"] = ("could not fetch or identify that as a " + kind +
                         ". Nothing was stored. If the URL is right, pass kind_hint to override "
                         "the route.")
    elif out["status"] == "saved":
        warning = _thin_metadata_warning(conn, atom_id)
        if warning:                      # a degraded success is still a success — but say so
            out["warning"] = warning
    return out
