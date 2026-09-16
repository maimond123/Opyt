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
import sqlite3

from . import ingest_common
from . import link_router
from ..ingestion import url_canon

_WHY_BASIS = {
    "sniffed": "the URL says so — a known host, or a DOI in its path. A fact, checked offline",
    "hint": "your kind hint; the URL host matched nothing known",
    # NOT "the page declares a citation_doi", which this said until 2026-09-16. `probed` covers
    # four different sources now — a declared tag, a DOI the page merely prints, bare `citation_*`
    # markup, and OpenAlex naming the paper that lives at this url — and the last of those routinely
    # fires on a page that declared NOTHING (`hal.science` answers 200 with a JS shell; most of the
    # rest answer 403). Asserting the page said so was then a statement about a page nobody read.
    "probed": ("it cost a fetch: either the page itself names a DOI, or an index records this url "
               "as that paper's page. Not derivable from the url alone"),
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
    status link tells the model nothing verifiable. The preview reads the local X session so it
    can show the actual post before the user confirms.

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
    except Exception:                       # a malformed payload drops the card, never the preview
        return None, None


def preview(conn: sqlite3.Connection, reference: str, *, kind_hint: str | None = None,
            enrich: bool = True) -> dict:
    """Where would this go, and do we already have it? No writes, ever.

    An X post may add a local-session preview card because a bare status link is otherwise
    unverifiable. `enrich=False` disables that read because the confirm path fetches the post.

    The other kinds use the reference the caller already supplied; preview does not fetch titles.

    ONE BOUNDED FETCH, and only on the article fallback. A publisher that declares `citation_doi`
    in its page head is a paper, and nothing about its URL says so — measured 2026-09-09,
    nature.com filed AlphaFold as `blog:nature.com/articles/…`, `who_id = blog:nature.com`,
    `what_kind = opinion`, with the author list scraped into the title. That atom never dedups
    against the same paper saved by DOI and never feeds `sync_paper_author_signals`, so none of
    its 34 authors becomes a candidate.

    It runs HERE rather than on confirm because the preview is what the user approves, and the
    whole point of the two-phase split is that a wrong route is silent. `classify_link_deep`
    bounds itself (5s, 64KB of the head), and it costs nothing on a link that already sniffed or
    carried a hint. The probe REWRITES `reference` to the `doi.org` form it found, so `save`
    mints the paper rather than the landing page it was handed.
    """
    ref = (reference or "").strip()
    kind, basis = link_router.classify_reference(ref, hint=kind_hint)
    # ⚠️ REFUSE THE PLATFORMS WE CANNOT READ, HERE, BEFORE ANY FETCH. Routing sends everything
    # unrecognised to `article`, so a youtube/podcast/linkedin link used to be fetched as a blog
    # post and come back `rejected` — "the content-quality gate found no substantive units (nav /
    # promo / boilerplate)". On a page that IS nav and promo that verdict is technically right and
    # reads as OPYT calling the user's video worthless. The honest answer is that OPYT does not
    # read video, audio or login-walled feeds yet, and it is knowable from the URL alone.
    #
    # `url_canon._EXCLUDED_HOSTS` has said which hosts these are since it was written ("Excluded
    # platforms must not fall through to the personal-blog default") — Hopper simply had no way
    # to ask, so it fell through. Placed BEFORE `classify_link_deep` because the probe is a real
    # network read, and reading the head of a page we have already decided not to store is pure
    # cost. Only the `fallback` route is checked: a DOI on one of these hosts is still a paper.
    if basis == "fallback" and (platform := url_canon.excluded_platform(ref)):
        return {"routable": False, "reference": ref, "kind": None, "saw": platform,
                "error": (f"OPYT cannot read {platform} pages yet — it stores written text, and "
                          "these carry video, audio or a login-walled feed the page itself does "
                          "not contain. Nothing was written, and nothing is wrong with the link. "
                          "If a transcript or write-up of it exists somewhere, that URL works.")}
    content_type = None
    if basis == "fallback" and (probed := link_router.classify_link_deep(ref)):
        kind, ref, content_type = probed
        basis = "probed"
    if kind is None:
        return {"routable": False, "reference": ref, "kind": None,
                "saw": ("not an http(s) URL" if ref else "empty reference"),
                "error": "cannot route this — Hopper takes a URL (an article, a paper, a github "
                         "repo, a Substack post, or an X post). Nothing was written."}

    atom_id = link_router.predicted_atom_id(ref, kind, content_type=content_type)
    present = bool(atom_id) and link_router.atom_present(conn, atom_id)
    out = {
        "routable": True, "reference": ref, "kind": kind, "why": _WHY_BASIS.get(basis, basis),
        "atom_id": atom_id, "already_present": present, "entry_mode": "user-saved",
    }
    if content_type:
        # Only the probe can assert this — it required a real fetch. Carried so `save` hands the
        # adapter the same fact the id was predicted from, instead of re-deriving from a url that
        # by definition does not carry it.
        out["content_type"] = content_type
    if kind == "substack":
        # The only kind whose id is not derivable offline — it keys on the post's numeric id.
        out["note"] = ("the atom id for a Substack post is only known after the fetch, so "
                       "'already present' cannot be answered here; the adapter dedups on it.")
    elif kind == "paper" and not atom_id:
        # A PubMed url: its PMID is not a DOI and the string carries no route to one, so the id is
        # only known after one lookup — which the free pre-check deliberately does not pay for.
        out["note"] = ("this paper's id is only known after the lookup, so 'already present' "
                       "cannot be answered here; the adapter dedups on it.")
    elif kind == "github":
        # The store keys on the API's canonical owner casing, which the URL may not match.
        out["note"] = ("the github atom id uses the API's canonical owner casing, so this id is a "
                       "best guess from the URL; a casing mismatch causes one re-fetch, not a twin.")
    if present:
        out["note"] = "already in the knowledge base — a confirm would be a no-op."
        return out
    if kind == "x" and enrich:
        card, problem = _x_preview_card(ref)
        if card:
            out["description"] = card       # what the atom will carry, verbatim
        elif problem:
            out["unreadable"] = problem
    return out


def save(conn: sqlite3.Connection, embedder, reference: str, *, kind_hint: str | None = None,
         confirm: bool = False) -> dict:
    """Route one reference into the atom KB. Two-phase: `confirm=False` (the default) PREVIEWS and
    writes nothing; `confirm=True` runs the ingest.

    Returns the preview dict, or on a confirm `{status, kind, atom_id, entry_mode, …}` where status
    is one of:
      • "already_present" — the atom was there. Nothing written.
      • "saved"           — a new (or changed) atom is in the store.
      • "rejected"        — fetched fine, but the content gate found no substantive units. Not an
                            error: the page was nav/promo/boilerplate. Nothing stored, by design.
      • "blocked"         — the host stopped us (a bot-check or a challenge shell). Retryable.
      • "failed"          — the fetch failed, or the URL is not what its kind claims.
      • "unroutable"      — not a URL at all.

    Fail-safe on every branch: a failure SKIPS. No partial atom, nothing marked processed, no
    guess. The next attempt starts clean.

    Does NOT run `resolve_entities`. The footprint callers re-resolve after a whole archive lands;
    one atom does not earn a graph pass, and the next real ingest picks it up.
    """
    from . import ingest_blog, ingest_x

    # `enrich=not confirm`: on a confirm the adapter is about to fetch the post anyway, so the
    # preview does not fetch the same tweet twice.
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
        status, atom_id = ingest_x.x_atom_from_url(conn, embedder, ref)
    else:
        res = link_router.mint_artifact(conn, embedder, ref, kind, entry_mode="user-saved",
                                        content_type=pre.get("content_type"))
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
        # NOT "pass kind_hint to override the route", which this said until 2026-09-06: the hint
        # is consulted only when the host matches nothing known, so it cannot override anything
        # that got as far as being identified. `a0ecd0e8` measured that it is live for exactly
        # ONE kind and wrote the correct version into `hopper_tools`' own docstring in the same
        # commit, leaving this copy behind.
        out["detail"] = ("could not fetch or identify that as a " + kind +
                         ". Nothing was stored.")
    elif out["status"] == "saved":
        warning = _thin_metadata_warning(conn, atom_id)
        if warning:                      # a degraded success is still a success — but say so
            out["warning"] = warning
    return out
