"""
pipeline/kb/hopper.py — one deposit surface: route a URL (paper, GitHub repo, Substack post,
article, X post) to the adapter that handles it, producing one atom.

Hopper may create an entity but never an Oracle; `add_oracle` is the human gate for that. Every
atom is stamped `entry_mode='user-saved'`, like a bookmark. Preview, then confirm: a wrong route
is silent, so the default call writes nothing and reports where the thing is headed.
"""

from __future__ import annotations

from pipeline.kb.schema import stored_atom_id
import sqlite3

from . import ingest_common
from . import link_mint, link_router
from ..platforms import url_canon

_WHY_BASIS = {
    "sniffed": "the URL says so — a known host, or a DOI in its path. A fact, checked offline",
    "hint": "your kind hint; the URL host matched nothing known",
    # `probed` includes an index naming the paper at this URL, so it can fire on a page that
    # declared nothing.
    "probed": ("it cost a fetch: either the page itself names a DOI, or an index records this url "
               "as that paper's page. Not derivable from the url alone"),
    "fallback": "nothing matched, so it is treated as a plain article",
}


def _thin_metadata_warning(conn: sqlite3.Connection, atom_id: str | None) -> str | None:
    """A warning if the atom landed with a body but no title and no date, else None.

    A throttled metadata fetch can cause this, and papers are immutable so it cannot be repaired.
    Read after the write, which Hopper makes synchronously.
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
    return ("stored WITHOUT metadata — no title, no date" + who + ". Its body IS indexed and "
            "searchable, but this atom will never answer 'who wrote this, and when'. Cause is "
            "usually a throttled metadata lookup while the document fetch succeeded. Papers are "
            "immutable once written, so re-saving will NOT repair it.")


def _x_preview_card(url: str) -> tuple[str | None, str | None]:
    """For an X post: `(description, problem)`, read free from the local X session.

    x.com serves a JS shell to unauthenticated fetchers, so this is the only way to show the post
    before confirm. The description is `derive_x`'s, so what is approved is what gets stored.
    """
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

    An X post may add a local-session preview card; `enrich=False` skips it. On the article
    fallback only, one bounded fetch probes for a paper behind the URL and rewrites `reference` to
    the `doi.org` form it found, so the route the user approves is the one `save` uses.
    """
    ref = (reference or "").strip()
    kind, basis = link_router.classify_reference(ref, hint=kind_hint)
    # Refuse video, audio and login-walled platforms before any fetch, so they are not gated as
    # empty articles. Only the fallback route: a DOI on one of these hosts is still a paper.
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

    atom_id = link_mint.predicted_atom_id(ref, kind, content_type=content_type)
    present = bool(atom_id) and bool(stored_atom_id(conn, atom_id))
    out = {
        "routable": True, "reference": ref, "kind": kind, "why": _WHY_BASIS.get(basis, basis),
        "atom_id": atom_id, "already_present": present, "entry_mode": "user-saved",
    }
    if content_type:
        # Carried so `save` hands the adapter the fact the id was predicted from.
        out["content_type"] = content_type
    if kind == "substack":
        # The only kind whose id is not derivable offline — it keys on the post's numeric id.
        out["note"] = ("the atom id for a Substack post is only known after the fetch, so "
                       "'already present' cannot be answered here; the adapter dedups on it.")
    elif kind == "paper" and not atom_id:
        # A PubMed URL: the DOI is only known after a lookup the free pre-check doesn't pay for.
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


def save(conn: sqlite3.Connection, reference: str, *, kind_hint: str | None = None,
         confirm: bool = False) -> dict:
    """Route one reference into the atom KB. `confirm=False` (the default) previews and writes
    nothing; `confirm=True` runs the ingest.

    Returns the preview dict, or on a confirm `{status, kind, atom_id, entry_mode, …}` where status
    is one of:
      • "already_present" — the atom was there. Nothing written.
      • "saved"           — a new (or changed) atom is in the store.
      • "blocked"         — the host stopped us (a bot-check or a challenge shell). Retryable.
      • "failed"          — the fetch failed, or the URL is not what its kind claims.
      • "unroutable"      — not a URL at all.

    A failure skips: no partial atom, nothing marked processed. Does not run `resolve_entities`;
    the next real ingest picks the atom up.
    """
    from . import ingest_blog, ingest_x

    # On a confirm the adapter fetches the post anyway.
    pre = preview(conn, reference, kind_hint=kind_hint, enrich=not confirm)
    if not pre["routable"]:
        return {**pre, "status": "unroutable"}
    if not confirm:
        return {**pre, "status": "preview", "next": "call again with confirm=True to store it."}
    if pre["already_present"]:
        # A shortcut, not the guarantee: every adapter re-checks. `promote_atom` makes the
        # returned `entry_mode` true for an atom another lane stored first.
        ingest_common.promote_atom(conn, pre["atom_id"], "user-saved")
        return {"status": "already_present", "kind": pre["kind"], "atom_id": pre["atom_id"],
                "entry_mode": "user-saved", "reference": pre["reference"]}

    ref, kind = pre["reference"], pre["kind"]
    if kind == "article":
        status, atom_id = ingest_blog.article_atom_from_url(
            conn, ref, entry_mode="user-saved")
    elif kind == "x":
        status, atom_id = ingest_x.x_atom_from_url(conn, ref)
    else:
        res = link_mint.mint_artifact(conn, ref, kind, entry_mode="user-saved",
                                        content_type=pre.get("content_type"))
        atom_id = res["atom_id"]
        # Translate `mint_artifact`'s vocabulary into this surface's.
        status = {"present": "present", "minted": "saved"}.get(res["status"], "failed")

    out = {"status": {"present": "already_present"}.get(status, status),
           "kind": kind, "atom_id": atom_id, "entry_mode": "user-saved", "reference": ref}
    if out["status"] == "blocked":
        out["detail"] = ("the host served a bot-check or challenge page instead of the article. "
                         "Nothing was stored; it is worth retrying later.")
    elif out["status"] == "failed":
        # kind_hint is only consulted for unknown hosts, so it cannot override an identified route.
        out["detail"] = ("could not fetch or identify that as a " + kind +
                         ". Nothing was stored.")
    elif out["status"] == "saved":
        warning = _thin_metadata_warning(conn, atom_id)
        if warning:                      # a degraded success is still a success, but say so
            out["warning"] = warning
    return out
