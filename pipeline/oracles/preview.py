"""
pipeline/oracles/preview.py — `add_oracle`'s resolve-only preview, and what it tells the host.

Phase one of `add_oracle` writes nothing: it says who a reference is, what each lookback window
will actually pull (the X clamp included), the real counts of a scholar's papers by year and by
subject, and what to ask the user next.
"""

from __future__ import annotations

from urllib.parse import urlparse

from pipeline.oracles import identify
from pipeline.oracles import roster
from pipeline.oracles import windows
from pipeline.oracles.reference import classify_reference, url_entity_id
from pipeline.oracles.topics import scholar_topic_counts, scholar_year_counts, topic_counts_for_id
from pipeline.platforms import connections
from pipeline.timeparse import utc_now


def _lookback_report(x_since, web_since, scholar_since=None, counts: dict | None = None) -> dict:
    """The human-facing windows, derived from the resolved datetimes rather than re-read off the
    preset strings — so what the user is told is what ran, including the X clamp they didn't ask
    for. This is the surface the consent invariant rests on.

    The three windows (X, web, papers) stay separate: each bounds a corpus with different limits.
    `counts`, when present, is the real distribution of the person's papers by year."""
    # Through the adapter's own clamp, so the report cannot drift from it.
    eff = windows.x_since(x_since, utc_now())
    clamped = x_since is not None and eff > x_since
    out = {
        "x": f"since {eff:%Y-%m-%d}" + (" (CLAMPED to the 2-year ceiling)" if clamped else
                                        "" if x_since is not None else " (6-month default)"),
        "x_since": eff.isoformat(),
        "web": f"since {web_since:%Y-%m-%d}" if web_since else "full archive",
        "web_since": web_since.isoformat() if web_since else None,
        "scholar": f"since {scholar_since:%Y-%m-%d}" if scholar_since else "whole corpus",
        "scholar_since": scholar_since.isoformat() if scholar_since else None,
        "note": "X is an ephemeral stream, capped at 2 years; the web archive and the paper "
                "corpus have no cap. The three windows are chosen separately — a short X window "
                "does not truncate either archive.",
    }
    if counts:
        out["scholar_counts"] = counts
        spans = " · ".join(f"{k} is {v}" for k, v in counts["by_window"].items())
        out["scholar_ask"] = (f"{counts['total']} papers across {counts['years']} years — "
                              f"{spans}. Pick a scholar_lookback, or 'all'.")
    return out


def _ask_which_one_they_meant(out: dict, reference: str, root_entity: str) -> None:
    """A URL pointing at one page, about to subscribe them to the whole site — say so first.

    `canonical_identity` collapses every URL on a host to that host, so confirming one article
    follows the whole site. Only the user knows which they meant, and confirm is the last moment
    to ask. Root URLs say nothing extra.
    """
    try:
        path = (urlparse(reference).path or "").strip("/")
    except Exception:
        return
    if not path:
        return
    site = root_entity.split(":", 1)[-1]
    out["ambiguous_scope"] = {"pointed_at": "one page", "would_follow": site}
    out["note"] = (
        f"They gave a link to ONE page, and confirming follows all of {site} from now on — every "
        f"post it has and every post it publishes later. ASK WHICH THEY MEANT BEFORE CONFIRMING, "
        f"in one line: follow {site} from now on, or just keep this one page? If they want the "
        f"one page, do NOT confirm — save it instead, which stores that page and subscribes them "
        f"to nothing.")


def of(conn, reference: str, match: dict | None,
             x_lookback: str | None, web_lookback: str | None,
             scholar_lookback: str | None = None) -> dict:
    """confirm=False preview: who this is and what confirm=True would do. No writes. An
    unresolvable reference returns `unresolved`, so a hallucinated handle has nothing to confirm.

    For a person already in the roster with an OpenAlex id, reports their papers counted by year
    (through the stored topic filter) and by subject."""
    cid = match["canonical_id"] if match else None
    counts = scholar_year_counts(conn, cid) if cid else None
    topic_counts = scholar_topic_counts(conn, cid) if cid else None
    x_connected = connections._connected_x()
    now = utc_now()
    try:                         # the confirm path's refusal, so a preview never errors on a preset
        check = windows.parse(x_lookback, web_lookback, scholar_lookback, now=now).check
    except ValueError as e:
        return {"error": f"{e}. Nothing was written."}
    x_since = windows.since("x", x_lookback, now) if x_connected and not check else None
    lookback = _lookback_report(x_since, windows.since("web", web_lookback, now),
                                windows.since("scholar", scholar_lookback, now), counts)
    if not x_connected or check:
        lookback.pop("x", None)
        lookback.pop("x_since", None)
    if x_connected and check:
        lookback["x"] = "since_last: a check of their newest posts, not a window"
    base = {"confirm_required": True, "reference": reference,
            "lookback": lookback,
            "on_confirm": "confirm=True confirms this Oracle and ingests what they carry: their "
                          "papers when they have an OpenAlex record, and — when they have accounts "
                          "to discover — their verified footprint and, when X is connected, their "
                          "X timeline. The X pull itself is free (your own browser session)."}
    if topic_counts:
        base["scholar_topics"] = topic_counts

    if match:                                        # already in the roster (dedup / Mode C)
        already = roster.on_roster(conn, match["canonical_id"])
        return {**base, "mode": "existing",
                "resolved": {"canonical_id": match["canonical_id"], "name": match.get("name"),
                             "members": [m["entity_id"] for m in match.get("members", [])],
                             "already_oracle": already},
                "note": ("Already one of your Oracles — confirm=True REFRESHES them "
                         "(re-pulls; atoms dedup, so nothing duplicates)." if already else
                         "Already in your roster from curation — confirm=True promotes them to an "
                         "Oracle and ingests their footprint.")}

    kind = classify_reference(reference)
    if kind == "canonical":
        return {"error": f"no entity for canonical_id {reference!r} — it may be stale or "
                         "hallucinated. Nothing was written."}
    if kind == "url":
        # The OpenAlex lookup runs here too, not only at confirm: telling the user "blog" and then
        # minting a venue would misreport a much larger pull.
        venue = identify.openalex_root(reference)
        if venue:
            oid = venue["openalex_id"]
            # Author and venue need different copy: a venue's corpus overruns one pull.
            author = oid.startswith("A")
            note = (f"That is a researcher's OpenAlex record — OPYT pulls their papers by that id, "
                    f"with real authors and DOIs. It carries {venue.get('works') or 'many'} works. "
                    f"Subjects narrow it and the choice PERSISTS into every later refresh."
                    if author else
                    f"That is a research venue, not a personal site — OPYT reads it "
                    f"through OpenAlex ({oid}), so papers arrive with "
                    f"their real authors and DOIs instead of being scraped as generic "
                    f"articles. It carries {venue.get('works') or 'many'} works. ASK "
                    f"WHICH SUBJECTS BEFORE CONFIRMING — confirm=True with no "
                    f"`scholar_topics` pulls none of its papers until the user picks the "
                    f"subjects they read.")
            out = {**base, "mode": "new",
                   "resolved": {"reference": reference, "name": venue.get("name"),
                                "root_entity": f"openalex:{oid}",
                                "platform": "openalex_author" if author else "openalex_venue",
                                "works": venue.get("works")},
                   "note": note}
            # The topic ask is keyed on the OpenAlex id, not a cluster: this root has no entity and
            # no `oracle_sources` row until confirm, and asking after the first pull is too late.
            topics = topic_counts_for_id(oid)
            if topics:
                out["scholar_topics"] = topics
            return out
        eid = url_entity_id(reference)
        out = {**base, "mode": "new",
               "resolved": {"reference": reference, "root_entity": eid,
                            "platform": "substack" if eid.startswith("substack:") else "blog"}}
        _ask_which_one_they_meant(out, reference, eid)
        return out
    # X handle — a cheap, read-only identity fetch for recognition (name / bio / followers).
    ident = identify.fetch_x_identity(reference)
    if not ident:
        return {"unresolved": reference,
                "note": "Could not resolve this X handle — check the spelling. Nothing was "
                        "written, so there is nothing to confirm."}
    return {**base, "mode": "new",
            "resolved": {"handle": ident.get("handle"), "name": ident.get("display_name"),
                         "bio": ident.get("bio"), "followers": ident.get("followers"),
                         "site": ident.get("site"), "root_entity": f"x:user:{ident['user_id']}"}}
