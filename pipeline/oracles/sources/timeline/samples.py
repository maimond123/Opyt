"""
pipeline/oracles/sources/timeline/samples.py — the light timeline pull over the Oracle candidate
list.

For each person the user has engaged with (follow, bookmark, like, List, subscription), pull one
shallow page of their own posts so the host can judge their field from their words. It gathers
evidence and never scores anyone. This module holds the queue and one page; the scouting scraper
(`oracles/sources/timeline/scouting.py`) owns the walk.

Constraints: zero tweets is a durable `empty` fact, never re-fetched; content never lands in
`atoms`, only through `probe_store`; text only, no media enrichment; no length gate on posts.
"""

from __future__ import annotations

import re
import sqlite3

from pipeline.ingestion import x_render as xt
from pipeline.kb import derive
from pipeline.kb.ingest_common import BASIS_OBSERVED, BODY_COMPLETE, BODY_PARTIAL, body_fields, changed_hash
from pipeline.kb.ingest_x_footprint import _filter_and_stitch
from pipeline.oracles import probe_store
from pipeline.oracles import ranking
from pipeline.oracles.sources.timeline import pages
from pipeline.platforms.x import originals as x_originals

_X_USER_RE = re.compile(r"^x:user:(\d+)$")

_ATOM_PREFIX = "xprobe"


# ── the queue ─────────────────────────────────────────────────────────────────

def x_user_id(members: list[str]) -> str | None:
    """The numeric X id in a resolved cluster, or None (no X identity: absent, not failed)."""
    for m in members:
        hit = _X_USER_RE.match(m or "")
        if hit:
            return hit.group(1)
    return None


def candidate_queue(conn: sqlite3.Connection, *, min_signals: int = 1) -> list[dict]:
    """The candidates due for a pull, in the screen's own rank order (`ranking.rank_candidates`).

    Excluded: confirmed Oracles (already in `atoms`), candidates with no X identity, and those
    below `min_signals`. Whether one was already sampled is scouting's to decide
    (`scouting.standing`).

    A candidate with no X id but an X handle they declared on Substack (`profile.x_handle`,
    `web/substack_samples`) is queued keyed `x:@<handle>` with no `user_id`: scouting's unit for it
    resolves the handle and mints the X account into their cluster, and the next ranking queues
    them by id."""

    out: list[dict] = []
    for cand in ranking.rank_candidates(conn):
        if cand.distinct_signals < min_signals:
            continue
        uid = x_user_id(cand.members)
        if uid:
            out.append({"who_id": f"x:user:{uid}", "user_id": uid,
                        "canonical_id": cand.canonical_id, "name": cand.name,
                        "handle": cand.handle, "distinct_signals": cand.distinct_signals})
        elif handle := cand.profile.get("x_handle"):
            out.append({"who_id": f"x:@{handle.lower()}", "user_id": None,
                        "canonical_id": cand.canonical_id, "name": cand.name, "handle": handle,
                        "links": cand.identity_links,
                        "distinct_signals": cand.distinct_signals})
    return out


# ── one candidate ─────────────────────────────────────────────────────────────

def _render_groups(groups: list[list[dict]], *, handle: str) -> list[dict]:
    """Stitched tweet groups → write-ready probe atoms. Pure: no DB or network."""
    out: list[dict] = []
    for group in groups:
        root = group[0]                       # chronologically first = the atom's canonical
        root_id = str(root.get("id", ""))
        if not root_id:
            continue
        is_thread = len(group) > 1
        article = root.get("article")
        is_article = bool(article) or any(xt._article_tweet_id(t) for t in group)
        # The timeline response sometimes carries the full Article body and sometimes its teaser.
        # Completeness follows the blocks actually received; no extra fetch belongs on this path.
        has_article_body = bool(xt._article_shape(article)[1]) if isinstance(article, dict) else False
        md = xt.tweet_to_markdown(root, article=article, thread_tweets=group if is_thread else None,
                                  source="x-probe", footer_label="Candidate probe")
        meta = derive.derive_x(root)
        out.append({
            "atom_id": f"{_ATOM_PREFIX}:{root_id}",
            "source_type": "x",
            "who_id": meta["who_id"],
            "when_ts": meta["when_ts"],
            "when_precision": meta["when_precision"],
            "source_url": root.get("url") or f"https://x.com/{handle}/status/{root_id}",
            "description": meta["description"],
            "payload": {
                "like_count": root.get("likeCount", 0),
                "reply_count": root.get("replyCount", 0),
                "is_thread": is_thread,
                "thread_len": len(group),
                "is_quote": bool(root.get("isQuote") or root.get("quoted_tweet")),
                "is_article": is_article,
                "has_media": any((t.get("extendedEntities") or {}).get("media") for t in group),
                "source_tags": meta["source_tags"],
                **body_fields(BODY_COMPLETE if not is_article or has_article_body
                              else BODY_PARTIAL, BASIS_OBSERVED),
            },
            "_markdown": md,
        })
    return out


def _prepare_groups(conn, groups: list[list[dict]], *,
                    who_id: str, handle: str) -> list[tuple[dict, str]]:
    """Render stitched groups, keeping those whose snapshot changed, for the probe store. Writes
    nothing in the store: returns each `(atom, snapshot)` to write."""
    atoms = _render_groups(groups, handle=handle)
    seen = probe_store.load_probe_hashes(conn, who_id)
    out = []
    for atom in atoms:
        md = atom.pop("_markdown")
        decided = changed_hash(atom["atom_id"], md, seen)
        if decided is None:                     # unchanged → no rewrite
            continue
        atom["raw_hash"] = decided
        seen[atom["atom_id"]] = atom["raw_hash"]
        out.append((atom, md))
    return out


def sample_page(conn, session, cand: dict, *, below_id: str | None = None) -> dict:
    """One request of a candidate's Originals timeline, rendered for the probe store:
    the scouting scraper's unit (`oracles.sources.timeline.scouting`). Records the attempt;
    writes nothing else. The caller writes the page with `write_sample`, with the candidate's
    standing (`probe_pulls`), since it knows whether the sample is finished.

    Raises what the request raises — `XUserUnavailable`, `XRateLimited`, `SyncAuthError`, a
    transport error — so the caller decides what each means for the candidate.

    Returns `{fetched, fresh, oldest_id, oldest_at, ended, written}`: `fresh` counts the page's
    posts other than the pin, `oldest_*` is where the next page resumes and how far back this one
    reached, `ended` says the timeline has nothing older, and `written` counts the atoms
    `write_sample` writes."""
    who_id, handle = cand["who_id"], (cand.get("handle") or "")
    walk = x_originals.fetch_user_originals(session, cand["user_id"], after=pages.X_FIRST_DAY,
                                     below_id=below_id)
    pin_id = str(walk.pinned["id"]) if walk.pinned else None
    fresh = [t for t in walk.tweets if str(t["id"]) != pin_id]
    oldest = min(fresh, key=lambda t: int(t["id"])) if fresh else None
    atoms = _prepare_groups(conn, _filter_and_stitch(walk.tweets), who_id=who_id, handle=handle)
    return {"fetched": len(walk.tweets), "fresh": len(fresh),
            "oldest_id": str(oldest["id"]) if oldest else None,
            "oldest_at": xt.parse_twitter_date(oldest.get("createdAt", "")) if oldest else None,
            "ended": walk.finished, "written": len(atoms), "atoms": atoms}


def write_sample(conn, page: dict) -> None:
    """Write the atoms `sample_page` rendered. Inside the caller's `write_tx` they commit with the
    candidate's standing."""
    for atom, md in page.get("atoms", ()):
        probe_store.store_probe_atom(conn, atom, md)
