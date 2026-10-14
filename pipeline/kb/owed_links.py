"""
pipeline/kb/owed_links.py — the links a post owes the referenced-links queue, and who linked
each one. Pure.

A post records them as `payload.links_owed` and fetches none of them; the reference sources
(`oracles/sources/references/`) capture them later. The payload field is the queue. Four kinds of
post owe links: an Oracle's X post (`owed_links`, from the timeline walk), an X post the user
saved (`saved_post_links`, from `ingest_x.build_x_atom`), and two that owe only their paper
links: an X post an Oracle linked (`referenced_post_links`, from `ingest_x.referenced_x_atom`)
and any blog, Substack or GitHub post (`long_form_links`, at `ingest_common.store_atom`).
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from pipeline.ingestion.x_render import has_link_card, parse_twitter_date
from pipeline.platforms.url_canon import host_of

from . import link_router


def _post_urls(tweet: dict) -> list[str]:
    """The tweet's expanded outbound urls; the text carries only t.co links."""
    return [u.get("expanded_url", "") for u in (tweet.get("entities") or {}).get("urls") or []
            if u.get("expanded_url")]


def _referenced_links(group: list[dict]) -> list[tuple[str, dict, dict | None]]:
    """Every link a group's posts carry → `(url, node, quoted)`: the Oracle's post that carried it,
    and the quoted post it sat in (None when it is in the Oracle's own text or a self-quote). A
    link in both counts once, as the Oracle's own."""
    oracle_id = str((group[0].get("author") or {}).get("id") or "")
    out: list[tuple[str, dict, dict | None]] = []
    seen: set[str] = set()
    for t in group:
        for u in _post_urls(t):
            if u not in seen:
                seen.add(u)
                out.append((u, t, None))
    for t in group:
        qt = t.get("quoted_tweet")
        if not isinstance(qt, dict):
            continue
        via = None if str((qt.get("author") or {}).get("id") or "") == oracle_id else qt
        for u in _post_urls(qt):
            if u not in seen:
                seen.add(u)
                out.append((u, t, via))
    return out


def _node_url(node: dict) -> str:
    handle = (node.get("author") or {}).get("userName") or "i"
    return node.get("url") or f"https://x.com/{handle}/status/{node.get('id')}"


def _reference(group: list[dict], node: dict, quoted: dict | None, post_atom_id: str) -> dict:
    """The `referenced_by` entry for one link (`schema.add_reference`). `by` is the post author's
    `who_id` — the join key to the entity graph; an Oracle's canonical id can be its blog's, so it
    is not this. `url`/`date` are the post that carried the link, `post` the atom that holds it
    (a thread's links all name the thread's atom). `via_quote` names the quoted post's author
    when the link sat in a quote — "Soren referenced it by quoting Nikunj"."""
    author = group[0].get("author") or {}
    when = parse_twitter_date(node.get("createdAt") or "")
    ref = {"by": f"x:user:{author.get('id')}", "handle": author.get("userName") or "",
           "post": post_atom_id, "url": _node_url(node),
           "date": when.date().isoformat() if when else ""}
    if quoted is not None:
        qa = quoted.get("author") or {}
        ref["via_quote"] = {"by": f"x:user:{qa.get('id')}", "handle": qa.get("userName") or "",
                            "url": _node_url(quoted)}
    return ref


def _is_own_post(url: str, tid: str | None, group: list[dict]) -> bool:
    """A link to one of the Oracle's own posts, which the footprint already holds as `xprofile:`.
    Read from the url's handle or the id being in this group."""
    handle = ((group[0].get("author") or {}).get("userName") or "").lower()
    seg = (urlparse(url).path.strip("/").split("/") or [""])[0].lower()
    return (bool(handle) and seg == handle) or str(tid) in {str(t.get("id")) for t in group}


def _carrier(url: str, node: dict) -> dict:
    """The post whose own links include `url`: the group's post, or the post it quotes."""
    if url in _post_urls(node) or not isinstance(node.get("quoted_tweet"), dict):
        return node
    return node["quoted_tweet"]


def owed_links(group: list[dict], post_atom_id: str, *,
               own_hosts: frozenset[str] = frozenset()) -> list[dict]:
    """The links one group owes the queue → `[{url, ref}]`, its `payload.links_owed`. Pure.

    A homepage, product, job or event link (`link_router.is_card_link`) is owed only for its
    card, and only when X gave its post none (`kb/link_cards.py`). Drops what a drain would only
    discard: any other non-reference (`reference_kind` None), the Oracle's own post, and an
    article on the Oracle's own blog."""
    out: list[dict] = []
    for u, node, quoted in _referenced_links(group):
        kind = link_router.reference_kind(u)
        if kind is None:
            if link_router.is_card_link(u) and not has_link_card(_carrier(u, node)):
                out.append({"url": u, "ref": _reference(group, node, quoted, post_atom_id)})
            continue
        if kind == "x":
            tid = link_router.parse_tweet_id(u)
            if not tid or _is_own_post(u, tid, group):
                continue
        elif kind == "article" and host_of(u) in own_hosts:
            continue
        out.append({"url": u, "ref": _reference(group, node, quoted, post_atom_id)})
    return out


def saved_post_links(norm: dict, chain: list | None, post_atom_id: str) -> list[dict]:
    """The links a post the user saved owes the queue → `[{url, ref}]`. Pure.

    Read from the post, the rest of its author's own thread, and the posts those quote. The chain
    also keeps ancestors by other people; their links are context, not the saved post's. Past
    that, the Oracle rule exactly, links to other X posts included."""
    author = str((norm.get("author") or {}).get("id") or "")
    group = [t for t in chain or [norm] if str((t.get("author") or {}).get("id") or "") == author]
    return owed_links(group, post_atom_id)


def referenced_post_links(norm: dict, post_atom_id: str) -> list[dict]:
    """The paper links an X post an Oracle linked owes the queue → `[{url, ref}]`. Pure.

    Paper links only, so its papers get lines: its other links are a second
    hop nobody chose, and stay text in it."""
    return [e for e in saved_post_links(norm, None, post_atom_id)
            if link_router.reference_kind(e["url"]) == "paper"]


# Long-form posts: the source types whose body is prose with inline links.
LONG_FORM = frozenset({"blog", "substack", "github"})

# A markdown link's target (not an image's), or a bare URL in the text.
_MD_LINK = re.compile(r"(?<!!)\[[^\]\n]*\]\((https?://[^)\s]+)\)")
_BARE_URL = re.compile(r"(?<![(\[\w/])https?://[^\s<>()\[\]\"']+")


def paper_urls(markdown: str, *, skip: str | None = None) -> list[str]:
    """The paper URLs a post's text links, in order, once each: markdown link targets (not images)
    and bare URLs whose shape is a paper's (`link_router.reference_kind`). Pure."""
    urls: list[str] = []
    for u in _MD_LINK.findall(markdown) + _BARE_URL.findall(markdown):
        u = u.rstrip(".,;:!?*_'\"")
        if u not in urls and u != skip and link_router.reference_kind(u) == "paper":
            urls.append(u)
    return urls


def long_form_links(atom: dict, markdown: str) -> list[dict] | None:
    """The paper links a blog, Substack or GitHub post owes the queue → `[{url, ref}]`, or None
    when this atom owes none this way. Pure.

    Paper links only: a long-form post's other links stay text in it. Read once, when the post is first built: a post
    whose payload already says what it owes (a stored post being rewritten, a card written in)
    is left alone, and so is an atom with no `entry_mode`, which is the candidate sample store's."""
    payload = atom.get("payload")
    if (atom.get("source_type") not in LONG_FORM or not atom.get("entry_mode")
            or not isinstance(payload, dict) or "links_owed" in payload):
        return None
    urls = paper_urls(markdown, skip=atom.get("source_url"))
    when = str(atom.get("when_ts") or "")[:10]
    ref = {"by": atom.get("who_id") or "", "handle": "", "post": atom["atom_id"],
           "url": atom.get("source_url") or "", "date": when}
    return [{"url": u, "ref": dict(ref)} for u in urls]
