"""
pipeline/kb/link_router.py — ONE url → which adapter, and the mint that follows.

Shared by `ingest_x_footprint` and Hopper (the hand-dump surface) so both route identically — a
second copy risks two sniffers drifting and silently sending a paper to the blog adapter. A
mis-route never raises; it sits wrong forever.

The split, and why it lands here:

  • `classify_link`      — the ARTIFACT sniffer, moved VERBATIM. github | substack | paper | None.
  • `classify_reference` — Hopper's WIDER vocabulary, layered on top: + "x" (a single post) and
                           + "article" (the bare-link catch-all). Deliberately a separate function
                           — see the warning on it.
  • `predicted_atom_id`  — the atom id derivable from the url ALONE, offline. The free
                           already-present check, and the in-flight key the vouch path needs.
  • `mint_artifact`      — the de-`self`'d half of `_dispatch_one`: url → atom, no vouches.

What deliberately did NOT move: the vouch bookkeeping (`_pending`, `_record`, `_on_written`). That
is per-Oracle-run state owned by `LinkDispatcher`, and a hand-dumped item has no vouching person —
it is the difference between the two callers, not their shared part.
"""

from __future__ import annotations

import re
import sqlite3
from urllib.parse import urlparse

import requests

# A link we turn into its OWN atom. Re-exported from ingest_x_footprint, whose substance filter
# also reads this tuple, so its contents are load-bearing there too.
_PAPER_HOSTS = ("arxiv.org", "doi.org", "semanticscholar.org", "biorxiv.org", "medrxiv.org",
                "openreview.net", "pubmed.ncbi.nlm.nih.gov", "aclanthology.org")

# A single post on X. Only x.com / twitter.com, optionally www.- or mobile.-prefixed — the mirror
# front-ends (fxtwitter, vxtwitter, nitter) are deliberately absent: nothing here has been verified
# against them, and a host we cannot fetch is better refused than mis-routed.
_X_POST_RE = re.compile(
    r"^https?://(?:www\.|mobile\.)?(?:x|twitter)\.com/[^/]+/status(?:es)?/(\d+)", re.I)

# The kinds a host model may assert. `classify_reference` accepts a hint only from this set, so a
# typo or a hallucinated kind falls through to the article catch-all instead of routing nowhere.
HINT_KINDS = ("github", "substack", "paper", "x", "article")


def classify_link(url: str) -> str | None:
    """The kind of dispatchable artifact a url points at — 'github', 'paper', or 'substack' — or
    None for a BARE link (news, a personal blog, a company site, another tweet). ONE classifier
    feeds both gates: `_dispatchable_link` (any non-None keeps the reaction alive) and the Step-3
    dispatcher (which MINTS only github + paper; Substack is the deferred Tier-2 tier).

    Pure URL-host matching — no LLM and no network, which is what makes it a fact rather than a
    guess, and what lets it outrank a host model's hint whenever it fires."""
    d = urlparse(url).netloc.lower()
    if "github.com" in d:
        return "github"
    if "substack.com" in d:
        return "substack"
    if any(h in d for h in _PAPER_HOSTS) or url.lower().split("?")[0].endswith(".pdf"):
        return "paper"
    return None


# Bounded probe sizing: read the <head>, not the page — a citation meta tag lives in the first
# few KB of any real article.
_DEEP_PROBE_TIMEOUT = 5              # seconds — a hung probe must not stall a footprint pull
_DEEP_PROBE_MAX_BYTES = 65_536       # 64KB cap on what we read before giving up on this page
_DEEP_PROBE_UA = {"User-Agent": "Mozilla/5.0 (compatible; OpytBot/1.0; +https://github.com/opyt)"}
_CITATION_DOI_RE = re.compile(rb'<meta[^>]+name=["\']citation_doi["\'][^>]+content=["\']([^"\']+)',
                              re.I)
_CITATION_ANY_RE = re.compile(rb'<meta[^>]+name=["\']citation_', re.I)


def classify_link_deep(url: str) -> tuple[str, str, str | None] | None:
    """NETWORK-BOUND fallback for 'paper' outside the free `_PAPER_HOSTS` list — deliberately kept
    out of `classify_link`'s free path; a caller must opt in as a bounded last resort, never a
    default per-url scan. Fetches once and checks a `citation_doi` meta tag (rewritten to a
    `doi.org/{doi}` url) or a PDF Content-Type / any other `citation_*` tag.

    Returns `(kind, mint_url, content_type)` — always `("paper", ...)`, or None if nothing
    paper-shaped was found. `mint_url` is what a caller should mint against, not necessarily `url`
    itself; `content_type` carries the raw header when it matters downstream (the PDF case), else
    None. Never raises — a network error or oversized page reads the same as "not a paper"."""
    try:
        with requests.get(url, timeout=_DEEP_PROBE_TIMEOUT, stream=True,
                           headers=_DEEP_PROBE_UA) as resp:
            if resp.status_code >= 400:
                return None
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "application/pdf" in ctype:
                return "paper", url, ctype
            if "text/html" not in ctype:
                return None                  # nothing here a citation meta tag could live in
            buf = b""
            for chunk in resp.iter_content(8192):
                buf += chunk
                m = _CITATION_DOI_RE.search(buf)
                if m:
                    doi = m.group(1).decode("utf-8", "replace").strip()
                    return "paper", f"https://doi.org/{doi}", None
                if len(buf) > _DEEP_PROBE_MAX_BYTES:
                    break
            if _CITATION_ANY_RE.search(buf):
                return "paper", url, None    # confirmed scholarly, but no DOI to rewrite to
    except requests.RequestException:
        pass
    return None


def parse_tweet_id(url: str) -> str | None:
    """The numeric tweet id in an X post url, or None. Query string and trailing path (`/photo/1`)
    are ignored — the id is the only identity."""
    m = _X_POST_RE.match((url or "").strip())
    return m.group(1) if m else None


def is_http_url(reference: str) -> bool:
    """Is this a fetchable http(s) url at all? The one thing that separates 'route it' from 'I
    cannot route this' — everything with a host gets SOME adapter, so a non-url (a bare phrase, a
    file path, a DOI with no scheme) is the only genuine unroutable."""
    try:
        p = urlparse((reference or "").strip())
    except ValueError:                      # malformed IPv6 literal etc. — not a url we can use
        return False
    return p.scheme in ("http", "https") and bool(p.netloc)


def classify_reference(reference: str, *, hint: str | None = None) -> tuple[str | None, str]:
    """Route ANY hand-supplied reference → `(kind, basis)`, where basis says WHY it routed that way:
    'sniffed' (a fact about the url), 'hint' (the host model's read), 'fallback' (the article
    catch-all), or 'none' (unroutable).

    Separate from `classify_link` on purpose: `classify_link`'s None is read as a decision by the
    X footprint substance filter (a bare-link post faces the 200-char naked bar), so teaching it
    to answer "x"/"article" would silently reverse that. The wider vocabulary lives here instead,
    where only Hopper reads it.

    Order is the design: a deterministic sniff outranks the hint (a url host is a fact, a model
    guess is not); the hint outranks the fallback (otherwise it would be dead code). A hint can't
    force an adapter that re-validates the url itself and refuses a mismatch — except Substack,
    which cannot tell a custom-domain Substack from a generic blog without a fetch, so a host
    model's hint genuinely helps there.

    Fail-safe: an unknown/garbled hint is IGNORED, and a non-url returns (None, 'none')."""
    ref = (reference or "").strip()
    if not is_http_url(ref):
        return None, "none"
    kind = classify_link(ref)
    if kind:
        return kind, "sniffed"
    if parse_tweet_id(ref):
        return "x", "sniffed"
    if hint in HINT_KINDS:
        return hint, "hint"
    return "article", "fallback"


def predicted_atom_id(url: str, kind: str, *, content_type: str | None = None) -> str | None:
    """The atom_id derivable from the url ALONE — no network, no DB — plus whatever `content_type`
    the caller already knows (see `ingest_papers._parse_paper_url`). None when the url cannot carry
    it, which is a real answer and not a failure:

      • substack — the atom keys on the post's NUMERIC id, which only the fetch knows.
      • github   — returned, but APPROXIMATE. The real key uses the API's canonical `owner.login`
                   casing, so a url written `github.com/Ggerganov/…` predicts an id the store may
                   hold as `github:ggerganov/…`. Safe for a cheap "already have it?" check (a miss
                   costs one re-fetch); NEVER safe to write with.

    Used for two different questions that happen to share an answer: the free already-present
    pre-check, and the in-flight key `LinkDispatcher` needs to spot a second reference to an
    artifact still buffered in the sink."""
    from . import ingest_papers

    if kind == "github":
        from . import ingest_github
        gh = ingest_github._github_owner_repo(url)
        return f"github:{gh[0]}/{gh[1]}" if gh else None
    if kind == "paper":
        # offline: deterministic id only (content_type is the one non-network fact allowed in).
        # Omitted rather than passed as None: `paper_from_url` defaults it the same way, and
        # every EXISTING caller's test double predates this parameter and doesn't expect it.
        kw = {"content_type": content_type} if content_type else {}
        parsed = ingest_papers.paper_from_url(url, enrich=False, **kw)
        return ingest_papers.paper_atom_id(parsed) if parsed else None
    if kind == "x":
        tid = parse_tweet_id(url)
        return f"x:{tid}" if tid else None
    if kind == "article":
        from .ingest_blog import _canon_post_url
        return _canon_post_url(url)
    return None                              # substack — the post id is behind the fetch


def atom_present(conn: sqlite3.Connection, atom_id: str) -> bool:
    """Is this atom already in the store? The pre-check that lets an unchanged re-run vouch WITHOUT
    a network fetch, and that separates atomize_paper's dedup-None (paper present → still vouch)
    from its failure-None (paper absent → do NOT vouch to a missing atom).

    It answers "is it DURABLE", not "did the mint succeed". Under a sink a submitted atom sits in
    RAM and this returns False for something that lands seconds later — see `submit_atom`'s
    told-not-asked note. Any caller deciding on landing must use `on_written`."""
    return conn.execute("SELECT 1 FROM atoms WHERE atom_id=?", (atom_id,)).fetchone() is not None


def mint_artifact(conn, embedder, url: str, kind: str, *, entry_mode: str = "author_referenced",
                  paper_seen: dict | None = None, gh_seen: dict | None = None,
                  sub_seen: dict | None = None, img_cache: dict | None = None,
                  sink=None, on_written=None, prefetched: dict | None = None,
                  in_flight=(), content_type: str | None = None) -> dict:
    """ONE github / paper / substack url → its artifact atom. The de-`self`'d minting half of
    `LinkDispatcher._dispatch_one`; the vouch bookkeeping stays with the dispatcher.

    `content_type` — the url's REAL Content-Type, when a caller already fetched it
    (`classify_link_deep`'s job, never this function's). Passed straight through to
    `ingest_papers.paper_from_url`; irrelevant to every kind but 'paper'.

    Returns `{"status", "atom_id", "used_prefetch"}` where status is:
      • "present"   — a pre-check found it durable. NO network was spent. `atom_id` is real.
      • "in-flight" — the caller told us it is already submitted-but-not-durable this window
                      (`in_flight`), so we did not re-mint or re-embed. `atom_id` is real.
      • "minted"    — the adapter ran and the atom exists, or will once the sink flushes.
      • "failed"    — nothing was written and nothing exists. Do NOT point an edge at it.

    Present is checked BEFORE in-flight, matching the pre-extraction order: an atom that is somehow
    both would otherwise queue a vouch against an `_on_written` that may never fire again.

    Only 'article' and 'x' are absent from this router — they are not artifact ADAPTERS but full
    ingest paths of their own (`ingest_blog.article_atom_from_url`, `ingest_x.x_atom_from_url`), and
    folding them in here would drag the content gate and a twitterapi key into the X footprint
    puller's import graph for no caller that wants them.

    Never raises for a bad link — every adapter here already degrades to None (fail-safe)."""
    from . import ingest_common, ingest_github, ingest_papers, ingest_substack

    out = {"status": "failed", "atom_id": None, "used_prefetch": False}

    aid = predicted_atom_id(url, kind, content_type=content_type)
    if aid:
        if atom_present(conn, aid):
            # Hopper's own pre-check is a shortcut, not the guarantee — this is the path a deposit
            # actually lands on for a github/paper URL, so the attestation has to be recorded here
            # too (RULED 2026-08-25). No-op under a machine `entry_mode`.
            ingest_common.promote_atom(conn, aid, entry_mode)
            return {**out, "status": "present", "atom_id": aid}
        if aid in in_flight:
            return {**out, "status": "in-flight", "atom_id": aid}
    elif kind in ("github", "paper"):
        # No derivable id means the url is not a repo / not a paper. Bail BEFORE the paid enrich —
        # the pre-extraction code did exactly this, and skipping it would make a junk `.pdf` link
        # buy an S2 round-trip for an atom that can never be keyed.
        return out

    if kind == "github":
        out["used_prefetch"] = bool(prefetched)
        minted = ingest_github.github_atom_from_url(   # → CANONICAL id (casing may differ) or None
            conn, embedder, url, entry_mode=entry_mode, seen=gh_seen, sink=sink,
            prefetched=prefetched, on_written=on_written)
        return {**out, "status": "minted", "atom_id": minted} if minted else out

    if kind == "substack":
        # Substack has no url-derived id, so the usual pre-check can't run and the adapter
        # collapses mint-and-present into one return value. Snapshot the hash ledger FIRST so a
        # repeat save can still be told apart from a real mint (asking after would answer "yes"
        # either way).
        if sub_seen is None:
            sub_seen = ingest_substack.schema.load_hashes(conn, "substack")
        before = set(sub_seen)
        minted = ingest_substack.substack_atom_from_url(   # mint / present → post id, or None
            conn, embedder, url, entry_mode=entry_mode, seen=sub_seen, img_cache=img_cache)
        if not minted:
            return out
        # Not sink-routed, so a returned id is ALREADY durable either way.
        return {**out, "status": ("present" if minted in before else "minted"), "atom_id": minted}

    if kind != "paper":
        return out

    # paper (arXiv / DOI / OpenReview / S2 / raw .pdf — all get full-body ingest)
    out["used_prefetch"] = bool(prefetched)
    kw = {"content_type": content_type} if content_type else {}          # see predicted_atom_id
    paper = (prefetched["paper"] if prefetched else                      # enrich to MINT
             ingest_papers.paper_from_url(url, **kw))
    if not paper:
        return out
    minted = ingest_papers.atomize_paper(
        conn, embedder, paper, entry_mode=entry_mode, seen=paper_seen, sink=sink,
        on_written=on_written, **({"fulltext": prefetched["fulltext"]} if prefetched else {}))
    # atomize_paper returns None for THREE different things: "deduped, already there" (the atom
    # exists → a caller may point at it), "the write failed", and "the S2 metadata fetch was blocked
    # with no full text, nothing written" (retried next run). Its return alone cannot separate them;
    # `atom_present` breaks the tie, because only the dedup case leaves an atom on disk.
    if minted is None and not (aid and atom_present(conn, aid)):
        return out
    return {**out, "status": "minted", "atom_id": aid}
