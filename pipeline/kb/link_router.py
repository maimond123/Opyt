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

# A link we turn into its OWN atom. This module is the only owner: `ingest_x_footprint` held a
# re-export alias until `8da077a6` (2026-09-05) deleted it and edited the twin comment there,
# leaving this one describing a link that no longer exists.
# A host where EVERY page is a paper, so the hostname alone is the whole test. A host that is only
# partly papers does not belong here — see `_PAPER_PATH_RES` directly below, and the 2026-09-16
# note on it for what putting one here costs.
_PAPER_HOSTS = ("arxiv.org", "doi.org", "biorxiv.org", "medrxiv.org",
                "openreview.net", "pubmed.ncbi.nlm.nih.gov", "aclanthology.org")

# Host AND path together — for hosts where only one section is papers. A bare host entry above
# would route every Hugging Face model page and every NCBI gene record to the paper adapter, which
# is the mis-route `_is_host` exists to prevent, one level down in the url.
#
# ⚠️ The default for a NEW host is this tuple, not the one above. SSRN, Zenodo, alphaxiv and
# Semantic Scholar were all added as bare hosts and all four were wrong in the same way — measured
# 2026-09-16, `zenodo.org/communities/ecodata`, `ssrn.com/en/index.cfm/aboutssrn/`,
# `papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=…` and
# `semanticscholar.org/author/Y-LeCun/1688882` every one routed `paper`. A community page, an
# about page and two AUTHOR pages are not papers, and the hopper tool's own offer bar states the
# rule these broke: "the path is part of the promise".
#
# Each pattern here is deliberately no wider than the parser that has to name the paper afterwards
# (`ingest_papers._SSRN_RE`, `_S2_RE`, `_ZENODO_RECORD_RE`, `_ARXIV_MIRROR_RE`): routing `paper`
# for a url no parser can key is an atom that can never be written, which is a promise broken
# later and further away.
_PAPER_PATH_RES = (
    re.compile(r"^https?://(?:www\.)?huggingface\.co/papers/", re.I),
    re.compile(r"^https?://(?:www\.|pmc\.)?ncbi\.nlm\.nih\.gov/(?:pmc/)?articles/", re.I),
    # MED (a PMID), PMC (a PMCID) and PPR (a preprint) — the three sources `_looked_up_doi` can
    # resolve. `/articles/PMC…` is their legacy form and still in circulation.
    re.compile(r"^https?://(?:www\.)?europepmc\.org/(?:article|abstract)/(?:MED|PMC|PPR)/", re.I),
    re.compile(r"^https?://(?:www\.)?europepmc\.org/articles?/PMC\d", re.I),
    re.compile(r"^https?://(?:www\.)?openalex\.org/W\d+", re.I),
    # `/paper/…` is the only shape `_S2_RE` reads; `api.…/CorpusID:N` is a real pasted form it
    # does not, kept here so scoping the host did not quietly demote it to a blog post.
    re.compile(r"^https?://(?:www\.|api\.)?semanticscholar\.org/(?:paper/|CorpusID:)", re.I),
    re.compile(r"^https?://(?:[\w.-]+\.)?ssrn\.com/\S*?abstract(?:_id)?=\d", re.I),
    re.compile(r"^https?://(?:www\.)?zenodo\.org/records?/\d", re.I),
    re.compile(r"^https?://(?:www\.)?alphaxiv\.org/(?:abs|pdf|overview)/", re.I),
)

# A DOI sitting in the PATH of any host — the structural half of the paper test, and the reason
# this list of hosts does not need to grow into a directory of publishers.
#
# Measured 2026-09-09: every one of ACS, JACS, Wiley, ACM, Taylor & Francis, SAGE, Springer and
# APS puts the DOI in the url, and none of their hosts was here — so `classify_link` returned None
# and a JACS paper was filed as a blog post under `who_id = blog:pubs.acs.org`. The publisher list
# that would have fixed that is unbounded and needs maintaining; the DOI's own shape does not.
#
# Deliberately NOT anchored on a `/doi/` segment even though ACS and Wiley both use one: Springer
# spells it `/article/10.1007/…` and APS `/abstract/10.1103/…`, so the segment is a convention and
# the DOI is the fact. `10.` + 4-9 digits + `/` is distinctive enough to search for on its own.
#
# It subsumes the doi.org case rather than sitting beside it — `https://doi.org/10.1021/x` matches
# this too — which is why `ingest_papers._DOI_RE` no longer names that host either.
_DOI_IN_PATH_RE = re.compile(r"/(10\.\d{4,9}/[^\s?#]+)")

# A single post on X. Only x.com / twitter.com, optionally www.- or mobile.-prefixed — the mirror
# front-ends (fxtwitter, vxtwitter, nitter) are deliberately absent: nothing here has been verified
# against them, and a host we cannot fetch is better refused than mis-routed.
_X_POST_RE = re.compile(
    r"^https?://(?:www\.|mobile\.)?(?:x|twitter)\.com/[^/]+/status(?:es)?/(\d+)", re.I)

# The kinds a host model may assert. `classify_reference` accepts a hint only from this set, so a
# typo or a hallucinated kind falls through to the article catch-all instead of routing nowhere.
# PRIVATE: this module is the only enforcer, and a public name is a promise to other modules that
# none of them took up in any commit since `9ebd80ca`. The `hopper` tool's `kind_hint` docstring
# names the same five values, and that is guidance for a host model about WHICH one to pass — not
# a second definition an import could replace.
_HINT_KINDS = ("github", "substack", "paper", "x", "article")


def _host(url: str) -> str:
    """The url's hostname, lowercased, port and userinfo stripped — `""` for anything without one.
    `.hostname` rather than `.netloc` because `netloc` carries `user:pw@` and `:8080` into the
    comparison, and only the hostname is the thing being identified."""
    try:
        return (urlparse(url or "").hostname or "").lower()
    except ValueError:                      # malformed IPv6 literal etc. — not a host we can name
        return ""


def _is_host(host: str, domain: str) -> bool:
    """Is `host` this domain, or a subdomain of it? The DNS-suffix boundary, and the reason this
    is not a substring test.

    `"substack.com" in netloc` is true for `not-substack.com` (a different registrable domain that
    merely contains the name) and for `github.com.evil.example` (the provider's name as a label
    under somebody else's domain). Both routed an arbitrary host into a provider's adapter, and a
    mis-route never raises — it sits wrong forever. Real subdomains (`carol.substack.com`,
    `www.github.com`) still match, which is the whole point of the suffix form."""
    return host == domain or host.endswith("." + domain)


def classify_link(url: str) -> str | None:
    """The kind of dispatchable artifact a url points at — 'github', 'paper', or 'substack' — or
    None for a BARE link (news, a personal blog, a company site, another tweet). ONE classifier
    feeds both gates: `_dispatchable_link` (any non-None keeps the reaction alive) and the Step-3
    dispatcher (which MINTS only github + paper; Substack is the deferred Tier-2 tier).

    Pure URL matching — no LLM and no network, which is what makes it a fact rather than a guess,
    and what lets it outrank a host model's hint whenever it fires.

    Two independent paper tests, and the second is what makes this generalize past a host list: a
    known paper HOST, a known paper host+PATH (`_PAPER_PATH_RES`, for hosts that are only partly
    papers), or a DOI in the url's own path (`_DOI_IN_PATH_RE`) on any host at all. The host list
    stays because arXiv, PubMed and OpenReview identify a paper WITHOUT a DOI in the url; the path
    test covers every publisher that puts one there, and every host that is papers in one section
    only."""
    host = _host(url)
    if _is_host(host, "github.com"):
        return "github"
    if _is_host(host, "substack.com"):
        return "substack"
    if (any(_is_host(host, h) for h in _PAPER_HOSTS)
            or any(r.search(url or "") for r in _PAPER_PATH_RES)
            or _DOI_IN_PATH_RE.search(url or "")
            or url.lower().split("?")[0].endswith(".pdf")):
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
# A DOI the page PRINTS rather than declares in a tag. Same bytes the citation scan already reads,
# so it costs nothing extra — and it is still a DECLARED identifier, which is the line this module
# does not cross: every attempt to INFER a paper's identity has measured badly (Crossref title
# search resolved 1 of 3 on 2026-09-09, returning a Faculty Opinions review of AlphaFold and a
# different paper called "Is Attention All You Need?"), and a wrong paper atom is immutable.
#
# `doi.org/` and a `DOI:` label are both required forms — a bare `10.1234/x` anywhere in 64KB of
# markup matches script payloads and analytics ids, and this pattern runs on pages that reached
# the ARTICLE fallback, i.e. mostly not papers at all.
_PRINTED_DOI_RE = re.compile(
    rb'(?:doi\.org/|\bdoi:\s*)(10\.\d{4,9}/[^\s"\'<>&]+)', re.I)


def classify_link_deep(url: str) -> tuple[str, str, str | None] | None:
    """NETWORK-BOUND fallback for 'paper' outside the free `_PAPER_HOSTS` list — deliberately kept
    out of `classify_link`'s free path; a caller must opt in as a bounded last resort, never a
    default per-url scan. Fetches once and checks a `citation_doi` meta tag (rewritten to a
    `doi.org/{doi}` url) or a PDF Content-Type / any other `citation_*` tag.

    Two sources, in order of authority: what the PAGE declares about itself, then — when the page
    declares nothing or will not be fetched at all — what OpenAlex records as living at that url.

    Returns `(kind, mint_url, content_type)` — always `("paper", ...)`, or None if nothing
    paper-shaped was found. `mint_url` is what a caller should mint against, not necessarily `url`
    itself; `content_type` carries the raw header when it matters downstream (the PDF case), else
    None. Never raises — a network error or oversized page reads the same as "not a paper"."""
    try:
        with requests.get(url, timeout=_DEEP_PROBE_TIMEOUT, stream=True,
                           headers=_DEEP_PROBE_UA) as resp:
            if resp.status_code >= 400:
                # They REFUSED us — which is not evidence about whether this is a paper, and is
                # the single most common answer here: 22 of the 37 urls this probe could not crack
                # answered 4xx (measured 2026-09-11). Fall through to the index below rather than
                # reading a closed door as "not a paper".
                return _openalex_fallback(url)
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "application/pdf" in ctype:
                return "paper", url, ctype
            if "text/html" not in ctype:
                # A SUCCESSFUL response that is an image or a json blob is a real answer: there is
                # no article here. Unlike a 4xx, this one is worth believing, so it costs no
                # further call.
                return None
            buf = b""
            for chunk in resp.iter_content(8192):
                buf += chunk
                m = _CITATION_DOI_RE.search(buf)
                if m:
                    doi = m.group(1).decode("utf-8", "replace").strip()
                    return "paper", f"https://doi.org/{doi}", None
                if len(buf) > _DEEP_PROBE_MAX_BYTES:
                    break
            # The page prints its DOI without declaring one. Checked AFTER the tag, never
            # instead: a `citation_doi` is the publisher naming this page's own paper, while a
            # printed DOI may be a reference to somebody else's — so the tag wins whenever both
            # are present, and this only runs when there is no tag to prefer.
            m = _PRINTED_DOI_RE.search(buf)
            if m:
                doi = m.group(1).decode("utf-8", "replace").strip().rstrip(".,;)")
                return "paper", f"https://doi.org/{doi}", None
            if _CITATION_ANY_RE.search(buf):
                # Scholarly, but naming no DOI — so this branch alone yields a `paper` whose id
                # nothing can derive, and `mint_artifact` has nothing to key an atom on. ASK THE
                # INDEX FIRST. This return sat above the fallback until 2026-09-16, which meant a
                # page that merely PRINTED `citation_author` was enough to skip the one source that
                # could name it: 8 of the 124 refused urls in the Set A+B sweeps, `jmlr.org`,
                # `eprint.iacr.org` and `repositorio.unal.edu.co/handle/unal/81443` (ResNet, which
                # OpenAlex names outright) among them.
                #
                # Keeping the bare `paper` as the FALLBACK's fallback is deliberate. It is the
                # honest answer for a page like `pure.au.dk/…/moloch-0`, which declares
                # `citation_author` + `citation_journal_title`, no DOI, and is unknown to OpenAlex:
                # a paper we cannot name stores nothing, while `article` would file it as a blog
                # post under `who_id = blog:pure.au.dk` — the mis-route this module exists to stop.
                return _openalex_fallback(url) or ("paper", url, None)
    except requests.RequestException:
        return _openalex_fallback(url)
    # The page told us nothing — or refused to talk to us at all, which is the COMMON case here:
    # of 37 urls this probe could not crack, 22 answered 403/404/429, identically for a browser
    # User-Agent (measured 2026-09-11), so there was never a page to read and no amount of trying
    # harder would have helped. Ask the index that already knows which paper lives at that url.
    #
    # Recovers 59 of the 96 refused urls in that sweep: institutional repositories
    # (`repositorio.unal.edu.co/handle/unal/81443` is ResNet) and aggregators.
    #
    # It does NOT cover publisher urls that name the article some private way. This comment claimed
    # "Elsevier's PII, MDPI's issue path" until 2026-09-16, written from the sweep's aggregate
    # recovery rate rather than from those forms; asked directly, with a control proving OpenAlex
    # was answering, `sciencedirect.com/science/article/pii/S0092867420302294`,
    # `mdpi.com/2072-6643/13/6/1815` and `dspace.mit.edu/handle/1721.1/7582` all returned None.
    # The index stores the landing pages it stores, and those are not among them.
    #
    # Reached ONLY from the article fallback, and that is a safety property rather than an
    # accident: every host with an authoritative route of its own — a DOI in the path, arXiv,
    # PubMed, PMC, Zenodo, SSRN — is already `paper` by the time we get here, so this can never
    # overrule `_pubmed_doi`'s "that record carries no DOI" with an index's guess. It is exactly
    # that ordering that keeps OpenAlex's one bad row out (pmid 526911, a 1979 record, is listed
    # there as a location of a 2026 deposit).
    #
    # Costs one cheap API call on urls that turn out NOT to be papers, which is most of what
    # reaches this function. Accepted because the page fetch above already spent up to 5s on them.
    return _openalex_fallback(url)


def _openalex_fallback(url: str) -> tuple[str, str, str | None] | None:
    """`classify_link_deep`'s second source: the DOI OpenAlex records at this url, or None."""
    from . import ingest_papers
    if doi := ingest_papers._openalex_doi_by_url(url):
        return "paper", f"https://doi.org/{doi}", None
    return None


def parse_tweet_id(url: str) -> str | None:
    """The numeric tweet id in an X post url, or None. Query string and trailing path (`/photo/1`)
    are ignored — the id is the only identity."""
    m = _X_POST_RE.match((url or "").strip())
    return m.group(1) if m else None


def _is_http_url(reference: str) -> bool:
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
    if not _is_http_url(ref):
        return None, "none"
    kind = classify_link(ref)
    if kind:
        return kind, "sniffed"
    if parse_tweet_id(ref):
        return "x", "sniffed"
    if hint in _HINT_KINDS:
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
    elif kind == "github":
        # No derivable id means the url is not a repo. Bail BEFORE the paid enrich — the
        # pre-extraction code did exactly this, and skipping it would make a junk link buy a round
        # trip for an atom that can never be keyed.
        #
        # 'paper' WAS here too, and that was a real bug for a month. It rests on "no derivable id
        # means not a paper", which stopped being true on 2026-08-13 — the same commit that wrote
        # this bail also made `pubmed.ncbi.nlm.nih.gov` a paper host, and a PMID is precisely an id
        # the string cannot yield. So every PubMed save returned `failed` while the hopper tool
        # advertised PubMed as a headline host, and `_pubmed_doi` (shipped 2026-09-09 to resolve
        # exactly these) was never once reached through here. 2026-09-11 added PMC, Europe PMC,
        # Zenodo and OpenAlex-work urls to the same dead path.
        #
        # The cost argument survives without it: `paper_from_url` only spends a request when
        # `_looked_up_doi` recognises the url, and returns None before the S2 call otherwise — so a
        # junk `.pdf` still costs nothing. Verified by `test_a_junk_paper_url_spends_no_request`.
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
    # The id the ADAPTER resolved, for the urls whose id was never in the string (PubMed, PMC,
    # Europe PMC, Zenodo, an OpenAlex work). Everything below keys on `aid`, so leaving it None
    # would report a successful mint as `atom_id: None` and re-report an already-stored paper as
    # `failed` — both silent, and both invisible to a test that only checks the arXiv/DOI forms.
    aid = aid or ingest_papers.paper_atom_id(paper)
    # The pre-check at the top could not run for those urls. Run it HERE, now that there is an id:
    # `atomize_paper` would dedup anyway (Policy B, before the paid embed), but it returns None for
    # dedup and for failure alike, so without this a re-save reads as `minted` and — the part that
    # matters — never records the user-saved attestation `promote_atom` exists for.
    if aid and atom_present(conn, aid):
        ingest_common.promote_atom(conn, aid, entry_mode)
        return {**out, "status": "present", "atom_id": aid}
    if aid and aid in in_flight:
        return {**out, "status": "in-flight", "atom_id": aid}
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
