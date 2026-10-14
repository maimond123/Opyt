"""
pipeline/kb/link_router.py — one url → which adapter. The mint that follows is `link_mint`.

Shared by `ingest_x_footprint` and Hopper so both route identically; a mis-route never raises, it
sits wrong forever.

  • `classify_link`      — the artifact sniffer: github | substack | paper | None.
  • `classify_reference` — Hopper's wider vocabulary: + "x" (a single post) and + "article".
"""

from __future__ import annotations

from pipeline.host_access import request as host_request
from pipeline.platforms.papers import lookup as paper_lookup
from pipeline.platforms.url_canon import host_of

import re
from urllib.parse import urlparse

import requests

# Hosts where every page is a paper, so the hostname alone is the test. A host that is only partly
# papers belongs in `_PAPER_PATH_RES`.
_PAPER_HOSTS = ("arxiv.org", "doi.org", "biorxiv.org", "medrxiv.org",
                "openreview.net", "pubmed.ncbi.nlm.nih.gov", "aclanthology.org")

# Host and path together, for hosts where only one section is papers. The default for a new host
# is this tuple. Each pattern is no wider than the parser that names the paper afterwards
# (`ingest_papers._SSRN_RE`, `_S2_RE`, `_ZENODO_RECORD_RE`, `_ARXIV_MIRROR_RE`), so every url
# routed `paper` can be keyed.
_PAPER_PATH_RES = (
    re.compile(r"^https?://(?:www\.)?huggingface\.co/papers/", re.I),
    re.compile(r"^https?://(?:www\.|pmc\.)?ncbi\.nlm\.nih\.gov/(?:pmc/)?articles/", re.I),
    # MED, PMC and PPR — the sources `_looked_up_doi` can resolve; `/articles/PMC…` is the legacy form.
    re.compile(r"^https?://(?:www\.)?europepmc\.org/(?:article|abstract)/(?:MED|PMC|PPR)/", re.I),
    re.compile(r"^https?://(?:www\.)?europepmc\.org/articles?/PMC\d", re.I),
    re.compile(r"^https?://(?:www\.)?openalex\.org/W\d+", re.I),
    # `api.…/CorpusID:N` is a real pasted form `_S2_RE` does not read; kept so it is not a blog post.
    re.compile(r"^https?://(?:www\.|api\.)?semanticscholar\.org/(?:paper/|CorpusID:)", re.I),
    re.compile(r"^https?://(?:[\w.-]+\.)?ssrn\.com/\S*?abstract(?:_id)?=\d", re.I),
    re.compile(r"^https?://(?:www\.)?zenodo\.org/records?/\d", re.I),
    re.compile(r"^https?://(?:www\.)?alphaxiv\.org/(?:abs|pdf|overview)/", re.I),
    # Report archives with no DOI, keyed on the report page (`ingest_papers._REPORT_RES`).
    re.compile(r"^https?://eprint\.iacr\.org/\d{4}/\d", re.I),
    re.compile(r"^https?://eccc\.weizmann\.ac\.il/report/\d{4}/\d", re.I),
)

# A DOI in the path of any host, so the host lists need not become a directory of publishers. Not
# anchored on a `/doi/` segment: publishers spell it differently (`/article/10.1007/…`). Also
# covers doi.org itself.
_DOI_IN_PATH_RE = re.compile(r"/(10\.\d{4,9}/[^\s?#]+)")

# A single post on X. Mirror front-ends (fxtwitter, nitter) are unverified, so they are refused.
_X_POST_RE = re.compile(
    r"^https?://(?:www\.|mobile\.)?(?:x|twitter)\.com/[^/]+/status(?:es)?/(\d+)", re.I)

# The kinds a host model may assert; any other hint falls through to the article catch-all.
_HINT_KINDS = ("github", "substack", "paper", "x", "article")


def _is_host(host: str, domain: str) -> bool:
    """Is `host` this domain, or a subdomain of it? A suffix test, not a substring test, so
    `not-substack.com` and `github.com.evil.example` do not match."""
    return host == domain or host.endswith("." + domain)


def classify_link(url: str) -> str | None:
    """The kind of dispatchable artifact a url points at — 'github', 'paper', or 'substack' — or
    None for a bare link. Read by the X footprint's keep gate (any non-None keeps a short post)
    and by `reference_kind`.

    Pure url matching, no network, so it outranks a host model's hint. A paper is a known paper
    host, a known host+path, or a DOI in the url's path on any host."""
    host = host_of(url)
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
# A DOI the page prints rather than declares in a tag. Still a stated identifier: this module never
# infers a paper's identity, since a wrong paper atom is immutable. Requires a `doi.org/` or `DOI:`
# prefix, because a bare `10.1234/x` matches script payloads and analytics ids.
_PRINTED_DOI_RE = re.compile(
    rb'(?:doi\.org/|\bdoi:\s*)(10\.\d{4,9}/[^\s"\'<>&]+)', re.I)


def classify_link_deep(url: str) -> tuple[str, str, str | None] | None:
    """Network-bound fallback for 'paper' beyond `classify_link`'s offline tests; a caller opts in
    as a bounded last resort. Two sources, in order of authority: what the page declares (a
    `citation_doi` tag, a PDF Content-Type, a printed DOI, any `citation_*` tag), then what OpenAlex
    records at that url.

    Returns `(kind, mint_url, content_type)` — always `("paper", ...)`, or None if nothing
    paper-shaped was found. `mint_url` is what a caller should mint against, not necessarily `url`
    itself; `content_type` carries the raw header when it matters downstream (the PDF case), else
    None. Never raises — a network error or oversized page reads the same as "not a paper"."""
    try:
        with host_request(requests.get, url, timeout=_DEEP_PROBE_TIMEOUT, stream=True,
                           headers=_DEEP_PROBE_UA) as resp:
            if resp.status_code >= 400:
                # A refusal says nothing about whether this is a paper; ask the index.
                return _openalex_fallback(url)
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "application/pdf" in ctype:
                return "paper", url, ctype
            if "text/html" not in ctype:
                # A successful non-HTML response is a real answer: no article here.
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
            # Checked after the tag: a printed DOI may cite somebody else's paper.
            m = _PRINTED_DOI_RE.search(buf)
            if m:
                doi = m.group(1).decode("utf-8", "replace").strip().rstrip(".,;)")
                return "paper", f"https://doi.org/{doi}", None
            if _CITATION_ANY_RE.search(buf):
                # Scholarly but naming no DOI: ask the index first. A bare `paper` is the last
                # resort — a paper we cannot name stores nothing, while `article` would file it as
                # a blog post.
                return _openalex_fallback(url) or ("paper", url, None)
    except requests.RequestException:
        return _openalex_fallback(url)
    # The page told us nothing: ask the index that records which paper lives at that url. Safe
    # only because this is reached from the article fallback alone — every host with its own
    # authoritative route is already `paper` before this runs.
    return _openalex_fallback(url)


def _openalex_fallback(url: str) -> tuple[str, str, str | None] | None:
    """`classify_link_deep`'s second source: the DOI OpenAlex records at this url, or None."""
    if doi := paper_lookup.openalex_doi_by_url(url):
        return "paper", f"https://doi.org/{doi}", None
    return None


def parse_tweet_id(url: str) -> str | None:
    """The numeric tweet id in an X post url, or None. Query string and trailing path (`/photo/1`)
    are ignored — the id is the only identity."""
    m = _X_POST_RE.match((url or "").strip())
    return m.group(1) if m else None


def _is_http_url(reference: str) -> bool:
    """Is this a fetchable http(s) url? A non-url is the only unroutable reference."""
    try:
        p = urlparse((reference or "").strip())
    except ValueError:                      # malformed IPv6 literal etc. — not a url we can use
        return False
    return p.scheme in ("http", "https") and bool(p.netloc)


def classify_reference(reference: str, *, hint: str | None = None) -> tuple[str | None, str]:
    """Route any hand-supplied reference → `(kind, basis)`, where basis is 'sniffed' (a fact about
    the url), 'hint' (the host model's read), 'fallback' (the article catch-all), or 'none'
    (unroutable).

    Kept separate from `classify_link`, whose None the X footprint reads as a decision. A sniff
    outranks the hint, the hint outranks the fallback; an unknown hint is ignored."""
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


# Hosts whose links are never an article to save: chat transcripts behind a login, audio/video,
# invites, and gists (excluded by choice). Paywalled news is left to the fetch's own verdict.
_NOT_AN_ARTICLE_HOSTS = ("gist.github.com", "gist.githubusercontent.com",
                         "chatgpt.com", "chat.openai.com", "grok.com", "claude.ai",
                         "youtube.com", "youtu.be", "podcasts.apple.com", "music.apple.com",
                         "open.spotify.com", "twitch.tv", "discord.gg", "discord.com")


# Links that are never a document, by their shape: the post that links one already carries its URL
# and the X card's title and blurb, so fetching it would only store a landing page. Measured on
# one store, these caught 41 of the 56 pages the old content gate rejected, and 119 of
# its 736 stored linked articles (93 bare homepages), none of them an article. No `download`:
# journals serve PDFs at `/download?id=…`, and a missed paper costs more than an app page.
_LINK_ONLY_SEGMENTS = frozenset({
    "jobs", "job", "careers", "career", "apply", "join", "events", "event", "tickets", "ticket",
    "register", "reg", "shop", "store", "checkout", "pricing",
    "signup", "sign-up", "signin", "sign-in", "login", "account", "forms", "get-involved",
    "advertise",
})
_LINK_ONLY_HOSTS = ("boards.greenhouse.io", "job-boards.greenhouse.io", "jobs.lever.co",
                    "jobs.ashbyhq.com", "eventbrite.com", "lu.ma", "luma.com", "forms.gle",
                    "drive.google.com", "lnkd.in")
_LINK_ONLY_SUBDOMAINS = ("jobs.", "careers.", "tickets.", "events.", "sso.", "login.", "shop.",
                         "store.")


def _link_only(url: str) -> bool:
    """Is this link's shape never a document — a bare homepage, a jobs / events / form / login /
    shop page? Whole path segments, not substrings, so `/p/joining-forces` still reads as a post.
    Not `discover.is_nav_path`: its `about` / `terms` are nav on an Oracle's own site but sit in
    real article paths elsewhere (`/about/newsroom/2026/…`)."""
    p = urlparse(url)
    if (p.path or "/") == "/":
        return True
    host = host_of(url)
    if (any(_is_host(host, h) for h in _LINK_ONLY_HOSTS) or host.startswith(_LINK_ONLY_SUBDOMAINS)
            or (host == "docs.google.com" and p.path.startswith("/forms"))):
        return True
    return any(s.lower() in _LINK_ONLY_SEGMENTS for s in p.path.split("/"))


def is_card_link(url: str) -> bool:
    """A link-only shape (`_link_only`) worth a card: its page is never fetched as a document, but
    its title and description say what it links (`kb.link_cards`). Not a link to X itself, nor one
    on a host that is never an article."""
    host = host_of(url)
    if not _is_http_url(url) or any(_is_host(host, h) for h in _NOT_AN_ARTICLE_HOSTS):
        return False
    if _is_host(host, "x.com") or _is_host(host, "twitter.com"):
        return False
    return _link_only(url)


def is_hosted_page(url: str) -> bool:
    """A page on a platform that hosts other people's pages (a job board, an event or form
    service, a drive), whose homepage describes the platform rather than the page."""
    host = host_of(url)
    return (any(_is_host(host, h) for h in _LINK_ONLY_HOSTS)
            or (host == "docs.google.com" and urlparse(url).path.startswith("/forms")))


def reference_kind(url: str) -> str | None:
    """What an Oracle's link becomes — 'github' | 'paper' | 'substack' | 'x' | 'article' — or None
    for a link that becomes nothing. Wider than `classify_link`, narrower than `classify_reference`
    (which routes every url because a person chose it).

    A link-only shape (`_link_only`) is None too, checked before the artifact sniff so a paper
    host's homepage (`arxiv.org/`) is not taken for a paper.

    'article' is a candidate, not a verdict: the caller runs the deep paper probe first, so a
    paper on a host no list names still becomes a paper. Offline, like `classify_link`."""
    host = host_of(url)
    if not _is_http_url(url) or any(_is_host(host, h) for h in _NOT_AN_ARTICLE_HOSTS):
        return None
    if _link_only(url):
        return None
    if _is_host(host, "x.com") or _is_host(host, "twitter.com"):
        return "x" if parse_tweet_id(url) else None
    return classify_link(url) or "article"
