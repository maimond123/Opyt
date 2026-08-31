"""
pipeline/ingestion/source_classify.py

A richer sibling to `discover_profile._classify_url`, which returns a bare type string
and is pinned by 3 live callers that expect exactly that. This returns a typed
`ProfileLink` carrying three extra facts hub expansion and atom-KB onboarding need:
``is_profile`` (account/home page vs an artifact like a repo or a post), ``shape``
("org" vs "personal" — org-shaped links become an affiliation edge, not the Oracle's
own atoms), and ``handle`` (for matching a surfaced profile into ``same_entity``).

Pure: stdlib only, no network. Unparseable/empty input returns ``None`` (fail-safe).
Kept separate from ``_classify_url`` so the live probe path stays byte-for-byte unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse


@dataclass(frozen=True)
class ProfileLink:
    """A classified link. ``url`` is the original string (callers canonicalize)."""

    type: str          # github|scholar|orcid|substack|blog|x|youtube|linkedin|paper|podcast|medium|gitlab
    is_profile: bool   # True = account/home; False = an artifact (repo/abs/watch/post)
    handle: str | None # account handle, else None
    shape: str         # "personal" | "org" | "unknown"
    url: str = ""      # the source url as given (so callers can build edges/sources)


# ── Host tables ───────────────────────────────────────────────────────────────

_YOUTUBE_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
_PODCAST_HOSTS = {"open.spotify.com", "spotify.com", "podcasts.apple.com", "apple.co",
                  "pod.link", "overcast.fm", "pca.st"}
_PAPER_HOSTS = {"arxiv.org", "doi.org", "biorxiv.org", "www.biorxiv.org",
                "medrxiv.org", "papers.ssrn.com", "openreview.net",
                "aclanthology.org", "dl.acm.org", "pubmed.ncbi.nlm.nih.gov"}

# First path segments on github.com / x.com that are NOT a user account.
_GITHUB_RESERVED = {"orgs", "about", "pricing", "features", "marketplace", "sponsors",
                    "settings", "login", "join", "explore", "topics", "trending",
                    "collections", "events", "apps", "team", "enterprise", "security",
                    "notifications", "new", "organizations", "site", "contact",
                    "customer-stories", "readme", "mobile", "nonprofit", "education",
                    "search", "pulls", "issues", "codespaces"}
_X_RESERVED = {"home", "explore", "notifications", "messages", "search", "settings",
               "i", "intent", "hashtag", "compose", "login", "logout", "signup",
               "tos", "privacy", "about", "share"}

# Second-level labels inside a country-code TLD (foo.co.uk → foo, not co).
_SLD_SECOND_LEVEL = {"co", "com", "org", "net", "gov", "edu", "ac", "gob", "go"}


def _host_of(parsed) -> str:
    host = (parsed.netloc or "").lower().split("@")[-1].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    if host in ("twitter.com", "mobile.twitter.com"):
        host = "x.com"
    return host


def _domain_label(host: str) -> str | None:
    """The registrable label of a bare host — the identity of a personal site.

    ``willcb.com`` → ``willcb``; ``blog.willcb.com`` → ``willcb``;
    ``foo.co.uk`` → ``foo``. Pure heuristic (no public-suffix list dep): take the
    label left of the TLD, stepping one further when the TLD is a ccTLD SLD.
    """
    parts = [p for p in host.split(".") if p]
    if len(parts) < 2:
        return parts[0] if parts else None
    if len(parts) >= 3 and parts[-2] in _SLD_SECOND_LEVEL:
        return parts[-3]
    return parts[-2]


def classify_source(url: str) -> ProfileLink | None:
    """Classify a URL into a typed ``ProfileLink``, or ``None`` if unusable."""
    if not url or not url.strip():
        return None
    raw = url.strip()
    # Non-web hrefs (mailto:/tel:/javascript:/…) are not profiles — real blog HTML is
    # full of them, and without this they'd urlparse into a bogus host+label.
    if raw.lower().startswith(("mailto:", "tel:", "javascript:", "data:", "file:",
                               "sms:", "ftp:", "#")):
        return None
    parsed = urlparse(raw if "://" in raw else "https://" + raw)
    host = _host_of(parsed)
    if not host:
        return None
    segs = [s for s in parsed.path.split("/") if s]
    path_l = parsed.path.lower()

    def mk(type_, is_profile, handle, shape):
        return ProfileLink(type=type_, is_profile=is_profile, handle=handle, shape=shape, url=raw)

    # ── arXiv AUTHOR listing (arxiv.org/a/name) is a research PROFILE, not a paper —
    #    check before the paper-host bucket, which would otherwise swallow it. ──
    if host == "arxiv.org" and segs and segs[0].lower() == "a":
        return mk("scholar", True, segs[1] if len(segs) >= 2 else None, "personal")

    # ── Papers / PDFs / DOIs — always artifacts, never a person's profile ──
    if host in _PAPER_HOSTS or host.endswith(".arxiv.org") or path_l.endswith(".pdf"):
        return mk("paper", False, None, "unknown")

    # ── GitHub ──
    if host == "github.com":
        if not segs:
            return mk("github", False, None, "unknown")            # site root
        first = segs[0].lower()
        if first == "orgs" and len(segs) >= 2:
            return mk("github", True, segs[1].lower(), "org")       # github.com/orgs/<org>
        if first in _GITHUB_RESERVED:
            return mk("github", False, None, "unknown")             # a feature page, not a user
        if len(segs) == 1:
            return mk("github", True, first, "personal")            # github.com/<user>
        return mk("github", False, first, "personal")              # github.com/<user>/<repo> = artifact

    if host == "gitlab.com":
        if segs and segs[0].lower() not in {"explore", "help", "-", "users"}:
            is_prof = len(segs) == 1
            return mk("gitlab", is_prof, segs[0].lower(), "personal")
        return mk("gitlab", False, None, "unknown")

    # ── X / Twitter ──
    if host == "x.com":
        if not segs:
            return mk("x", False, None, "unknown")
        first = segs[0].lower()
        if first in _X_RESERVED:
            return mk("x", False, None, "unknown")
        if len(segs) >= 3 and segs[1].lower() == "status":
            return mk("x", False, first, "personal")               # a tweet = artifact
        return mk("x", True, first, "personal")                    # x.com/<handle>

    # ── YouTube ──
    if host in _YOUTUBE_HOSTS:
        if host == "youtu.be":
            return mk("youtube", False, None, "unknown")           # short video link
        if not segs:
            return mk("youtube", False, None, "unknown")
        first = segs[0]
        if first.startswith("@"):
            return mk("youtube", True, first[1:].lower(), "personal")
        if first.lower() in ("channel", "c", "user") and len(segs) >= 2:
            return mk("youtube", True, segs[1].lower(), "personal")
        return mk("youtube", False, None, "unknown")               # watch/playlist/results

    # ── Substack (subdomain owns identity) ──
    if host.endswith(".substack.com"):
        sub = host[: -len(".substack.com")]
        if not sub or sub == "www":
            return mk("substack", False, None, "unknown")
        is_prof = not (segs and segs[0].lower() == "p")            # /p/<slug> = a post
        return mk("substack", is_prof, sub, "personal")

    # ── Medium (bucketed as blog per _classify_url; keep the handle) ──
    if host == "medium.com":
        if segs and segs[0].startswith("@"):
            is_prof = len(segs) == 1
            return mk("blog", is_prof, segs[0][1:].lower(), "personal")
        return mk("blog", False, None, "unknown")
    if host.endswith(".medium.com"):
        sub = host[: -len(".medium.com")]
        return mk("blog", not segs, sub or None, "personal")

    # ── Scholar / academic identity ──
    if host.startswith("scholar.google."):
        if "citations" in path_l:
            user = (parse_qs(parsed.query).get("user") or [None])[0]
            return mk("scholar", bool(user), user, "personal")
        return mk("scholar", False, None, "unknown")               # a search, not a profile
    if host.endswith("semanticscholar.org"):
        if segs and segs[0].lower() == "author":
            return mk("scholar", True, segs[1] if len(segs) >= 2 else None, "personal")
        return mk("scholar", False, None, "unknown")
    if host == "orcid.org":
        return mk("orcid", bool(segs), segs[0] if segs else None, "personal")
    # dblp — the canonical CS bibliography; author pages live at /pid/… or /pers/….
    if host == "dblp.org" or host.endswith(".dblp.org"):
        if segs and segs[0].lower() in ("pid", "pers"):
            return mk("scholar", True, segs[-1].replace(".html", "").lower(), "personal")
        return mk("scholar", bool(segs), None, "unknown")
    # ResearchGate / Academia.edu — profile pages (lower-signal, mostly reposts).
    if host.endswith("researchgate.net"):
        if len(segs) >= 2 and segs[0].lower() == "profile":
            return mk("scholar", True, segs[1].lower(), "personal")
        return mk("scholar", False, None, "unknown")
    if host == "academia.edu" or host.endswith(".academia.edu"):
        sub = host[: -len(".academia.edu")] if host != "academia.edu" else ""
        if sub and sub != "www":
            return mk("scholar", True, sub, "personal")           # {name}.academia.edu
        if segs:
            return mk("scholar", True, segs[0].lower(), "personal")  # academia.edu/{id}
        return mk("scholar", False, None, "unknown")

    # ── LinkedIn ──
    if host.endswith("linkedin.com"):
        if segs and segs[0].lower() == "in" and len(segs) >= 2:
            return mk("linkedin", True, segs[1].lower(), "personal")
        if segs and segs[0].lower() in ("company", "school") and len(segs) >= 2:
            return mk("linkedin", True, segs[1].lower(), "org")
        return mk("linkedin", False, None, "unknown")

    # ── Podcasts (discovered but not part of PROFILE_SCOPE / atom ingest) ──
    if host in _PODCAST_HOSTS:
        return mk("podcast", False, None, "unknown")

    # ── Default: a personal site / blog. Bare host = the profile; a path = a page. ──
    return mk("blog", len(segs) == 0, _domain_label(host), "unknown")
