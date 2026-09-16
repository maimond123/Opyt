"""
pipeline/ingestion/url_canon.py

`canonical_identity(url)` collapses a URL to its platform-aware *trust unit* —
the string the trust graph treats as one node. Two URLs that are "the same
account" must canonicalize to the same identity; two different accounts on the
same host must not.

The unit is platform-specific because where "identity" lives differs:
  - substack:           the SUBDOMAIN owns identity   → someuser.substack.com
  - github/x/medium:    host + first path SEGMENT      → github.com/someuser
  - academic profiles:  the platform account ID (query or path)
  - everything else:    the bare host                  → someuser.ai

This is the squatter-defense primitive: `github.com/someuser` and
`github.com/someoneelse` are distinct nodes, so a squatter cannot inherit a
real account's inbound trust edges just by sharing a host.

Derived from `discover_profile._normalize_url()`; this goes further (drops the
scheme, lowercases the host, aliases twitter↔x, extracts the path unit).
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

# Hosts where the first path segment is the identity (the account handle).
_PATH_PLATFORMS = {"github.com", "x.com", "medium.com", "gitlab.com"}

# Excluded platforms must not fall through to the personal-blog default.
_EXCLUDED_HOSTS = {
    "youtube.com", "youtu.be", "linkedin.com", "spotify.com",
    "podcasts.apple.com", "apple.co", "pod.link", "overcast.fm", "pca.st",
}


def parse_url(url: str):
    """Parse a web URL or bare host at the identity boundary; reject unusable inputs."""
    if not url or not url.strip():
        return None
    raw = url.strip()
    if "://" not in raw and (":" in raw or raw.startswith(("/", "#"))):
        return None
    if any(c.isspace() for c in raw):
        return None
    try:
        parsed = urlparse(raw if "://" in raw else "https://" + raw)
        host = parsed.hostname
        _ = parsed.port  # validate a supplied port at the same boundary
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not host:
        return None
    if "://" not in raw and "." not in host:
        return None
    if any(host == excluded or host.endswith("." + excluded) for excluded in _EXCLUDED_HOSTS):
        return None
    return parsed


def excluded_platform(url: str) -> str | None:
    """The excluded host this URL belongs to, or None — `_EXCLUDED_HOSTS` asked by NAME.

    Same fact `parse_url` already acts on, exposed because two callers need it for opposite
    reasons. `parse_url` folds it into "unusable" and returns None, which is right for identity:
    a youtube URL names no trust unit. But a caller deciding WHAT TO SAY needs to tell "this is
    a video platform we do not read" apart from "this is not a URL", and an empty identity cannot.

    Added 2026-09-15 for `hopper`, which had no way to ask. Its router falls everything unknown
    through to `article`, so a youtube link was fetched as if it were a blog post and came back
    "the content-quality gate found no substantive units (nav / promo / boilerplate)" — which
    reads as OPYT judging the video worthless rather than as OPYT not doing video. The comment
    above `_EXCLUDED_HOSTS` had stated the rule since it was written; nothing let Hopper read it.
    """
    parsed = urlparse(url.strip() if url else "")
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if not host:
        return None
    return next((e for e in _EXCLUDED_HOSTS if host == e or host.endswith("." + e)), None)


def canonical_identity(url: str) -> str:
    """Return the account's trust unit, or an empty string for unusable/excluded URLs."""
    parsed = parse_url(url)
    if parsed is None:
        return ""
    host = parsed.hostname.removeprefix("www.")

    # twitter.com and x.com are the same platform — alias to x.com.
    if host == "twitter.com" or host == "mobile.twitter.com":
        host = "x.com"

    segs = [s for s in parsed.path.split("/") if s]

    # Academic account identifiers live in queries or deeper paths, not the bare host.
    if host == "scholar.google.com":
        user = (parse_qs(parsed.query).get("user") or [None])[0]
        return f"{host}/citations?user={user}" if user else ""
    if host == "semanticscholar.org" or host.endswith(".semanticscholar.org"):
        return f"semanticscholar.org/author/{segs[-1]}" if len(segs) >= 2 and segs[0] == "author" else ""
    if host == "orcid.org":
        return f"{host}/{segs[0]}" if segs and re.fullmatch(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", segs[0]) else ""
    if host == "dblp.org" or host.endswith(".dblp.org"):
        return "dblp.org/" + "/".join(segs).removesuffix(".html") if len(segs) >= 2 and segs[0] in {"pid", "pers"} else ""
    if host == "researchgate.net" or host.endswith(".researchgate.net"):
        return f"researchgate.net/profile/{segs[1].lower()}" if len(segs) >= 2 and segs[0] == "profile" else ""
    if host == "academia.edu" or host.endswith(".academia.edu"):
        if segs and not segs[0].isdigit():
            return f"{host}/{segs[0].lower()}"
        return host if host != "academia.edu" and not segs else ""
    if host == "arxiv.org" and len(segs) >= 2 and segs[0] == "a":
        return f"{host}/a/{segs[1]}"
    if host == "github.com" and len(segs) >= 2 and segs[0] == "orgs":
        return f"{host}/{segs[1].lower()}"

    # Substack / Medium subdomain forms: the subdomain IS the identity.
    if host.endswith(".substack.com") or host.endswith(".medium.com"):
        return host

    # Handle-on-path platforms: host + first segment.
    if host in _PATH_PLATFORMS and segs:
        return f"{host}/{segs[0].lower()}"

    # Default: the bare host is the identity (personal sites, blogs).
    return host
