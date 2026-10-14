"""`canonical_identity(url)` collapses a URL to its platform-aware trust unit — the string the
trust graph treats as one node. Two URLs for the same account must canonicalize alike; two
accounts on the same host must not, so a squatter cannot inherit a real account's trust edges.

  - substack:           the subdomain               → someuser.substack.com
  - github/x/medium:    host + first path segment   → github.com/someuser
  - academic profiles:  the platform account ID (query or path)
  - everything else:    the bare host               → someuser.ai
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


def host_of(url: str) -> str:
    """The url's host, lowercased, without userinfo, port or a leading `www.` — `""` when it has
    none. A url with no scheme (`example.com/x`, `//example.com/x`) is read as one with."""
    url = url or ""
    if "://" not in url and not url.startswith("//"):
        url = "//" + url
    try:
        host = urlparse(url).hostname or ""
    except ValueError:                      # malformed IPv6 literal etc. — not a host we can name
        return ""
    return host.removeprefix("www.")


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
    """The excluded host this URL belongs to, or None.

    `parse_url` folds this into "unusable"; a caller choosing what to tell the user needs to tell
    "a platform we do not read" apart from "not a URL".
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

    # Substack / Medium subdomain forms: the subdomain is the identity.
    if host.endswith(".substack.com") or host.endswith(".medium.com"):
        return host

    # Handle-on-path platforms: host + first segment.
    if host in _PATH_PLATFORMS and segs:
        return f"{host}/{segs[0].lower()}"

    # Default: the bare host is the identity (personal sites, blogs).
    return host
