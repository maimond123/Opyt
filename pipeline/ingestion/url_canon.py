"""
pipeline/ingestion/url_canon.py

`canonical_identity(url)` collapses a URL to its platform-aware *trust unit* —
the string the trust graph treats as one node. Two URLs that are "the same
account" must canonicalize to the same identity; two different accounts on the
same host must not.

The unit is platform-specific because where "identity" lives differs:
  - substack:           the SUBDOMAIN owns identity   → someuser.substack.com
  - github/x/medium:    host + first path SEGMENT      → github.com/someuser
  - youtube:            the channel handle/id          → youtube.com/@andrejsomeuser
  - everything else:    the bare host                  → someuser.ai

This is the squatter-defense primitive: `github.com/someuser` and
`github.com/someoneelse` are distinct nodes, so a squatter cannot inherit a
real account's inbound trust edges just by sharing a host.

Derived from `discover_profile._normalize_url()`; this goes further (drops the
scheme, lowercases the host, aliases twitter↔x, extracts the path unit).
"""

from __future__ import annotations

from urllib.parse import urlparse

# Hosts where the first path segment is the identity (the account handle).
_PATH_PLATFORMS = {"github.com", "x.com", "medium.com", "gitlab.com"}

# Hosts treated as YouTube; channel identity lives in the path.
_YOUTUBE_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}


def canonical_identity(url: str) -> str:
    """Return the platform-aware trust unit for `url` (no scheme, lowercased host).

    Returns "" for falsy/unparseable input so callers can filter cheaply.
    """
    if not url or not url.strip():
        return ""

    u = url.strip()
    if "://" not in u:
        u = "https://" + u  # urlparse needs a scheme to populate netloc

    parsed = urlparse(u)
    host = (parsed.netloc or "").lower()
    if not host:
        return ""
    host = host.split("@")[-1]   # drop any user:pass@ prefix
    host = host.split(":")[0]    # drop :port
    if host.startswith("www."):
        host = host[4:]

    # twitter.com and x.com are the same platform — alias to x.com.
    if host == "twitter.com" or host == "mobile.twitter.com":
        host = "x.com"

    segs = [s for s in parsed.path.split("/") if s]

    # Substack / Medium subdomain forms: the subdomain IS the identity.
    if host.endswith(".substack.com") or host.endswith(".medium.com"):
        return host

    # YouTube: identity is the channel handle (@x), or /channel|c|user/<id>.
    if host in _YOUTUBE_HOSTS:
        if segs:
            first = segs[0].lower()
            if first.startswith("@"):
                return f"youtube.com/{first}"
            if first in ("channel", "c", "user") and len(segs) >= 2:
                return f"youtube.com/{first}/{segs[1].lower()}"
        return "youtube.com"

    # Handle-on-path platforms: host + first segment.
    if host in _PATH_PLATFORMS and segs:
        return f"{host}/{segs[0].lower()}"

    # Default: the bare host is the identity (personal sites, blogs).
    return host


def resolve_channel_url(url: str) -> str | None:
    """Resolve any YouTube URL to its canonical channel URL.

    Discovery surfaces watch/playlist URLs (a video the person linked), but a YouTube
    IDENTITY is the channel — so `discover_profile` resolves before trust runs, or two
    links to the same creator's videos read as two different sources. Short-circuits with
    NO network when the URL is already channel-shaped (/@handle, /channel, /c, /user).
    Returns None when it cannot resolve; the caller keeps the original URL (fail-safe:
    a failed lookup must not lose the link it was handed).
    """
    cid = canonical_identity(url)
    if cid and cid != "youtube.com":
        return url  # already channel-specific — no fetch needed

    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "extract_flat": "in_playlist", "playlistend": 1}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return None
    ch = info.get("channel_url") or info.get("uploader_url")
    if not ch:
        entries = info.get("entries") or []
        if entries:
            ch = entries[0].get("channel_url") or entries[0].get("uploader_url")
    return ch
