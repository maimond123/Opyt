"""The strings that identify one person inside a URL, so the blog-path admission gates can
recognize another host or path belonging to the same person.

The host token is the first host label (`karpathy.ai` → karpathy, `carol.substack.com` → carol),
not the one before the TLD, which would read `karpathy.github.io` as "github".
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from pipeline.platforms.url_canon import host_of

# A token must survive as a whole host label or path segment, so short ones match far too much:
# `mit` in meche.mit.edu, `ng` for Andrew Ng, the `j` in "Markus J. Buehler". Four characters is
# where a name stops colliding with routing.
_MIN_TOKEN = 4

# Host labels that name a site role rather than a person (`blog.example.com` would otherwise match
# `/blog/` everywhere). A stop-word list for tokens, not a host allow-list.
_GENERIC_LABELS = frozenset({"blog", "www", "web", "site", "home", "news", "docs", "mail",
                             "page", "pages", "index", "about"})

_WORD_RE = re.compile(r"[a-z]+")


def tokens_for(origin_url: str, display_name: str | None = None) -> frozenset[str]:
    """The identity tokens for the person whose site is `origin_url`.

    Two sources: the origin's first host label, and the words of the display name. The name arm
    covers an Oracle rooted at an institutional page (host "meche", name "Buehler").
    """
    out = set()
    label = (host_of(origin_url).split(".") or [""])[0]
    if len(label) >= _MIN_TOKEN and label not in _GENERIC_LABELS:
        out.add(label)
    for word in _WORD_RE.findall((display_name or "").lower()):
        if len(word) >= _MIN_TOKEN and word not in _GENERIC_LABELS:
            out.add(word)
    return frozenset(out)


def host_carries_token(url: str, tokens) -> bool:
    """True when the URL's host names one of `tokens` — the person has another home here.

    The only check that may promote a link to its own source; a path match must not
    (`cs.stanford.edu/people/karpathy/…` would register all of Stanford CS as his site).
    """
    return bool(tokens) and any(label in tokens for label in host_of(url).split("."))


def url_carries_token(url: str, tokens) -> bool:
    """True when the person's own work lives at this URL: their token is the host, or a path
    segment they own the area under.

    Matches whole segments, never substrings, and only a non-final segment (or a `~`/`@`-prefixed
    one): their own area has content beneath it, while an article about them ends on their name.
    """
    if host_carries_token(url, tokens):
        return True
    if not tokens:
        return False
    segs = [s for s in urlparse(
        url if "://" in (url or "") else "https://" + (url or "")).path.split("/") if s]
    return any(seg.lstrip("~@").lower() in tokens and (seg[0] in "~@" or i < len(segs) - 1)
               for i, seg in enumerate(segs))
