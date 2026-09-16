"""
pipeline/ingestion/identity_tokens.py — the strings that identify ONE person inside a URL.

Both admission gates on the blog path ask the same question — "is this the person's own thing?"
— and both used to answer it with a HOST rule. `_probe_blog_profile` kept an outbound link as a
declared identity only if it landed on one of three named platforms; `link_discovery._is_owned`
kept a hub link only if its host was the origin or a subdomain of it. Neither rule can express
"another host belonging to the same person", which is where a writer's actual corpus lives.

Measured on karpathy.ai, 2026-09-09, over 55 of his own links and 35 that are not his:

    same host / subdomain (the old rule)        3 / 55 kept,  0 false positives
    token in the host OR a whole path segment  48 / 55 kept,  1 false positive

The one false positive is `wired.com/2015/01/karpathy/` — a journalist's article about him. That
is the same trade the platform arm already accepts by ruling (`_BLOG_IDENTITY_TYPES`, David,
2026-09-05: "the occasional mention of another person is an accepted tradeoff").

WHY THE FIRST HOST LABEL AND NOT THE ONE BEFORE THE TLD. `_openalex_root` takes the label before
the TLD because it is asking who PUBLISHES a venue. This asks who a site BELONGS to, and that is
the first label: `karpathy.ai` → karpathy, `carol.substack.com` → carol, `meche.mit.edu` → meche.
The other rule would read `karpathy.github.io` as "github" and hand every GitHub URL on the page
a token match.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

# A token must survive as a whole host label or path segment, so short ones match far too much:
# `mit` in meche.mit.edu, `ng` for Andrew Ng, the `j` in "Markus J. Buehler". Four characters is
# where a name stops colliding with routing.
_MIN_TOKEN = 4

# Host labels that name a SITE ROLE rather than a person. Without these, `blog.example.com` yields
# the token "blog" and then matches `/blog/` on every other site on the page. This is a stop-word
# list for one derived string, not a host allow-list — it never decides whether a URL is admitted,
# only whether a word can stand for a person.
_GENERIC_LABELS = frozenset({"blog", "www", "web", "site", "home", "news", "docs", "mail",
                             "page", "pages", "index", "about"})

_WORD_RE = re.compile(r"[a-z]+")


def _host(url: str) -> str:
    u = url if "://" in (url or "") else "https://" + (url or "")
    h = (urlparse(u).netloc or "").lower().split("@")[-1].split(":")[0]
    return h[4:] if h.startswith("www.") else h


def tokens_for(origin_url: str, display_name: str | None = None) -> frozenset[str]:
    """The identity tokens for the person whose site is `origin_url`.

    Two sources, both already in hand at every call site: the origin's first host label, and the
    words of the display name (`<title>` at the blog probe, `author_name` at link discovery). The
    name arm carries the case the host arm cannot — an Oracle rooted at an institutional page,
    like the `blog:meche.mit.edu` in the live store, whose host says "meche" and whose name says
    "Buehler".
    """
    out = set()
    label = (_host(origin_url).split(".") or [""])[0]
    if len(label) >= _MIN_TOKEN and label not in _GENERIC_LABELS:
        out.add(label)
    for word in _WORD_RE.findall((display_name or "").lower()):
        if len(word) >= _MIN_TOKEN and word not in _GENERIC_LABELS:
            out.add(word)
    return frozenset(out)


def host_carries_token(url: str, tokens) -> bool:
    """True when the URL's HOST names one of `tokens` — the person has another HOME here.

    The stronger of the two checks, and the one that may promote a link to its own SOURCE:
    `karpathy.github.io` is his whole blog, so registering it costs one source and its feed
    yields the posts. A PATH match must never promote — `cs.stanford.edu/people/karpathy/…`
    would register all of Stanford CS as his site.
    """
    return bool(tokens) and any(label in tokens for label in _host(url).split("."))


def url_carries_token(url: str, tokens) -> bool:
    """True when the person's own work lives at this URL: their token is the host, or it names a
    path segment they OWN THE AREA UNDER.

    Two rules, and the second is what separates their work from writing about them.

    WHOLE SEGMENTS, never a substring. `karpathy` as a substring also matches
    `techcrunch.com/2017/…/tesla-hires-andrej-karpathy-to-lead-ai` and two more press pieces — 6
    false positives measured on karpathy.ai, versus 1 for whole-segment matching.

    A PREFIX, not the final slug. A person's own area has their content beneath it
    (`cs.stanford.edu/people/karpathy/advice.html`); an article ABOUT them ends on their name
    (`wired.com/2015/01/karpathy/`, `theblock.co/post/gajesh`). Requiring the token to be
    non-final — or `~`/`@`-prefixed, the two shapes that spell "this person's area" outright —
    took the measured false positives from 1 to 0 and cost nothing: the only links it drops are
    `github.com/karpathy` and `twitter.com/karpathy`, which are PROFILES that
    `_probe_blog_profile`'s platform arm already admits, not content this function should reach.
    """
    if host_carries_token(url, tokens):
        return True
    if not tokens:
        return False
    segs = [s for s in urlparse(
        url if "://" in (url or "") else "https://" + (url or "")).path.split("/") if s]
    return any(seg.lstrip("~@").lower() in tokens and (seg[0] in "~@" or i < len(segs) - 1)
               for i, seg in enumerate(segs))
