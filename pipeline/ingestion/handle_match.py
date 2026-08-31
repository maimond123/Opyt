"""
pipeline/ingestion/handle_match.py

Ownership-by-namesake for hub expansion. A profile surfaced on the Oracle's own
X-vouched, self-curated page (their blog / Substack) is almost certainly theirs
if its handle matches a handle the Oracle already owns — a near-namesake
coincidence *on their own page* is negligible. This module decides "does this
handle match the Oracle?" crisply enough to auto-trust on.

"Tight fuzzy" is deliberately narrow, so it can only ever *promote*:
  * normalized-equality always matches (any length that clears the generic gate);
  * edit-distance ≤ 1 matches ONLY when the candidate is ≥ 6 chars — a 1-edit
    window on a 5-char handle is proportionally too loose;
  * a generic / short candidate (``blog``, ``admin``, < 5 chars) never matches,
    so a squatter can't graduate on a throwaway namesake.

Uses edit-distance ≤ 1 rather than ``difflib``'s similarity ratio (used at classify.py:299)
since it needs no magic cutoff. Pure stdlib.
"""

from __future__ import annotations

import re

# Non-identifying tokens that must never auto-trust, even at length ≥ 5. These are
# page/section/platform words, not people.
GENERIC_HANDLES = frozenset({
    "blog", "home", "about", "index", "admin", "user", "users", "news", "team",
    "info", "contact", "page", "pages", "site", "sites", "www", "docs", "app",
    "apps", "api", "me", "posts", "post", "feed", "feeds", "rss", "official",
    "support", "help", "mail", "email", "dev", "test", "demo", "hello",
    "account", "accounts", "profile", "profiles", "login", "signup", "shop",
    "store", "podcast", "podcasts", "newsletter", "subscribe", "author",
    "authors", "staff", "editor", "press", "media", "careers", "jobs", "legal",
    "privacy", "terms", "status", "assets", "static", "images", "download",
    "search", "explore", "settings", "channel", "watch", "public", "main",
})

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def normalize_handle(h: str | None) -> str:
    """Lowercase, strip a leading '@', drop everything non-[a-z0-9].

    ``"@Gergely_Orosz"`` → ``"gergelyorosz"``. Empty / None → ``""``.
    """
    if not h:
        return ""
    s = str(h).strip().lower()
    if s.startswith("@"):
        s = s[1:]
    return _NON_ALNUM.sub("", s)


def is_generic(h: str | None, *, min_len: int = 5) -> bool:
    """A handle too short or too generic to safely auto-trust on."""
    n = normalize_handle(h)
    return len(n) < min_len or n in GENERIC_HANDLES


def _edit_distance_le1(a: str, b: str) -> bool:
    """True iff Levenshtein(a, b) ≤ 1 — equal, one substitution, or one indel.

    O(n), no dependency. The exact, testable core of "tight" fuzzy matching.
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        # equal or a single substitution
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    # lengths differ by exactly 1 — allow a single insertion/deletion.
    if la > lb:
        a, b, la, lb = b, a, lb, la          # a is now the shorter
    i = j = 0
    skipped = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            if skipped:
                return False
            skipped = True
            j += 1                            # consume one extra char in the longer
    return True


def match_known_handle(candidate: str | None, known: set[str]) -> str | None:
    """Return the *original* known handle the candidate matches, or None.

    Exposing which handle matched lets callers build an auditable reason string
    ("Handle 'willccbb' matches the Oracle's 'willccbb'") instead of a bare bool.
    """
    c = normalize_handle(candidate)
    if not c or is_generic(c):
        return None
    norm_known = [(normalize_handle(k), k) for k in known]

    # 1. Normalized equality — the strong signal, any non-generic length.
    for nk, orig in norm_known:
        if nk and nk == c:
            return orig

    # 2. Tight fuzzy — one edit, but only for ≥ 6-char candidates.
    if len(c) >= 6:
        for nk, orig in norm_known:
            if nk and _edit_distance_le1(c, nk):
                return orig
    return None


