"""
pipeline/robots.py — may Opyt fetch this URL, by its site's robots.txt (RFC 9309).

`check` is the question every web read asks, from the one transport they share
(`host_access.request`). It reads each origin's robots.txt once a day and raises `Refused` when
the rules disallow the URL for the fetcher's own User-Agent or for `*`; a disallow in either
group refuses. A robots.txt that answers 5xx, 429 or nothing at all refuses too (RFC 9309 reads
an unreachable robots.txt as a complete disallow), and is asked again on the next read. Any other
4xx means the site has no robots.txt: everything is allowed.

`Refused` is a `requests.RequestException`, so every caller already skips it the way it skips a
failed fetch: no write, nothing marked done, a retry on the caller's own schedule.

Documented APIs are not crawled pages, and robots.txt does not govern them (`governs`): several
of the hosts below disallow every path to crawlers while documenting the API for exactly this use.

Deciding: `governs`, `agents`, `parse`, `verdict`, `allows`. Doing: `rules_for`, `check`.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

import requests

from pipeline.ingestion.utils import log

# Documented APIs, as (host, path prefix), called as their documentation describes. Measured:
# export.arxiv.org, pub.orcid.org and eutils.ncbi.nlm.nih.gov disallow `/` for `*`,
# and zenodo.org disallows `/api`.
_APIS = (
    ("api.github.com", "/"),
    ("api.openalex.org", "/"),
    ("api.semanticscholar.org", "/"),
    ("export.arxiv.org", "/api/"),
    ("eutils.ncbi.nlm.nih.gov", "/entrez/eutils/"),
    ("www.ebi.ac.uk", "/europepmc/webservices/"),
    ("zenodo.org", "/api/"),
    ("pub.orcid.org", "/"),
)
# Substack's JSON API sits on every publication's own host, custom domains included.
_SUBSTACK_API = "/api/v1/"

_TTL_S = 24 * 3600
_TIMEOUT_S = 15
_PRODUCT = re.compile(r"([A-Za-z][\w.-]*)/\d")


class Refused(requests.RequestException):
    """robots.txt disallows the URL, or could not be read."""


@dataclass(frozen=True)
class Rule:
    allow: bool
    length: int                  # the pattern's length: the longest matching rule wins
    pattern: re.Pattern


@dataclass(frozen=True)
class Rules:
    """One robots.txt: each user-agent's rules (lowercased product token → rules), its groups
    combined as RFC 9309 says. No entry for an agent means no rule for it."""
    by_agent: dict[str, tuple[Rule, ...]] = field(default_factory=dict)


ALLOW_ALL = Rules()


def governs(url: str) -> bool:
    """Is `url` a page robots.txt speaks for, rather than a documented API?"""
    parts = urlsplit(url)
    host, path = (parts.hostname or "").lower(), parts.path or "/"
    if path.startswith(_SUBSTACK_API):
        return False
    return not any(host == h and path.startswith(p) for h, p in _APIS)


def agents(user_agent: str) -> tuple[str, ...]:
    """The product tokens a User-Agent names, lowercased:
    `Mozilla/5.0 (compatible; OPYT/1.0)` → `("mozilla", "opyt")`."""
    return tuple(dict.fromkeys(t.lower() for t in _PRODUCT.findall(user_agent or "")))


def _compile(pattern: str) -> re.Pattern:
    """A path pattern as a regex anchored at the path's start: `*` is any run of characters and a
    final `$` ends the path."""
    anchored = pattern.endswith("$")
    body = unquote(pattern[:-1] if anchored else pattern)
    return re.compile("".join(".*" if ch == "*" else re.escape(ch) for ch in body)
                      + (r"\Z" if anchored else ""))


def parse(text: str) -> Rules:
    """robots.txt text → its rules. A run of `user-agent` lines opens a group and the
    `allow`/`disallow` lines after it are that group's; other records are ignored. An empty
    `disallow` adds no rule."""
    by_agent: dict[str, list[Rule]] = {}
    group: list[str] = []
    in_rules = False
    for raw in (text or "").splitlines():
        key, sep, value = raw.split("#", 1)[0].partition(":")
        if not sep:
            continue
        key, value = key.strip().lower(), value.strip()
        if key == "user-agent":
            if in_rules:
                group, in_rules = [], False
            group.append(value.split("/", 1)[0].strip().lower())
        elif key in ("allow", "disallow") and group:
            in_rules = True
            if value:
                rule = Rule(key == "allow", len(value), _compile(value))
                for agent in group:
                    by_agent.setdefault(agent, []).append(rule)
    return Rules({agent: tuple(rules) for agent, rules in by_agent.items()})


def verdict(status: int | None, text: str) -> Rules | None:
    """A robots.txt response → its rules; None when it could not be read and the origin is closed
    for now. RFC 9309: a 4xx is no robots.txt; a 5xx or no answer is unreachable. A 429 is read
    as unreachable too, as Google does: it says try later, not that no rules exist."""
    if status is None or status >= 500 or status == 429:
        return None
    if 200 <= status < 300:
        return parse(text)
    return ALLOW_ALL


def _allowed(rules: tuple[Rule, ...], path: str) -> bool:
    best, allow = -1, True
    for rule in rules:
        if rule.pattern.match(path) and (rule.length > best or (rule.length == best and rule.allow)):
            best, allow = rule.length, rule.allow
    return allow


def allows(rules: Rules, url: str, user_agents: tuple[str, ...]) -> bool:
    """May a fetcher naming `user_agents` read `url` under `rules`? Every named agent's group and
    the `*` group must allow it; within a group the longest matching rule wins, and `allow` wins a
    tie. `/robots.txt` itself is always allowed."""
    parts = urlsplit(url)
    path = unquote(parts.path or "/") + (f"?{unquote(parts.query)}" if parts.query else "")
    if path == "/robots.txt":
        return True
    return all(_allowed(rules.by_agent.get(agent, ()), path) for agent in (*user_agents, "*"))


# ── doing ─────────────────────────────────────────────────────────────────────────────────────

_CACHE: dict[str, tuple[float, Rules]] = {}


def _read(origin: str, user_agent: str) -> Rules | None:
    """Fetch `origin`'s robots.txt as `user_agent` → its rules, or None when unreachable."""
    try:
        resp = requests.get(f"{origin}/robots.txt", headers={"User-Agent": user_agent},
                            timeout=_TIMEOUT_S)
    except requests.RequestException:
        return None
    return verdict(resp.status_code, resp.text)


def rules_for(origin: str, user_agent: str, now: float) -> Rules | None:
    """`origin`'s rules, read at most once a day. An unreachable answer is not kept: the next
    read asks again."""
    hit = _CACHE.get(origin)
    if hit is not None and now - hit[0] < _TTL_S:
        return hit[1]
    rules = _read(origin, user_agent)
    if rules is not None:
        _CACHE[origin] = (now, rules)
    return rules


def check(url: str, user_agent: str) -> None:
    """Raise `Refused` unless robots.txt lets `user_agent` fetch `url`."""
    if not governs(url):
        return
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    rules = rules_for(origin, user_agent, time.monotonic())
    if rules is None:
        log(f"  [robots] {origin}/robots.txt unreachable; {url} waits for a later read")
        raise Refused(f"robots.txt unreachable for {origin}")
    if not allows(rules, url, agents(user_agent)):
        log(f"  [robots] disallowed for {user_agent!r}: {url}")
        raise Refused(f"robots.txt disallows {url}")
