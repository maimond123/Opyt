"""Read a web page's name and one-line description: its title and meta description, the fields a
link card shows. One GET; a page built by JavaScript is rendered first (`browser/render.filled`).

No domain decisions here: which page to read, and what a thin answer means, are the caller's
(`kb/link_cards.py`).
"""
from __future__ import annotations

import html
from dataclasses import dataclass

import requests
import trafilatura

from pipeline.host_access import request as host_request
from pipeline.platforms.browser import render

# A rate limit or a server error may pass on its own; any other refusal is the page's answer.
_TRY_AGAIN = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class Head:
    title: str = ""
    description: str = ""


def _clean(text: str | None) -> str:
    return " ".join(html.unescape(text or "").split())


def read_head(url: str) -> Head | None:
    """The page's title and description. An empty `Head` when the page answered with nothing to
    say (a 404, a login wall, no tags); None when the request failed and may succeed later."""
    try:
        resp = host_request(requests.get, url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
    except Exception:
        return None
    if resp.status_code in _TRY_AGAIN:
        return None
    if resp.status_code != 200:
        return Head()
    page = render.filled(resp.text, str(getattr(resp, "url", "") or url))
    # A parser failure is a page with nothing readable, the same answer as no tags.
    try:
        meta = trafilatura.metadata.extract_metadata(page)
    except Exception:
        return Head()
    if meta is None:
        return Head()
    return Head(_clean(meta.title), _clean(meta.description))
