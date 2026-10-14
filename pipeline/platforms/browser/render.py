"""
pipeline/platforms/browser/render.py — a page's finished HTML, for sites built by JavaScript.

A plain GET returns what the server sent. A JavaScript-built site (aleximas.com: 1,130 bytes, two
script tags, no links) sends an empty shell and builds the page in the browser. `rendered_html`
opens such a page in a headless Chrome on a throwaway, cookie-less profile and reads the DOM once
it settles. `is_js_shell` decides when that is worth doing; callers fetch plainly first.

It uses the user's installed Chromium-family browser. No browser means no render, and the caller
keeps the plain HTML.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from pipeline.ingestion.utils import log
from pipeline.platforms.browser import cdp, cookies

# How long a page may take to finish building, and how often its link count is sampled while it
# does. A site that loads its data after the load event (aleximas.com fetches `data.js`) is
# settled once two samples agree.
_LOAD_TIMEOUT_S = 20.0
_SETTLE_POLL_S = 0.5
_SETTLE_MAX_S = 6.0

_COUNT_LINKS = "document.querySelectorAll('a[href]').length"

# A shell has scripts and next to nothing a reader could see. danintheory.com, a plain site, has
# ~65 links and ~6 KB of text; aleximas.com's shell has 0 links and a page title. The bars sit far
# from both, so a thin-but-real page (a one-paragraph home with a few links) is not rendered.
_SHELL_MAX_LINKS = 2
_SHELL_MAX_TEXT = 300

_SCRIPT = re.compile(r"<script\b", re.I)
_LINK = re.compile(r"<a\b[^>]*\bhref\s*=", re.I)
_INVISIBLE = re.compile(r"<(script|style|noscript|template|head)\b.*?</\1\s*>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_OUTER_HTML = "document.documentElement.outerHTML"


def is_js_shell(html: str) -> bool:
    """Is this plain-fetched page an empty shell that only a browser would fill in?

    A pure decision on the HTML string. True sends the page to `rendered_html`; False keeps it.
    """
    if not html or not _SCRIPT.search(html):
        return False                                   # no scripts: what arrived is the page
    if len(_LINK.findall(html)) > _SHELL_MAX_LINKS:
        return False
    text = _TAG.sub(" ", _INVISIBLE.sub(" ", html))
    return len("".join(text.split())) <= _SHELL_MAX_TEXT


def filled(html: str | None, url: str) -> str | None:
    """The page as a reader sees it: `html` itself, or its render when it is a shell that a
    browser can fill. The one place every page fetch asks this, so the rule is written once."""
    if html is None or not is_js_shell(html):
        return html
    return rendered_html(url) or html


def _browser_binary() -> Path | None:
    """The user's first installed Chromium-family browser, or None when there is none."""
    for backend in cookies.installed_backends():
        if (path := backend.app_path()) is not None:
            return path
    return None


def _evaluate(page: cdp.Page, expression: str):
    out = page.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
    return (out.get("result") or {}).get("value")


def rendered_html(url: str) -> str | None:
    """`url`'s HTML after its scripts have run, or None when no browser is available or the
    render fails. Never raises: a failed render leaves the caller with its plain fetch."""
    binary = _browser_binary()
    if binary is None:
        return None
    try:
        with cdp.controlled_browser(binary) as browser:        # ephemeral profile, no cookies
            page = cdp.new_page(browser, "about:blank")
            page.call("Page.enable")                           # listen before navigating
            page.call("Runtime.enable")
            page.call("Page.navigate", {"url": url})
            page.wait_for_event("Page.loadEventFired", timeout=_LOAD_TIMEOUT_S)
            deadline, last = time.monotonic() + _SETTLE_MAX_S, -1
            while time.monotonic() < deadline:
                count = _evaluate(page, _COUNT_LINKS) or 0
                if count == last:
                    break
                last = count
                time.sleep(_SETTLE_POLL_S)
            return _evaluate(page, _OUTER_HTML)
    except Exception as exc:                                   # the plain fetch stands
        log(f"  [render] {url} could not be rendered: {type(exc).__name__}: {exc}")
        return None
