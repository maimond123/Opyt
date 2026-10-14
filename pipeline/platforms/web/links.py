"""Extract resolved links and their anchor text from one HTML page. Pure: the caller fetches the
page. Every domain is kept, because deciding which links belong to the person is the discovery
layer's job.
"""
from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import urljoin


class _LinkExtractor(HTMLParser):
    """Pull every ``<a href>`` off a page with its visible anchor text, resolving relative hrefs
    against the page URL. The anchor text (nested tags included) is the LLM URL-triage's main
    signal."""

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.links: list[dict] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = next((v for (k, v) in attrs if k == "href" and v), None)
        # Skip in-page anchors and non-navigational schemes (mailto/tel/js) up front.
        if not href or href.startswith("#") or href.startswith(("mailto:", "tel:", "javascript:")):
            return
        self._href = urljoin(self.base_url, href)
        self._text = []

    def handle_data(self, data):
        if self._href is not None and data.strip():
            self._text.append(data.strip())

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append({"url": self._href, "anchor": " ".join(self._text)[:200]})
            self._href = None
            self._text = []


def extract_links(html: str, base_url: str) -> list[dict]:
    """Every ``<a href>`` on a page as ``{url, anchor}``, de-duped on url (first anchor wins). A
    malformed-HTML parse error returns whatever was parsed rather than sinking discovery."""
    lx = _LinkExtractor(base_url)
    try:
        lx.feed(html or "")
    except Exception:
        pass
    seen: set[str] = set()
    out: list[dict] = []
    for link in lx.links:
        if link["url"] in seen:
            continue
        seen.add(link["url"])
        out.append(link)
    return out
