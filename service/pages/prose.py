"""
service/pages/prose.py — a source's markdown as HTML that is safe to put on a page and shaped
for reading.

Content is scraped from the web, so every body passes through `_CLEAN` before the page adds its own
markup. Nothing a source wrote reaches the page as a script, a style or an attribute outside
`_SAFE_ATTRS`; `render.CSP` tells the browser the same.
"""
from __future__ import annotations

import html
from urllib.parse import urljoin, urlparse

from lxml import html as lhtml
from lxml_html_clean import Cleaner
from markdown_it import MarkdownIt

_MD = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
_SAFE_ATTRS = frozenset({"href", "src", "alt", "title", "colspan", "rowspan", "start", "datetime",
                         "lang", "dir"})
_CLEAN = Cleaner(scripts=True, javascript=True, comments=True, style=True, inline_style=True,
                 meta=True, page_structure=True, embedded=True, frames=True, forms=True,
                 annoying_tags=True, safe_attrs_only=True, safe_attrs=_SAFE_ATTRS)

_BADGE_HOSTS = ("img.shields.io", "badge.fury.io", "badgen.net")
_PROMO = {"subscribe now", "subscribe", "share", "leave a comment", "upgrade to paid"}


def to_html(md: str, base: str | None = None, link_base: str | None = None,
            anchors: dict[str, str] | None = None) -> tuple[str, int]:
    """`(html, words)`: the markdown rendered, cleaned of anything active, then shaped for reading
    (media grids, video, link cards, quoted posts, figures). A link whose `url_key` is in
    `anchors` goes to that anchor on the page instead."""
    root = lhtml.fragment_fromstring(_MD.render(md), create_parent="div")
    _CLEAN(root)
    _shape(root, base, link_base, anchors or {})
    words = len(root.text_content().split())
    out = html.escape(root.text or "") + "".join(lhtml.tostring(c, encoding="unicode")
                                                  for c in root)
    return f'<div class="prose">{out}</div>', words


def url_key(url: str) -> str:
    """A link as an answer and the page's sources both write it, give or take a final slash."""
    return url.strip().rstrip("/").lower()


def _shape(root, base: str | None, link_base: str | None, anchors: dict[str, str]) -> None:
    for img in root.iter("img"):
        src = img.get("src") or ""
        if any(h in src for h in _BADGE_HOSTS):
            _drop(img)
            continue
        if base and src and not urlparse(src).scheme:
            img.set("src", urljoin(base, src))
        img.set("loading", "lazy")
        img.set("decoding", "async")
        img.set("referrerpolicy", "no-referrer")
    for a in list(root.iter("a")):
        href = a.get("href") or ""
        if url_key(href) in anchors:
            a.set("href", anchors[url_key(href)])
            continue
        if link_base and href and not urlparse(href).scheme and not href.startswith("#"):
            a.set("href", urljoin(link_base, href))
        a.set("target", "_blank")
        a.set("rel", "noopener noreferrer")
        if a.text_content().strip().lower() in _PROMO and len(a) == 0:
            parent = a.getparent()
            if parent is not None and parent.tag == "p" and \
                    parent.text_content().strip() == a.text_content().strip():
                _drop(parent)
    for h in list(root.iter("h2", "h3")):
        label = h.text_content().strip()
        if label == "Media":
            _media(h)
        elif label == "Links":
            h.set("class", "label")
    for bq in list(root.iter("blockquote")):
        _linkcard(bq)
    for p in list(root.iter("p")):
        _quoting(p)
        imgs = p.findall(".//img")
        if len(imgs) == 1 and not (p.text_content() or "").strip():
            p.set("class", "fig")


def _media(heading) -> None:
    """The `## Media` block of a post: its photos and videos as one grid, the heading gone."""
    parent = heading.getparent()
    nodes, sib = [], heading.getnext()
    while sib is not None and sib.tag == "p" and not sib.text_content().strip() \
            and sib.find(".//img") is not None:
        nodes.append(sib)
        sib = sib.getnext()
    tiles = []
    for p in nodes:
        for img in p.iter("img"):
            link = img.getparent() if img.getparent().tag == "a" else None
            href = link.get("href") if link is not None else ""
            if img.get("alt") == "video" and href and ".mp4" in href:
                v = lhtml.Element("video", controls="", preload="none", playsinline="",
                                  poster=img.get("src"), src=href)
                tiles.append(v)
            else:
                tiles.append(img)
    if tiles:
        grid = lhtml.Element("div", {"class": f"media n{min(len(tiles), 4)}"})
        for t in tiles:
            t.tail = None
            grid.append(t)
        heading.addprevious(grid)
    for p in nodes:
        parent.remove(p)
    _drop(heading)


def _linkcard(bq) -> None:
    """Stored link cards (`> **[title](url)**`, then a description line) as cards you can click.

    A snapshot writes several cards into one quote, and the same page twice when a post linked
    it with and without a trailing slash; each becomes one card, the first copy kept."""
    cards: list[list[str]] = []
    for p in bq:
        if p.tag != "p" or (p.text or "").strip():
            return
        for child in p:
            a = child.find("a") if child.tag == "strong" else None
            if a is not None and child.text_content().strip() == a.text_content().strip():
                cards.append([a.get("href") or "", a.text_content().strip(), child.tail or ""])
            elif cards:
                cards[-1][2] += child.text_content() + (child.tail or "")
            else:
                return
    if not cards:
        return
    seen: set[str] = set()
    for href, title, desc in cards:
        key = href.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        card = lhtml.Element("a", {"class": "linkcard", "href": href, "target": "_blank",
                                   "rel": "noopener noreferrer"})
        for cls, text in (("lc-host", host(href)), ("lc-title", title),
                          ("lc-desc", " ".join(desc.split()))):
            if text:
                span = lhtml.Element("span", {"class": cls})
                span.text = text
                card.append(span)
        bq.addprevious(card)
    _drop(bq)


def _quoting(p) -> None:
    """`**Quoting** [@who](url):` and the blockquote after it, as one inset quoted post."""
    strong = p.find("strong")
    if strong is None or strong.text_content().strip() != "Quoting":
        return
    nxt = p.getnext()
    box = lhtml.Element("div", {"class": "quoted"})
    p.addprevious(box)
    p.set("class", "q-by")
    box.append(p)
    if nxt is not None and nxt.tag == "blockquote":
        for child in list(nxt):
            box.append(child)
        nxt.getparent().remove(nxt)


def _drop(el) -> None:
    """Remove `el` but keep the text that followed it."""
    parent = el.getparent()
    if parent is None:
        return
    prev = el.getprevious()
    if el.tail:
        if prev is not None:
            prev.tail = (prev.tail or "") + el.tail
        else:
            parent.text = (parent.text or "") + el.tail
    parent.remove(el)


def host(url: str) -> str:
    """A URL's site, without a leading `www.`."""
    name = urlparse(url or "").netloc.lower()
    return name[4:] if name.startswith("www.") else name
