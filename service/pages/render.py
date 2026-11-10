"""
service/pages/render.py — a page of sources as one HTML document: the items a conversation used,
each drawn from its stored snapshot in one shared layout.

Pure: items, title, answer and dates in, HTML out. This file draws what is around the items (the
bar, the title, the answer, the contents, the colophon); `cards.py` draws each item's card.

Content is scraped from the web, so every body is cleaned in `prose.py` before the page adds its
own markup, and `CSP` tells the browser the same: the page's own three scripts are the only ones
it runs.

The third script is the page's comments (`comments.js`, styled by `comments.css`). It draws
everything about them in the browser, from what `comments.py` answers, so nothing here knows a
comment.
"""
from __future__ import annotations

import base64
import hashlib
import html
from datetime import date
from pathlib import Path

from service.pages import cards, prose

_CSS = (Path(__file__).parent / "page.css").read_text()
_COMMENTS_CSS = (Path(__file__).parent / "comments.css").read_text()
_COMMENTS_SCRIPT = (Path(__file__).parent / "comments.js").read_text()
_SCRIPT = (Path(__file__).parent / "page.js").read_text()

_PLURALS = {"x": ("X post", "X posts"), "substack": ("Substack post", "Substack posts"),
            "blog": ("blog post", "blog posts"), "paper": ("paper", "papers"),
            "github": ("GitHub repo", "GitHub repos")}

_TILE = ('<svg class="tile" viewBox="0 0 64 64" aria-hidden="true"><rect class="tile-bg" '
         'width="64" height="64"/><path class="tile-glyph" d="M18 16 L36 32 L18 48" '
         'stroke-width="7" fill="none"/><rect class="tile-cursor" x="40" y="41" width="11" '
         'height="7"/></svg>')


# The reading faces a reader can pick: (key, name, CSS stack, Google Fonts family), the default
# first. Wallace et al. (ACM TOCHI 2022, 16 fonts) found no face fastest for everyone, so the
# choice is the reader's. EB Garamond had the fastest average and is the default;
# Franklin Gothic won the most head-to-head pairings, Montserrat close behind; Noto Sans was the
# most preferred.
_FONTS = [
    ("garamond", "Garamond", '"EB Garamond", Garamond, Georgia, serif',
     "EB+Garamond:ital,wght@0,400..700;1,400..700"),
    ("franklin", "Franklin", '"Libre Franklin", "Franklin Gothic", "Helvetica Neue", Arial, sans-serif',
     "Libre+Franklin:ital,wght@0,400..700;1,400..700"),
    ("noto", "Noto Sans", '"Noto Sans", "Helvetica Neue", Arial, sans-serif',
     "Noto+Sans:ital,wght@0,400..700;1,400..700"),
    ("montserrat", "Montserrat", '"Montserrat", "Helvetica Neue", Arial, sans-serif',
     "Montserrat:ital,wght@0,400..700;1,400..700"),
]
_BRAND_FONTS = "Martian+Mono:wght@600..800&family=IBM+Plex+Mono:wght@400;500;600"


def _font_href(family: str, brand: bool = False) -> str:
    extra = f"&family={_BRAND_FONTS}" if brand else ""
    return f"https://fonts.googleapis.com/css2?family={family}{extra}&display=swap"


def _font_css() -> str:
    """`--read` for each face but the default, which `page.css` sets."""
    return "".join(f':root[data-font="{k}"]{{--read:{stack}}}' for k, _, stack, _ in _FONTS[1:])


def _font_map() -> str:
    """The non-default faces' stylesheets, keyed by name, for the script that loads a stored pick."""
    return "{" + ",".join(f'"{k}":"{_font_href(fam)}"' for k, _, _, fam in _FONTS[1:]) + "}"


def _font_menu() -> str:
    """One item per face, each named in its own face."""
    return "".join(
        f'<button type="button" role="menuitemradio" aria-checked="false" data-font="{k}" '
        f'data-href="{"" if n == 0 else html.escape(_font_href(fam))}" '
        f'style="font-family:{html.escape(stack)}">{name}'
        f'{"<span class=hint>default</span>" if n == 0 else ""}</button>'
        for n, (k, name, stack, fam) in enumerate(_FONTS))


# Before the page draws: the reader's stored theme and font, so the page never flashes the default.
_HEAD_SCRIPT = ('try{var r=document.documentElement,t=localStorage.getItem("opyt-theme"),'
                'f=localStorage.getItem("opyt-font"),F=' + _font_map() + ';'
                'if(t==="dark"||t==="light")r.setAttribute("data-theme",t);'
                'if(f&&F[f]){r.setAttribute("data-font",f);var l=document.createElement("link");'
                'l.rel="stylesheet";l.href=F[f];document.head.appendChild(l)}}catch(e){}')


def page(items: list[dict], *, title: str, note: str | None, answer: str | None, made: date,
         days: int) -> str:
    """The whole page: the bar, the title, the answer, one card per item, the contents beside
    them, the colophon. `days` is how long a page is kept after it was last opened."""
    drawn = [cards.card(i, n, len(items)) for n, i in enumerate(items, 1)]
    body = "".join(c["html"] for c in drawn)
    note_html = f'<p class="note">{html.escape(note)}</p>' if note else ""
    answer_html = _answer(answer, items) if answer else ""
    return f"""{_head(title, _COMMENTS_CSS)}
<body>
<nav class="bar">
  {_BRAND}
  <div class="end">
    <span class="expires">Kept {days} days after last opened</span>
    <div class="fontpick">
      <button class="theme-btn" id="font-btn" type="button" aria-haspopup="true" aria-expanded="false" aria-controls="font-menu"><span class="aa" aria-hidden="true">Aa</span><span class="lbl">Font</span></button>
      <div class="menu" id="font-menu" role="menu" hidden>
        <p class="menu-note">Different fonts read faster for different people. Pick yours.</p>
        {_font_menu()}
      </div>
    </div>
    <button class="theme-btn" id="theme-btn" type="button" aria-label="Switch color theme"><svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M8 1 A7 7 0 0 1 8 15 Z" fill="currentColor"/></svg><span class="lbl" id="theme-name">dark</span></button>
    <button class="share" id="share-btn" type="button" data-days="{days}"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 10V1.8M4.9 4.6 8 1.5l3.1 3.1M3 8v6h10V8" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>Share</button>
  </div>
</nav>
<div class="toast" id="toast" role="status" aria-live="polite"></div>
<main>
  <header class="hero">
    <p class="eyebrow">{len(items)} sources · {cards.day(made)}</p>
    <h1>{html.escape(title.strip())}</h1>
    {note_html}
    <p class="meta">{html.escape(_census(items))}</p>
  </header>
  {answer_html}
  {body}
</main>
{_contents([c["nav"] for c in drawn])}
<footer class="colophon">
  <p>Gathered by <a href="https://useopyt.com" target="_blank" rel="noopener">Opyt</a> from the sources your assistant used. Every item links back to where it was published.<br>This page is private to whoever has its link. It is deleted {days} days after it was last opened.</p>
  {_WORDMARK}
</footer>
<script>{_SCRIPT}</script>
<script>{_COMMENTS_SCRIPT}</script>
</body>
</html>
"""


def _answer(md: str, items: list[dict]) -> str:
    """The host's answer as a card above the sources. A link to a source on the page goes to its
    card below, so each citation opens the source whole, here. It is item 0 to the comments, so
    its text can be commented on like a source's."""
    on_page = {prose.url_key(i["source_url"]): f"#s{n}" for n, i in enumerate(items, 1)
               if i.get("source_url")}
    body, _ = prose.to_html(md, anchors=on_page)
    return f"""
  <section class="answer" id="s0" aria-label="Answer">
    <div class="head">{_ANSWER_GLYPH}<span>Answer</span></div>
    <div class="body">{body}</div>
  </section>"""


_ANSWER_GLYPH = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 4h18v2.2H3V4Zm0 '
                 '5h18v2.2H3V9Zm0 5h18v2.2H3V14Zm0 5h11v2.2H3V19Z"/></svg>')


def gone(days: int) -> str:
    """What a link shows once its page has expired or been taken down, and what an unknown link
    shows: the two are not told apart. `days` is how long a page is kept after it was last
    opened."""
    return f"""{_head("Page expired")}
<body>
<nav class="bar">
  {_BRAND}
</nav>
<main>
  <header class="hero">
    <h1>This page has expired</h1>
    <p class="note">A reader view is deleted {days} days after it was last opened, and whoever made it can take it down sooner. Ask them for a new link.</p>
  </header>
</main>
<footer class="colophon">
  <p>Reader views like this are made by <a href="https://useopyt.com" target="_blank" rel="noopener">Opyt</a> from the sources an assistant used.</p>
  {_WORDMARK}
</footer>
</body>
</html>
"""


_BRAND = (f'<a class="brand" href="https://useopyt.com" target="_blank" rel="noopener">{_TILE}'
          f'<span class="name">Opyt</span></a>')
_WORDMARK = ('<div class="wordmark" aria-hidden="true"><svg viewBox="14 70 1147 320" '
             'preserveAspectRatio="xMidYMax meet"><text class="wm-text" x="0" y="317" '
             'font-size="312" vector-effect="non-scaling-stroke">opyt</text><g class="tile '
             'wm-tile" transform="translate(916 72) scale(3.828)"><rect class="tile-bg" '
             'width="64" height="64"/><path class="tile-glyph" d="M18 16 L36 32 L18 48" '
             'stroke-width="7" fill="none"/><rect class="tile-cursor" x="40" y="41" width="11" '
             'height="7"/></g></svg></div>')


def _head(title: str, more_css: str = "") -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>{html.escape(title)} · Opyt</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{_font_href(_FONTS[0][3], brand=True)}" rel="stylesheet">
<script>{_HEAD_SCRIPT}</script>
<style>{_CSS}{_font_css()}{more_css}</style>
</head>"""


_LIST_ICON = ('<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2 4h12M2 8h8M2 12h10" '
              'fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>')


def _contents(entries: list[tuple[str, str, str, list[tuple[str, str]]]]) -> str:
    """The contents: a rail of ticks at the side that opens into the list on hover or click, and
    on a narrow screen a button that opens the same list from the bottom. A source is a long
    tick, a section inside it a short one."""
    ticks, rows = [], []
    for sid, kind, label, heads in entries:
        ticks.append(f'<span class="tick l1" data-for="{sid}"></span>')
        rows.append(f'<li class="l1"><a href="#{sid}" data-for="{sid}">'
                    f'{cards.GLYPHS.get(kind, cards.GLYPHS["blog"])}'
                    f'<span>{html.escape(label)}</span></a></li>')
        for hid, text in heads:
            ticks.append(f'<span class="tick l2" data-for="{hid}"></span>')
            rows.append(f'<li class="l2"><a href="#{hid}" data-for="{hid}" data-in="{sid}">'
                        f'<span>{html.escape(text)}</span></a></li>')
    return (f'<nav class="contents" id="contents" aria-label="Contents">'
            f'<button class="rail" id="rail" type="button" aria-label="Contents" '
            f'aria-expanded="false">{"".join(ticks)}</button>'
            f'<div class="panel"><div class="panel-head">Contents</div>'
            f'<ol>{"".join(rows)}</ol></div></nav>'
            f'<button class="contents-btn" id="contents-btn" type="button" '
            f'aria-controls="contents" aria-expanded="false">{_LIST_ICON}<span>Contents</span>'
            f'</button>')


def _census(items: list[dict]) -> str:
    counts: dict[str, int] = {}
    for i in items:
        k = i.get("source_type") or "blog"
        counts[k] = counts.get(k, 0) + 1
    parts = []
    for k in _PLURALS:
        if counts.get(k):
            one, many = _PLURALS[k]
            parts.append(f"{counts[k]} {one if counts[k] == 1 else many}")
    return " · ".join(parts)


def _script_hash(script: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(script.encode()).digest()).decode() + "'"


# The Content-Security-Policy a page is served with. Scripts: only the page's own three, by hash.
# Styles: its own, its font sheets, and the font menu's `style` attributes. A source's images and
# video load from anywhere on https; a frame only from the page's own address (its PDFs); and the
# scripts call only the page's own address (its comments).
CSP = "; ".join((
    "default-src 'none'",
    f"script-src {_script_hash(_HEAD_SCRIPT)} {_script_hash(_SCRIPT)} "
    f"{_script_hash(_COMMENTS_SCRIPT)}",
    "connect-src 'self'",
    "style-src 'unsafe-inline' https://fonts.googleapis.com",
    "font-src https://fonts.gstatic.com",
    "img-src 'self' https: data:",
    "media-src https:",
    "frame-src 'self'",
    "base-uri 'none'",
    "form-action 'none'",
    "frame-ancestors 'none'",
))
