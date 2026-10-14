"""
service/pages/cards.py — one card per item, drawn from its stored snapshot.

An item is what `open()` returns for an atom (its `source_type`, `source_url`, `when_ts`,
`description`, `raw` markdown and `payload`), plus `pdf_url` for a paper whose PDF the page shows.
Every item gets the same frame (kind, byline, body, link back to the original); only the body
differs by source.
"""
from __future__ import annotations

import html
import re
from datetime import date
from urllib.parse import urlparse

import yaml

from service.pages import prose

_WPM = 230

_FOOTER = re.compile(r"\n---\n+\*[^\n]*\[Original post\]\([^)]*\)\*\s*$")
_THREAD_LINE = re.compile(r"^> \*\*Thread\*\* · \[(\d+) posts?\]\([^)]*\)\s*$", re.M)
_POST_NO = re.compile(r"^\*\*\[(\d+)/(\d+)\]\*\*\s*")
_BARE_URL = re.compile(r"(?<![(<\[\"'=])\b(https?://[^\s<>()\[\]]+[^\s<>()\[\].,;:!?'\"])")

# Source marks, ink only: the brand keeps color for the logo's cursor and a heading's dot.
GLYPHS = {
    "x": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18.9 1.15h3.68l-8.04 9.19L24 '
         '22.85h-7.4l-5.8-7.59-6.64 7.59H.47l8.6-9.83L0 1.15h7.6l5.24 6.93Zm-1.29 19.5h2.04L6.49 '
         '3.24H4.3Z"/></svg>',
    "substack": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M22.54 8.24H1.46V5.4h21.08v'
                '2.84ZM1.46 10.81V24L12 18.11 22.54 24V10.81H1.46ZM22.54 0H1.46v2.84h21.08V0Z"/>'
                '</svg>',
    "github": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 .3a12 12 0 0 0-3.8 23.38'
              'c.6.12.83-.26.83-.57v-2.02c-3.34.73-4.04-1.61-4.04-1.61-.55-1.39-1.33-1.76-1.33-1.76'
              '-1.09-.74.08-.73.08-.73 1.2.09 1.84 1.24 1.84 1.24 1.07 1.83 2.8 1.3 3.49 1 .1-.78.42'
              '-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.14-.3-.54-1.52.1'
              '-3.18 0 0 1-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.28-1.55 3.29-1.23 3.29-1.23.64 1.66'
              '.24 2.88.12 3.18a4.65 4.65 0 0 1 1.23 3.22c0 4.61-2.8 5.63-5.48 5.92.42.36.81 1.1.81'
              ' 2.22v3.29c0 .32.21.7.82.58A12 12 0 0 0 12 .3"/></svg>',
    "paper": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 1h9.5L20 6.5V23H5V1Zm9 1.8V7'
             'h4.2L14 2.8ZM7.5 10h10v1.6h-10V10Zm0 3.6h10v1.6h-10v-1.6Zm0 3.6h7v1.6h-7v-1.6Z"/>'
             '</svg>',
    "blog": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 1a11 11 0 1 0 0 22 11 11 0 0'
            ' 0 0-22Zm7.9 10h-3.47a17 17 0 0 0-1.6-6.98A9 9 0 0 1 19.9 11ZM12 3.06c.9 1.1 2.2 3.5'
            ' 2.43 7.94H9.57C9.8 6.56 11.1 4.16 12 3.06ZM9.17 4.02A17 17 0 0 0 7.57 11H4.1a9 9 0 0'
            ' 1 5.07-6.98ZM4.1 13h3.47a17 17 0 0 0 1.6 6.98A9 9 0 0 1 4.1 13ZM12 20.94c-.9-1.1-2.2'
            '-3.5-2.43-7.94h4.86c-.23 4.44-1.53 6.84-2.43 7.94Zm2.83-.96a17 17 0 0 0 1.6-6.98h3.47'
            'a9 9 0 0 1-5.07 6.98Z"/></svg>',
}


# ── one item ────────────────────────────────────────────────────────────────────────────────────

def card(item: dict, n: int, total: int) -> dict:
    """`{nav, html}` for one item: its entry in the contents and its card."""
    kind = item.get("source_type") or "blog"
    meta, body = _front(item.get("raw") or "")
    body = _FOOTER.sub("", body)
    draw = _BODIES.get(kind, _article)
    d = draw(item, meta, body)
    sid = f"s{n}"
    d["html"], heads = _anchor(d["html"], sid)
    url = item.get("source_url") or meta.get("url") or ""
    title = f'<h2 class="title">{html.escape(d["title"])}</h2>' if d.get("title") else ""
    card = f"""
  <article class="item item-{html.escape(kind)}" id="{sid}">
    <div class="head">{GLYPHS.get(kind, GLYPHS["blog"])}<span>{html.escape(d["kind"])}</span><span class="idx">{n:02d} / {total:02d}</span></div>
    {title}
    {d["byline"]}
    <div class="body">{d["html"]}</div>
    <div class="foot">{_via(item.get("payload") or {})}{d.get("stat", "")}<a class="open" href="{html.escape(url)}" target="_blank" rel="noopener">{html.escape(d["open"])} <span aria-hidden="true">↗</span></a></div>
  </article>"""
    # A post has no title, so its entry leads with who wrote it.
    label = d.get("title") or f'{d["who"]}: {d.get("lead") or ""}'
    return {"nav": (sid, kind, label, heads), "html": card}


_HEADING = re.compile(r"<h([1-3])>(.*?)</h\1>", re.S)
_MAX_HEADS = 8


def _anchor(body: str, sid: str) -> tuple[str, list[tuple[str, str]]]:
    """Ids on a body's section headings, and those headings for the contents.

    One level only, at most `_MAX_HEADS`: the highest that holds two or more headings, so a
    README's lone title or a blog's lone "Subscribe" banner is not taken for its sections. A
    heading the page labels itself (`Links`) carries a class and is skipped."""
    levels = [lvl for lvl, _ in _HEADING.findall(body)]
    top = next((lvl for lvl in sorted(set(levels)) if levels.count(lvl) >= 2), None)
    if top is None:
        return body, []
    heads: list[tuple[str, str]] = []

    def name(m: re.Match) -> str:
        text = " ".join(html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).split())
        if m.group(1) != top or not text or len(heads) >= _MAX_HEADS:
            return m.group(0)
        hid = f"{sid}-h{len(heads) + 1}"
        heads.append((hid, text))
        return f'<h{top} id="{hid}">{m.group(2)}</h{top}>'

    return _HEADING.sub(name, body), heads


def _x(item: dict, meta: dict, body: str) -> dict:
    """A post, a thread, or a long X article. A thread's posts are one card, numbered."""
    count = _THREAD_LINE.search(body)
    body = _THREAD_LINE.sub("", _drop_heading(body))
    posts = [p.strip() for p in re.split(r"\n---\n", body) if p.strip()]
    parts = []
    for p in posts:
        # No "2/5" of our own: authors number their threads ("1/N") and two counts disagree.
        inner, _ = prose.to_html(_BARE_URL.sub(r"<\1>", _POST_NO.sub("", p)))
        parts.append(f'<div class="post">{inner}</div>')
    handle = (meta.get("author") or "").lstrip("@")
    name = meta.get("author_name") or handle
    payload = item.get("payload") or {}
    if count:
        kind = f"X thread · {count.group(1)} posts"
    elif payload.get("is_article"):
        kind = "X article"
    else:
        kind = "X post"
    likes = payload.get("like_count") or meta.get("likes")
    return {"kind": kind, "who": name,
            "lead": _lead(item.get("description")),
            "byline": _byline(name, f"@{handle}" if handle else "", _when(item.get("when_ts")),
                              mono=True),
            "html": "".join(parts), "open": "Open on X",
            "stat": f'<span class="stat">{int(likes):,} likes</span>' if likes else ""}


def _article(item: dict, meta: dict, body: str) -> dict:
    """A Substack post or a blog post: the title, the byline, the whole text."""
    title, body = _title(body)
    url = item.get("source_url") or meta.get("url") or ""
    host = prose.host(url)
    who = meta.get("author_name") or ""
    if not who or who.startswith("http"):
        who = host
    sub = meta.get("publication") or (host if who != host else "")
    # A page's <title> often ends with the site's or the author's name ("… | Merit Systems",
    # "… — Christian Catalini"), and the post often opens by repeating its title.
    title = _bare_title(title, who, sub, host, host.split(".")[0])
    body = _drop_repeat(body, title)
    # A blog writes its images and links relative to itself ("/_next/image?…").
    inner, words = prose.to_html(body, base=url, link_base=url)
    if item.get("source_type") == "substack":
        kind, open_label = "Substack", "Open on Substack"
    else:
        kind, open_label = "Blog", f"Open on {host}" if host else "Open original"
    extra = f"{_minutes(words)} min read"
    return {"kind": kind, "title": title or _lead(item.get("description")), "who": who,
            "html": inner, "open": open_label,
            "byline": _byline(who, sub, _when(item.get("when_ts") or meta.get("date")),
                              extra=extra)}


def _paper(item: dict, meta: dict, body: str) -> dict:
    """A paper: its authors and abstract, then the PDF itself, figures and all."""
    payload = item.get("payload") or {}
    title, rest = _title(body)
    title = payload.get("title") or title
    authors = re.search(r"^\*\*Authors:\*\*\s*(.+)$", rest, re.M)
    abstract = payload.get("abstract") or _section(rest, "Abstract")
    venue = " · ".join(str(v) for v in (payload.get("venue"), payload.get("year")) if v)
    if not venue and "arxiv.org" in (item.get("source_url") or ""):
        venue = "arXiv preprint"
    who = authors.group(1).strip() if authors else ""
    first = who.split(",")[0] if who else "Unknown authors"
    abstract_html = prose.to_html(abstract)[0] if abstract else ""
    # `pdf_url` is the copy the page frames; `pdf_source` is where it was published.
    pdf = item.get("pdf_url")
    if pdf:
        source = item.get("pdf_source") or pdf
        viewer = (f'<div class="pdf"><div class="pdf-bar"><span class="pdf-tag">PDF</span>'
                  f'<span class="pdf-src">{html.escape(prose.host(source) + urlparse(source).path)}'
                  f'</span><a href="{html.escape(source)}" target="_blank" rel="noopener">'
                  f'Open ↗</a></div>'
                  f'<iframe src="{html.escape(pdf)}#view=FitH" loading="lazy" '
                  f'title="{html.escape(title or "Paper")}"></iframe></div>')
    else:
        viewer = ('<p class="nopdf">This paper has no open PDF. Read it where it was '
                  'published.</p>')
    sub = (f'<p class="authors">{html.escape(who)}</p>' if who else "")
    abstract_block = (f'<div class="abstract"><div class="label">Abstract</div>'
                      f'{abstract_html}</div>') if abstract_html else ""
    url = item.get("source_url") or ""
    return {"kind": "Paper" + (f" · {venue}" if venue else ""), "title": title, "who": first,
            "html": abstract_block + viewer,
            "byline": f'<div class="byline paper-by">{sub}<span class="when">'
                      f'{day(_when(item.get("when_ts"))) if item.get("when_ts") else ""}'
                      f'</span></div>',
            "open": f"Open on {prose.host(url)}" if url else "Open original"}


def _repo(item: dict, meta: dict, body: str) -> dict:
    """A GitHub repository: what it is, its numbers, and its README."""
    payload = item.get("payload") or {}
    name, rest = _title(body)
    desc = re.search(r"^> (.+)$", rest, re.M)
    readme = rest.split("## README", 1)[1] if "## README" in rest else ""
    url = item.get("source_url") or meta.get("url") or ""
    owner_repo = urlparse(url).path.strip("/")
    inner, _ = prose.to_html(readme, base=f"https://raw.githubusercontent.com/{owner_repo}/HEAD/",
                      link_base=f"https://github.com/{owner_repo}/blob/HEAD/")
    chips = []
    if payload.get("stars") is not None:
        chips.append(f'★ {int(payload["stars"]):,}')
    for key in ("code_language", "license"):
        if payload.get(key):
            chips.append(str(payload[key]))
    if payload.get("forks") is not None:
        chips.append(f'{int(payload["forks"]):,} forks')
    stats = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    lead = f'<p class="lead">{html.escape(desc.group(1))}</p>' if desc else ""
    owner = meta.get("owner") or owner_repo.split("/")[0]
    return {"kind": "GitHub repo", "title": name or owner_repo, "who": owner, "html": inner,
            "byline": f'{lead}<div class="stats">{stats}</div>',
            "open": "Open on GitHub"}


_BODIES = {"x": _x, "substack": _article, "blog": _article, "paper": _paper, "github": _repo}


# ── small decisions ─────────────────────────────────────────────────────────────────────────────

def _front(raw: str) -> tuple[dict, str]:
    """A snapshot's frontmatter as a dict, and the body after it."""
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    try:
        meta = yaml.safe_load(raw[3:end]) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), raw[end + 4:].lstrip("\n")


def _title(body: str) -> tuple[str, str]:
    """The body's leading `# Title` line, and the body without it."""
    m = re.match(r"\s*# (.+)\n", body)
    return (m.group(1).strip(), body[m.end():]) if m else ("", body)


def _drop_heading(body: str) -> str:
    return _title(body)[1]


_TITLE_TAIL = re.compile(r"\s+[|—–·-]\s+([^|—–·]+)$")


def _bare_title(title: str, *names: str) -> str:
    """`title` without a trailing " | Name" when Name is the author's or the site's own."""
    m = _TITLE_TAIL.search(title)
    if m and _squash(m.group(1)) in {_squash(n) for n in names if n}:
        return title[:m.start()].strip()
    return title


def _drop_repeat(body: str, title: str) -> str:
    """The body without an opening heading that only repeats the title."""
    m = re.search(r"^#{1,3} (.+)\n?", body, re.M)
    if m and m.start() < 600 and _squash(m.group(1)) == _squash(title):
        return body[:m.start()] + body[m.end():]
    return body


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _section(body: str, name: str) -> str:
    m = re.search(rf"^## {name}\s*\n(.*?)(?=^## |\Z)", body, re.M | re.S)
    return m.group(1).strip() if m else ""


def _lead(description: str | None) -> str:
    """A post's first words, from its stored description (`@who (Name) · text…`)."""
    text = (description or "").split(" · ", 1)[-1]
    return " ".join(text.split())[:140]


def _byline(who: str, sub: str, when: date | None, *, mono: bool = False,
            extra: str = "") -> str:
    words = re.findall(r"[A-Za-z0-9]+", who)
    # A site named by its domain ("anthropic.com") is one name, not two words.
    initials = (words[0][0] if "." in who and " " not in who else
                "".join(w[0] for w in words[:2])).upper() if words else "·"
    sub_html = (f'<span class="{"handle" if mono else "sub"}">{html.escape(sub)}</span>'
                if sub else "")
    bits = [b for b in (day(when) if when else "", extra) if b]
    # The dot travels with what follows it, so a wrapped byline never ends on a dangling "·".
    tail = "".join(f'<span class="bit"><span class="sep">·</span>{html.escape(b)}</span>'
                   for b in bits)
    return (f'<div class="byline"><span class="av" aria-hidden="true">{html.escape(initials)}'
            f'</span><span class="who">{html.escape(who)}</span>{sub_html}{tail}</div>')


def _via(payload: dict) -> str:
    """Who in the user's circle linked this, when someone did."""
    handles = []
    for ref in payload.get("referenced_by") or []:
        h = isinstance(ref, dict) and ref.get("handle")
        if h and f"@{h}" not in handles:
            handles.append(f"@{h}")
    if not handles:
        return ""
    return f'<span class="via">Linked by {html.escape(", ".join(handles[:2]))}</span>'


def _when(value) -> date | None:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def day(d: date | None) -> str:
    return f"{d:%b} {d.day}, {d.year}" if d else ""


def _minutes(words: int) -> int:
    return max(1, round(words / _WPM))
