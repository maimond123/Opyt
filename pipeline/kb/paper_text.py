"""
pipeline/kb/paper_text.py — a paper's full text for `open()`: read when the model opens, never kept.

A paper record holds its
abstract and a linked paper only a line, so the KB stays the size of those. When the model opens a
paper, or a post that links papers, the full text is read then and handed back in that response;
it is never written to the store. The extracted text is cached as a file
outside search (`opyt_home()/paper_text/`), so a second open downloads nothing, and a paper with no
open PDF is not asked again for a day.

Clients cap one tool result (Claude Code refuses one over ~25k tokens), so the papers one open
returns share `BUDGET` characters, and one open reads at most `MAX_READS` papers it has not cached.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, timedelta

from opyt_core.paths import opyt_path

from . import ingest_papers, link_cards, link_mint, link_router, owed_links

BUDGET = 90_000          # characters of full text one open returns, across its papers
MAX_READS = 3            # papers one open reads from the network; each can take a 30 s download
_RETRY_DAYS = 1          # a paper with no open PDF is asked again after this
_DIR = "paper_text"


def _file(key: str, ext: str):
    return opyt_path(_DIR, hashlib.sha256(key.encode()).hexdigest()[:32] + ext)


def cached(key: str) -> str | None:
    """The full text already read for this paper, or None."""
    try:
        return _file(key, ".txt").read_text()
    except OSError:
        return None


def _asked_lately(key: str) -> bool:
    try:
        stamp = _file(key, ".none").read_text().strip()
        return date.fromisoformat(stamp) > date.today() - timedelta(days=_RETRY_DAYS)
    except (OSError, ValueError):
        return False


def read(paper: dict, key: str) -> str | None:
    """The paper's full text: the cached file, else read now from its open PDF
    (`ingest_papers.resolve_fulltext`) and cached. None when it has no open PDF, an answer kept for
    `_RETRY_DAYS`. A cache that cannot be written never stops the read."""
    if (text := cached(key)) is not None:
        return text
    if _asked_lately(key):
        return None
    text = ingest_papers.resolve_fulltext(paper)
    try:
        _file(key, ".txt").parent.mkdir(parents=True, exist_ok=True)
        if text:
            _file(key, ".txt").write_text(text)
            _file(key, ".none").unlink(missing_ok=True)
        else:
            _file(key, ".none").write_text(date.today().isoformat())
    except OSError:
        pass
    return text or None


def section(text: str, limit: int) -> tuple[str, bool]:
    """The `## Full text` block an open appends, clipped to `limit`, and whether it was clipped."""
    clipped = len(text) > limit
    shown = text[:limit]
    note = f"\n\n[First {limit:,} of {len(text):,} characters.]" if clipped else ""
    return f"\n\n## Full text\n\n{shown}{note}\n", clipped


def _url_of(ref: str) -> str | None:
    """A paper's id (`paper:arXiv:…`, `paper:DOI:…`, `paper:openreview:…`, `paper:url:…`) or URL →
    the URL to look it up by; None for anything else."""
    if ref.startswith(("http://", "https://")):
        return ref if link_router.reference_kind(ref) == "paper" else None
    kind, _, rest = ref.partition(":")
    space, _, ident = rest.partition(":")
    if kind != "paper" or not ident:
        return None
    return {"arXiv": f"https://arxiv.org/abs/{ident}", "DOI": f"https://doi.org/{ident}",
            "openreview": f"https://openreview.net/forum?id={ident}",
            "url": ident if ident.startswith(("http://", "https://")) else f"https://{ident}"}.get(space)


def record_id(conn: sqlite3.Connection, ref: str) -> str | None:
    """The stored record a paper id or URL names under another id (a URL, or an arXiv link to a
    paper keyed on its DOI), or None."""
    url = _url_of(ref)
    aid = link_mint.predicted_atom_id(url, "paper") if url else None
    if aid and aid != ref and conn.execute("SELECT 1 FROM atoms WHERE atom_id=?", (aid,)).fetchone():
        return aid
    return None


def open_unstored(ref: str) -> dict | None:
    """`open()` of a paper with no record, read live by its id or URL: metadata, abstract and full
    text in one response, nothing stored. None when `ref` names no paper we can identify."""
    url = _url_of(ref)
    paper = ingest_papers.paper_from_url(url) if url else None
    if not paper:
        return None
    text = read(paper, ingest_papers.paper_atom_id(paper) or url)
    raw = ingest_papers.paper_to_markdown_full(paper, None)
    clipped = True
    if text:
        block, clipped = section(text, BUDGET)
        raw += block
    return {"atom_id": ref, "stored": False, "source_type": "paper", "what_kind": "artifact",
            "source_url": paper.get("url") or url, "description": paper.get("title") or "",
            "raw": raw, "raw_available": True,
            "body_state": "partial" if clipped else "complete", "body_basis": "observed",
            "payload": {"title": paper.get("title") or "", "year": paper.get("year"),
                        "venue": paper.get("venue") or "",
                        "authors": [a.get("name") for a in paper.get("authors") or []
                                    if isinstance(a, dict) and a.get("name")][:20],
                        "has_fulltext": bool(text)}}


def _record(conn, aid: str):
    return conn.execute("SELECT atom_id, when_ts, when_precision, source_url, payload FROM atoms "
                        "WHERE atom_id=? AND source_type='paper'", (aid,)).fetchone()


def of_record(row: sqlite3.Row) -> str | None:
    """A stored paper record's full text (`read`), rebuilt from its own columns and payload."""
    payload = json.loads(row["payload"] or "{}")
    return read(ingest_papers.paper_from_atom(row, payload), row["atom_id"])


def linked(conn: sqlite3.Connection, atom_id: str, raw: str | None) -> list[dict]:
    """The papers a post links, in order — the paper links in its text, then any record naming the
    post as its referrer — each with its full text while `BUDGET` lasts. A paper past the budget,
    or past `MAX_READS` network reads, comes back with the id `open()` reads it by."""
    entries: list[dict] = []
    seen: set[str] = set()
    for url in owed_links.paper_urls(raw or ""):
        aid = link_mint.predicted_atom_id(url, "paper")
        rec = _record(conn, aid) if aid else None
        key = rec["atom_id"] if rec else (aid or url)
        if key not in seen:
            seen.add(key)
            entries.append({"url": url, "open_id": key, "_rec": rec})
    for (aid,) in conn.execute(
            "SELECT atom_id FROM atoms WHERE source_type='paper' AND EXISTS (SELECT 1 FROM "
            "json_each(payload, '$.referenced_by') r WHERE json_extract(r.value, '$.post') = ?)",
            (atom_id,)):
        if aid not in seen:
            seen.add(aid)
            rec = _record(conn, aid)
            entries.append({"url": rec["source_url"], "open_id": aid, "_rec": rec})

    left, reads, out = BUDGET, 0, []
    for e in entries:
        rec = e.pop("_rec")
        title = (json.loads(rec["payload"] or "{}").get("title") if rec else None) or (
            (link_cards.stored(conn, e["url"]) or link_cards.Card("", "")).title)
        e.update(title=title or "", full_text=None)
        text = None
        if left > 0 and (cached(e["open_id"]) is not None or reads < MAX_READS):
            reads += cached(e["open_id"]) is None
            paper = None if rec else ingest_papers.paper_from_url(e["url"])
            if rec:
                text = of_record(rec)
            elif paper:
                e["title"] = e["title"] or paper.get("title") or ""
                text = read(paper, e["open_id"])
            if not rec and not paper:
                e["note"] = "No registry identifies this paper; its link and line are in the post."
            elif not text:
                e["note"] = "No open PDF; open(open_id) returns its abstract or line."
        else:
            e["note"] = "Past this open's size budget; open(open_id) returns its full text."
        if text:
            if len(text) <= left or not out or all(o["full_text"] is None for o in out):
                shown = text[:left]
                e.update(full_text=shown, chars=len(text))
                if len(shown) < len(text):
                    e["note"] = f"First {len(shown):,} of {len(text):,} characters."
                left -= len(shown)
            else:
                e["note"] = "Past this open's size budget; open(open_id) returns its full text."
        out.append(e)
    return out
