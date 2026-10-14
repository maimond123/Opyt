"""
service/pages/pdfs.py — a paper's open PDF, fetched by the box so its page can show it from the
page's own address (a PDF framed from another site often refuses to show).

`urls` decides where to look, in the library's own order (`fulltext_pdf_urls`); `fetch` reads one
into a file. The URLs come from whoever made the page, so every hop, redirects included, must be
HTTPS: the box's own services answer plain HTTP on loopback, and a page must not be a way to call
them. Every hop also asks robots.txt (`pipeline/robots.py`, the rule the install's reads follow).
A PDF the box cannot get leaves its paper showing the abstract and a link.
"""
from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from pipeline import robots
from pipeline.platforms.papers.pdf import fulltext_pdf_urls

MAX_BYTES = 50 * 1024 * 1024
# Seconds for one PDF, redirects included; a request already started may run one `_TIMEOUT` past
# it, which keeps a page well inside the install's wait (`client._PAGE_TIMEOUT`).
_DEADLINE = 45
_TIMEOUT = (10, 20)       # connect, read: per socket operation
_HOPS = 5
_UA = {"User-Agent": "opyt-pages/1.0"}


def urls(item: dict) -> list[str]:
    """Where a paper item's open PDF may be, most reliable first; none for anything else."""
    if item.get("source_type") != "paper":
        return []
    payload = item.get("payload") or {}
    ids = dict(payload.get("external_ids") or {})
    atom_id = item.get("atom_id") or ""
    if atom_id.startswith("paper:arXiv:"):
        # A copy served by an install older than its `external_ids` lacks them, but an arXiv
        # paper's id is its arXiv id.
        ids.setdefault("ArXiv", atom_id.removeprefix("paper:arXiv:"))
    return fulltext_pdf_urls({"externalIds": ids,
                              "openAccessPdf": {"url": payload.get("pdf_url")}})


def fetch(url: str, dest: Path) -> bool:
    """Read the PDF at `url` into `dest`: True when it is there. At most `MAX_BYTES`, within
    `_DEADLINE`; a body that is not a PDF is not kept. Never raises."""
    part = dest.with_name(dest.name + ".part")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        size, deadline = 0, time.monotonic() + _DEADLINE
        with _get(url, deadline) as resp, open(part, "wb") as out:
            for chunk in resp.iter_content(1 << 16):
                if size == 0 and b"%PDF-" not in chunk[:1024]:
                    return False
                size += len(chunk)
                if size > MAX_BYTES or time.monotonic() > deadline:
                    return False
                out.write(chunk)
        if size == 0:
            return False
        part.replace(dest)
        return True
    except Exception:
        return False
    finally:
        part.unlink(missing_ok=True)


def _get(url: str, deadline: float) -> requests.Response:
    """The response `url` ends at, following redirects one hop at a time, each HTTPS, allowed by
    its robots.txt and started before `deadline`. Raises on anything else."""
    for _ in range(_HOPS):
        if urlparse(url).scheme != "https":
            raise ValueError(f"not https: {url}")
        if time.monotonic() > deadline:
            raise TimeoutError(url)
        robots.check(url, _UA["User-Agent"])
        resp = requests.get(url, stream=True, timeout=_TIMEOUT, headers=_UA,
                            allow_redirects=False)
        if not resp.is_redirect:
            resp.raise_for_status()
            return resp
        resp.close()
        url = urljoin(url, resp.headers["location"])
    raise ValueError("too many redirects")
