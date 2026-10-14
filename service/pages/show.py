"""
service/pages/show.py — a page as its reader gets it: drawn from its stored items on every
request, so a design change reaches pages already made. Opening a page starts its days again.
An unknown or expired link gets nothing here; the route answers it with `render.gone`.
"""
from __future__ import annotations

from pathlib import Path

from service.pages import render, store


def page(page_id: str, now: float) -> str | None:
    """The page's HTML, or None when no live page has this id."""
    got = store.read(page_id, now)
    if got is None:
        return None
    store.opened(page_id, now)
    items = [_framed(page_id, n, item) for n, item in enumerate(got.items, 1)]
    return render.page(items, title=got.title, note=got.note, answer=got.answer,
                       made=store.day(got.made), days=store.DAYS)


def pdf(page_id: str, n: int, now: float) -> Path | None:
    """The file of a live page's `n`th PDF, or None."""
    if store.read(page_id, now) is None:
        return None
    path = store.pdf_path(page_id, n)
    return path if path.is_file() else None


def _framed(page_id: str, n: int, item: dict) -> dict:
    """The item, with the copy of its PDF to frame (`pdf_url`) while the page keeps one. A PDF
    that is not there leaves the paper's abstract and link."""
    if item.get("pdf_source") and store.pdf_path(page_id, n).is_file():
        return {**item, "pdf_url": f"{store.path(page_id)}/pdf/{n}"}
    return item
