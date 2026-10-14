"""
service/pages/make.py — make a page of sources. Both ways in call it: the `/v1/pages` route for an
install (its owner token) and the connector's `page` tool (a Google account).

Check the items, fetch each paper's PDF beside the page, store the page, hand back its path. A
PDF the box cannot get never fails the page: that paper shows its abstract and its link.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from service.pages import pdfs, store

MAX_ITEMS = 50
DEFAULT_TITLE = "Sources"
# What a page draws an item from (`render.py`). Anything else a client sends is dropped, so no
# client can set the PDF a page frames.
_FIELDS = ("atom_id", "source_type", "source_url", "when_ts", "description", "raw")


class Refused(ValueError):
    """Nothing to make a page of. The message says why."""


def create(owner: str, items: list[dict], title: str | None, note: str | None,
           answer: str | None, now: float) -> dict:
    """`{id, path, days}` of a new page showing `answer` (markdown), then `items` in their
    order, made by `owner`'s library: it is deleted `days` days after it was last opened. Raises
    `Refused` before anything is written."""
    if not items:
        raise Refused("a page needs at least one item")
    if len(items) > MAX_ITEMS:
        raise Refused(f"a page holds at most {MAX_ITEMS} items, and this had {len(items)}")
    kept = [_kept(i) for i in items]
    page_id = store.new_id()
    if store.has_room():
        _fetch_pdfs(page_id, kept)
    store.save(store.Page(page_id, owner, (title or "").strip() or DEFAULT_TITLE,
                          (note or "").strip() or None, (answer or "").strip() or None, kept,
                          now))
    return {"id": page_id, "path": store.path(page_id), "days": store.DAYS}


def _kept(item: dict) -> dict:
    kept = {k: item[k] for k in _FIELDS if isinstance(item.get(k), str)}
    kept["payload"] = item["payload"] if isinstance(item.get("payload"), dict) else {}
    return kept


def _fetch_pdfs(page_id: str, items: list[dict]) -> None:
    """Every paper's PDF fetched beside the page, at the same time. A paper whose PDF arrived
    keeps where it came from, as `pdf_source`."""
    papers = [(n, item) for n, item in enumerate(items, 1) if pdfs.urls(item)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(lambda p: _first_pdf(page_id, *p), papers))
    for (_, item), source in zip(papers, got):
        if source:
            item["pdf_source"] = source


def _first_pdf(page_id: str, n: int, item: dict) -> str | None:
    """The first of the paper's PDF URLs the box could read, now kept as the page's `n`th PDF."""
    return next((u for u in pdfs.urls(item) if pdfs.fetch(u, store.pdf_path(page_id, n))), None)
