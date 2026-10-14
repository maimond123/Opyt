"""
service/reading.py — one served read: run against an owner's export, capped, and counted.

Both ways in call these: the HTTP routes (`app.py`, a reader token) and the connector's tools
(`connector.py`, a Google account). The caps and the read meter live here once, so a read costs
the same either way and is counted the same.

Each read calls the function a local reader calls against a registered peer
(`pipeline.retrieval.reads.run_kb_search(..., kb=owner)` and its siblings), so within the caps
the answer equals that call made in-process.
"""
from __future__ import annotations

from pipeline.retrieval import reads
from service import usage

# Responses are capped, not requests: one call can return everything. Each cap also keeps one
# answer inside the client's tool-result limit: about 150,000 characters in claude.ai and Claude
# Desktop, 25,000 tokens by default in Claude Code (claude.com/docs/connectors/building). Measured
# on a library of 8,937 items: a search hit is about 2,500 characters (3,152 at most),
# so 25 hits stay under 81,000; 100 hits were 229,000-250,000 and failed in claude.ai.
K_MAX = 25
# How deep a search pages (`offset`): each page reruns the ranking down to its last hit, so the
# depth bounds one call's work. Ten pages of 25.
SEARCH_DEPTH_MAX = 250
# One page of an item's text, about 15,000 tokens. A longer text is read in pages (`offset`):
# 25 of those 8,937 items were over 100,000 characters.
OPEN_CHARS_MAX = 60_000
# Feed items are light cards: 200 measured 87,000 characters, so 100 stay near 45,000.
FEED_MAX = 100


def search(owner: str, reader: str, query: str, *, k: int = 8, offset: int | None = None,
           **filters) -> dict:
    """`run_kb_search` over `owner`'s export: one page of at most `K_MAX` hits, from `offset`.
    `filters` are its own keyword arguments (`tags`, `who`, `as_kb`, …).

    A caller that pages (the connector) passes `offset`, and gets `next_offset` back. The ranking
    is rerun down to the page's last hit and the earlier hits dropped; a keyword score does not
    depend on how deep the run goes, so consecutive pages continue one ranking. A caller that
    cannot page (a share-link reader's install, `offset` None) gets the first page. An offset past
    `SEARCH_DEPTH_MAX` answers an error and is not counted."""
    size, start = min(k, K_MAX), offset or 0
    if not 0 <= start < SEARCH_DEPTH_MAX:
        return {"error": f"offset {start} is outside 0-{SEARCH_DEPTH_MAX - 1}. Pass 0, or the "
                         f"`next_offset` of the previous page; to reach other results, narrow "
                         f"the search (a person, dates, a source type)."}
    envelope = reads.run_kb_search(query, k=min(start + size, SEARCH_DEPTH_MAX), kb=owner,
                                   **filters)
    hits = envelope["hits"][start:]
    nxt = next_offset(start, len(hits), envelope["trace"]["ranked"])
    envelope["hits"] = hits
    envelope["notices"] = [_page_notice(n, start, len(hits), nxt, paged=offset is not None,
                                        clamped=k > K_MAX,
                                        at_least=envelope["trace"]["pool_saturated"])
                           for n in envelope["notices"]]
    if offset is not None and (start or nxt is not None):
        envelope["next_offset"] = nxt
    return _counted(envelope, owner, reader, "search", results="hits")


def next_offset(start: int, shown: int, ranked: int) -> int | None:
    """Where the page after this one starts: None when nothing ranked is left, or the depth is
    reached."""
    nxt = start + shown
    return nxt if shown and ranked > nxt and nxt < SEARCH_DEPTH_MAX else None


def _page_notice(notice: dict, start: int, shown: int, nxt: int | None, *, paged: bool,
                 clamped: bool, at_least: bool) -> dict:
    """The "results cut" notice, true of this page: the engine's says "raise k", which stops
    working at `K_MAX`. Every other notice passes unchanged. `at_least`: the ranking stopped
    collecting at its pool (`trace.pool_saturated`), so `ranked` is a floor, not a total."""
    if notice.get("code") != "results_truncated":
        return notice
    ranked = f"more than {notice['ranked']}" if at_least else str(notice["ranked"])
    if paged:
        shown_as = f"Showing results {start + 1}-{start + shown} of {ranked} that matched."
        message = (f"{shown_as} Pass offset={nxt} for the next ones, or narrow the search."
                   if nxt is not None else
                   f"{shown_as} Search pages stop at {SEARCH_DEPTH_MAX}; narrow the search (a "
                   f"person, dates, a source type) to reach the others.")
    elif clamped:
        message = (f"Showing the top {shown} of {ranked} that matched; this service returns at "
                   f"most {K_MAX} per search. Narrow the search to reach the others.")
    else:
        return notice
    return {**notice, "showing": shown, "message": message}


def open_atom(owner: str, reader: str, atom_id: str, as_kb: str | None = None,
              offset: int = 0) -> dict:
    """One atom from `owner`'s export, with one page of its raw text: `OPEN_CHARS_MAX` characters
    from `offset` (see `page`). An offset outside the text answers an error and is not counted."""
    envelope = reads.kb_open(atom_id, kb=owner, as_kb=as_kb)
    raw = envelope.get("raw")
    if isinstance(raw, str):
        if not 0 <= offset <= max(len(raw) - 1, 0):
            return {"atom_id": atom_id, "error": (
                f"offset {offset} is outside this item's text, which has {len(raw)} characters; "
                f"pass 0, or the `next_offset` an earlier answer gave.")}
        envelope.update(page(raw, offset))
    return _counted(envelope, owner, reader, "open")


def open_whole(owner: str, reader: str, atom_id: str) -> dict:
    """One atom from `owner`'s export with all of its raw text, counted as one read: what a
    reader-view page shows (`connector.reader_view`). A page is a document, not a tool answer, so
    `OPEN_CHARS_MAX` does not apply."""
    return _counted(reads.kb_open(atom_id, kb=owner), owner, reader, "open")


def page(raw: str, offset: int) -> dict:
    """What changes in an `open` answer to carry the part of `raw` from `offset`: nothing for a
    text that fits one page, else that page, where it sits, and where the next one starts.

    `next_offset` is None on the last page. A page has `body_state` `partial`, so it is never
    quoted as the whole source."""
    end = offset + OPEN_CHARS_MAX
    if offset == 0 and end >= len(raw):
        return {}
    return {"raw": raw[offset:end], "raw_offset": offset, "raw_chars": len(raw),
            "next_offset": end if end < len(raw) else None, "body_state": "partial"}


def aggregate(owner: str, reader: str, scope: dict | None = None,
              as_kb: str | None = None) -> dict:
    """The state-of-play skeleton over `owner`'s export."""
    # No cap: `kb_aggregate` is already bounded (tests/service/test_caps.py asserts it).
    # `sample` is deliberately not forwarded; it is a question about your own corpus.
    envelope = reads.kb_aggregate(scope=scope, kb=owner, as_kb=as_kb)
    return _counted(envelope, owner, reader, "aggregate")


def feed(owner: str, reader: str, *, limit: int = 50, **cursor) -> dict:
    """The newest arrivals in `owner`'s export, `limit` clamped to 1..`FEED_MAX`. `cursor` is
    `kb_feed`'s own keyword arguments. Raises `ValueError` on a malformed date."""
    envelope = reads.kb_feed(limit=max(1, min(limit, FEED_MAX)), kb=owner, **cursor)
    return _counted(envelope, owner, reader, "feed", results="items")


def _counted(envelope: dict, owner: str, reader: str, tool: str,
             results: str | None = None) -> dict:
    """Count a served read on the meter and return its answer. `results` names the list whose
    emptiness makes it a zero-result read."""
    usage.record_usage(owner, reader, tool,
                       zero_results=results is not None and not envelope[results])
    return envelope
