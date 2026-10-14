"""The four reads' envelopes, pinned byte for byte.

`search`, `open`, `aggregate` and `feed` are what a host reads, so their keys, key order, values,
notice wording and `cite` links are behavior. The service's routes are pinned the same way in
`tests/service/test_wire_contract.py`, and riders in `test_server_riders_pins.py`; this file pins
the flows underneath both, over the user's own store, a file peer and an unreadable `kb=`.

Only values that differ per run are normalised: the temp home (`<home>`) and timestamps
(`<time>`). A failure means an envelope changed. Recapture (`OPYT_RETRIEVAL_CAPTURE=1`) only for
an intended change.

The reads are reached through `_READS`, so a move switches that line and nothing else.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from pipeline.retrieval import reads as kb_entry
from opyt_core.paths import opyt_db
from pipeline.kb import schema
from pipeline.retrieval import peers
from tests.kb.test_search_envelope import _add, _corpus

_READS = {"search": kb_entry.run_kb_search, "open": kb_entry.kb_open,
          "aggregate": kb_entry.kb_aggregate, "feed": kb_entry.kb_feed}

GOLDEN = Path(__file__).with_name("retrieval_envelopes.json")
CAPTURE = os.environ.get("OPYT_RETRIEVAL_CAPTURE") == "1"
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?")
_captured: dict[str, str] = {}


def _check(case: str, envelope: dict, home: Path) -> None:
    text = json.dumps(envelope, ensure_ascii=False)       # key order kept: it is behavior
    text = _TIME.sub("<time>", text.replace(str(home), "<home>"))
    if CAPTURE:
        _captured[case] = text
        return
    want = json.loads(GOLDEN.read_text())[case]
    assert text == want, f"{case}: the envelope changed\n got: {text}\nwant: {want}"


@pytest.fixture(scope="module", autouse=True)
def _write_golden():
    yield
    if CAPTURE and _captured:
        old = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        old.update(_captured)
        GOLDEN.write_text(json.dumps(old, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


@pytest.fixture()
def store(kb_home):
    conn = schema.connect()
    _corpus(conn)
    _add(conn, "web:crawl/agents", "blog", "artifact", "blog:crawl",
         ["ai-agents"], "a long crawl page about agent framework tools and an autonomous library",
         entry_mode="frontier", when_ts="2025-02-03")
    _add(conn, "x:2", "x", "opinion", "x:user:2", [],
         "an undated note on agent tools", entry_mode="user-saved", when_ts="")
    conn.execute("INSERT OR IGNORE INTO entities (entity_id, name, canonical_id) "
                 "VALUES ('github:root', 'Root Author', 'github:root')")
    conn.commit()
    conn.close()
    return kb_home


SEARCHES = {
    "search_default": dict(query="agent framework"),
    "search_short": dict(query="framework", k=2),
    "search_sentence": dict(query="what do people think about autonomous tools"),
    "search_tags_normalized_and_missing": dict(query="agent", tags=["AI Agents", "!!!", "nope"]),
    "search_who_unresolved": dict(query="agent", who="@nobody_here"),
    "search_who_resolves": dict(query="agent", who="root"),
    "search_who_id_list": dict(query="agent", who_id=["github:root", "x:user:1"]),
    "search_date_window_undated": dict(query="agent", date_from="2024-06", date_to="2025"),
    "search_filters_match_nothing": dict(query="agent", source_type="substack"),
    "search_scoped_frontier": dict(query="agent", entry_mode="frontier", k=20),
    "search_scoped_list": dict(query="agent", entry_mode=["user-saved"]),
    "search_kind_and_source": dict(query="agent", what_kind="artifact", source_type="github"),
}


@pytest.mark.parametrize("case", sorted(SEARCHES))
def test_search(store, case):
    _check(case, _READS["search"](**SEARCHES[case]), store)


def test_search_of_an_empty_store(kb_home):
    _check("search_empty_store", _READS["search"]("agent"), kb_home)


def test_a_malformed_date_raises_before_anything_opens(store):
    with pytest.raises(ValueError) as e:
        _READS["search"]("agent", date_from="5/11/2026")
    _check("search_bad_date_error", {"error": str(e.value)}, store)


def test_open(store):
    _check("open_github", _READS["open"]("github:root/agentkit"), store)
    _check("open_undated_x", _READS["open"]("x:2"), store)
    _check("open_unknown", _READS["open"]("nope:1"), store)
    _check("open_as_kb", _READS["open"]("x:1", as_kb="alex"), store)
    _check("open_as_kb_me_refused", _READS["open"]("x:1", as_kb="me"), store)


def test_aggregate(store):
    _check("aggregate_whole", _READS["aggregate"](), store)
    _check("aggregate_scoped_sample",
           _READS["aggregate"]({"tags": ["ai-agents"], "date_to": "2024"}, sample=3), store)
    _check("aggregate_scope_nothing", _READS["aggregate"]({"source_type": "substack"}), store)


def test_feed(store):
    first = _READS["feed"](limit=2)
    _check("feed_seed_page", first, store)
    _check("feed_after_cursor", _READS["feed"](since="2000-01-01 00:00:00", since_id="", limit=2),
           store)
    _check("feed_all_modes", _READS["feed"](entry_mode=["frontier", "user-saved"]), store)
    _check("feed_dated_window", _READS["feed"](date_from="2025"), store)


def test_a_file_peer(store):
    peers.add("friend", opyt_db(), "Friend's KB")
    _check("peer_search", _READS["search"]("agent framework", kb="friend"), store)
    _check("peer_search_who_unresolved", _READS["search"]("agent", who="@x", kb="friend"), store)
    _check("peer_open", _READS["open"]("x:1", kb="friend"), store)
    _check("peer_aggregate", _READS["aggregate"](kb="friend"), store)
    _check("peer_feed", _READS["feed"](kb="friend", as_kb="alex"), store)


def test_an_unreadable_kb(store):
    peers.add("gone", store / "missing.db")
    for verb, call in (("search", lambda kb: _READS["search"]("agent", kb=kb)),
                       ("open", lambda kb: _READS["open"]("x:1", kb=kb)),
                       ("aggregate", lambda kb: _READS["aggregate"](kb=kb)),
                       ("feed", lambda kb: _READS["feed"](kb=kb))):
        _check(f"unknown_{verb}", call("nobody"), store)
        _check(f"unreadable_{verb}", call("gone"), store)
