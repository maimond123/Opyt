"""`payload` on a hit — the read side of everything the capture audit captured.

Adapters record more per atom than the text — `body_state`/`body_basis`, `source_tags`,
`code_language`, `citationCount`, `stars`, `like_count`. All of it lands in `atoms.payload`, and
the column is in every retrieved row (`SELECT a.*`), so it has to be decoded on the way out to be
readable.

Two things are under test, and they are different in kind:

  1. **Verbatim pass-through.** Whatever the adapter wrote comes back, unfiltered. The failure this
     prevents is a read-side allowlist — the shape where capturing a new field silently requires a
     second edit somewhere else before anyone can see it. `test_payload_key_names.py` pins the same
     rule structurally; these prove the behavior.
  2. **The `body_state` promotion.** Those two keys are moved to the top level, not copied. They
     are not data about the source like `stars`; they qualify the snippet itself, and the failure
     sizes are not comparable — a missing `stars` is a duller answer, a missing
     `body_state: "partial"` quotes a paywall teaser as the whole essay.
"""
from __future__ import annotations

import json

from pipeline.retrieval import reads as kb_entry
from pipeline.kb import schema
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.ingest_common import snapshot_hash
from pipeline.retrieval.engine import search_atoms

# One atom per source, each with the payload shape its OWN adapter writes. The point of the
# spread is that there is no common schema to normalize to — the host gets what the source had.
_ATOMS = [
    ("github:root/agentkit", "github", "artifact", "github:root",
     "an autonomous agent framework with tools",
     {"stars": 60000, "code_language": "Python", "source_tags": ["ai-agents"],
      "body_state": "complete", "body_basis": "observed"}),
    ("x:1", "x", "opinion", "x:user:1",
     "thoughts on rollup and proof systems in crypto",
     {"like_count": 412, "is_thread": True,
      "body_state": "complete", "body_basis": "stated"}),
    ("paper:2401.00001", "papers", "artifact", "scholar:1",
     "a library for autonomous agent evaluation",
     {"citationCount": 87, "venue": "NeurIPS",
      "body_state": "partial", "body_basis": "observed"}),
]


def _seed(conn, atoms=_ATOMS):
    for atom_id, source_type, what_kind, who, snap, payload in atoms:
        raw_hash = snapshot_hash(snap)
        store_atom(conn, dict(
            atom_id=atom_id, source_type=source_type, what_kind=what_kind, who_id=who,
            when_ts="2024-05-01", when_precision="day", about_entities=[],
            source_url=f"https://e/{atom_id}", raw_hash=raw_hash,
            description=f"{atom_id} card", payload=payload, entry_mode="user-saved",
        ), snap)


def _by_id(hits):
    return {h["atom_id"]: h for h in hits}


# ── verbatim, and source-shaped ───────────────────────────────────────────────

def test_each_source_returns_its_own_payload_shape(kb_home):
    """Three sources, three different key sets, all intact. There is no normalization step and
    no fixed schema — `payload` is the source's own vocabulary, handed over as-is."""
    conn = schema.connect(); _seed(conn); conn.close()
    hits = _by_id(kb_entry.run_kb_search("agent OR rollup OR library", k=8)["hits"])

    gh = hits["github:root/agentkit"]["payload"]
    assert gh["stars"] == 60000 and gh["code_language"] == "Python"
    assert gh["source_tags"] == ["ai-agents"]

    x = hits["x:1"]["payload"]
    assert x["like_count"] == 412 and x["is_thread"] is True

    paper = hits["paper:2401.00001"]["payload"]
    assert paper["citationCount"] == 87 and paper["venue"] == "NeurIPS"


def test_an_unknown_key_passes_through_untouched(kb_home):
    """The anti-allowlist test. A key no reader has ever heard of must arrive anyway — otherwise
    every future captured field needs a second edit on the read path before it becomes visible,
    which is exactly the write-only trap this change closes."""
    conn = schema.connect()
    _seed(conn, [(
        "github:root/novel", "github", "artifact", "github:root",
        "a novel agent framework",
        {"a_field_invented_tomorrow": {"nested": [1, 2]}, "body_state": "complete",
         "body_basis": "observed"},
    )])
    conn.close()
    hit = _by_id(kb_entry.run_kb_search("framework", k=8)["hits"])["github:root/novel"]
    assert hit["payload"]["a_field_invented_tomorrow"] == {"nested": [1, 2]}


def test_the_engine_hit_carries_the_payload(kb_home):
    """One layer below `run_kb_search`: the engine's own hit already carries the decoded payload
    and the promoted `body_state`, so no caller has to decode the column itself."""
    conn = schema.connect()
    _seed(conn)
    try:
        hits = {h.atom_id: h for h in search_atoms(conn, "autonomous agent framework").hits}
        assert hits["github:root/agentkit"].payload["stars"] == 60000
        assert hits["github:root/agentkit"].body_state == "complete"
    finally:
        conn.close()


# ── the promotion: moved, not copied ──────────────────────────────────────────

def test_body_state_is_top_level_and_gone_from_payload(kb_home):
    """Both directions, because the move is the decision. Copying would leave two spellings of
    "is this the whole body" free to drift apart."""
    conn = schema.connect(); _seed(conn); conn.close()
    hit = _by_id(kb_entry.run_kb_search("agent", k=8)["hits"])["github:root/agentkit"]

    assert hit["body_state"] == "complete" and hit["body_basis"] == "observed"
    assert "body_state" not in hit["payload"] and "body_basis" not in hit["payload"]


def test_a_partial_body_says_so_end_to_end(kb_home):
    """This atom's stored body is knowingly short of the source, and a host about to quote the
    snippet can see that before it does."""
    conn = schema.connect(); _seed(conn); conn.close()
    hit = _by_id(kb_entry.run_kb_search("library", k=8)["hits"])["paper:2401.00001"]
    assert hit["body_state"] == "partial"


def test_the_stored_column_still_holds_body_state(kb_home):
    """The strip is a read-side projection only. `schema.py`'s pending sweep queries
    `json_extract(payload,'$.body_state')` against the stored column, so removing the key from
    the row — rather than from the returned dict — would silently break that sweep."""
    conn = schema.connect()
    _seed(conn)
    try:
        stored = conn.execute("SELECT payload FROM atoms WHERE atom_id='paper:2401.00001'"
                              ).fetchone()["payload"]
        assert json.loads(stored)["body_state"] == "partial"
    finally:
        conn.close()


# ── open() agrees with search ─────────────────────────────────────────────────

def test_open_reports_the_same_three_fields_as_search(kb_home):
    """`open()` is the call made immediately before asserting what a source says, so it is the
    last place "this is only the teaser" can still be seen. Two decoders that disagreed would
    make the check depend on which call the host happened to make."""
    conn = schema.connect(); _seed(conn); conn.close()
    hit = _by_id(kb_entry.run_kb_search("library", k=8)["hits"])["paper:2401.00001"]
    opened = kb_entry.kb_open("paper:2401.00001")

    assert opened["body_state"] == hit["body_state"] == "partial"
    assert opened["body_basis"] == hit["body_basis"] == "observed"
    assert opened["payload"] == hit["payload"] == {"citationCount": 87, "venue": "NeurIPS"}


# ── fail-safe: a bad payload costs its extras, never the hit ──────────────────

def _corrupt(conn, atom_id, value):
    conn.execute("UPDATE atoms SET payload=? WHERE atom_id=?", (value, atom_id))
    conn.commit()


def test_a_null_payload_degrades_to_empty(kb_home):
    conn = schema.connect(); _seed(conn); _corrupt(conn, "x:1", None); conn.close()
    hit = _by_id(kb_entry.run_kb_search("rollup", k=8)["hits"])["x:1"]
    assert hit["payload"] == {} and hit["body_state"] is None and hit["body_basis"] is None


def test_a_malformed_payload_degrades_to_empty(kb_home):
    conn = schema.connect(); _seed(conn)
    _corrupt(conn, "x:1", "{not json at all")
    conn.close()
    hit = _by_id(kb_entry.run_kb_search("rollup", k=8)["hits"])["x:1"]
    assert hit["payload"] == {} and hit["body_state"] is None


def test_a_non_object_payload_degrades_to_empty(kb_home):
    # Valid JSON, wrong shape. `json.loads` succeeds and hands back a list, and `.pop(key)` on a
    # list raises — the type check is what stops a parse success becoming a crash.
    conn = schema.connect(); _seed(conn)
    _corrupt(conn, "x:1", "[1, 2, 3]")
    conn.close()
    hit = _by_id(kb_entry.run_kb_search("rollup", k=8)["hits"])["x:1"]
    assert hit["payload"] == {}


def test_a_bad_payload_never_drops_the_atom(kb_home):
    """The Fail-safe invariant, stated as the thing that must NOT happen. A decode error is our
    problem; losing the search result makes it the user's. The extras are optional — the hit is not.
    Checked on `open()` too, since that is the path a citation depends on."""
    conn = schema.connect(); _seed(conn)
    _corrupt(conn, "x:1", "{not json at all")
    conn.close()
    assert "x:1" in _by_id(kb_entry.run_kb_search("rollup", k=8)["hits"])
    opened = kb_entry.kb_open("x:1")
    assert opened["raw_available"] is True and opened["payload"] == {}
