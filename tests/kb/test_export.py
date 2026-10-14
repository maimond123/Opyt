"""The export builder — the projection a foreign reader queries (pipeline/sharing/export.py).

Three properties, in the order they matter:

  1. FIDELITY. The same query against the live store and against its export returns the same
     hits, in the same order, with the same spans and scores. First, because if the projection
     loses fidelity then nothing built on top of it means anything — and because this is what
     catches whatever the hand-written allow-list missed.
  2. NO LEAK. The export contains exactly the allow-listed objects and nothing that names the
     owner's filesystem or looks like a credential. A PERMANENT assertion, not a one-time audit:
     a table added to `schema.py` must fail this test rather than silently ship in every export.
  3. SELF-CONTAINMENT. Opened as the ONLY thing in a second `$OPYT_HOME` — no snapshots, no
     settings, no live store — all three read tools still work, `open()` included. That is what
     "no download of the KB is required to query it" rests on.
"""
from __future__ import annotations

import json
import re
import sqlite3

import pytest

from pipeline.retrieval import reads as kb_entry
from pipeline.forget import atoms as forget
from pipeline.kb import schema
from pipeline.sharing import export
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.ingest_common import snapshot_hash
from pipeline.retrieval.engine import search_atoms
from pipeline.oracles import roster

A = "github:root/agentkit"
B = "github:stranger/agents"
C = "x:1"
D = "paper:2401.00001"
E = "substack:post1"
F = "x:2"
G = "blog:home"


def _add(conn, atom_id, source_type, what_kind, who_id, topics, snapshot,
         *, entry_mode="oracle-footprint", when_ts="2024-05-01", when_precision="day",
         body_state="complete", extra_payload=None):
    raw_hash = snapshot_hash(snapshot)
    payload = {"source_tags": topics, "body_state": body_state, "body_basis": "observed"}
    payload.update(extra_payload or {})
    store_atom(conn, dict(
        atom_id=atom_id, source_type=source_type, what_kind=what_kind, who_id=who_id,
        when_ts=when_ts, when_precision=when_precision, about_entities=[],
        source_url=f"https://example/{atom_id}", raw_hash=raw_hash,
        description=f"{atom_id} card", payload=payload, entry_mode=entry_mode,
    ), snapshot)


def _corpus(conn, *, extra_payload=None):
    """Seven atoms spanning every source type, both kinds, three entry modes, a year-precision
    date, an undated atom, and a two-platform author cluster — so every filter arm the fidelity
    sweep exercises has something to bite on."""
    _add(conn, A, "github", "artifact", "github:root", ["ai-agents"],
         "an autonomous agent framework with tools for building agents",
         entry_mode="user-saved", when_ts="2024-05-01", extra_payload=extra_payload)
    _add(conn, B, "github", "artifact", "github:stranger", ["ai-agents"],
         "an agent framework library", when_ts="2024-07-09", body_state="partial")
    _add(conn, C, "x", "opinion", "x:user:1", ["crypto"],
         "thoughts on rollup and proof systems", entry_mode="user-saved", when_ts="2025-01-15")
    _add(conn, D, "paper", "artifact", "scholar:1", ["ai-agents"],
         "a paper about an agent framework and its library",
         when_ts="2024-01-01", when_precision="year")
    _add(conn, E, "substack", "opinion", "substack:carol", ["web-dev"],
         "a react dashboard for the web", entry_mode="user-saved", when_ts="2025-03-02")
    _add(conn, F, "x", "opinion", "x:user:2", ["crypto", "ai-agents"],
         "autonomous agents and crypto proof", entry_mode="author_referenced", when_ts="2025-06-01")
    _add(conn, G, "blog", "opinion", "blog:example.com", [],
         "web tools", entry_mode="user-saved", when_ts="", when_precision="")

    schema.upsert_entity(conn, "github:root", name="Root Dev")
    schema.upsert_entity(conn, "github:stranger", name="Stranger")
    schema.upsert_entity(conn, "x:user:1", name="Alice",
                         profile={"handle": "alice", "bio": "a private-ish scraped bio",
                                  "followers": 4210, "verified": True})
    schema.upsert_entity(conn, "x:user:2", name="Bob", profile={"handle": "bob"})
    schema.upsert_entity(conn, "substack:carol", name="Carol")
    schema.upsert_entity(conn, "scholar:1", name="A Researcher")
    schema.upsert_entity(conn, "blog:example.com", name="Example Blog")
    roster.upsert_oracle(conn, "x:user:1", name="Alice")
    conn.commit()


# Every shape the read path has: each pre-filter, each date bound,
# a handle that resolves, one that does not, and a query that matches nothing.
_QUERIES = [
    dict(query="agent framework"),
    dict(query="autonomous"),
    dict(query="library"),
    dict(query="crypto rollup proof"),
    dict(query="react dashboard"),
    dict(query="web tools"),
    dict(query="agent"),
    dict(query="agent", k=2),
    dict(query="zzzznothingmatchesthis"),
    dict(query="agent", tags=["ai-agents"]),
    dict(query="proof", tags=["crypto"]),
    dict(query="agent", tags=["no-such-tag"]),
    dict(query="agent framework", what_kind="artifact"),
    dict(query="agent framework", source_type="github"),
    dict(query="proof", source_type="x"),
    dict(query="agent", who_id="github:stranger"),
    dict(query="agent", who_id=["github:root", "github:stranger"]),
    dict(query="agent framework", date_from="2025-01-01"),
    dict(query="agent framework", date_to="2024-12-31"),
    dict(query="agent framework", entry_mode="user-saved"),
]


@pytest.fixture()
def built(tmp_path, monkeypatch):
    """A live store under one home, and its export beside it. Returns (live_home, export_path)."""
    live = tmp_path / "live"
    monkeypatch.setenv("OPYT_HOME", str(live))
    conn = schema.connect()
    _corpus(conn)
    conn.close()
    out = export.build_export(tmp_path / "export.db")
    return live, tmp_path / "export.db", out


def _reader_home(tmp_path, monkeypatch, export_path, name="reader"):
    """A second `$OPYT_HOME` holding ONLY the export as its store — no settings, nothing beside
    it. Copied rather than moved so the pristine export stays available to other assertions
    (a writable `schema.connect()` runs the idempotent DDL and would edit it in place)."""
    home = tmp_path / name
    home.mkdir()
    (home / "opyt.db").write_bytes(export_path.read_bytes())
    monkeypatch.setenv("OPYT_HOME", str(home))
    return home


# ── 1. fidelity ──────────────────────────────────────────────────────────────────

def _hit_tuple(h):
    """Everything about a hit that a reader can see, so a projection that quietly lost a column
    fails here rather than somewhere downstream."""
    return (h.atom_id, h.source_type, h.what_kind, h.who_id, h.who_name, h.when_ts,
            h.when_precision, h.source_url, h.description, h.snippet, h.snippet_span,
            round(h.score, 9), h.body_state, h.body_basis,
            h.entry_mode, json.dumps(h.payload, sort_keys=True))


def test_search_is_identical_against_the_export(built):
    """I7: same query, same hits, same order. The whole build rests on this one."""
    _live_home, export_path, _ = built
    live = schema.connect()
    # read_only, no DDL — the same call the query service will make (schema.connect:879).
    exp = schema.connect(export_path, read_only=True)
    try:
        for spec in _QUERIES:
            kwargs = {k: v for k, v in spec.items() if k != "query"}
            a = search_atoms(live, spec["query"], **kwargs)
            b = search_atoms(exp, spec["query"], **kwargs)
            assert [_hit_tuple(h) for h in a.hits] == [_hit_tuple(h) for h in b.hits], spec
            assert (a.ranked, a.candidates, a.cutoff, a.fts_query) == \
                   (b.ranked, b.candidates, b.cutoff, b.fts_query), spec
    finally:
        live.close()
        exp.close()


def test_the_export_indexes_its_own_bodies(built):
    """The export's `bodies_fts` is built from `schema.py`'s DDL over the bodies it copied, with
    the triggers, so a search of the copy reads the same index a local search does."""
    _live_home, export_path, _ = built
    exp = sqlite3.connect(export_path)
    try:
        exp.execute("INSERT INTO bodies_fts(bodies_fts, rank) VALUES ('integrity-check', 1)")
        triggers = {r[0] for r in exp.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'")}
        assert triggers == {"bodies_ai", "bodies_ad", "bodies_au"}
        assert exp.execute("SELECT COUNT(*) FROM bodies_fts WHERE bodies_fts MATCH 'agent'"
                           ).fetchone()[0] > 0
    finally:
        exp.close()


def test_the_export_index_and_triggers_are_schemas_own(built):
    """Built from `schema.BODIES_INDEX_DDL`, not re-declared: the export's `bodies_fts` and its
    three triggers must be word-for-word what a fresh store gets, and every exported body must be
    in the index, or a reader's search silently disagrees with the owner's."""
    _live_home, export_path, _ = built
    fresh = sqlite3.connect(":memory:")
    fresh.execute("CREATE TABLE bodies (body_id INTEGER PRIMARY KEY, atom_id TEXT UNIQUE, text)")
    fresh.executescript(schema.BODIES_INDEX_DDL)
    want = dict(fresh.execute(
        "SELECT name, sql FROM sqlite_master WHERE name IN "
        "('bodies_fts', 'bodies_ai', 'bodies_ad', 'bodies_au')"))
    fresh.close()
    exp = sqlite3.connect(export_path)
    try:
        got = dict(exp.execute(
            "SELECT name, sql FROM sqlite_master WHERE name IN "
            "('bodies_fts', 'bodies_ai', 'bodies_ad', 'bodies_au')"))
        assert got == want and len(got) == 4
        exp.execute("INSERT INTO bodies_fts(bodies_fts, rank) VALUES ('integrity-check', 1)")
        assert exp.execute("SELECT COUNT(*) FROM bodies_fts_docsize").fetchone()[0] == \
               exp.execute("SELECT COUNT(*) FROM bodies").fetchone()[0] == 7
    finally:
        exp.close()


def test_the_manifest_counts_exactly_the_carried_tables(built):
    """The manifest reports what crossed and nothing else: no `chunks`, no `kb_meta`, and no
    `embed` block, since a reader needs no model to search a whole-item index."""
    _live_home, _export_path, manifest = built
    assert set(manifest["tables"]) == set(export._CARRY)
    assert "embed" not in manifest


def test_who_resolution_survives_the_profile_key_filter(built, tmp_path, monkeypatch):
    """`profile` is cut to `{handle}`, and `handle` is the ONLY arm that resolves an X author
    (X ids are numeric, so the handle lives nowhere else). Cutting one key too many would show
    up as `who=` silently matching nobody."""
    _live_home, export_path, _ = built
    live = kb_entry.run_kb_search("proof", who="alice")
    _reader_home(tmp_path, monkeypatch, export_path)
    foreign = kb_entry.run_kb_search("proof", who="alice")
    assert [h["atom_id"] for h in live["hits"]] == [h["atom_id"] for h in foreign["hits"]] == [C]
    assert foreign["insights"]["resolved_who"] == live["insights"]["resolved_who"]


def test_aggregate_is_identical_against_the_export(built, tmp_path, monkeypatch):
    """Including `trusted_atoms`, which joins `oracles` — the table the parent plan excluded by
    name. Excluded, this count reads a permanent zero and no caller can tell that from 'nobody
    here is an oracle'."""
    _live_home, export_path, _ = built
    live = kb_entry.kb_aggregate()
    _reader_home(tmp_path, monkeypatch, export_path)
    foreign = kb_entry.kb_aggregate()
    assert live == foreign
    assert live["trusted_atoms"] > 0


def test_every_row_keeps_the_live_stores_rowid(tmp_path, monkeypatch):
    """Ties in a read fall back to rowid order (`aggregate`'s newest twelve sort on `when_ts`, and
    day-precision dates tie), and a delta keys rows by rowid, so a fresh export and a patched
    served copy must both take rowids from the live store. A forgotten atom leaves a gap, which a
    renumbering copy would close."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "live"))
    live = schema.connect()
    _corpus(live)
    forget.atom(live, B, confirm=True)
    export.build_export(tmp_path / "export.db")
    exp = schema.connect(tmp_path / "export.db", read_only=True)
    keys = {"atoms": "atom_id", "bodies": "atom_id", "entities": "entity_id",
            "oracles": "canonical_id"}
    try:
        for table, key in keys.items():
            rows = f"SELECT rowid, {key} FROM {table} ORDER BY rowid"
            assert live.execute(rows).fetchall() == exp.execute(rows).fetchall(), table
    finally:
        live.close()
        exp.close()


# ── 2. leak audit ────────────────────────────────────────────────────────────────

# FTS5 creates these itself from the one CREATE VIRTUAL TABLE; they are not separately copied.
# No `bodies_fts_content`: the index is external-content, so the text is stored once, in `bodies`.
_FTS_SHADOW = {f"bodies_fts_{part}" for part in ("data", "idx", "docsize", "config")}
# SHAPES, never words. An earlier draft matched `secret`/`password`/`api_key` as substrings and a
# 40-character word run; measured against the real 2,804-atom store it flagged 1,118 rows, every
# one of them prose — a tweet discussing secrets, a long repo name. A leak audit that cries wolf on
# a thousand rows is one nobody reads, so it must match things that can ONLY be a credential.
_CREDENTIALS = re.compile(
    r"(sk-[A-Za-z0-9]{20,}"                       # OpenAI / Anthropic style
    r"|gh[pousr]_[A-Za-z0-9]{30,}"                # GitHub personal/OAuth/server tokens
    r"|xox[baprs]-[0-9A-Za-z-]{10,}"              # Slack
    r"|AKIA[0-9A-Z]{16}"                          # AWS access key id
    r"|Bearer\s+[A-Za-z0-9._~+/-]{20,}"           # a literal Authorization header value
    r"|eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}" # a JWT
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----)")      # a private key block


def _all_values(conn):
    for (table,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info('{table}')")]
        for row in conn.execute(f"SELECT * FROM '{table}'"):
            for col, value in zip(cols, row):
                if isinstance(value, str):
                    yield table, col, value


def test_the_export_holds_exactly_the_allow_listed_objects(built):
    """EQUALITY, not a deny-list. A table added to `schema.py` lands here as a failure — which is
    the whole point: the default for a new table must be 'excluded', enforced by a red test rather
    than by someone remembering to add it to an exclusion list."""
    _live_home, export_path, _ = built
    conn = sqlite3.connect(export_path)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    finally:
        conn.close()
    assert tables == set(export._CARRY) | {"bodies_fts"} | _FTS_SHADOW
    # A delta patches exactly `TABLES` and the index built from them, so a table an export holds
    # outside them would never update.
    assert tables == set(export.TABLES) | {"bodies_fts"} | _FTS_SHADOW


def test_no_value_in_the_export_names_a_filesystem_or_a_credential(built):
    """I4. No column holds a path by construction, so an absolute path in an export means
    something wrote one — which is exactly the class of thing nobody notices until
    the file is already on someone else's machine."""
    _live_home, export_path, _ = built
    conn = sqlite3.connect(export_path)
    try:
        offenders = [(t, c, v[:80]) for t, c, v in _all_values(conn)
                     if v.startswith(("/Users/", "/home/", "/var/", "C:\\"))]
        assert not offenders, offenders
        # Every table, `bodies` included, and no exemption: a snapshot body that really does carry
        # a live token is precisely what an owner needs told BEFORE the file leaves their machine.
        creds = [(t, c, v[:80]) for t, c, v in _all_values(conn) if _CREDENTIALS.search(v)]
        assert not creds, creds
    finally:
        conn.close()


def test_the_json_columns_are_key_filtered(built):
    """`payload` and `profile` are verbatim passthroughs from the adapters, so the allow-list is
    the only thing standing between a future adapter's field and every export."""
    _live_home, export_path, _ = built
    conn = sqlite3.connect(export_path)
    try:
        for (blob,) in conn.execute("SELECT payload FROM atoms WHERE payload IS NOT NULL"):
            assert set(json.loads(blob)) <= export._PAYLOAD_KEYS
        for (blob,) in conn.execute("SELECT profile FROM entities WHERE profile IS NOT NULL"):
            assert set(json.loads(blob)) <= export._PROFILE_KEYS
        assert conn.execute("SELECT COUNT(*) FROM entities WHERE identity_links IS NOT NULL")\
                   .fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM oracles WHERE source IS NOT NULL"
                            ).fetchone()[0] == 0
        # `ingest_from`/`ingest_to` do not exist — coverage is per (Oracle x source), in
        # `oracle_sources`, which the export does not carry at all.
        assert "ingest_from" not in {r[1] for r in conn.execute("PRAGMA table_info(oracles)")}
    finally:
        conn.close()


def test_a_papers_open_pdf_survives_the_export(tmp_path, monkeypatch):
    """The connector makes a page of sources from this copy, and fetches each paper's PDF from
    where these two keys say it is."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "live"))
    where = {"external_ids": {"DOI": "10.1/x", "ArXiv": "2401.00002"},
             "pdf_url": "https://journal.test/x.pdf"}
    conn = schema.connect()
    _add(conn, "paper:DOI:10.1/x", "paper", "artifact", "scholar:1", [],
         "a paper", extra_payload=where)
    conn.commit()
    conn.close()
    export.build_export(tmp_path / "export.db")
    out = sqlite3.connect(tmp_path / "export.db")
    try:
        payload = json.loads(out.execute("SELECT payload FROM atoms").fetchone()[0])
    finally:
        out.close()
    assert {k: payload[k] for k in where} == where


def test_a_key_nobody_allow_listed_is_dropped_and_REPORTED(tmp_path, monkeypatch):
    """Dropping is right; dropping in silence is not. A new adapter field would otherwise vanish
    from every export with nothing anywhere saying so."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "live"))
    conn = schema.connect()
    _corpus(conn, extra_payload={"an_adapter_added_this": "value"})
    conn.close()
    manifest = export.build_export(tmp_path / "export.db")
    assert manifest["payload_keys_dropped"] == {"an_adapter_added_this": 1}
    assert manifest["profile_keys_dropped"] == {"bio": 1, "followers": 1, "verified": 1}


# ── 3. self-containment ──────────────────────────────────────────────────────────

def test_cold_read_of_the_export_alone(built, tmp_path, monkeypatch):
    """I6: a second home holding ONLY the export. Nothing may reach back to a local file."""
    _live_home, export_path, manifest = built
    _reader_home(tmp_path, monkeypatch, export_path)

    found = kb_entry.run_kb_search("agent framework", k=5)
    assert [h["atom_id"] for h in found["hits"]]
    assert kb_entry.kb_aggregate()["total"] == 7

    opened = kb_entry.kb_open(A)
    assert opened["raw_available"] is True
    assert "autonomous agent framework" in opened["raw"]
    assert opened["body_state"] == "complete"
    assert manifest["tables"]["bodies"] == 7


def test_open_returns_the_same_body_from_either_store(built, tmp_path, monkeypatch):
    """A body opened from the live store and from its export must be indistinguishable to
    `open()`. That equality is what lets the retrieval code stay single-implementation."""
    _live_home, export_path, _ = built
    local = kb_entry.kb_open(B)
    _reader_home(tmp_path, monkeypatch, export_path)
    foreign = kb_entry.kb_open(B)
    assert local["raw"] == foreign["raw"] is not None
    assert local == foreign
