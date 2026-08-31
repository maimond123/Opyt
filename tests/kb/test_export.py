"""The export builder — the projection a foreign reader queries (pipeline/kb/export.py).

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

Design record: docs/plans/2026-08-26-foreign-kb-export-builder-phase1.md.
"""
from __future__ import annotations

import json
import re
import sqlite3

import pytest

from opyt_core import kb as kb_entry
from pipeline.kb import export, schema
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.raw_store import write_snapshot
from pipeline.kb.retrieve import search_atoms

A = "github:root/agentkit"
B = "github:stranger/agents"
C = "x:1"
D = "paper:2401.00001"
E = "substack:post1"
F = "x:2"
G = "blog:home"


def _add(conn, emb, atom_id, source_type, what_kind, who_id, topics, snapshot,
         *, entry_mode="oracle-footprint", when_ts="2024-05-01", when_precision="day",
         body_state="complete", extra_payload=None):
    raw_ref, raw_hash = write_snapshot(source_type, atom_id, snapshot)
    payload = {"source_tags": topics, "body_state": body_state, "body_basis": "observed"}
    payload.update(extra_payload or {})
    store_atom(conn, emb, atom=dict(
        atom_id=atom_id, source_type=source_type, what_kind=what_kind, who_id=who_id,
        when_ts=when_ts, when_precision=when_precision, about_entities=[],
        source_url=f"https://example/{atom_id}", raw_ref=raw_ref, raw_hash=raw_hash,
        description=f"{atom_id} card", payload=payload, entry_mode=entry_mode,
    ), snapshot_text=snapshot)


def _corpus(conn, emb, *, extra_payload=None):
    """Seven atoms spanning every source type, both kinds, three entry modes, a year-precision
    date, an undated atom, and a two-platform author cluster — so every filter arm the fidelity
    sweep exercises has something to bite on."""
    _add(conn, emb, A, "github", "artifact", "github:root", ["ai-agents"],
         "an autonomous agent framework with tools for building agents",
         entry_mode="user-saved", when_ts="2024-05-01", extra_payload=extra_payload)
    _add(conn, emb, B, "github", "artifact", "github:stranger", ["ai-agents"],
         "an agent framework library", when_ts="2024-07-09", body_state="partial")
    _add(conn, emb, C, "x", "opinion", "x:user:1", ["crypto"],
         "thoughts on rollup and proof systems", entry_mode="user-saved", when_ts="2025-01-15")
    _add(conn, emb, D, "paper", "artifact", "scholar:1", ["ai-agents"],
         "a paper about an agent framework and its library",
         when_ts="2024-01-01", when_precision="year")
    _add(conn, emb, E, "substack", "opinion", "substack:carol", ["web-dev"],
         "a react dashboard for the web", entry_mode="user-saved", when_ts="2025-03-02")
    _add(conn, emb, F, "x", "opinion", "x:user:2", ["crypto", "ai-agents"],
         "autonomous agents and crypto proof", entry_mode="author_referenced", when_ts="2025-06-01")
    _add(conn, emb, G, "blog", "opinion", "blog:example.com", [],
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
    schema.upsert_oracle(conn, "x:user:1", name="Alice")
    conn.commit()


# Every shape the read path has: both arms alone and fused, each pre-filter, each date bound,
# a handle that resolves, one that does not, and a query that matches nothing.
_QUERIES = [
    dict(query="agent framework", mode="hybrid"),
    dict(query="autonomous", mode="bm25"),
    dict(query="library", mode="bm25"),
    dict(query="crypto rollup proof", mode="hybrid"),
    dict(query="react dashboard", mode="semantic"),
    dict(query="web tools", mode="hybrid"),
    dict(query="agent", mode="semantic"),
    dict(query="agent", mode="hybrid", k=2),
    dict(query="zzzznothingmatchesthis", mode="hybrid"),
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
def built(tmp_path, monkeypatch, fake_embedder):
    """A live store under one home, and its export beside it. Returns (live_home, export_path)."""
    live = tmp_path / "live"
    monkeypatch.setenv("OPYT_HOME", str(live))
    conn = schema.connect()
    _corpus(conn, fake_embedder)
    conn.close()
    out = export.build_export(tmp_path / "export.db")
    return live, tmp_path / "export.db", out


def _reader_home(tmp_path, monkeypatch, export_path, name="reader"):
    """A second `$OPYT_HOME` holding ONLY the export as its store — no kb_raw/, no settings, no
    snapshots. Copied rather than moved so the pristine export stays available to other assertions
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
            h.when_precision, h.source_url, h.description, h.snippet, h.chunk_seq, h.chunk_span,
            h.bm25_rank, h.sem_rank, round(h.score, 9), h.body_state, h.body_basis,
            h.entry_mode, json.dumps(h.payload, sort_keys=True))


def test_search_is_identical_against_the_export(built, fake_embedder):
    """I7: same query, same hits, same order. The whole build rests on this one."""
    _live_home, export_path, _ = built
    live = schema.connect()
    # read_only, no DDL — the same call the query service will make (schema.connect:879).
    exp = schema.connect(export_path, read_only=True)
    try:
        for spec in _QUERIES:
            kwargs = {k: v for k, v in spec.items() if k != "query"}
            a = search_atoms(live, spec["query"], fake_embedder, **kwargs)
            b = search_atoms(exp, spec["query"], fake_embedder, **kwargs)
            assert [_hit_tuple(h) for h in a.hits] == [_hit_tuple(h) for h in b.hits], spec
            assert (a.effective_mode, a.ranked, a.candidates, a.cutoff, a.fts_query) == \
                   (b.effective_mode, b.ranked, b.candidates, b.cutoff, b.fts_query), spec
    finally:
        live.close()
        exp.close()


def test_the_vector_blobs_and_their_declared_width_copy_exactly(built):
    """The sharpest fidelity check available, and free — no embedder involved.

    A vector arm that reads a copied-wrong blob does not error, it returns confidently wrong
    neighbours: `np.frombuffer` will happily reshape a float16 buffer as float32 and hand back
    numbers. So assert the bytes AND the width the store declares them at, rather than trusting a
    ranking comparison to notice."""
    _live_home, export_path, _ = built
    from pipeline.kb.embed import stored_dtype
    live = schema.connect()
    exp = schema.connect(export_path, read_only=True)
    try:
        rows = "SELECT chunk_id, vector FROM chunks ORDER BY chunk_id"
        assert live.execute(rows).fetchall() == exp.execute(rows).fetchall()
        assert stored_dtype(live) == stored_dtype(exp)
    finally:
        live.close()
        exp.close()


def test_who_resolution_survives_the_profile_key_filter(built, tmp_path, monkeypatch,
                                                        fake_embedder):
    """`profile` is cut to `{handle}`, and `handle` is the ONLY arm that resolves an X author
    (X ids are numeric, so the handle lives nowhere else). Cutting one key too many would show
    up as `who=` silently matching nobody."""
    _live_home, export_path, _ = built
    monkeypatch.setattr(kb_entry, "get_kb_embedder", lambda: fake_embedder)
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


# ── 2. leak audit ────────────────────────────────────────────────────────────────

# FTS5 creates these itself from the one CREATE VIRTUAL TABLE; they are not separately copied.
_FTS_SHADOW = {"chunks_fts_data", "chunks_fts_idx", "chunks_fts_content", "chunks_fts_docsize",
               "chunks_fts_config"}
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
    assert tables == set(export._CARRY) | {export._FTS_TABLE, "kb_raw"} | _FTS_SHADOW


def test_no_value_in_the_export_names_a_filesystem_or_a_credential(built):
    """I4. `raw_ref` is relative by construction (`kb_raw/x/x_1.md`), so an absolute path in an
    export means something wrote one — which is exactly the class of thing nobody notices until
    the file is already on someone else's machine."""
    _live_home, export_path, _ = built
    conn = sqlite3.connect(export_path)
    try:
        offenders = [(t, c, v[:80]) for t, c, v in _all_values(conn)
                     if v.startswith(("/Users/", "/home/", "/var/", "C:\\"))]
        assert not offenders, offenders
        # Every table, `kb_raw` included, and no exemption: a snapshot body that really does carry
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
        assert conn.execute("SELECT COUNT(*) FROM chunks WHERE embed_text IS NOT NULL")\
                   .fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM entities WHERE identity_links IS NOT NULL")\
                   .fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM oracles WHERE source IS NOT NULL OR ingest_from IS NOT NULL "
            "OR ingest_to IS NOT NULL").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_key_nobody_allow_listed_is_dropped_and_REPORTED(tmp_path, monkeypatch, fake_embedder):
    """Dropping is right; dropping in silence is not. A new adapter field would otherwise vanish
    from every export with nothing anywhere saying so."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "live"))
    conn = schema.connect()
    _corpus(conn, fake_embedder, extra_payload={"an_adapter_added_this": "value"})
    conn.close()
    manifest = export.build_export(tmp_path / "export.db")
    assert manifest["payload_keys_dropped"] == {"an_adapter_added_this": 1}
    assert manifest["profile_keys_dropped"] == {"bio": 1, "followers": 1, "verified": 1}


# ── 3. self-containment ──────────────────────────────────────────────────────────

def test_cold_read_of_the_export_alone(built, tmp_path, monkeypatch, fake_embedder):
    """I6: a second home holding ONLY the export. Nothing may reach back to a local file."""
    _live_home, export_path, manifest = built
    monkeypatch.setattr(kb_entry, "get_kb_embedder", lambda: fake_embedder)
    home = _reader_home(tmp_path, monkeypatch, export_path)
    assert not (home / "kb_raw").exists()

    found = kb_entry.run_kb_search("agent framework", k=5)
    assert [h["atom_id"] for h in found["hits"]]
    assert kb_entry.kb_aggregate()["total"] == 7

    opened = kb_entry.kb_open(A)
    assert opened["raw_available"] is True
    assert "autonomous agent framework" in opened["raw"]
    assert opened["body_state"] == "complete"
    assert manifest["tables"]["kb_raw"] == 7


def test_open_returns_the_same_body_from_either_store(built, tmp_path, monkeypatch):
    """The two homes of a body — files beside a live store, a `kb_raw` table inside an export —
    must be indistinguishable to `open()`. That equality is what lets the retrieval code stay
    single-implementation."""
    _live_home, export_path, _ = built
    local = kb_entry.kb_open(B)
    _reader_home(tmp_path, monkeypatch, export_path)
    foreign = kb_entry.kb_open(B)
    assert local["raw"] == foreign["raw"] is not None
    assert {k: v for k, v in local.items() if k != "raw_path"} == \
           {k: v for k, v in foreign.items() if k != "raw_path"}


def test_the_export_records_the_embedder_a_reader_must_match(built):
    """§10's open fork, pinned as a fact rather than left implicit: the vector arm needs a query
    vector in the SAME subspace, and `storage_dtype` is what the blobs get decoded with —
    reading a float16 blob as float32 is silent garbage, not an error."""
    _live_home, _export_path, manifest = built
    assert manifest["embed"]["model"] == "fake-bow"
    assert manifest["embed"]["provider"] == "local"
    assert manifest["embed"]["storage_dtype"] == "float16"
    assert manifest["embed"]["dim"] == 11


def test_a_reader_whose_embedder_differs_fails_loudly(built, tmp_path, monkeypatch):
    """The blocker Phase 1 surfaces and deliberately does NOT paper over: querying a store with
    the wrong model must RAISE, not return plausible-looking garbage. This is the LOCAL path, and
    it still raises: a local store whose recorded model disagrees with this install's config is a
    misconfiguration. The foreign path is where 'the store's model is not mine' became routine,
    and it is answered by `embed.embedder_for_store` — which follows the store rather than
    checking it, at the one layer that knows a store is somebody else's."""
    from pipeline.kb.embed import SubspaceError
    from tests.kb.conftest import FakeEmbedder

    _live_home, export_path, _ = built
    _reader_home(tmp_path, monkeypatch, export_path)
    wrong = FakeEmbedder(["totally", "different", "vocabulary"])
    wrong.model = "some-other-model"
    monkeypatch.setattr(kb_entry, "get_kb_embedder", lambda: wrong)
    with pytest.raises(SubspaceError):
        kb_entry.run_kb_search("agent framework")
