"""Reading someone else's knowledge base — `kb=` over a registered peer (pipeline/kb/peers.py).

Phase 1 built the artifact: an export that is fidelity-identical to the live store and
self-contained. Phase 2 makes a SECOND knowledge base readable from it, on the same disk, with no
network. What these tests hold, in the order they matter:

  1. ROUTING. `kb=None`/"me" reads the local store, byte-for-byte the behaviour that shipped;
     any other name reads that peer's, and the two corpora do not intersect — so a hit can only
     have come from the store the caller asked for.
  2. PROVENANCE (X2). Every hit says which knowledge base it came from, local ones included, and
     a foreign read says so in a notice. An unlabelled foreign atom is just a bookmark.
  3. NO WRITE BACK (X1, I3). A foreign read leaves the reader's atoms untouched and the peer's
     file byte-identical. Neither is enforced by discipline: they are different databases, and a
     peer is opened read-only.
  4. NOTHING RAISES OUT OF A TOOL (P3). `kb=` is a string a host model typed. Unregistered, or
     registered and missing, returns an empty envelope of the normal shape plus a notice.

Design record: docs/plans/2026-08-26-foreign-kb-export-builder-phase1.md (Part 2).
"""
from __future__ import annotations

import sqlite3

import pytest

from opyt_core import kb as kb_entry
from pipeline.kb import export, peers, schema
from pipeline.kb.embed import (SubspaceError, embedder_for_store, embedder_from_meta,
                               ensure_kb_meta)
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.raw_store import write_snapshot


def _forget(name: str) -> bool:
    """Delete a peer row, reporting whether one was there. Was `peers.remove`, deleted 2026-08-28
    for having no production caller — nothing revokes a peer through the registry."""
    conn = schema.connect()
    try:
        n = conn.execute("DELETE FROM peers WHERE name = ?", (name,)).rowcount
        conn.commit()
    finally:
        conn.close()
    return n > 0

# Disjoint on every axis a hit is identified by — id, author, topic and vocabulary — so no
# assertion below can pass by accident on a store it did not mean to read.
LOCAL_A = "x:local1"
LOCAL_B = "github:mine/dashboard"
PEER_A = "x:peer1"
PEER_B = "github:theirs/rollup"


def _add(conn, emb, atom_id, source_type, what_kind, who_id, topics, snapshot):
    raw_ref, raw_hash = write_snapshot(source_type, atom_id, snapshot)
    store_atom(conn, emb, atom=dict(
        atom_id=atom_id, source_type=source_type, what_kind=what_kind, who_id=who_id,
        when_ts="2025-04-01", when_precision="day", about_entities=[],
        source_url=f"https://example/{atom_id}", raw_ref=raw_ref, raw_hash=raw_hash,
        description=f"{atom_id} card", entry_mode="user-saved",
        payload={"source_tags": topics, "body_state": "complete", "body_basis": "observed"},
    ), snapshot_text=snapshot)


def _local_corpus(conn, emb):
    _add(conn, emb, LOCAL_A, "x", "opinion", "x:user:100", ["web-dev"],
         "a react dashboard for the web")
    _add(conn, emb, LOCAL_B, "github", "artifact", "github:mine", ["web-dev"],
         "react dashboard tools")
    schema.upsert_entity(conn, "x:user:100", name="Mine", profile={"handle": "mine"})
    schema.upsert_entity(conn, "github:mine", name="Mine")
    conn.commit()


def _peer_corpus(conn, emb):
    _add(conn, emb, PEER_A, "x", "opinion", "x:user:200", ["crypto"],
         "crypto rollup proof systems")
    _add(conn, emb, PEER_B, "github", "artifact", "github:theirs", ["crypto"],
         "a rollup proof library")
    schema.upsert_entity(conn, "x:user:200", name="Theirs", profile={"handle": "theirs"})
    schema.upsert_entity(conn, "github:theirs", name="Theirs")
    schema.upsert_oracle(conn, "x:user:200", name="Theirs")
    conn.commit()


@pytest.fixture()
def two_kbs(tmp_path, monkeypatch, fake_embedder):
    """A reader's own store, and a peer's export registered in it. Returns (reader_home, export).

    The peer is built under its OWN `$OPYT_HOME` and then exported, which is the real sequence —
    the export is the only thing that crosses, and its snapshot files stay behind in a directory
    the reader never learns the name of."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "peer"))
    conn = schema.connect()
    _peer_corpus(conn, fake_embedder)
    conn.close()
    export_path = tmp_path / "peer-export.db"
    export.build_export(export_path)

    reader_home = tmp_path / "reader"
    monkeypatch.setenv("OPYT_HOME", str(reader_home))
    conn = schema.connect()
    _local_corpus(conn, fake_embedder)
    conn.close()
    peers.add("peer", export_path, "The Peer's KB")
    return reader_home, export_path


# ── 1. the registry ──────────────────────────────────────────────────────────────

def test_registry_round_trip(two_kbs, tmp_path):
    _reader_home, export_path = two_kbs
    assert [(p["name"], p["label"]) for p in peers.list_peers()] == [("peer", "The Peer's KB")]
    # Stored absolute: a relative path would name a different file per working directory.
    assert peers.list_peers()[0]["location"] == str(export_path.resolve())

    peers.add("peer", export_path, "Renamed")          # idempotent on name, updates the label
    assert len(peers.list_peers()) == 1
    assert peers.list_peers()[0]["label"] == "Renamed"

    assert _forget("peer") is True
    assert _forget("peer") is False               # revocation is just a row delete
    assert peers.list_peers() == []


def test_add_preserves_a_url_verbatim(two_kbs):
    """A URL must NOT go through `Path.resolve()` — that mangles `https://host/v1/kb/x` into a
    filesystem path under the working directory, silently, and the row then names a file that
    will never exist. The trailing slash is stripped so the transport can join paths without
    checking."""
    peers.add("remote", "https://api.example.com/v1/kb/david/", label="David over HTTPS")
    row = peers.get("remote")
    assert row["location"] == "https://api.example.com/v1/kb/david"
    assert peers.is_remote(row["location"]) is True


def test_a_local_peer_is_still_resolved_to_an_absolute_path(two_kbs):
    _reader_home, export_path = two_kbs
    assert peers.is_remote(peers.get("peer")["location"]) is False
    assert peers.get("peer")["location"] == str(export_path.resolve())


def test_the_reader_token_round_trips(two_kbs):
    """The token lives in the registry beside the peer it opens, not in `.env`: `.env` is
    per-install and a reader can hold tokens for several knowledge bases at once, so a single
    environment variable has nowhere to put the second one."""
    peers.add("remote", "https://api.example.com/v1/kb/david", token="tok-abc")
    assert peers.get("remote")["token"] == "tok-abc"

    peers.add("remote", "https://api.example.com/v1/kb/david", token="tok-def")
    assert peers.get("remote")["token"] == "tok-def"


def test_get_is_none_for_a_name_nobody_registered(two_kbs):
    assert peers.get("nobody-registered-this") is None


def test_open_peer_refuses_a_remote_row(two_kbs):
    """`open_peer` opens FILES. A remote peer reaching here means an entry point skipped the
    branch that routes it over HTTP, so the message names that rather than reporting a missing
    file — which is what `schema.connect` would have said, several frames later, about a path
    that was never a path."""
    peers.add("remote", "https://api.example.com/v1/kb/david", token="t")
    with pytest.raises(peers.PeerUnavailable) as e:
        peers.open_peer("remote")
    assert "HTTP" in str(e.value)


def test_the_local_store_cannot_be_registered_as_a_peer(two_kbs):
    """`kb="me"` is how a caller says 'my own store', so a row under that name would be
    registered, listed, and permanently unopenable — dead state nothing reports."""
    _reader_home, export_path = two_kbs
    with pytest.raises(ValueError):
        peers.add("me", export_path)


@pytest.mark.parametrize("call", [
    lambda: kb_entry.run_kb_search("rollup", mode="bm25", kb="nobody"),
    lambda: kb_entry.kb_aggregate(kb="nobody"),
    lambda: kb_entry.kb_open(PEER_A, kb="nobody"),
])
def test_an_unknown_kb_is_an_empty_result_never_an_exception(two_kbs, call):
    """P3 at the tool boundary. `kb=` is a string a host model typed, so a name that does not
    resolve is bad input, not a crash — and the answer has to name the ones that do resolve, or
    the host has nothing to correct itself with."""
    out = call()
    if "hits" in out:
        assert out["hits"] == []
    notice = out["notices"][0] if "notices" in out else out
    text = notice.get("message") or notice.get("error")
    assert "nobody" in text and "peer" in text


def test_a_registered_peer_whose_file_is_gone_reports_that_it_is_gone(two_kbs):
    """The other half of 'cannot be read right now': registered, but the export moved or was
    deleted. Same empty envelope, and the message distinguishes it from a name that was never
    registered — otherwise the fix looks like 'register it' when it is already registered."""
    _reader_home, export_path = two_kbs
    export_path.unlink()
    out = kb_entry.run_kb_search("rollup", mode="bm25", kb="peer")
    assert out["hits"] == []
    assert out["notices"][0]["code"] == "kb_unknown"
    assert "could not be opened" in out["notices"][0]["message"]


# ── 2. routing ───────────────────────────────────────────────────────────────────

def _ids(out):
    return sorted(h["atom_id"] for h in out["hits"])


def test_search_routes_to_the_store_the_caller_named(two_kbs):
    local_default = kb_entry.run_kb_search("react dashboard rollup proof", mode="bm25", k=8)
    local_named = kb_entry.run_kb_search("react dashboard rollup proof", mode="bm25", k=8, kb="me")
    foreign = kb_entry.run_kb_search("react dashboard rollup proof", mode="bm25", k=8, kb="peer")

    assert _ids(local_default) == _ids(local_named) == [LOCAL_B, LOCAL_A]
    assert _ids(foreign) == [PEER_B, PEER_A]
    assert not set(_ids(local_default)) & set(_ids(foreign))


def test_open_and_aggregate_route_the_same_way(two_kbs):
    foreign = kb_entry.kb_open(PEER_A, kb="peer")
    assert foreign["raw"] == "crypto rollup proof systems"
    assert foreign["kb"] == "peer"

    # An atom id means nothing outside its own store. Opening a peer's id locally is a miss, not
    # a silent hand-back of something else — this is the known, accepted round-trip contract.
    assert kb_entry.kb_open(PEER_A)["error"] == "not found"

    assert kb_entry.kb_aggregate()["total"] == 2
    theirs = kb_entry.kb_aggregate(kb="peer")
    assert theirs["total"] == 2
    assert theirs["kb"] == "peer"
    # Counted off the OWNER's confirmations, which is why `oracles` is carried into an export.
    assert theirs["trusted_atoms"] == 1
    assert kb_entry.kb_aggregate()["trusted_atoms"] == 0


def test_a_foreign_body_has_no_path_on_this_machine(two_kbs):
    """`raw_path` is a statement about THIS filesystem, and a peer's snapshots were never written
    here — `resolve_ref` would rehydrate the ref against the reader's own home and hand back a
    path to nothing. `raw` carries the body either way."""
    assert kb_entry.kb_open(PEER_A, kb="peer")["raw_path"] is None
    assert kb_entry.kb_open(LOCAL_A)["raw_path"] is not None


# ── 3. provenance (X2) ───────────────────────────────────────────────────────────

def test_every_hit_says_which_knowledge_base_it_came_from(two_kbs):
    """On the HIT, not only the envelope: a host that lifts one card into a document carries the
    attribution with it. Local hits carry "me" for the same reason — provenance a host has to
    infer from a MISSING key is provenance it will get wrong."""
    local = kb_entry.run_kb_search("react dashboard", mode="bm25", k=8)
    foreign = kb_entry.run_kb_search("rollup proof", mode="bm25", k=8, kb="peer")

    assert {h["kb"] for h in local["hits"]} == {"me"}
    assert {h["kb"] for h in foreign["hits"]} == {"peer"}
    assert local["trace"]["kb"] == "me" and foreign["trace"]["kb"] == "peer"

    codes = {n["code"] for n in foreign["notices"]}
    assert "foreign_kb" in codes
    said = next(n for n in foreign["notices"] if n["code"] == "foreign_kb")["message"]
    assert "The Peer's KB" in said
    assert "foreign_kb" not in {n["code"] for n in local["notices"]}


def test_an_unresolvable_handle_says_whose_knowledge_base_lacks_them(two_kbs):
    """'Nobody matching X is in this knowledge base' is ambiguous the moment there are two, and
    the reader is the one who has to tell them apart."""
    out = kb_entry.run_kb_search("rollup", mode="bm25", who="nosuchperson", kb="peer")
    notice = next(n for n in out["notices"] if n["code"] == "who_unresolved")
    assert "The Peer's KB" in notice["message"]
    assert notice["kb"] == "peer"


def test_notices_about_the_readers_own_install_stay_out_of_a_foreign_read(two_kbs, monkeypatch):
    """`rails_budget_paused` reports the READER's paused collection rails. Inside somebody else's
    results it says their corpus is missing recent material, which is not true and not knowable
    from here."""
    from pipeline.kb import rail_budgets
    monkeypatch.setattr(rail_budgets, "paused_today",
                        lambda: [{"rail": "x", "label": "X backfill", "spent_usd": 1.0,
                                  "ceiling_usd": 1.0}])
    local = kb_entry.run_kb_search("react", mode="bm25", kb="me")
    foreign = kb_entry.run_kb_search("rollup", mode="bm25", kb="peer")
    assert "rails_budget_paused" in {n["code"] for n in local["notices"]}
    assert "rails_budget_paused" not in {n["code"] for n in foreign["notices"]}


def test_an_empty_peer_does_not_tell_the_reader_to_onboard(two_kbs, tmp_path):
    """`onboard` sets up the READER's install and would not put an atom in somebody else's store,
    so the local sentence sends them to fix the wrong thing.

    The peer is emptied by hand because `build_export` refuses to build an export of a store that
    has never ingested — an empty peer is a STATE a reader can meet (a corpus the owner cleared
    out), not an export the builder will produce."""
    _reader_home, export_path = two_kbs
    hollow = tmp_path / "hollow.db"
    hollow.write_bytes(export_path.read_bytes())
    conn = sqlite3.connect(hollow)
    conn.execute("DELETE FROM atoms")
    conn.commit()
    conn.close()
    peers.add("hollow", hollow)

    out = kb_entry.run_kb_search("anything", mode="bm25", kb="hollow")
    notice = out["notices"][0]
    assert notice["code"] == "store_empty" and notice["kb"] == "hollow"
    assert "onboard" not in notice["message"]


def test_exporting_a_store_that_never_ingested_refuses(tmp_path, monkeypatch):
    """The precondition said once, on the owner's machine. Without it the build died on a SQL
    syntax error two hundred lines in, because `kb_meta` — which the allow-list calls REQUIRED —
    is written on first ingest and simply does not exist yet."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "fresh"))
    schema.connect().close()
    with pytest.raises(ValueError, match="never ingested"):
        export.build_export(tmp_path / "nothing.db")
    assert not (tmp_path / "nothing.db.building").exists()


def test_the_frontier_notice_does_not_ride_a_foreign_result(two_kbs, monkeypatch):
    """Frontier's queue is the reader's OWN staged artifacts. Riding it on a foreign search tells
    them their backlog grew because they looked at somebody else's KB."""
    from mcp_server import atoms_tools

    class _Mcp:
        def __init__(self):
            self.tools = {}

        def tool(self):
            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn
            return deco

    monkeypatch.setattr("mcp_server.frontier_tools.notice",
                        lambda: {"message": "3 new artifacts staged"}, raising=False)
    atoms_tools._reset_session()
    mcp = _Mcp()
    atoms_tools.register_atoms_tools(mcp)
    assert "frontier" in mcp.tools["search"]("react", mode="bm25")

    atoms_tools._reset_session()
    assert "frontier" not in mcp.tools["search"]("rollup", mode="bm25", kb="peer")


# ── 4. no write back (X1, I3) ────────────────────────────────────────────────────

def test_a_foreign_session_writes_nothing_anywhere(two_kbs):
    """X1 is satisfied by CONSTRUCTION, so this pins it rather than implementing it: a foreign read
    opens a DIFFERENT database, and there is no write path from it into the reader's `atoms` — so a
    peer's atom can never reach HUMAN_ATTESTED or seed the reader's standing queries. I3 is the
    other direction, and it is SQLite's job: `open_peer` connects read-only.

    The reader's legitimate route to keep something foreign stays `hopper(source_url)` against
    their own store, and every hit card already carries `source_url`."""
    reader_home, export_path = two_kbs
    conn = schema.connect()
    before = conn.execute("SELECT atom_id, raw_hash, version FROM atoms ORDER BY atom_id")\
                 .fetchall()
    conn.close()
    peer_bytes = export_path.read_bytes()

    kb_entry.run_kb_search("rollup proof", mode="bm25", kb="peer")
    kb_entry.kb_open(PEER_A, kb="peer")
    kb_entry.kb_aggregate(kb="peer")

    conn = schema.connect()
    after = conn.execute("SELECT atom_id, raw_hash, version FROM atoms ORDER BY atom_id")\
                .fetchall()
    peer_ids = conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_id IN (?, ?)",
                            (PEER_A, PEER_B)).fetchone()[0]
    conn.close()
    assert after == before and peer_ids == 0
    assert export_path.read_bytes() == peer_bytes
    assert not (reader_home / "kb_raw" / "x" / f"{PEER_A.replace(':', '_')}.md").exists()


# ── 5. the embedder follows the store ────────────────────────────────────────────

def test_embedder_for_store_takes_its_identity_from_the_store(tmp_path, monkeypatch):
    """The Phase-1 deferral, now due. Locally, a store whose model differs from this install's
    config is a misconfiguration; on a peer's store it is the routine case, because the model is
    a fact the OWNER recorded when they paid to embed their corpus."""
    from pipeline.kb.embed import _resolve_config

    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "h"))
    conn = schema.connect()
    ensure_kb_meta(conn, model="somebody-elses-model", dim=1234,
                   provider=_resolve_config()["provider"], query_instruction="Theirs: ")
    emb = embedder_for_store(conn)
    conn.close()
    assert (emb.model, emb.dim, emb.query_instruction) == \
           ("somebody-elses-model", 1234, "Theirs: ")


def test_embedder_from_meta_takes_a_dict_rather_than_a_connection(monkeypatch):
    """The same rule, reachable without a store. A remote reader has no file to open — their
    peer is an HTTPS prefix — so the meta they follow arrives from `GET /v1/kb/{owner}/meta` as
    a dict. Same construction, one layer down; `embedder_for_store` is now the file-shaped
    caller of this."""
    from pipeline.kb.embed import _resolve_config

    emb = embedder_from_meta({"model": "somebody-elses-model", "dim": 1234,
                              "provider": _resolve_config()["provider"],
                              "query_instruction": "Theirs: "})
    assert (emb.model, emb.dim, emb.query_instruction) == \
           ("somebody-elses-model", 1234, "Theirs: ")


def test_embedder_from_meta_refuses_a_provider_this_install_cannot_speak():
    """Raises naming BOTH providers, because the reader has to be told which of the two they can
    change — and the only thing they can change is which install they are on."""
    with pytest.raises(SubspaceError) as e:
        embedder_from_meta({"model": "m", "dim": 8, "provider": "a-provider-nobody-configures",
                            "query_instruction": ""})
    from pipeline.kb.embed import _resolve_config
    assert "a-provider-nobody-configures" in str(e.value)
    assert _resolve_config()["provider"] in str(e.value)


def test_a_provider_this_install_cannot_speak_raises_rather_than_guessing(tmp_path, monkeypatch):
    """`provider` is the one axis that cannot follow the store: it is paired with `endpoint` and a
    key, and this install has one of each. Guessing an endpoint would produce vectors from some
    other model — confident garbage, not an error."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "h"))
    conn = schema.connect()
    ensure_kb_meta(conn, model="m", dim=8, provider="a-provider-nobody-configures")
    with pytest.raises(SubspaceError):
        embedder_for_store(conn)
    conn.close()


def test_the_query_is_embedded_for_the_store_being_read(two_kbs, monkeypatch, fake_embedder):
    """Which FACTORY each path uses, asserted without paying for a call: the local path keeps
    `get_kb_embedder` and its loud `assert_model`, the foreign path follows the store."""
    seen = []
    monkeypatch.setattr(kb_entry, "get_kb_embedder", lambda: seen.append("local") or fake_embedder)
    monkeypatch.setattr(kb_entry, "embedder_for_store",
                        lambda conn: seen.append("store") or fake_embedder)
    monkeypatch.setattr(kb_entry, "assert_model", lambda conn, emb, **kw: None)

    kb_entry.run_kb_search("react dashboard", mode="semantic")
    assert seen == ["local"]
    kb_entry.run_kb_search("rollup proof", mode="semantic", kb="peer")
    assert seen == ["local", "store"]


def test_a_peer_whose_vectors_this_install_cannot_reach_degrades_to_the_keyword_arm(
        two_kbs, monkeypatch):
    """Half a search plus a sentence saying which half is missing beats a stack trace (P3). BM25
    needs no vectors at all, so there is a real answer to give. The LOCAL path keeps raising —
    there, the same mismatch is this install misconfigured, and a wrong-subspace answer is
    confident garbage rather than a smaller result."""
    def _no_endpoint(conn):
        raise SubspaceError("this knowledge base was embedded on 'x'; your install speaks 'y'")

    monkeypatch.setattr(kb_entry, "embedder_for_store", _no_endpoint)
    out = kb_entry.run_kb_search("rollup proof", mode="semantic", k=8, kb="peer")
    assert _ids(out) == [PEER_B, PEER_A]
    assert out["trace"]["ran"] == "bm25"
    assert "vector_arm_unavailable" in {n["code"] for n in out["notices"]}
