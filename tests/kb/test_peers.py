"""Reading someone else's knowledge base — `kb=` over a registered peer (pipeline/retrieval/peers.py).

An export is fidelity-identical to the live store and self-contained, so a second knowledge
base can be read from it on the same disk, with no network. What these tests hold, in the order
they matter:

  1. Routing. `kb=None`/"me" reads the local store;
     any other name reads that peer's, and the two corpora do not intersect — so a hit can only
     have come from the store the caller asked for.
  2. Provenance. Every hit says which knowledge base it came from, local ones included, and
     a foreign read says so in a notice. An unlabelled foreign atom is just a bookmark.
  3. No write back. A foreign read leaves the reader's atoms untouched and the peer's
     file byte-identical. Neither is enforced by discipline: they are different databases, and a
     peer is opened read-only.
  4. Nothing raises out of a tool. `kb=` is a string a host model typed. Unregistered, or
     registered and missing, returns an empty envelope of the normal shape plus a notice.
"""
from __future__ import annotations

import sqlite3

import pytest

from pipeline.retrieval import reads as kb_entry
from pipeline.kb import schema
from pipeline.retrieval import peers
from pipeline.sharing import export
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.ingest_common import snapshot_hash
from pipeline.oracles import roster


# Disjoint on every axis a hit is identified by — id, author, topic and vocabulary — so no
# assertion below can pass by accident on a store it did not mean to read.
LOCAL_A = "x:local1"
LOCAL_B = "github:mine/dashboard"
PEER_A = "x:peer1"
PEER_B = "github:theirs/rollup"


def _add(conn, atom_id, source_type, what_kind, who_id, topics, snapshot):
    raw_hash = snapshot_hash(snapshot)
    store_atom(conn, dict(
        atom_id=atom_id, source_type=source_type, what_kind=what_kind, who_id=who_id,
        when_ts="2025-04-01", when_precision="day", about_entities=[],
        source_url=f"https://example/{atom_id}", raw_hash=raw_hash,
        description=f"{atom_id} card", entry_mode="user-saved",
        payload={"source_tags": topics, "body_state": "complete", "body_basis": "observed"},
    ), snapshot)


def _local_corpus(conn):
    _add(conn, LOCAL_A, "x", "opinion", "x:user:100", ["web-dev"],
         "a react dashboard for the web")
    _add(conn, LOCAL_B, "github", "artifact", "github:mine", ["web-dev"],
         "react dashboard tools")
    schema.upsert_entity(conn, "x:user:100", name="Mine", profile={"handle": "mine"})
    schema.upsert_entity(conn, "github:mine", name="Mine")
    conn.commit()


def _peer_corpus(conn):
    _add(conn, PEER_A, "x", "opinion", "x:user:200", ["crypto"],
         "crypto rollup proof systems")
    _add(conn, PEER_B, "github", "artifact", "github:theirs", ["crypto"],
         "a rollup proof library")
    schema.upsert_entity(conn, "x:user:200", name="Theirs", profile={"handle": "theirs"})
    schema.upsert_entity(conn, "github:theirs", name="Theirs")
    roster.upsert_oracle(conn, "x:user:200", name="Theirs")
    conn.commit()


@pytest.fixture()
def two_kbs(tmp_path, monkeypatch):
    """A reader's own store, and a peer's export registered in it. Returns (reader_home, export).

    The peer is built under its OWN `$OPYT_HOME` and then exported, which is the real sequence —
    the export is the only thing that crosses, and its snapshot files stay behind in a directory
    the reader never learns the name of."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "peer"))
    conn = schema.connect()
    _peer_corpus(conn)
    conn.close()
    export_path = tmp_path / "peer-export.db"
    export.build_export(export_path)

    reader_home = tmp_path / "reader"
    monkeypatch.setenv("OPYT_HOME", str(reader_home))
    conn = schema.connect()
    _local_corpus(conn)
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

    assert peers.remove("peer") is True
    assert peers.remove("peer") is False          # revocation is just a row delete
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


# ── 1a. two knowledge bases wanting one name ─────────────────────────────────────
#
# `peers.token` is the only copy of a reader bearer token in existence — the service keeps just
# its sha256 and hands the clear text over once, at redemption. So a blind upsert in `add` would
# not merely drop a row, it would destroy an unrecoverable credential, restorable only by a
# fresh grant code from that owner. `location` is what tells a re-registration apart from a
# collision, and these tests pin that distinction.

def test_re_registering_the_same_kb_replaces_its_token_and_label(two_kbs):
    """Same name, same location means
    the same knowledge base registered again — a revoked token being replaced by a fresh grant.
    One row, new credential."""
    where = "https://api.example.com/v1/kb/david"
    assert peers.add("alex", where, "Alex's KB", token="tok-abc") == "alex"
    assert peers.add("alex", where, "Alex renamed", token="tok-def") == "alex"

    assert [p["name"] for p in peers.list_peers() if p["name"].startswith("alex")] == ["alex"]
    assert peers.get("alex")["token"] == "tok-def"
    assert peers.get("alex")["label"] == "Alex renamed"


def test_a_second_kb_under_a_taken_name_is_suffixed_and_the_first_is_untouched(two_kbs):
    """Two different people both called `alex`. The first one's token must survive, because
    nothing else holds a copy of it."""
    peers.add("alex", "https://one.example.com/v1/kb/a1", "First Alex", token="tok-first")
    landed = peers.add("alex", "https://two.example.com/v1/kb/a2", "Second Alex",
                       token="tok-second")

    assert landed == "alex-2"
    assert peers.get("alex")["token"] == "tok-first"
    assert peers.get("alex")["location"] == "https://one.example.com/v1/kb/a1"
    assert peers.get("alex-2")["token"] == "tok-second"


def test_the_returned_name_is_the_one_get_resolves(two_kbs):
    """The return value is the whole reporting channel — `add` never prompts, so a caller that
    ignored it would tell the reader to use a name that resolves to somebody else's KB."""
    for i in range(3):
        landed = peers.add("alex", f"https://h{i}.example.com/v1/kb/a", token=f"t{i}")
        assert peers.get(landed)["token"] == f"t{i}"
    assert [p["name"] for p in peers.list_peers() if p["name"].startswith("alex")] == [
        "alex", "alex-2", "alex-3"]


def test_re_registering_an_already_suffixed_kb_lands_back_on_its_own_row(two_kbs):
    """The suffix rule is applied at every candidate, not just the first. Otherwise a re-redeem
    of the second Alex walks past `alex-2` and mints `alex-3`, then `alex-4` — a new row per
    attempt, with the reader's working name silently going stale."""
    peers.add("alex", "https://one.example.com/v1/kb/a1", token="tok-first")
    peers.add("alex", "https://two.example.com/v1/kb/a2", token="tok-second")

    assert peers.add("alex", "https://two.example.com/v1/kb/a2", token="tok-fresh") == "alex-2"
    assert peers.get("alex-2")["token"] == "tok-fresh"
    assert peers.get("alex-3") is None


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
    lambda: kb_entry.run_kb_search("rollup", kb="nobody"),
    lambda: kb_entry.kb_aggregate(kb="nobody"),
    lambda: kb_entry.kb_open(PEER_A, kb="nobody"),
])
def test_an_unknown_kb_is_an_empty_result_never_an_exception(two_kbs, call):
    """At the tool boundary, `kb=` is a string a host model typed, so a name that does not
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
    out = kb_entry.run_kb_search("rollup", kb="peer")
    assert out["hits"] == []
    assert out["notices"][0]["code"] == "kb_unknown"
    assert "could not be opened" in out["notices"][0]["message"]


def test_a_copy_made_before_whole_item_search_is_answered_as_unavailable(two_kbs):
    """An export from an older Opyt has no `bodies_fts`, so this search cannot read it. It is
    one check where a foreign store is opened, not a second read path: the reader is told the
    copy refreshes on its owner's next push."""
    _reader_home, export_path = two_kbs
    conn = sqlite3.connect(export_path)
    for name in ("bodies_ai", "bodies_ad", "bodies_au"):
        conn.execute(f"DROP TRIGGER {name}")
    conn.execute("DROP TABLE bodies_fts")
    conn.commit()
    conn.close()
    out = kb_entry.run_kb_search("rollup", kb="peer")
    assert out["hits"] == []
    assert out["notices"][0]["code"] == "kb_unknown"
    assert "older Opyt" in out["notices"][0]["message"]


# ── 2. routing ───────────────────────────────────────────────────────────────────

def _ids(out):
    return sorted(h["atom_id"] for h in out["hits"])


def test_search_routes_to_the_store_the_caller_named(two_kbs):
    local_default = kb_entry.run_kb_search("react dashboard rollup proof", k=8)
    local_named = kb_entry.run_kb_search("react dashboard rollup proof", k=8, kb="me")
    foreign = kb_entry.run_kb_search("react dashboard rollup proof", k=8, kb="peer")

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
    # Counted off the owner's confirmations, which is why `oracles` is carried into an export.
    assert theirs["trusted_atoms"] == 1
    assert kb_entry.kb_aggregate()["trusted_atoms"] == 0


# ── 3. provenance ───────────────────────────────────────────────────────────

def test_every_hit_says_which_knowledge_base_it_came_from(two_kbs):
    """On the hit, not only the envelope: a host that lifts one card into a document carries the
    attribution with it. Local hits carry "me" for the same reason — provenance a host has to
    infer from a missing key is provenance it will get wrong."""
    local = kb_entry.run_kb_search("react dashboard", k=8)
    foreign = kb_entry.run_kb_search("rollup proof", k=8, kb="peer")

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
    out = kb_entry.run_kb_search("rollup", who="nosuchperson", kb="peer")
    notice = next(n for n in out["notices"] if n["code"] == "who_unresolved")
    assert "The Peer's KB" in notice["message"]
    assert notice["kb"] == "peer"


def test_an_empty_peer_does_not_tell_the_reader_to_onboard(two_kbs, tmp_path):
    """`onboard` sets up the reader's install and would not put an atom in somebody else's store,
    so the local sentence sends them to fix the wrong thing.

    The peer is emptied by hand because `build_export` refuses to build an export of a store that
    has never ingested — an empty peer is a state a reader can meet (a corpus the owner cleared
    out), not an export the builder will produce."""
    _reader_home, export_path = two_kbs
    hollow = tmp_path / "hollow.db"
    hollow.write_bytes(export_path.read_bytes())
    conn = sqlite3.connect(hollow)
    conn.execute("DELETE FROM atoms")
    conn.commit()
    conn.close()
    peers.add("hollow", hollow)

    out = kb_entry.run_kb_search("anything", kb="hollow")
    notice = out["notices"][0]
    assert notice["code"] == "store_empty" and notice["kb"] == "hollow"
    assert "onboard" not in notice["message"]


def test_exporting_a_store_that_never_ingested_refuses(tmp_path, monkeypatch):
    """The precondition said once, on the owner's machine: a store with no atoms has nothing to
    share. A store with atoms does export (`tests/kb/test_export.py`)."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "fresh"))
    schema.connect().close()
    with pytest.raises(ValueError, match="nothing yet"):
        export.build_export(tmp_path / "nothing.db")
    assert not (tmp_path / "nothing.db.building").exists()


def test_the_latest_notice_does_not_ride_a_foreign_result(two_kbs, monkeypatch):
    """`latest` counts the reader's OWN arrivals. Riding it on a foreign search tells them their
    library grew because they looked at somebody else's KB."""
    from tests import mcp_session

    search = mcp_session.session().tools()["search"]
    monkeypatch.setattr("pipeline.arrivals.notice.notice",
                        lambda conn, now: {"message": "3 new items arrived"})
    assert "latest" in search("react")
    assert "latest" not in search("rollup", kb="peer")


# ── 4. no write back ────────────────────────────────────────────────────

def test_a_foreign_session_writes_nothing_anywhere(two_kbs):
    """This pins a property that holds by construction: a foreign read opens a different
    database, and there is no write path from it into the reader's `atoms` — so a peer's atom can
    never reach HUMAN_ATTESTED or seed the reader's standing queries. The peer's own file is
    protected by SQLite: `open_peer` connects read-only.

    The reader's legitimate route to keep something foreign stays `hopper(source_url)` against
    their own store, and every hit card already carries `source_url`."""
    reader_home, export_path = two_kbs
    conn = schema.connect()
    before = conn.execute("SELECT atom_id, raw_hash, version FROM atoms ORDER BY atom_id")\
                 .fetchall()
    conn.close()
    peer_bytes = export_path.read_bytes()

    kb_entry.run_kb_search("rollup proof", kb="peer")
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
