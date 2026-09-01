"""The push rail — the served copy stays current without anybody asking it to.

R3's LAZY GATE is the whole subject: push when somebody has READ since the last push AND the
local store has CHANGED since it. Both terms, and the truth table below is the test. Demand alone
would ship an identical 117 MB every time anyone reads; change alone would push for readers who
do not exist.

Run against the REAL service in-process, through the same `publisher` fixture `opyt-push` uses,
because the second thing these tests hold is that the rail and the CLI are one implementation. A
rail with its own upload sequence would be a second copy of the code every reader's fidelity
depends on.
"""
from __future__ import annotations

import sqlite3

from opyt_core import push
from opyt_core.paths import opyt_path
from pipeline.kb import push_catchup, schema
from service import store, uploads
from tests.kb.test_export import _add

NEW = "github:pushed/by-the-rail"


def _read_once(publisher):
    """Demand, made true the only way it can be — a reader actually reading."""
    with publisher.on_service():
        publisher.svc.client.post(f"/v1/kb/{publisher.svc.owner}/search",
                                  json={"query": "agent"}, headers=publisher.svc.reader_hdr)


def _ingest_one(publisher, atom_id=NEW):
    """Change, made true the only way it can be — an atom arriving."""
    conn = schema.connect()
    _add(conn, publisher.emb, atom_id, "github", "artifact", "github:railed", ["ai-agents"],
         "a library of autonomous agent tools the rail has not published yet")
    conn.close()


def _served_ids(publisher) -> set[str]:
    with publisher.on_service():
        served = uploads.export_path(publisher.svc.owner)
        conn = sqlite3.connect(f"file:{served}?mode=ro", uri=True)
    try:
        return {r[0] for r in conn.execute("SELECT atom_id FROM atoms")}
    finally:
        conn.close()


def _mark_published(publisher):
    """Put the rail in its steady state: an export already served, and a watermark matching the
    store as it stands. The fixture uploads before the rail exists, so the watermark is missing."""
    push_catchup.write_watermark(push_catchup.store_position())


# ── the gate's truth table ───────────────────────────────────────────────────────

def test_an_install_that_never_shared_is_a_silent_no_op(publisher, monkeypatch):
    """The absent token IS the consent gate. This rail has no marker of its own, because the yes
    it would ask for was already given, explicitly, when the user shared at all."""
    monkeypatch.delenv("OPYT_SERVICE_TOKEN", raising=False)
    monkeypatch.setattr("pipeline.credentials.get_credential", lambda service: None)

    assert push_catchup.run_push_catchup()["status"] == "not_shared"


def test_a_first_publish_that_never_happened_is_pushed_regardless_of_demand(publisher,
                                                                            monkeypatch):
    """The retry net, and it is not a special case for its own sake: without it a failed first
    publish is PERMANENT, because demand can never become true against an export that does not
    exist. Self-service makes this ordinary — `share` registers, then publishes, and the second
    call is the one that can die. Nobody has read and nothing has changed, and it still pushes."""
    with publisher.on_service():
        fresh = publisher.svc.client.post("/v1/register", json={"label": "Leo"}).json()
        assert store.publish_demand(fresh["owner"])["last_upload_at"] is None
    monkeypatch.setattr("pipeline.credentials.get_credential", lambda s: fresh["token"])

    assert push_catchup.run_push_catchup()["status"] == "ok"
    with publisher.on_service():
        assert uploads.export_path(fresh["owner"]).exists()


def test_unpublishing_does_not_look_like_a_first_publish(publisher):
    """The bypass above must not become a way for the rail to silently RE-publish something the
    owner just stopped sharing. `clear_upload` zeroes the bytes and keeps `last_published_at`, so
    an unpublished knowledge base reads as published-and-unread rather than never-published — and
    demand can never become true again, because unpublish deleted every reader."""
    _mark_published(publisher)
    with publisher.on_service():
        publisher.svc.client.post("/v1/unpublish", headers=publisher.svc.owner_hdr)
    _ingest_one(publisher)

    assert push_catchup.run_push_catchup()["status"] == "no_demand"
    with publisher.on_service():
        assert not uploads.export_path(publisher.svc.owner).exists()


def test_no_demand_skips_even_when_the_store_moved(publisher):
    """The term that keeps this cheap. Without it every ingest ships 117 MB to nobody."""
    _mark_published(publisher)
    _ingest_one(publisher)

    assert push_catchup.run_push_catchup()["status"] == "no_demand"
    assert NEW not in _served_ids(publisher)


def test_no_change_skips_even_when_somebody_read(publisher):
    """The term that makes the gate self-quieting. Demand alone never goes false on its own — a
    reader who reads twice would ship the identical file twice."""
    _mark_published(publisher)
    _read_once(publisher)

    assert push_catchup.run_push_catchup()["status"] == "no_change"


def test_both_terms_true_publishes(publisher):
    """The one case that pushes, and it is asserted on the SERVED file rather than on a status:
    the point of the rail is that a reader's next read sees the new atom."""
    _mark_published(publisher)
    _read_once(publisher)
    _ingest_one(publisher)

    assert push_catchup.run_push_catchup()["status"] == "ok"
    assert NEW in _served_ids(publisher)


def test_the_gate_goes_quiet_again_after_the_push_consumes_the_demand(publisher):
    """Self-quieting is the property that ruled out the token-existence gate, which has no
    consuming step: a read makes demand true, the push consumes it, and it goes false again."""
    _mark_published(publisher)
    _read_once(publisher)
    _ingest_one(publisher)
    assert push_catchup.run_push_catchup()["status"] == "ok"

    _ingest_one(publisher, "github:pushed/and-again")
    assert push_catchup.run_push_catchup()["status"] == "no_demand"


# ── the watermark ────────────────────────────────────────────────────────────────

def test_the_watermark_is_written_only_on_success(publisher, monkeypatch):
    """A watermark written before or regardless would make a failed push look like a completed
    one — and the store would then have to change AGAIN before anything retried."""
    _mark_published(publisher)
    before = push_catchup.read_watermark()
    _read_once(publisher)
    _ingest_one(publisher)

    real, fail = push.publish, [True]
    monkeypatch.setattr(push, "publish", lambda *a, **kw: (
        {"status": "upload_failed", "message": "nope"} if fail[0] else real(*a, **kw)))

    assert push_catchup.run_push_catchup()["status"] == "upload_failed"
    assert push_catchup.read_watermark() == before

    fail[0] = False
    assert push_catchup.run_push_catchup()["status"] == "ok"
    assert push_catchup.read_watermark() != before


def test_a_re_embed_moves_the_watermark_without_any_new_atom(publisher):
    """Why the watermark carries `MAX(chunks.chunk_id)` and not just `MAX(atoms.ingested_at)`.
    A re-chunk writes new chunk rows for atoms that already exist, so an `ingested_at`-only
    comparison would call the store unchanged and never publish the new vectors."""
    _mark_published(publisher)
    _read_once(publisher)
    conn = schema.connect()
    try:
        row = conn.execute("SELECT atom_id, text FROM chunks LIMIT 1").fetchone()
        conn.execute("INSERT INTO chunks (atom_id, seq, text) VALUES (?, 99, ?)",
                     (row["atom_id"], row["text"]))
        conn.commit()
    finally:
        conn.close()

    assert push_catchup.run_push_catchup()["status"] == "ok"


def test_an_unwritable_watermark_costs_a_redundant_push_not_a_missed_one(publisher,
                                                                        monkeypatch):
    """Fail-safe direction. A marker that cannot be written leaves the rail thinking the store
    moved, which pushes again next session — bandwidth, not a stale copy."""
    _mark_published(publisher)
    _read_once(publisher)
    _ingest_one(publisher)
    monkeypatch.setattr(push_catchup.Path, "write_text",
                        lambda self, *a, **kw: (_ for _ in ()).throw(OSError("read-only")))

    assert push_catchup.run_push_catchup()["status"] == "ok"


# ── the seam ─────────────────────────────────────────────────────────────────────

def test_the_rail_calls_the_same_publish_the_cli_does(publisher, monkeypatch):
    """One implementation, two callers. A rail with its own build/upload/verify sequence would be
    a second copy of the code every reader's fidelity depends on."""
    _mark_published(publisher)
    _read_once(publisher)
    _ingest_one(publisher)
    called = {}

    real = push.publish

    def spy(*a, **kw):
        called["hit"] = True
        return real(*a, **kw)

    monkeypatch.setattr(push, "publish", spy)
    assert push_catchup.run_push_catchup()["status"] == "ok"
    assert called["hit"] is True


def test_a_dead_service_exits_quietly(publisher, monkeypatch):
    """Fail-safe: an unreachable service is not this rail's problem to solve, and it must never
    be the reason a session is worse. No raise, no watermark, and a status naming the cause."""
    _mark_published(publisher)
    before = push_catchup.read_watermark()

    def dead(*a, **kw):
        raise push.RequestException("connection refused")

    monkeypatch.setattr(push.requests, "get", dead)

    res = push_catchup.run_push_catchup()
    assert res["status"] == "unreachable"
    assert push_catchup.read_watermark() == before


def test_force_publishes_with_neither_term_true(publisher):
    """The escape hatch for a person who just asked. Nobody has read and nothing has changed."""
    _mark_published(publisher)
    assert push_catchup.run_push_catchup()["status"] == "no_demand"
    assert push_catchup.run_push_catchup(force=True)["status"] == "ok"


def test_the_watermark_honours_a_sandboxed_home(publisher):
    """Resolved at call time, not bound at import: a marker bound at import resolves to the real
    `~/.opyt` under a sandboxed `$OPYT_HOME`, which is how a test writes to a user's real home."""
    _mark_published(publisher)
    assert push_catchup._watermark_path() == opyt_path("push_watermark")
    assert str(publisher.home) in str(push_catchup._watermark_path())
