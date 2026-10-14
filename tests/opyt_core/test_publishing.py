"""Publishing — the served copy stays current without anybody asking it to (`publishing.py`, the
scrapers process's `publishing` quota).

The gate is the subject: push when the local store has CHANGED since the last push, whether or
not anybody read, and never once the owner withdrew the copy. The truth table below is the test.

Run against the REAL service in-process, through the same `publisher` fixture `test_push.py`
uses, because the second thing these tests hold is that the pass adds a GATE and nothing else —
`publish.publish` is the implementation and the pass decides only whether to call it. A pass with
its own upload sequence would be a second copy of the code every reader's fidelity depends on.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pytest

from opyt_core import service_http
from opyt_core.paths import opyt_path
from pipeline.forget import atoms as forget
from pipeline.kb import schema
from pipeline.sharing import client, publish, publishing, push_gate
from pipeline.timeparse import utc_now
from pipeline.worker.home import LOCAL_HOME_ID, WORKER_DB_ENV, WORKER_HOME_ID_ENV
from pipeline.worker.job_queue import RailJobStore
from pipeline.worker.quota_loop.choose import Picked
from pipeline.worker.quota_loop.loop import Fetched, QuotaLoop
from pipeline.worker.quota_loop.retries import PLATFORM_PAUSES_S, Outages
from pipeline.worker.quota_loop.serial import Serial
from pipeline.worker.quota_loop.source import Unit
from pipeline.worker.registry import LINE, RAILS
from service import uploads
from tests.kb.test_export import _add

NEW = "github:pushed/by-a-pass"


def _ingest_one(publisher, atom_id=NEW):
    """Change, made true the only way it can be — an atom arriving."""
    conn = schema.connect()
    _add(conn, atom_id, "github", "artifact", "github:railed", ["ai-agents"],
         "a library of autonomous agent tools no pass has published yet")
    conn.close()


def _served_ids(publisher) -> set[str]:
    with publisher.on_service():
        served = uploads.export_path(publisher.svc.owner)
        conn = sqlite3.connect(f"file:{served}?mode=ro", uri=True)
    try:
        return {r[0] for r in conn.execute("SELECT atom_id FROM atoms")}
    finally:
        conn.close()


def _pass(at: datetime | None = None) -> dict:
    """One publishing pass, as the loop runs it (fetch, then record), whether or not it is due."""
    src = publishing.Publications()
    unit = Unit("publishing:pass", at or utc_now())
    fetched = src.fetch(None, unit, None)
    src.record(None, unit, fetched)
    return fetched.result


def _mark_published(publisher):
    """Put publishing in its steady state: an export already served, and a watermark matching the
    store as it stands. The fixture uploads before any pass, so the watermark is missing."""
    push_gate.write_watermark(push_gate.store_position())


# ── the gate's truth table ───────────────────────────────────────────────────────

def test_an_install_that_never_shared_is_offered_no_pass(publisher, monkeypatch):
    """The absent token IS the consent gate. Publishing has no marker of its own, because the yes
    it would ask for was already given, explicitly, when the user shared at all."""
    monkeypatch.delenv("OPYT_SERVICE_TOKEN", raising=False)
    monkeypatch.setattr("pipeline.credentials.get_credential", lambda service: None)

    src, now = publishing.Publications(), utc_now()
    assert src.next_unit(None, now) is None and src.next_due(None, now) is None


def test_a_first_publish_that_died_is_retried(publisher, monkeypatch):
    """The retry net: `share` registers, then owes the first push, and the push is the call that
    can die. The debt stays until a push lands, so the next pass publishes with nothing changed."""
    with publisher.on_service():
        fresh = publisher.svc.client.post("/v1/register", json={"label": "Leo"}).json()
    monkeypatch.setattr("pipeline.credentials.get_credential", lambda s: fresh["token"])
    push_gate.owe()

    assert _pass()["status"] == "ok"
    with publisher.on_service():
        assert uploads.export_path(fresh["owner"]).exists()


def test_a_withdrawn_copy_is_not_published_again(publisher):
    """`unshare` deletes the copy and the watermark (`forget_publication`); a store that moves
    afterwards must not silently re-publish what the owner just stopped sharing. (The service
    deleted the owner token too, `service/deletion.py`; the pass never gets as far as asking.)"""
    _mark_published(publisher)
    with publisher.on_service():
        publisher.svc.client.post("/v1/unpublish", headers=publisher.svc.owner_hdr)
    push_gate.forget_publication()
    _ingest_one(publisher)

    assert _pass()["status"] == "not_published"
    with publisher.on_service():
        assert not uploads.export_path(publisher.svc.owner).exists()


def test_a_moved_store_publishes_without_anybody_reading(publisher):
    """Asserted on the SERVED file rather than on a status: the point of publishing is that a
    reader's next read sees the new atom."""
    _mark_published(publisher)
    _ingest_one(publisher)

    assert _pass()["status"] == "ok"
    assert NEW in _served_ids(publisher)


def test_an_unchanged_store_skips_without_asking_the_service(publisher, monkeypatch):
    """A pass runs every hour, so a quiet one costs one local read and no request."""
    _mark_published(publisher)

    def no_network(*a, **kw):
        raise AssertionError("a pass on an unchanged store asked the service")

    monkeypatch.setattr(service_http.requests, "get", no_network)
    assert _pass()["status"] == "no_change"


def test_the_gate_goes_quiet_once_the_push_lands(publisher):
    _mark_published(publisher)
    _ingest_one(publisher)
    assert _pass()["status"] == "ok"
    assert _pass()["status"] == "no_change"


def test_deleting_an_older_atom_removes_it_from_the_served_copy(publisher):
    conn = schema.connect()
    target = conn.execute("SELECT atom_id FROM bodies ORDER BY body_id LIMIT 1").fetchone()[0]
    conn.execute("UPDATE atoms SET ingested_at = '2000-01-01' WHERE atom_id = ?", (target,))
    conn.commit()
    _mark_published(publisher)
    before = push_gate.read_watermark()
    assert target in _served_ids(publisher)

    assert forget.atom(conn, target, confirm=True)["status"] == "forgotten"
    conn.close()

    after = push_gate.store_position()
    assert after["atoms_at"] == before["atoms_at"]
    assert after["atom_count"] == before["atom_count"] - 1
    assert _pass()["status"] == "ok"
    assert target not in _served_ids(publisher)


# ── the watermark ────────────────────────────────────────────────────────────────

def test_the_watermark_is_written_only_on_success(publisher, monkeypatch):
    """A watermark written before or regardless would make a failed push look like a completed
    one — and the store would then have to change AGAIN before anything retried."""
    _mark_published(publisher)
    before = push_gate.read_watermark()
    _ingest_one(publisher)

    real, fail = publish.publish, [True]
    monkeypatch.setattr(publish, "publish", lambda *a, **kw: (
        {"status": "upload_failed", "message": "nope"} if fail[0] else real(*a, **kw)))

    assert _pass()["status"] == "upload_failed"
    assert push_gate.read_watermark() == before

    fail[0] = False
    assert _pass()["status"] == "ok"
    assert push_gate.read_watermark() != before


def test_a_changed_atom_moves_the_watermark_without_any_new_atom(publisher):
    """Why the watermark carries `MAX(atoms.ingested_at)` and not just the atom count. A
    re-observed atom whose text changed adds no row, so a count-only comparison would call the
    store unchanged and never publish the new body. The corpus is backdated first so the re-store
    is a later second than anything the published watermark saw."""
    conn = schema.connect()
    try:
        conn.execute("UPDATE atoms SET ingested_at = '2000-01-01 00:00:00'")
        conn.commit()
        _mark_published(publisher)
        count = push_gate.read_watermark()["atom_count"]
        _add(conn, "github:root/agentkit", "github", "artifact", "github:root", ["ai-agents"],
             "an autonomous agent framework, rewritten since the last push")
    finally:
        conn.close()

    assert push_gate.store_position()["atom_count"] == count
    assert _pass()["status"] == "ok"
    with publisher.on_service():
        served = uploads.export_path(publisher.svc.owner)
        out = sqlite3.connect(f"file:{served}?mode=ro", uri=True)
    try:
        text = out.execute("SELECT text FROM bodies WHERE atom_id = 'github:root/agentkit'"
                           ).fetchone()[0]
    finally:
        out.close()
    assert "rewritten since the last push" in text


def test_a_rewrite_in_the_same_second_as_the_push_still_moves_the_watermark(publisher):
    """Why the watermark carries `atom_writes` (every `upsert_atom` bumps `version`). An atom
    rewritten within the second the watermark was taken leaves `MAX(ingested_at)` and the count
    where they were, so without it the new body would wait for some later change to be pushed.
    The clock is pinned to the watermark's second by stamping every atom with it."""
    _mark_published(publisher)
    before = push_gate.read_watermark()
    conn = schema.connect()
    try:
        _add(conn, "github:root/agentkit", "github", "artifact", "github:root", ["ai-agents"],
             "an autonomous agent framework, rewritten within the same second")
        conn.execute("UPDATE atoms SET ingested_at = ?", (before["atoms_at"],))
        conn.commit()
    finally:
        conn.close()

    after = push_gate.store_position()
    assert (after["atoms_at"], after["atom_count"]) == (before["atoms_at"], before["atom_count"])
    assert after["atom_writes"] == before["atom_writes"] + 1
    assert _pass()["status"] == "ok"


def test_an_unwritable_watermark_costs_a_redundant_push_not_a_missed_one(publisher,
                                                                        monkeypatch):
    """Fail-safe direction. A marker that cannot be written leaves the pass thinking the store
    moved, which pushes again next session — bandwidth, not a stale copy."""
    _mark_published(publisher)
    _ingest_one(publisher)
    monkeypatch.setattr(push_gate.Path, "write_text",
                        lambda self, *a, **kw: (_ for _ in ()).throw(OSError("read-only")))

    assert _pass()["status"] == "ok"


# ── the seam ─────────────────────────────────────────────────────────────────────

def test_the_pass_calls_the_one_publish(publisher, monkeypatch):
    """One implementation, two callers (the pass, and the operator by hand). A pass with its own
    build/upload/verify sequence would be a second copy of the code every reader's fidelity
    depends on."""
    _mark_published(publisher)
    _ingest_one(publisher)
    called = {}

    real = publish.publish

    def spy(*a, **kw):
        called["hit"] = True
        return real(*a, **kw)

    monkeypatch.setattr(publish, "publish", spy)
    assert _pass()["status"] == "ok"
    assert called["hit"] is True


def test_a_dead_service_exits_quietly(publisher, monkeypatch):
    """Fail-safe: an unreachable service is not this pass's problem to solve, and it must never
    be the reason a session is worse. No raise, no watermark, and a status naming the cause."""
    _mark_published(publisher)
    before = push_gate.read_watermark()
    _ingest_one(publisher)

    def dead(*a, **kw):
        raise service_http.RequestException("connection refused")

    monkeypatch.setattr(service_http.requests, "get", dead)

    res = _pass()
    assert res["status"] == "unreachable"
    assert push_gate.read_watermark() == before


def test_the_watermark_honours_a_sandboxed_home(publisher):
    """Resolved at call time, not bound at import: a marker bound at import resolves to the real
    `~/.opyt` under a sandboxed `$OPYT_HOME`, which is how a test writes to a user's real home."""
    _mark_published(publisher)
    assert push_gate._watermark_path() == opyt_path("push_watermark")
    assert str(publisher.home) in str(push_gate._watermark_path())


# ── when a pass is owed ──────────────────────────────────────────
_NOW = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
_EVERY = publishing.EVERY


def test_publishing_that_never_passed_is_due_now():
    assert publishing.due_at(None, None, _NOW) == _NOW


def test_a_push_owed_since_the_last_pass_is_due_now():
    assert publishing.due_at(_NOW - _EVERY / 2, _NOW - _EVERY / 4, _NOW) == _NOW
    assert publishing.due_at(_NOW - _EVERY / 2, _NOW - _EVERY / 2, _NOW) == _NOW   # same instant


def test_otherwise_a_pass_is_due_every_hour():
    last = _NOW - timedelta(minutes=20)
    assert _EVERY == timedelta(hours=1)
    assert publishing.due_at(last, None, _NOW) == last + _EVERY
    assert publishing.due_at(last, last - _EVERY, _NOW) == last + _EVERY


def test_the_hour_after_a_share_or_connection_passes_every_five_minutes():
    """2026-10-14: a phone opened during onboarding read a copy 118 items behind the Mac, and
    would have until the next hourly pass."""
    last = _NOW - timedelta(minutes=2)
    assert publishing.FAST == timedelta(minutes=5)
    assert publishing.due_at(last, None, _NOW, fast_since=_NOW - timedelta(minutes=30)) == \
        last + publishing.FAST
    assert publishing.due_at(last, None, _NOW, fast_since=_NOW - publishing.FAST_FOR) == \
        last + _EVERY


def test_the_fast_hour_marker_is_read_back(publisher):
    assert push_gate.fast_since() is None
    push_gate.start_fast_hour()
    assert push_gate.fast_since() is not None


def test_an_owed_push_whose_pass_failed_waits_for_the_next_pass(publisher, monkeypatch):
    """The owed marker survives a failed pass (so the push is still owed), but the pass's time is
    newer than the marker, so the next pass waits its turn instead of retrying in a hot loop."""
    _mark_published(publisher)
    _ingest_one(publisher)
    push_gate.owe()
    monkeypatch.setattr(client, "tokens", lambda acct: (_ for _ in ()).throw(
        client.Failed("unreachable", "connection refused")))
    src = publishing.Publications()
    now = utc_now() + timedelta(seconds=1)
    unit = src.next_unit(None, now)
    assert unit is not None
    fetched = src.fetch(None, unit, None)
    assert src.record(None, unit, fetched).status == "unreachable"

    assert push_gate.owed_at() is not None
    later = now + timedelta(minutes=10)
    assert src.next_unit(None, later) is None
    assert src.next_due(None, later) == push_gate.passed_at() + _EVERY


# ── a sharing install is given the line ───────────

def _line_job(publisher):
    return RailJobStore(publisher.home / "rail_jobs.db").get(LOCAL_HOME_ID, LINE)


def test_a_pass_gives_a_sharing_home_the_line_job(publisher, monkeypatch):
    """An install that shares asks the service for the web's tool calls, and the worker launches
    only what has a row. An install that shared before the job existed has none, and nothing it
    does later would write one, so the pass every sharing install runs does. The row is one the
    worker can launch, due now."""
    monkeypatch.delenv(WORKER_DB_ENV, raising=False)
    monkeypatch.delenv(WORKER_HOME_ID_ENV, raising=False)
    _mark_published(publisher)
    assert _line_job(publisher) is None

    assert _pass()["status"] == "no_change"

    job = _line_job(publisher)
    assert LINE in RAILS and job is not None and job.due_at <= time.time()


def test_an_hourly_pass_never_brings_the_line_job_forward(publisher, monkeypatch):
    """The job keeps its own hourly cadence once it has a row. A pass that asked for it every
    time would run it twice an hour on an install with no web writer."""
    monkeypatch.delenv(WORKER_DB_ENV, raising=False)
    monkeypatch.delenv(WORKER_HOME_ID_ENV, raising=False)
    _mark_published(publisher)
    _pass()
    store = RailJobStore(publisher.home / "rail_jobs.db")
    store.finish(store.claim_next(rails=(LINE,)), 0, cadence=RAILS[LINE].cadence)
    scheduled = _line_job(publisher).due_at

    _pass()

    assert _line_job(publisher).due_at == scheduled > time.time()


class _Killed(BaseException):
    """What a stop that kills the process mid-upload looks like from inside the pass."""


def test_a_publication_killed_mid_upload_leaves_no_watermark_and_is_redone(publisher,
                                                                           monkeypatch):
    """Watermark after success (lanes plan, slice 7): a pass that never returned from its upload
    writes nothing, so the push stays owed and the next pass exports and uploads again."""
    _mark_published(publisher)
    before = push_gate.read_watermark()
    _ingest_one(publisher)
    push_gate.owe()

    real = client.upload

    def killed(*a, **kw):
        raise _Killed()

    monkeypatch.setattr(client, "upload", killed)
    src = publishing.Publications()
    now = utc_now() + timedelta(seconds=1)
    unit = src.next_unit(None, now)
    with pytest.raises(_Killed):
        src.fetch(None, unit, None)                  # killed: `record` is never reached
    assert push_gate.read_watermark() == before
    assert push_gate.owed_at() is not None and push_gate.passed_at() is None

    monkeypatch.setattr(client, "upload", real)
    again = src.next_unit(None, now)
    assert again is not None, "the killed publication must be due again at once"
    fetched = src.fetch(None, again, None)
    src.record(None, again, fetched)
    assert fetched.result["status"] == "ok" and NEW in _served_ids(publisher)
    assert push_gate.read_watermark() != before and push_gate.owed_at() is None


def test_a_lost_lease_after_the_upload_skips_the_record(publisher):
    """The loop never calls `record` once the quota's lease is lost, so the watermark stays put
    and the next holder's pass publishes again: a redundant push, never a missed one."""
    _mark_published(publisher)
    before = push_gate.read_watermark()
    _ingest_one(publisher)
    src = publishing.Publications()
    unit = Unit("publishing:pass", utc_now())
    fetched = src.fetch(None, unit, None)
    assert fetched.result["status"] == "ok"

    loop = QuotaLoop(publishing.QUOTA, [src], executor=Serial(), outages=Outages(PLATFORM_PAUSES_S),
                     connect=lambda: sqlite3.connect(":memory:"))
    loop.ctx.lease_lost = lambda: True
    with sqlite3.connect(":memory:") as conn:
        ran = loop.settle(conn, Picked(src, unit), Fetched(fetched))
    assert ran.result == "lease_lost"
    assert push_gate.read_watermark() == before and push_gate.passed_at() is None
