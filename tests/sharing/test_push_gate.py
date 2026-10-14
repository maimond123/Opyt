"""`push_gate.decide` — one test per row of the gate's table. Plain values: no store, no service."""
from __future__ import annotations

from datetime import datetime, timezone

from pipeline.sharing import push_gate

HERE = {"atoms_at": "2026-10-04 12:00:00", "atom_count": 7, "atom_writes": 9}
MOVED = {**HERE, "atom_count": 8}


def test_a_moved_store_pushes_whether_or_not_anybody_read():
    assert push_gate.decide(MOVED, HERE) is None


def test_an_unchanged_store_is_no_change():
    assert push_gate.decide(HERE, dict(HERE))["status"] == "no_change"


def test_an_unreadable_store_pushes():
    assert push_gate.decide(None, HERE) is None


def test_nothing_served_and_nothing_owed_is_not_published():
    """After `unshare` withdrew the copy: a moved store must not re-publish it."""
    assert push_gate.decide(MOVED, None)["status"] == "not_published"


def test_a_push_owed_with_nothing_served_pushes():
    """A first share, a share after `unshare`, and a first push that died (still owed)."""
    assert push_gate.decide(HERE, None, owed=True) is None


def test_a_push_owed_on_an_unchanged_store_is_no_change():
    assert push_gate.decide(HERE, dict(HERE), owed=True)["status"] == "no_change"


def test_what_a_pass_leaves_behind(tmp_path, monkeypatch):
    """`record`: every pass leaves its time; a push moves the watermark and pays an owed push;
    `no_change` only pays it; a failure or `not_published` changes nothing else, so the next pass
    retries."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    at = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
    assert push_gate.passed_at() is None and push_gate.owed_at() is None
    push_gate.owe()
    push_gate.record("upload_failed", HERE, at)
    push_gate.record("not_published", HERE, at)
    assert push_gate.owed_at() is not None and push_gate.read_watermark() is None
    assert push_gate.passed_at() == at

    push_gate.record("no_change", HERE, at)
    assert push_gate.owed_at() is None and push_gate.read_watermark() is None

    push_gate.owe()
    push_gate.record("ok", HERE, at)
    assert push_gate.owed_at() is None and push_gate.read_watermark() == HERE

    push_gate.owe()
    push_gate.forget_publication()
    assert push_gate.owed_at() is None and push_gate.read_watermark() is None
