"""Changes-only uploads, end to end: `publish.publish` against the real service in-process.

The first push sends the whole export, the next sends a delta, and the served copy then holds
exactly what a fresh export of the owner's store holds: the same rows under the same rowids, a
search index in step with its bodies, and the digest it records. Every way a delta can be
refused that a whole upload fixes (a mismatch, a service ahead of the owner, a service older than
the route) ends in a whole upload in the same push, and a refusal a whole upload would not fix is
reported instead.
"""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from opyt_core import service_http
from opyt_core.paths import opyt_path
from pipeline.kb import schema
from pipeline.sharing import export, fingerprint, publish, push_gate
from pipeline.sharing.account import Account
from service import uploads
from tests.kb.test_export import _add
from tests.opyt_core.conftest import URL


def _push(publisher) -> dict:
    return publish.publish(Account(URL, publisher.svc.owner_token))


def _ingest(publisher, atom_id, text="a library of autonomous agent tools added later"):
    conn = schema.connect()
    _add(conn, atom_id, "github", "artifact", "github:later", ["ai-agents"], text)
    conn.close()


def _served(publisher):
    with publisher.on_service():
        return uploads.export_path(publisher.svc.owner)


def _rows(path) -> dict:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {t: conn.execute(f"SELECT rowid, * FROM {t} ORDER BY rowid").fetchall()
                for t in export.TABLES}
    finally:
        conn.close()


def _assert_served_is_a_fresh_export(publisher, tmp_path):
    """Built from the owner's store as it stands, compared row by row; the served index checked
    against its bodies by FTS5; and the copy's recorded digest that of the fresh build."""
    fresh = tmp_path / "fresh.db"
    export.build_export(fresh)
    served = _served(publisher)
    assert _rows(served) == _rows(fresh)
    conn = sqlite3.connect(served)
    try:
        conn.execute("INSERT INTO bodies_fts(bodies_fts, rank) VALUES ('integrity-check', 1)")
        assert fingerprint.stamped(conn) == fingerprint.of_copy(fresh).digest
    finally:
        conn.close()


def _posts(monkeypatch):
    """Every POST the owner sends, by path, while still sending it."""
    sent, real = [], service_http.requests.post

    def post(url, **kw):
        sent.append(url.removeprefix(URL))
        return real(url, **kw)

    monkeypatch.setattr(service_http.requests, "post", post)
    return sent


def _answer(monkeypatch, status, detail):
    """The delta route answers `status`; every other route is the real service. Returns the paths
    posted to, in order."""
    sent, real = [], service_http.requests.post

    def post(url, **kw):
        sent.append(url.removeprefix(URL))
        if url.endswith("/delta"):
            return SimpleNamespace(status_code=status, text=detail,
                                   json=lambda: {"detail": detail})
        return real(url, **kw)

    monkeypatch.setattr(service_http.requests, "post", post)
    return sent


# ── whole, then changes only ─────────────────────────────────────────────────────

def test_the_first_push_is_whole_and_the_next_is_a_delta(publisher, tmp_path, monkeypatch):
    sent = _posts(monkeypatch)
    first = _push(publisher)
    assert first["status"] == "ok" and first["sent"] == "whole"
    assert push_gate.read_held().digest == first["digest"]
    _assert_served_is_a_fresh_export(publisher, tmp_path)

    _ingest(publisher, "github:later/one")
    second = _push(publisher)
    assert second["status"] == "ok" and second["sent"] == "delta"
    assert second["bytes"] * 2 < first["bytes"]
    assert sent == [f"/v1/upload/{publisher.svc.owner}",
                    f"/v1/upload/{publisher.svc.owner}/delta"]
    _assert_served_is_a_fresh_export(publisher, tmp_path)
    assert push_gate.read_held().digest == second["digest"]
    assert not opyt_path("tmp", "export-push.db").exists()
    assert not opyt_path("tmp", "export-push.delta").exists()


def test_an_export_that_did_not_change_sends_nothing(publisher, monkeypatch):
    """The store moved (the gate said push) but nothing an export carries did."""
    assert _push(publisher)["sent"] == "whole"
    sent = _posts(monkeypatch)
    res = _push(publisher)
    assert res["status"] == "ok" and res["sent"] == "nothing" and res["bytes"] == 0
    assert sent == []


def test_several_deltas_in_a_row_stay_a_fresh_export(publisher, tmp_path):
    assert _push(publisher)["sent"] == "whole"
    for i in range(3):
        _ingest(publisher, f"github:later/{i}", f"agent framework number {i} for crypto proofs")
        assert _push(publisher)["sent"] == "delta"
        _assert_served_is_a_fresh_export(publisher, tmp_path)


def test_a_delta_of_half_the_export_or_more_goes_whole(publisher, tmp_path):
    assert _push(publisher)["sent"] == "whole"
    _ingest(publisher, "github:later/big",
            " ".join(f"agent library chapter {i} about tools and frameworks." for i in range(4000)))
    assert _push(publisher)["sent"] == "whole"
    _assert_served_is_a_fresh_export(publisher, tmp_path)


@pytest.mark.parametrize("delta_bytes, export_bytes, worth", [
    (1, 100, True),
    (49, 100, True),
    (50, 100, False),
    (90, 100, False),
])
def test_a_delta_is_worth_sending_only_under_half_the_export(delta_bytes, export_bytes, worth):
    assert publish.worth_a_delta(delta_bytes, export_bytes) is worth


# ── what a whole upload fixes ────────────────────────────────────────────────────

def test_a_delta_that_misses_a_change_is_refused_and_the_push_goes_whole(publisher, tmp_path,
                                                                        monkeypatch):
    """A bug on the owner's side: the delta leaves out the new atom's body. The service's digest
    check refuses it before anything is written, and the same push sends the whole export."""
    assert _push(publisher)["sent"] == "whole"
    _ingest(publisher, "github:later/one")
    real = fingerprint.diff
    monkeypatch.setattr(fingerprint, "diff", lambda old, new: fingerprint.Changes(
        {**real(old, new).write, "bodies": []}, real(old, new).delete))
    sent = _posts(monkeypatch)

    res = _push(publisher)
    assert res["status"] == "ok" and res["sent"] == "whole"
    assert sent == [f"/v1/upload/{publisher.svc.owner}/delta", f"/v1/upload/{publisher.svc.owner}"]
    _assert_served_is_a_fresh_export(publisher, tmp_path)


def test_a_service_ahead_of_the_owner_gets_the_whole_export(publisher, tmp_path, monkeypatch):
    """A pass that lost its lease after the service accepted a push never recorded it, so the
    owner cuts the next delta against an older copy. The base check refuses it."""
    assert _push(publisher)["sent"] == "whole"
    older = push_gate.read_held()
    _ingest(publisher, "github:later/one")
    assert _push(publisher)["sent"] == "delta"
    push_gate.write_held(older)

    _ingest(publisher, "github:later/two")
    sent = _posts(monkeypatch)
    res = _push(publisher)
    assert res["status"] == "ok" and res["sent"] == "whole"
    assert sent == [f"/v1/upload/{publisher.svc.owner}/delta", f"/v1/upload/{publisher.svc.owner}"]
    _assert_served_is_a_fresh_export(publisher, tmp_path)


def test_a_service_older_than_the_route_gets_the_whole_export(publisher, tmp_path, monkeypatch):
    """An install updated before the service: its delta route answers 404, as a deployed service
    without it does (the connector mounted at `/` answers 404 too, measured)."""
    assert _push(publisher)["sent"] == "whole"
    _ingest(publisher, "github:later/one")
    sent = _answer(monkeypatch, 404, "Not Found")

    res = _push(publisher)
    assert res["status"] == "ok" and res["sent"] == "whole"
    assert sent == [f"/v1/upload/{publisher.svc.owner}/delta", f"/v1/upload/{publisher.svc.owner}"]
    _assert_served_is_a_fresh_export(publisher, tmp_path)


def test_a_refusal_a_whole_upload_would_not_fix_is_reported(publisher, monkeypatch):
    """A patched copy over the cap is refused with 413, and a whole upload would be too: the push
    reports it and the next pass tries again, rather than sending hundreds of MB to be refused."""
    assert _push(publisher)["sent"] == "whole"
    held = push_gate.read_held()
    _ingest(publisher, "github:later/one")
    sent = _answer(monkeypatch, 413, "larger than the 500 MB this service serves")

    res = _push(publisher)
    assert res["status"] == "upload_failed" and "413" in res["message"]
    assert sent == [f"/v1/upload/{publisher.svc.owner}/delta"]
    assert push_gate.read_held() == held


def test_unsharing_forgets_what_the_service_held(publisher):
    """`unshare` deletes the copy; a later share must send the whole export, not a delta against
    a copy that is gone."""
    assert _push(publisher)["sent"] == "whole"
    assert push_gate.read_held() is not None
    push_gate.forget_publication()
    assert push_gate.read_held() is None
    assert _push(publisher)["sent"] == "whole"
