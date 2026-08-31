"""What is served changes atomically, and reading it never changes it.

Two invariants that are easy to state and easy to lose. P3: a partial upload is never served —
so an upload killed mid-body must leave the previous export queryable, not a truncated file that
opens and answers wrongly. I3: a reader never writes to somebody else's knowledge base — so the
served file must be byte-identical after a full session of reads.
"""
from __future__ import annotations

import hashlib

import pytest

from pipeline.kb import peers
from service import uploads


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_completed_upload_is_served_immediately_and_peers_points_at_it(svc):
    served = uploads.export_path(svc.owner)
    assert served.exists()
    assert svc.upload["sha256"] == _sha(svc.export) == _sha(served)
    assert svc.upload["bytes"] == svc.export.stat().st_size

    row = next(p for p in peers.list_peers() if p["name"] == svc.owner)
    assert row["location"] == str(served)

    r = svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"},
                        headers=svc.reader_hdr)
    assert r.status_code == 200 and r.json()["hits"]


def test_an_upload_killed_mid_body_leaves_the_previous_export_intact(svc):
    """The rename is the commit, so a half-written upload has nothing to roll back.

    Driven through `Receiver` rather than through HTTP, and that is not a shortcut: `TestClient`
    buffers a request body before handing it to the app, so an exception from the body iterator
    is raised in the transport and NO chunk ever reaches the handler — measured. A test that
    posted a dying generator would pass without the interesting bytes ever being written, which
    is worse than not testing it. Driving the receiver directly is the one way to get a genuinely
    HALF-WRITTEN file on disk. The HTTP half of the property has its own test below.
    """
    served = uploads.export_path(svc.owner)
    before = _sha(served)
    good = svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"},
                           headers=svc.reader_hdr).json()

    rx = uploads.Receiver(svc.owner)
    rx.write(b"SQLite format 3\x00" + b"\x00" * 65536)   # a plausible, truncated database
    assert rx.tmp.exists() and rx.tmp.stat().st_size > 0, "nothing was actually written"
    assert _sha(served) == before, "the served file changed before the commit"
    rx.abort()

    assert not rx.tmp.exists()
    assert _sha(served) == before

    after = svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"},
                            headers=svc.reader_hdr).json()
    assert after["hits"] == good["hits"]


def test_a_request_that_fails_before_the_handler_serves_nothing_new(svc):
    """The HTTP half: a body that dies takes the request down and leaves no scratch file and no
    change to what is served. This is what `TestClient` CAN drive — the failure lands before the
    handler rather than inside it — so it pins the endpoint's behavior, not the receiver's."""
    served = uploads.export_path(svc.owner)
    before = _sha(served)

    def dies():
        yield b"SQLite format 3\x00"
        raise ConnectionError("the client went away")

    with pytest.raises(BaseException):
        svc.client.post(f"/v1/upload/{svc.owner}", content=dies(), headers=svc.owner_hdr)

    assert _sha(served) == before
    assert not served.with_name(f"{svc.owner}.db.uploading").exists()
    assert svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"},
                           headers=svc.reader_hdr).status_code == 200


def test_a_second_upload_replaces_rather_than_merges(svc, tmp_path):
    """An export is a projection, not a log of changes to one (I11). The second upload here is a
    DIFFERENT, smaller file; if anything merged, the served bytes would be neither one."""
    smaller = tmp_path / "smaller.db"
    smaller.write_bytes(svc.export.read_bytes()[:8192])

    r = svc.client.post(f"/v1/upload/{svc.owner}", content=smaller.read_bytes(),
                        headers=svc.owner_hdr)
    assert r.status_code == 200
    assert r.json()["sha256"] == _sha(smaller) == _sha(uploads.export_path(svc.owner))
    assert r.json()["bytes"] == 8192


def test_the_served_file_is_byte_identical_after_a_session_of_reads(svc, emb):
    """I3, enforced by SQLite rather than by discipline: `peers.open_peer` is the only place a
    peer store is opened and it opens read-only, so this holds for every read path at once —
    including the idempotent DDL that a writable connect would have run."""
    from tests.service.conftest import query_vector

    served = uploads.export_path(svc.owner)
    before = _sha(served)

    hits = svc.client.post(f"/v1/kb/{svc.owner}/search",
                           json={"query": "agent framework",
                                 "query_vector": query_vector(emb, "agent framework")},
                           headers=svc.reader_hdr).json()["hits"]
    for h in hits:
        svc.client.post(f"/v1/kb/{svc.owner}/open", json={"atom_id": h["atom_id"]},
                        headers=svc.reader_hdr)
    svc.client.post(f"/v1/kb/{svc.owner}/aggregate", json={}, headers=svc.reader_hdr)
    svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "crypto proof"},
                    headers=svc.reader_hdr)

    assert _sha(served) == before


def test_an_owner_cannot_upload_to_a_name_that_is_not_theirs(svc):
    r = svc.client.post("/v1/upload/somebody-else", content=b"x", headers=svc.owner_hdr)
    assert r.status_code == 403
    assert not uploads.exports_dir().joinpath("somebody-else.db").exists()


@pytest.mark.parametrize("name", ["../escape", "me", "Has-Caps", "", "with space", ".hidden"])
def test_an_owner_name_that_would_become_a_bad_filename_is_refused(name):
    """Validated at the boundary it arrives through — a URL path segment that becomes a file
    name. An allow-list, because it is the only version of this check that cannot be argued
    with; `me` is in the list for a different reason, being the name a reader uses for their own
    store and therefore permanently unopenable as a peer."""
    with pytest.raises(uploads.BadOwner):
        uploads.valid_owner(name)
