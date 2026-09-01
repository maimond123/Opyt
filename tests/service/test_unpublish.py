"""An owner stops sharing: every reader cut off AND the served copy deleted.

The point of these tests is the SECOND half. Revoking tokens was already covered by
`test_auth.py`; what was missing until now is that the export file itself goes, because
`revoke` alone leaves 117 MB of somebody's reading history on this disk with no reader, no
removal path, and no expiry. R1 made the consent maximal (the whole KB, standing), so without
this the consent was also irrevocable.
"""
from __future__ import annotations

import pytest

from service import store, uploads
from tests.service.conftest import OWNER, query_vector


def test_unpublish_cuts_every_reader_and_deletes_the_export(svc, emb):
    """The whole feature in one pass: a working reader, one call, then neither half remains."""
    body = {"query": "agent framework", "k": 5, "mode": "hybrid",
            "query_vector": query_vector(emb, "agent framework")}
    before = svc.client.post(f"/v1/kb/{OWNER}/search", json=body, headers=svc.reader_hdr)
    assert before.status_code == 200 and before.json()["hits"]
    assert uploads.export_path(OWNER).exists()

    out = svc.client.post("/v1/unpublish", headers=svc.owner_hdr)
    assert out.status_code == 200, out.text
    assert out.json() == {"owner": OWNER, "readers_revoked": 1, "export_deleted": True}

    assert not uploads.export_path(OWNER).exists()
    after = svc.client.post(f"/v1/kb/{OWNER}/search", json=body, headers=svc.reader_hdr)
    assert after.status_code != 200


def test_a_second_reader_is_cut_too(svc, emb):
    """`revoke_all_readers` is one DELETE, not a loop over a list read a moment earlier — so a
    grant redeemed after the first reader still goes."""
    code = svc.client.post("/v1/grant", json={"label": "second"},
                           headers=svc.owner_hdr).json()["code"]
    second = svc.client.post("/v1/redeem",
                             json={"code": code, "install_id": "install-2"}).json()["token"]

    assert svc.client.post("/v1/unpublish",
                           headers=svc.owner_hdr).json()["readers_revoked"] == 2
    r = svc.client.get(f"/v1/kb/{OWNER}/meta",
                       headers={"Authorization": f"Bearer {second}"})
    assert r.status_code == 401


def test_the_owner_token_survives_so_re_sharing_is_not_a_re_mint(svc):
    """Deliberate: only READER tokens go. Nothing re-uploads without a person asking it to, and
    taking the owner token would make re-sharing an operator step on a self-service path."""
    svc.client.post("/v1/unpublish", headers=svc.owner_hdr)

    assert store.resolve_token(svc.owner_token) is not None
    again = svc.client.post(f"/v1/upload/{OWNER}", content=svc.export.read_bytes(),
                            headers=svc.owner_hdr)
    assert again.status_code == 200
    assert uploads.export_path(OWNER).exists()


def test_re_publishing_does_not_resurrect_the_old_readers(svc, emb):
    """Why burning the routing key was NOT built: there is no reader left to resurrect. The old
    token is gone from `tokens`, so it answers 401 against a freshly re-uploaded export."""
    svc.client.post("/v1/unpublish", headers=svc.owner_hdr)
    svc.client.post(f"/v1/upload/{OWNER}", content=svc.export.read_bytes(),
                    headers=svc.owner_hdr)

    r = svc.client.post(f"/v1/kb/{OWNER}/search",
                        json={"query": "agent", "k": 5, "mode": "bm25"},
                        headers=svc.reader_hdr)
    assert r.status_code == 401


def test_unpublish_is_idempotent(svc):
    """The caller wants the same end state either way, so a second call is not an error — it
    just reports that there was nothing left to remove."""
    svc.client.post("/v1/unpublish", headers=svc.owner_hdr)
    second = svc.client.post("/v1/unpublish", headers=svc.owner_hdr)
    assert second.status_code == 200
    assert second.json() == {"owner": OWNER, "readers_revoked": 0, "export_deleted": False}


def test_a_reader_token_cannot_unpublish(svc):
    """The destructive action needs the owner role, like `grant` and `revoke`."""
    r = svc.client.post("/v1/unpublish", headers=svc.reader_hdr)
    assert r.status_code == 403
    assert uploads.export_path(OWNER).exists()


def test_the_registry_row_goes_before_the_file(svc):
    """`remove` deregisters first so no row ever names a deleted file — the mirror of `commit`
    registering only after the rename."""
    from pipeline.kb import peers
    assert peers.get(OWNER) is not None
    svc.client.post("/v1/unpublish", headers=svc.owner_hdr)
    assert peers.get(OWNER) is None
