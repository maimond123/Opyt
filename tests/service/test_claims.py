"""One name, one publisher, forever.

The collision this closes (docs/plans/2026-08-28-finish-the-deploy-and-open-the-owner-path.md,
item 3): `tokens` has no uniqueness on `owner` and the served file is `exports/<owner>.db`, so a
second owner token named 'dave' could silently replace the first dave's knowledge base — no error
at any layer, and every reader's saved `kb='dave'` peer row would start answering with the second
dave's atoms. Two checks close it: the mint refuses a claimed name, and the upload refuses a
token that does not hold the claim. The second is the one that protects the file, because a
pre-claims database can already hold duplicate names the mint never gets asked about.
"""
from __future__ import annotations

import hashlib
import secrets

import pytest

from service import store, uploads


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """A store-only `$OPYT_HOME` — these tests need `service.db`, not a served export."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))


def test_a_second_owner_token_for_a_name_is_refused(home):
    store.mint_token("dave", "owner")
    with pytest.raises(store.NameClaimed):
        store.mint_token("dave", "owner")


def test_revoking_the_owner_token_does_not_release_the_name(home):
    """The claim outlives the token. A name released on revocation would hand every reader's
    saved peer row to whoever claims it next — the same failure the mint refusal closes,
    arriving by a different route."""
    t = store.mint_token("dave", "owner")
    assert store.revoke("dave", store.token_hash(t))
    with pytest.raises(store.NameClaimed):
        store.mint_token("dave", "owner")


def test_reclaim_rotates_only_after_revocation(home):
    """The rotation path for a leaked owner token: revoke, then reclaim. While the live token
    stands, the claim cannot move."""
    t = store.mint_token("dave", "owner")
    with pytest.raises(store.NameClaimed):
        store.mint_token("dave", "owner", reclaim=True)

    store.revoke("dave", store.token_hash(t))
    t2 = store.mint_token("dave", "owner", reclaim=True)
    assert store.claim_holder("dave") == store.token_hash(t2)


def test_reader_tokens_are_not_claims(home):
    """A knowledge base has one publisher and any number of readers, so the reader path must
    never touch `owner_claims` — a second redeem for the same owner is the normal case."""
    store.mint_token("dave", "owner")
    store.mint_token("dave", "reader", label="friend one")
    store.mint_token("dave", "reader", label="friend two")
    assert len(store.list_tokens("dave")) == 3


def test_a_pre_claims_database_claims_names_on_connect(home):
    """The seed in `_DDL` is the migration for owner tokens minted before the table existed:
    dropping the claim row and reconnecting restores it, pointed at the token."""
    t = store.mint_token("dave", "owner")
    conn = store.connect()
    try:
        conn.execute("DELETE FROM owner_claims")
        conn.commit()
    finally:
        conn.close()
    assert store.claim_holder("dave") == store.token_hash(t)


def test_an_upload_from_a_non_holder_token_is_refused(svc):
    """The check at the harm site. The rogue row is inserted by hand because that is exactly the
    population it exists for — duplicate names the fixed mint can no longer create."""
    rogue = secrets.token_urlsafe(32)
    conn = store.connect()
    try:
        conn.execute("INSERT INTO tokens (token_sha256, owner, role) VALUES (?, ?, 'owner')",
                     (store.token_hash(rogue), svc.owner))
        conn.commit()
    finally:
        conn.close()

    served = uploads.export_path(svc.owner)
    before = hashlib.sha256(served.read_bytes()).hexdigest()

    r = svc.client.post(f"/v1/upload/{svc.owner}", content=b"SQLite format 3\x00",
                        headers={"Authorization": f"Bearer {rogue}"})
    assert r.status_code == 403
    assert hashlib.sha256(served.read_bytes()).hexdigest() == before

    r = svc.client.post(f"/v1/upload/{svc.owner}", content=svc.export.read_bytes(),
                        headers=svc.owner_hdr)
    assert r.status_code == 200, "the claim holder must still be able to publish"
