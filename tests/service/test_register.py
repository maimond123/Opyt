"""Anyone with Opyt installed can publish — the endpoint R5 rules into existence.

`DEPLOY.md` §7 used to say there was deliberately no endpoint that mints an owner token, because
one that does hands out the right to publish. That was two arguments wearing one sentence. The
load-bearing half was the permanent NAME claim: a routing key was the string a reader typed, so
handing one out gave away `karpathy` forever, since `_claim_name` never releases. R4 deleted that
half by making the key an assigned address. What is left is resource abuse, which R5a rules a
quota question handled after the fact, by the operator, reading `/v1/stats`.

So these tests hold two things: a registered token really can publish and grant, under a key that
is routable and unclaimable; and the claims machinery that used to be the gate is still doing its
other job, guarding the upload against a second owner token for one name.
"""
from __future__ import annotations

from pipeline.kb import peers
from service import store, uploads


def _register(client, label=None):
    r = client.post("/v1/register", json={"label": label})
    assert r.status_code == 200, r.text
    return r.json()


def test_a_registered_token_publishes_and_grants(svc):
    """The whole feature in one pass, and nobody opened a shell. The reader that comes out the
    far end reads the export the new owner uploaded — which is the only proof that the key the
    service assigned actually routes."""
    me = _register(svc.client, "Leo")
    hdr = {"Authorization": f"Bearer {me['token']}"}

    up = svc.client.post(f"/v1/upload/{me['owner']}", content=svc.export.read_bytes(),
                         headers=hdr)
    assert up.status_code == 200, up.text

    code = svc.client.post("/v1/grant", json={}, headers=hdr).json()["code"]
    redeemed = svc.client.post("/v1/redeem",
                               json={"code": code, "install_id": "i-1"}).json()
    assert redeemed["suggested_name"] == "Leo"

    r = svc.client.post(f"/v1/kb/{me['owner']}/search", json={"query": "agent"},
                        headers={"Authorization": f"Bearer {redeemed['token']}"})
    assert r.status_code == 200 and r.json()["hits"]


def test_the_assigned_key_is_a_usable_route_and_never_the_local_name(svc):
    """The key becomes a URL path segment AND a filename, so it has to satisfy `_OWNER` without
    anybody sanitising it later. `"me"` is the one name `peers` reserves — a knowledge base
    served under it would be registered, listed and permanently unopenable."""
    for _ in range(20):
        owner = _register(svc.client)["owner"]
        assert uploads.valid_owner(owner) == owner
        assert owner != peers.LOCAL_KB


def test_two_registrations_are_two_knowledge_bases(svc):
    """No shared namespace, so no negotiation and no squatting. Two people asking for the same
    display name get the same label and different keys, which is exactly what R4 bought."""
    a, b = _register(svc.client, "Alex"), _register(svc.client, "Alex")

    assert a["owner"] != b["owner"] and a["token"] != b["token"]
    assert store.owner_label(a["owner"]) == store.owner_label(b["owner"]) == "Alex"


def test_a_key_collision_retries_instead_of_failing(svc, monkeypatch):
    """`NameClaimed` is not dead code after R5 — it is the collision signal this loop reads. At
    48 bits a real collision is a birthday event nobody will see, so it is forced here: the first
    key is one that is already claimed, and the caller must never learn that happened."""
    taken = _register(svc.client)["owner"]
    keys = iter([taken, "aabbccddeeff"])
    monkeypatch.setattr(store.secrets, "token_hex", lambda n: next(keys))

    assert _register(svc.client)["owner"] == "aabbccddeeff"


def test_a_registered_token_cannot_touch_another_knowledge_base(svc):
    """Scope is the token's own `owner` column, so self-service adds no new surface: a stranger's
    token reaches exactly one knowledge base, the one it was minted for."""
    hdr = {"Authorization": f"Bearer {_register(svc.client)['token']}"}

    r = svc.client.post(f"/v1/upload/{svc.owner}", content=b"x", headers=hdr)
    assert r.status_code == 403
    assert uploads.export_path(svc.owner).stat().st_size == svc.upload["bytes"]


def test_the_claim_still_guards_the_upload(svc):
    """The half of the claims machinery that was never the gate. Nothing in `tokens` makes
    `owner` unique, and a pre-claims database can already hold two owner tokens for one name — so
    agreeing strings are not enough, and `claim_holder` is what stops the second token replacing
    the first one's served file."""
    conn = store.connect()
    try:
        second = store.mint_token("interloper", "owner")
        conn.execute("UPDATE tokens SET owner = ? WHERE token_sha256 = ?",
                     (svc.owner, store.token_hash(second)))
        conn.commit()
    finally:
        conn.close()

    r = svc.client.post(f"/v1/upload/{svc.owner}", content=b"x",
                        headers={"Authorization": f"Bearer {second}"})
    assert r.status_code == 403
    assert "different owner token" in r.json()["detail"]


def _granted_but_unpublished(svc):
    """A reader holding a real token for a knowledge base with no export yet. ORDINARY rather
    than exotic: `share` returns the invite immediately and pushes detached, so the link is live
    for the minute or two the upload takes."""
    me = _register(svc.client, "Eager")
    hdr = {"Authorization": f"Bearer {me['token']}"}
    code = svc.client.post("/v1/grant", json={}, headers=hdr).json()["code"]
    token = svc.client.post("/v1/redeem",
                            json={"code": code, "install_id": "i-2"}).json()["token"]
    return me["owner"], {"Authorization": f"Bearer {token}"}


def test_every_read_endpoint_answers_a_missing_export_the_same_way(svc):
    """One boundary check, four endpoints. Before `_served` existed, only `meta` 404'd: `search`,
    `open` and `aggregate` returned 200 carrying the LOCAL entry point's answer, which is written
    for a person at their own install."""
    owner, hdr = _granted_but_unpublished(svc)

    one = svc.client.get(f"/v1/kb/{owner}/meta", headers=hdr)
    assert one.status_code == 404
    for path, body in (("search", {"query": "x"}), ("open", {"atom_id": "a"}),
                       ("aggregate", {"scope": None})):
        r = svc.client.post(f"/v1/kb/{owner}/{path}", json=body, headers=hdr)
        assert r.status_code == 404, path
        # The same sentence from all four, so a host cannot learn a different fact per endpoint.
        assert r.json()["detail"] == one.json()["detail"], path


def test_a_reader_never_learns_what_else_this_service_serves(svc):
    """The leak `_served` closes, and self-service is what made it matter: the local entry
    point's sentence for an unreadable `kb=` NAMES every knowledge base registered on the box —
    which here is every published routing key — and tells the caller to omit `kb` and search their
    own, which a reader of a served export cannot do. Harmless prose locally, the server's
    registry crossing a trust boundary here."""
    owner, hdr = _granted_but_unpublished(svc)

    for path, body in (("search", {"query": "x"}), ("aggregate", {"scope": None})):
        detail = svc.client.post(f"/v1/kb/{owner}/{path}", json=body, headers=hdr).json()["detail"]
        assert svc.owner not in detail, path
        assert "Registered knowledge bases" not in detail, path
        assert "Omit `kb`" not in detail, path


def test_register_needs_no_token(svc):
    """Unauthenticated by construction: the whole point is that nobody has to be let in first."""
    assert svc.client.post("/v1/register", json={}).status_code == 200
