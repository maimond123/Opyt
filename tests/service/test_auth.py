"""Who may call this service, and what happens the moment they may not.

Four properties, and each one is a single row or a single string comparison rather than a
permissions engine — which is the design claim this file is really checking. If any of these
needed a policy layer to hold, the model would be wrong.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from service import store
from tests.service.conftest import LABEL


def test_a_read_with_no_token_is_refused(svc):
    for path in ("search", "open", "aggregate"):
        r = svc.client.post(f"/v1/kb/{svc.owner}/{path}", json={"query": "x", "atom_id": "x"})
        assert r.status_code == 401, path


def test_a_reader_token_cannot_reach_another_knowledge_base(svc):
    """Scope is the `owner` column, so this is one string comparison and there is nothing else
    to get wrong. A second knowledge base is registered so the 403 is about the TOKEN's scope and
    not about the name being unknown to the service."""
    conn = store.connect()
    other = store.mint_token("stranger", "reader")
    conn.close()

    mine = svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"},
                           headers=svc.reader_hdr)
    assert mine.status_code == 200

    theirs = svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"},
                             headers={"Authorization": f"Bearer {other}"})
    assert theirs.status_code == 403


def test_an_owner_token_does_not_read_and_a_reader_token_does_not_upload(svc):
    """The roles name what a token is FOR. Neither direction is a policy decision that could
    have gone the other way without a second caller appearing to justify it."""
    r = svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"},
                        headers=svc.owner_hdr)
    assert r.status_code == 403

    r = svc.client.post(f"/v1/upload/{svc.owner}", content=b"x", headers=svc.reader_hdr)
    assert r.status_code == 403

    r = svc.client.post("/v1/grant", json={}, headers=svc.reader_hdr)
    assert r.status_code == 403


def test_a_revoked_token_fails_on_the_very_next_request(svc):
    """Revocation is a row delete, so there is no refresh cycle to wait out and no window during
    which a revoked reader still works. The two requests below are separated by one DELETE."""
    ok = svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"},
                         headers=svc.reader_hdr)
    assert ok.status_code == 200

    listed = svc.client.get("/v1/tokens", headers=svc.owner_hdr).json()["tokens"]
    reader_sha = next(t["token_sha256"] for t in listed if t["role"] == "reader")
    gone = svc.client.post("/v1/revoke", json={"token_sha256": reader_sha}, headers=svc.owner_hdr)
    assert gone.json() == {"revoked": True}

    after = svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"},
                            headers=svc.reader_hdr)
    assert after.status_code == 401


def test_an_owner_cannot_revoke_another_owners_token(svc):
    """Scoped in the WHERE clause rather than checked beforehand: naming a stranger's token hash
    deletes nothing, and the answer teaches nothing about whether it existed."""
    conn = store.connect()
    victim = store.mint_token("stranger", "reader")
    conn.close()

    r = svc.client.post("/v1/revoke", json={"token_sha256": store.token_hash(victim)},
                        headers=svc.owner_hdr)
    assert r.json() == {"revoked": False}

    conn = store.connect()
    assert store.resolve_token(victim) is not None
    conn.close()


def test_a_grant_code_can_be_redeemed_exactly_once(svc):
    code = svc.client.post("/v1/grant", json={}, headers=svc.owner_hdr).json()["code"]

    first = svc.client.post("/v1/redeem", json={"code": code, "install_id": "a"})
    assert first.status_code == 200 and first.json()["owner"] == svc.owner

    second = svc.client.post("/v1/redeem", json={"code": code, "install_id": "b"})
    assert second.status_code == 404


def test_redeem_hands_back_the_owners_display_name_to_register_under(svc):
    """R4's other half. `owner` is the routing key — after R5 an opaque hex string — so a client
    that registered the peer under it would leave the reader typing `kb='a3f9c2e1'`. The owner
    named themselves once, on their own token, and every reader starts from that."""
    code = svc.client.post("/v1/grant", json={}, headers=svc.owner_hdr).json()["code"]
    body = svc.client.post("/v1/redeem", json={"code": code, "install_id": "a"}).json()

    assert body["suggested_name"] == LABEL
    assert body["owner"] == svc.owner


def test_an_owner_who_registered_without_a_label_suggests_nothing(svc):
    """Null rather than an invented name: `opyt-redeem` falls back to the routing key, which is
    at least a string that resolves. Guessing one here would put a name in the reader's registry
    that the owner never chose."""
    hdr = {"Authorization": f"Bearer {store.mint_token('nameless', 'owner')}"}
    code = svc.client.post("/v1/grant", json={}, headers=hdr).json()["code"]
    body = svc.client.post("/v1/redeem", json={"code": code, "install_id": "a"}).json()

    assert body["suggested_name"] is None


def test_two_simultaneous_redeems_of_one_code_produce_exactly_one_token(svc):
    """The property the conditional UPDATE exists for. A read-then-write would have a window
    between the check and the claim where both callers see NULL; the row count is what decides
    here, so SQLite's own write lock serializes them and the loser updates zero rows.

    Threads rather than a mocked race: the seam being tested is SQLite's, and stubbing it would
    test the stub. `check_same_thread=False` is deliberately NOT set anywhere — each caller opens
    its own connection, which is what a second process would do too."""
    code = svc.client.post("/v1/grant", json={}, headers=svc.owner_hdr).json()["code"]

    def redeem(i):
        conn = store.connect()
        try:
            return store.redeem_grant(code, f"racer-{i}")
        except store.GrantUnavailable:
            return None
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(redeem, range(8)))

    won = [r for r in results if r is not None]
    assert len(won) == 1

    conn = store.connect()
    try:
        # A prefix the fixture's own reader token does not share, so this counts the
        # tokens THIS race minted rather than every reader that ever redeemed.
        minted = [t for t in store.list_tokens(svc.owner)
                  if (t["install_id"] or "").startswith("racer-")]
        assert len(minted) == 1
    finally:
        conn.close()


def test_a_code_that_was_never_minted_and_a_spent_one_answer_the_same(svc):
    """One failure type for both. Telling them apart is information a stranger holding a wrong
    code should not get, and the holder of a real code does the same thing either way."""
    code = svc.client.post("/v1/grant", json={}, headers=svc.owner_hdr).json()["code"]
    svc.client.post("/v1/redeem", json={"code": code})

    spent = svc.client.post("/v1/redeem", json={"code": code})
    never = svc.client.post("/v1/redeem", json={"code": "not-a-real-code"})
    assert spent.status_code == never.status_code == 404
    assert spent.json()["detail"] == never.json()["detail"]


def test_the_database_holds_no_usable_credential(svc):
    """`tokens` and `grant_codes` store hashes. Asserted by reading the whole file: a token that
    leaked into any column, index or free page would be findable here."""
    raw = (svc.home / "service.db").read_bytes()
    assert svc.owner_token.encode() not in raw
    assert svc.reader_token.encode() not in raw
    assert store.token_hash(svc.reader_token).encode() in raw


def test_a_grant_code_survives_being_a_command_line_argument(svc):
    """`opyt-redeem <url> <code>` puts the code in an argparse POSITIONAL, and argparse reads a
    leading `-` as an option. `secrets.token_urlsafe` produced one about 1 code in 64, so ~1.6% of
    readers met a usage error naming the wrong problem. The alphabet is the fix, and this asserts
    it TOTALLY rather than sampling: every character, over enough codes to be worth the second."""
    for _ in range(200):
        code = store.mint_grant(svc.owner)
        assert code.isalnum() and len(code) >= 40, code
