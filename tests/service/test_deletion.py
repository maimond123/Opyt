"""Deleting a library: `POST /v1/unpublish` and the operator's command both run
`deletion.delete_library`, which leaves nothing on this service keyed by the library.

The export, the upload, the grant and the redeem are the real ones (`conftest.svc`). The
completeness check reads the schema, so a table added later with an `owner` column fails here
until the deletion covers it.
"""
from __future__ import annotations

import asyncio

import pytest

import time

from pipeline.retrieval import peers, reads
from service import account_peers, accounts, db, deletion, grants, line, tokens, uploads
from service.pages import make, store as pages
from tests.service.conftest import OWNER, agent_key

SUB = "google-123"


def link_google(sub: str, owner: str) -> None:
    """An account linked to `owner`, and a second sign-in for it in flight."""
    conn = db.connect()
    conn.execute("INSERT INTO accounts (google_sub, owner) VALUES (?, ?)", (sub, owner))
    conn.execute("INSERT INTO account_links (owner, device_code, user_code, verification_url, "
                 "expires_at) VALUES (?, 'dev', 'CODE', 'https://www.google.com/device', 9e9)",
                 (owner,))
    conn.commit()
    conn.close()


def rows_for(owner: str) -> dict[str, int]:
    """How many rows each table with an `owner` column holds for `owner`."""
    conn = db.connect()
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        keyed = [t for t in tables
                 if "owner" in {c[1] for c in conn.execute(f"PRAGMA table_info({t})")}]
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t} WHERE owner = ?", (owner,)).fetchone()[0]
                for t in keyed}
    finally:
        conn.close()


@pytest.fixture()
def full(svc):
    """The library with a row in every owner-keyed table: a read counted, an invite spent and one
    pending, an agent invite pending, a Google account linked and one it was shared with, a
    reader-view page made, a call left
    for its Mac and the Mac's ask for it; and a second library beside it."""
    body = {"query": "agent framework", "k": 5}
    found = svc.client.post(f"/v1/kb/{OWNER}/search", json=body, headers=svc.reader_hdr)
    assert found.json()["hits"]
    svc.client.post("/v1/grant", json={"label": "pending"}, headers=svc.owner_hdr)
    svc.client.post("/v1/grant", json={"label": "an agent", "agent": True}, headers=svc.owner_hdr)
    link_google(SUB, OWNER)
    shared = svc.client.post("/v1/grant", json={"label": "a friend"}, headers=svc.owner_hdr)
    assert account_peers.accept("google-friend", shared.json()["code"])["status"] == "accepted"
    svc.page = make.create(OWNER, [reads.kb_open("x:1", kb=OWNER)], "A page", None, None,
                           time.time())
    asyncio.run(line.call(OWNER, "hopper", {"reference": "https://example.com/a"}))
    assert svc.client.get("/v1/line/next", params={"wait": 0},
                          headers=svc.owner_hdr).json()["call"]
    other = tokens.mint_token("other", "owner")
    svc.client.post("/v1/upload/other", content=svc.export.read_bytes(),
                    headers={"Authorization": f"Bearer {other}"})
    link_google("google-other", "other")
    svc.other_token = other
    svc.search_body = body
    return svc


def test_unpublish_leaves_nothing_keyed_by_the_library(full):
    before = rows_for(OWNER)
    assert all(before.values()), before          # every table really had a row to delete

    out = full.client.post("/v1/unpublish", headers=full.owner_hdr)

    assert out.status_code == 200, out.text
    # The reader's key and the one a friend's Google account holds.
    assert out.json() == {"owner": OWNER, "readers_revoked": 2, "export_deleted": True,
                          "deleted": True}
    assert set(rows_for(OWNER)) <= set(deletion.PAGES + deletion.ACCESS + deletion.RECORDS)
    assert not any(rows_for(OWNER).values()), rows_for(OWNER)
    assert not uploads.export_path(OWNER).exists()
    # The page's link is gone with its folder (its PDFs).
    assert full.client.get(full.page["path"]).status_code == 404
    assert not pages.pdf_path(full.page["id"], 0).parent.exists()
    assert peers.get(OWNER) is None
    assert accounts.owner_of(SUB) is None


def test_another_library_is_untouched(full):
    before = rows_for("other")
    full.client.post("/v1/unpublish", headers=full.owner_hdr)
    assert rows_for("other") == before
    assert uploads.export_path("other").exists() and accounts.owner_of("google-other") == "other"


def test_every_token_is_refused_afterwards_the_owners_too(full):
    """The builder's own token goes last, so after a deletion it is refused on every route: a
    second unpublish, a grant, an upload, a link."""
    full.client.post("/v1/unpublish", headers=full.owner_hdr)

    assert full.client.post(f"/v1/kb/{OWNER}/search", json=full.search_body,
                            headers=full.reader_hdr).status_code == 401
    for method, path in (("post", "/v1/unpublish"), ("post", "/v1/grant"), ("get", "/v1/tokens"),
                         ("post", "/v1/link"), ("post", f"/v1/upload/{OWNER}")):
        got = getattr(full.client, method)(path, headers=full.owner_hdr,
                                           **({"json": {}} if method == "post" else {}))
        assert got.status_code == 401, (path, got.status_code)


def test_a_pending_invite_dies_too(full):
    code = full.client.post("/v1/grant", json={}, headers=full.owner_hdr).json()["code"]
    full.client.post("/v1/unpublish", headers=full.owner_hdr)
    assert full.client.post("/v1/redeem", json={"code": code}).status_code == 404


def test_an_agent_key_and_an_agent_invite_nobody_redeemed_leave_no_row_behind(full):
    """The agent's key counts with the readers', and the invite that would have made another
    agent dies with the rest. Another library's agent and agent invite are untouched."""
    agent = agent_key(full)
    pending = full.client.post("/v1/grant", json={"label": "later", "agent": True},
                               headers=full.owner_hdr).json()["code"]
    theirs = tokens.mint_token("other", "agent")
    grants.mint_grant("other", agent=True)
    others = rows_for("other")
    assert rows_for(OWNER)["agent_grants"] == 3     # the fixture's, the redeemed one, `pending`

    out = full.client.post("/v1/unpublish", headers=full.owner_hdr).json()

    assert out["readers_revoked"] == 3              # the two readers' keys and the agent's
    assert not any(rows_for(OWNER).values()), rows_for(OWNER)
    assert full.client.get(f"/v1/kb/{OWNER}/feed",
                           headers={"Authorization": f"Bearer {agent}"}).status_code == 401
    assert full.client.post("/v1/redeem", json={"code": pending}).status_code == 404
    assert rows_for("other") == others and tokens.resolve_token(theirs) is not None


def test_an_agent_key_ends_in_the_first_step_with_the_readers(full, monkeypatch):
    """Access ends first: when the deletion stops before the copy is removed, the agent can
    already neither read nor act, and no agent invite is left to make another."""
    agent = agent_key(full)

    def stuck(owner):
        raise OSError("disk went away")
    monkeypatch.setattr(uploads, "remove", stuck)
    with pytest.raises(OSError):
        deletion.delete_library(OWNER)

    assert tokens.resolve_token(agent) is None and rows_for(OWNER)["agent_grants"] == 0
    assert tokens.resolve_token(full.owner_token) is not None


def test_a_partial_upload_left_by_a_restart_goes_too(full):
    partial = uploads.export_path(OWNER).with_name(f"{OWNER}.db.uploading")
    partial.write_bytes(b"half an export")
    full.client.post("/v1/unpublish", headers=full.owner_hdr)
    assert not partial.exists()


def test_a_deletion_that_stopped_part_way_is_finished_by_calling_again(full, monkeypatch):
    """Readers and links went, then the file could not be removed. The owner token is still good,
    so the builder's next unshare finishes the job."""
    real = uploads.remove

    def stuck(owner):
        raise OSError("disk went away")
    monkeypatch.setattr(uploads, "remove", stuck)
    with pytest.raises(OSError):
        deletion.delete_library(OWNER)
    assert full.client.post(f"/v1/kb/{OWNER}/search", json=full.search_body,
                            headers=full.reader_hdr).status_code == 401
    assert accounts.owner_of(SUB) is None
    assert tokens.resolve_token(full.owner_token) is not None

    monkeypatch.setattr(uploads, "remove", real)
    assert full.client.post("/v1/unpublish", headers=full.owner_hdr).json()["deleted"] is True
    assert not any(rows_for(OWNER).values())


def test_a_reader_token_cannot_delete(full):
    assert full.client.post("/v1/unpublish", headers=full.reader_hdr).status_code == 403
    assert uploads.export_path(OWNER).exists() and all(rows_for(OWNER).values())


# ── the operator's command ─────────────────────────────────────────────────────────

def test_the_command_finds_a_library_by_its_google_account(full, capsys):
    assert deletion.main(["--google-sub", SUB]) == 0
    assert '"deleted": true' in capsys.readouterr().out
    assert not any(rows_for(OWNER).values()) and not uploads.export_path(OWNER).exists()
    assert uploads.export_path("other").exists()


def test_the_command_takes_a_routing_key(full):
    assert deletion.main(["--owner", OWNER]) == 0
    assert not any(rows_for(OWNER).values())


@pytest.mark.parametrize("args", [["--google-sub", "nobody"], ["--owner", "no-such-key"],
                                  ["--owner", "Not A Key!"]])
def test_the_command_refuses_what_it_cannot_find_and_deletes_nothing(full, args, capsys):
    before = rows_for(OWNER)
    assert deletion.main(args) == 1
    assert capsys.readouterr().err
    assert rows_for(OWNER) == before
