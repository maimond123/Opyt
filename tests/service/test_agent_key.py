"""The agent key: a third kind of key, which an owner makes for their own agent.

It comes from redeeming an agent invite, reads its own library as a reader's key does, and passes
`access.agent_token`, the dependency the routes an agent acts through hang on. It is never an
owner's key: every owner-only route refuses it. A reader's key stays what it was.

The export, the upload, the grant and the redeem are the real ones (`conftest.svc`).
"""
from __future__ import annotations

import re

from fastapi import Depends, FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from service import access, db, tokens, uploads
from service.app import app
from tests.service.conftest import agent_key


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _grant(svc, **body) -> str:
    return svc.client.post("/v1/grant", json=body, headers=svc.owner_hdr).json()["code"]


def _counted(owner: str, token: str) -> dict[str, int]:
    """The reads the meter holds under this token, by tool."""
    conn = db.connect()
    try:
        return {r["tool"]: r["n"] for r in conn.execute(
            "SELECT tool, n FROM usage_daily WHERE owner = ? AND reader = ?",
            (owner, tokens.token_hash(token)))}
    finally:
        conn.close()


# ── an agent invite ──────────────────────────────────────────────────────────────

def test_an_agent_invite_redeems_once_into_an_agent_key(svc):
    code = _grant(svc, label="Instinct", agent=True)

    first = svc.client.post("/v1/redeem", json={"code": code, "install_id": "i-1"})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["role"] == "agent" and body["owner"] == svc.owner
    row = tokens.resolve_token(body["token"])
    assert (row["role"], row["owner"], row["label"], row["install_id"]) == (
        "agent", svc.owner, "Instinct", "i-1")

    assert svc.client.post("/v1/redeem", json={"code": code}).status_code == 404


def test_a_reader_invite_still_redeems_into_a_reader_key_with_the_answer_it_always_had(svc):
    """An install older than agent keys reads this answer, so it gains no field. An explicit
    `agent: false` is the same invite."""
    for body in ({"label": "Maya"}, {"label": "Maya", "agent": False}):
        r = svc.client.post("/v1/redeem", json={"code": _grant(svc, **body)})
        assert list(r.json()) == ["owner", "token", "suggested_name", "notice"]
        assert tokens.resolve_token(r.json()["token"])["role"] == "reader"


def test_the_kind_of_an_invite_is_the_owners_choice_not_the_redeemers(svc):
    """Nothing in a redeem asks for a role: a reader invite redeemed with `agent` in the body is
    still a reader's key."""
    r = svc.client.post("/v1/redeem", json={"code": _grant(svc), "agent": True, "role": "agent"})
    assert "role" not in r.json()
    assert tokens.resolve_token(r.json()["token"])["role"] == "reader"


# ── what an agent key reads ──────────────────────────────────────────────────────

def test_an_agent_key_reads_its_own_library_and_is_counted_under_itself(svc):
    agent = agent_key(svc)
    hdr, kb = _bearer(agent), f"/v1/kb/{svc.owner}"

    hits = svc.client.post(f"{kb}/search", json={"query": "agent"},
                           headers=hdr).json()["hits"]
    assert hits and {h["kb"] for h in hits} == {svc.owner}
    opened = svc.client.post(f"{kb}/open", json={"atom_id": hits[0]["atom_id"]}, headers=hdr)
    assert opened.status_code == 200 and opened.json()["raw"]
    assert svc.client.post(f"{kb}/aggregate", json={}, headers=hdr).json()["total"] > 0
    assert svc.client.get(f"{kb}/feed", headers=hdr).json()["items"]

    assert _counted(svc.owner, agent) == {"search": 1, "open": 1, "aggregate": 1, "feed": 1}
    assert _counted(svc.owner, svc.reader_token) == {}


def test_an_agent_key_reads_no_other_library(svc):
    """Two libraries, an agent of each: the key's own `owner` is its whole scope."""
    other = tokens.mint_token("other", "owner")
    svc.client.post("/v1/upload/other", content=svc.export.read_bytes(), headers=_bearer(other))
    mine = _bearer(agent_key(svc))
    theirs = _bearer(tokens.mint_token("other", "agent"))

    for hdr, own, foreign in ((mine, svc.owner, "other"), (theirs, "other", svc.owner)):
        assert svc.client.get(f"/v1/kb/{own}/feed", headers=hdr).status_code == 200
        for path in ("search", "open", "aggregate"):
            r = svc.client.post(f"/v1/kb/{foreign}/{path}", headers=hdr,
                                json={"query": "agent", "atom_id": "x:1"})
            assert r.status_code == 403, (foreign, path)
        assert svc.client.get(f"/v1/kb/{foreign}/feed", headers=hdr).status_code == 403


# ── what an agent key may not do ─────────────────────────────────────────────────

def _depends_on(route: APIRoute, dependency) -> bool:
    todo = list(route.dependant.dependencies)
    while todo:
        dep = todo.pop()
        if dep.call is dependency:
            return True
        todo += dep.dependencies
    return False


def test_every_owner_only_route_refuses_an_agent_key(svc):
    """Read from the app, so a route added later under `owner_token` is held to this too. An
    agent acts through its own routes, never the install's: it cannot upload, mint or revoke
    by these, list tokens, link Google, make a page from sent items, or delete the library."""
    owner_only = [r for r in app.routes
                  if isinstance(r, APIRoute) and _depends_on(r, access.owner_token)]
    assert {"/v1/upload/{owner}", "/v1/upload/{owner}/delta", "/v1/grant", "/v1/revoke",
            "/v1/tokens", "/v1/unpublish", "/v1/link", "/v1/pages",
            "/v1/pages/{page_id}/delete"} <= {r.path for r in owner_only}
    hdr = _bearer(agent_key(svc))
    before = tokens.list_tokens(svc.owner)

    for route in owner_only:
        path = re.sub(r"\{(\w+)\}", lambda m: svc.owner if m.group(1) == "owner" else "x",
                      route.path)
        for method in route.methods:
            got = svc.client.request(method, path, headers=hdr,
                                     **({"json": {}} if method == "POST" else {}))
            assert got.status_code == 403, (method, route.path, got.status_code)
            assert got.json() == {"detail": "this action needs an owner token"}

    assert tokens.list_tokens(svc.owner) == before and uploads.export_path(svc.owner).exists()


def test_the_acting_dependency_takes_an_agent_key_and_no_other(svc):
    """`access.agent_token` on a route of its own: it hands the route the agent's row, and
    refuses a reader's key (anyone the owner shared with), the owner's own, none, and an agent's
    once it is revoked."""
    probe = FastAPI()

    @probe.post("/act")
    def act(auth: dict = Depends(access.agent_token)) -> dict:
        return {"owner": auth["owner"], "role": auth["role"], "label": auth["label"]}

    client = TestClient(probe)
    agent = agent_key(svc, "Instinct")

    assert client.post("/act", headers=_bearer(agent)).json() == {
        "owner": svc.owner, "role": "agent", "label": "Instinct"}
    for refused in (svc.reader_hdr, svc.owner_hdr):
        got = client.post("/act", headers=refused)
        assert got.status_code == 403 and got.json() == {
            "detail": "this action needs an agent token"}
    assert client.post("/act").status_code == 401

    assert tokens.revoke(svc.owner, tokens.token_hash(agent))
    assert client.post("/act", headers=_bearer(agent)).status_code == 401


# ── listed and revoked with the readers ──────────────────────────────────────────

def test_an_agent_key_is_listed_with_its_role_and_revoked_like_a_readers(svc):
    agent = agent_key(svc, "Instinct")
    listed = svc.client.get("/v1/tokens", headers=svc.owner_hdr).json()["tokens"]
    assert sorted((t["role"], t["label"]) for t in listed) == [
        ("agent", "Instinct"), ("owner", "David's KB"), ("reader", "a reader")]

    sha = next(t["token_sha256"] for t in listed if t["role"] == "agent")
    assert svc.client.post("/v1/revoke", json={"token_sha256": sha},
                           headers=svc.owner_hdr).json() == {"revoked": True}
    assert svc.client.get(f"/v1/kb/{svc.owner}/feed", headers=_bearer(agent)).status_code == 401
    assert svc.client.get(f"/v1/kb/{svc.owner}/feed", headers=svc.reader_hdr).status_code == 200
