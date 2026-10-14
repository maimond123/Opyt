"""The owner's agent's routes (`/v1/library/…`): every tool the connector offers beyond reading,
for an agent that calls web addresses and holds an agent key.

Each route is a second thin entry to a flow the connector's tool already enters, so the test that
matters is that the two give the same answer; the rest is who may call, and what an agent may not
do that its owner may.
"""
from __future__ import annotations

import pytest

from service import line, tokens
from tests.service.conftest import Mac, agent_key
from tests.service.test_connector import (  # noqa: F401  (`lib` and `signed_in` are fixtures)
    SAVED, URL, WRITES, as_the_local_tool, call, calls, lib, signed_in, until)

# Every route, with a body it accepts.
ROUTES = (("get", "status", None),
          ("post", "reader_view", {"atom_ids": ["x:1"]}),
          ("post", "share", {}),
          ("post", "unshare", {"reader": "a reader"}),
          ("post", "hopper", {"reference": URL}),
          ("post", "forget", {"atom_id": "x:1"}),
          ("post", "watch", {"action": "list"}),
          ("post", "add_oracle", {"reference": "@karpathy"}))


def send(svc, method: str, tool: str, body: dict | None, headers: dict | None):
    kw = {"headers": headers} if headers else {}
    if method == "get":
        return svc.client.get(f"/v1/library/{tool}", **kw)
    return svc.client.post(f"/v1/library/{tool}", json=body, **kw)


@pytest.fixture()
def agent(lib):
    """An agent key for the signed-in account's library: both ways in to the same flows."""
    return {"Authorization": f"Bearer {agent_key(lib)}"}


def act(svc, key: dict, tool: str, /, **body) -> dict:
    r = svc.client.post(f"/v1/library/{tool}", json=body, headers=key)
    assert r.status_code == 200, r.text
    return r.json()


def test_only_an_agent_key_reaches_them(svc):
    for method, tool, body in ROUTES:
        for headers, refused in ((None, 401), (svc.reader_hdr, 403), (svc.owner_hdr, 403)):
            assert send(svc, method, tool, body, headers).status_code == refused, tool
    assert calls() == [], "a refused call left something for the Mac"


def test_an_agent_of_a_library_that_has_not_arrived_is_told_so(svc):
    ghost = {"Authorization": f"Bearer {tokens.mint_token('ghost', 'agent')}"}
    for method, tool, body in ROUTES:
        assert send(svc, method, tool, body, ghost).status_code == 404, tool


def test_each_write_answers_the_agent_as_it_answers_the_connector(lib, agent):
    with Mac(lib, as_the_local_tool) as mac:
        for tool, args in WRITES:
            assert act(lib, agent, tool, **args) == call(tool, **args), tool
    # The same call reached the Mac from both ways in.
    assert [(c["tool"], c["args"]) for c in mac.taken[::2]] == [
        (c["tool"], c["args"]) for c in mac.taken[1::2]]


def test_the_tools_this_service_answers_itself_answer_the_agent_as_the_connector(lib, agent):
    told = lib.client.get("/v1/library/status", headers=agent).json()
    assert told == call("status") and told["status"] == "ready" and told["items"] > 0
    assert act(lib, agent, "share") == call("share")
    assert act(lib, agent, "unshare", reader="a reader") == call("unshare", reader="a reader")
    assert act(lib, agent, "forget", atom_id="x:1") == call("forget", atom_id="x:1")
    assert act(lib, agent, "forget", atom_id="x:1")["status"] == "preview"
    assert calls() == [], "a preview left something for the Mac"


def test_a_reader_view_made_by_an_agent_is_a_page_anyone_with_the_link_reads(lib, agent):
    made = act(lib, agent, "reader_view", atom_ids=["x:1", "nope:1"], title="Rollups")
    assert made["status"] == "made" and made["missing"] == ["nope:1"]
    path = made["url"][made["url"].index("/p/"):]
    assert lib.client.get(path).status_code == 200
    assert act(lib, agent, "reader_view", delete=made["url"])["status"] == "deleted"


def test_an_agent_invites_a_person_and_never_another_agent(lib, agent):
    """`agent` is not an argument here: an agent that could mint agent keys could outlive the
    owner cutting it off."""
    before = len(tokens.list_tokens(lib.owner))
    got = act(lib, agent, "share", confirm=True, for_whom="Maya", agent=True)
    assert got["status"] == "shared"
    redeemed = lib.client.post("/v1/redeem", json={"code": got["invite"].split("#")[1]}).json()
    assert "role" not in redeemed
    assert tokens.resolve_token(redeemed["token"])["role"] == "reader"
    assert len(tokens.list_tokens(lib.owner)) == before + 1


def test_an_agent_cuts_a_reader_off(lib, agent):
    assert act(lib, agent, "unshare", reader="a reader", confirm=True)["status"] == "reader_revoked"
    assert lib.client.get(f"/v1/kb/{lib.owner}/feed", headers=lib.reader_hdr).status_code == 401


def test_an_outcome_that_came_late_is_told_to_the_agent_once(lib, agent):
    assert act(lib, agent, "hopper", reference=URL)["status"] == "later"
    with Mac(lib, as_the_local_tool):
        until(lambda: calls()[0]["state"] == line.DONE, "the Mac never saved the link")
    told = lib.client.get("/v1/library/status", headers=agent).json()
    assert told["earlier"] == [f"{URL} is saved in the library."]
    assert "earlier" not in lib.client.get("/v1/library/status", headers=agent).json()
    assert SAVED["status"] == told["changes"][0]["answer"]["status"]
