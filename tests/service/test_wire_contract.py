"""The service's HTTP API, pinned byte for byte.

An install and the hosted service can run different versions, so every `/v1/*` route, its request
and its response is a contract between versions. These tests drive each of the 16 routes once,
plus the refusals a client meets, and compare the exact response bytes with
`wire_contract.json`, captured from the code on 2026-10-04 (`223b11f0`) before the sharing
restructure moved anything. The delta route and its four cases were added on 2026-10-13
(docs/plans/2026-10-13-shared-copy-deltas.md), ruled by David. Recaptured on 2026-10-13 for the
one other ruled contract change: whole-item search deleted `GET /meta` and the search body's
`mode`/`query_vector` (docs/plans/2026-10-13-whole-item-search.md).
The agent key's cases (an agent grant and redeem, an agent key on read and owner-only routes) and
the `offset` cases for search and open were added the same day
(docs/plans/2026-10-13-connector-writes.md §13, §14.7). They added one optional field to three
request bodies, which `openapi` lists; no other existing case changed. So were the two line
routes and their eight cases (§14.2 of that plan), and the eight routes an owner's agent acts
through with their fifteen (§13): the published schema gained them, and no other existing case
changed. Recaptured on 2026-10-13 for two sentences in `/v1/library/share`'s answers: where an
invite is redeemed now names claude.ai and ChatGPT through the connector, which David approved
(docs/plans/2026-10-13-one-share.md, part 2). No field and no other case changed.
Recaptured again the same day for part 1 of that plan, which David approved ("yes build this"):
`GET /v1/tokens` gained `google_linked_at`, and `POST /v1/unlink` and its two cases were added.
No other case changed.
Recaptured on 2026-10-14 for reader view's answer, which David asked for ("do 2 and 3, then 1
too"): the page body and the agent's reader view body gained an optional `answer`, which `openapi`
lists, and the reader view's `made` message gained a sentence telling the host not to also make
an artifact (docs/plans/2026-10-14-reader-view-answer.md). No other case changed.

Only values that differ on every run are normalised: bearer tokens and grant codes (`<secret>`),
token hashes and file digests (`<sha256>`), timestamps (`<time>`), keys `register` assigns
(`<key>`), the temp home (`<home>`), and the export's size (`<bytes>`), which follows the store's
schema rather than this service. Everything else, key order included, is compared as sent.
One ordering is canonicalised: `/v1/tokens` lists tokens by `created_at, token_sha256`, and two
tokens minted in the same second fall back to their random hashes, so its rows are compared
sorted by role and label, re-serialised with the separators FastAPI uses.

A failure here means a response changed. Change the golden file only for a contract change that
was ruled on, by running this file with `OPYT_WIRE_CAPTURE=1`, and say so in the commit.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date
import os
import re
import shutil
from pathlib import Path

import pytest

from pipeline.sharing import delta, fingerprint
from service import db, line
from tests.service.conftest import agent_key

GOLDEN = Path(__file__).with_name("wire_contract.json")
CAPTURE = os.environ.get("OPYT_WIRE_CAPTURE") == "1"

_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?")
_SHA = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
_SECRET = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{43}(?![A-Za-z0-9_-])")

_captured: dict[str, list] = {}


def _normalise(text: str, *, keys=(), home=None, size=None) -> str:
    if home is not None:
        text = text.replace(str(home), "<home>")
    if size is not None:
        text = re.sub(rf"\b{size}\b", "<bytes>", text)
    for key in keys:
        text = text.replace(key, "<key>")
    text = _TIME.sub("<time>", text)
    text = _SHA.sub("<sha256>", text)
    return _SECRET.sub("<secret>", text)


def _check(case: str, r, *, keys=(), home=None, size=None, body: str | None = None) -> None:
    got = [r.status_code, r.headers.get("content-type"),
           _normalise(r.text if body is None else body, keys=keys, home=home, size=size)]
    if CAPTURE:
        _captured[case] = got
        return
    want = json.loads(GOLDEN.read_text())[case]
    assert got == want, f"{case}: the response changed\n got: {got}\nwant: {want}"


@pytest.fixture(scope="module", autouse=True)
def _write_golden():
    yield
    if CAPTURE and _captured:
        old = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        old.update(_captured)
        GOLDEN.write_text(json.dumps(old, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _stranger(svc):
    """A second knowledge base: registered, granted, redeemed, never uploaded."""
    me = svc.client.post("/v1/register", json={"label": "Leo"}).json()
    code = svc.client.post("/v1/grant", json={"label": "a friend"},
                           headers=_bearer(me["token"])).json()["code"]
    reader = svc.client.post("/v1/redeem", json={"code": code, "install_id": "i-2"}).json()
    return me, reader


# ── the four reads ───────────────────────────────────────────────────────────────

def test_search(svc):
    _check("search", svc.client.post(
        f"/v1/kb/{svc.owner}/search", headers=svc.reader_hdr,
        json={"query": "agent framework", "k": 3, "as_kb": "x"}))


def test_open(svc):
    _check("open", svc.client.post(f"/v1/kb/{svc.owner}/open", headers=svc.reader_hdr,
                                   json={"atom_id": "github:root/agentkit", "as_kb": "x"}))
    _check("open_missing", svc.client.post(f"/v1/kb/{svc.owner}/open", headers=svc.reader_hdr,
                                           json={"atom_id": "nope:1"}))


def test_aggregate(svc):
    _check("aggregate", svc.client.post(f"/v1/kb/{svc.owner}/aggregate",
                                        headers=svc.reader_hdr, json={"as_kb": "x"}))


def test_feed(svc):
    _check("feed", svc.client.get(f"/v1/kb/{svc.owner}/feed", headers=svc.reader_hdr,
                                  params={"limit": 3, "as_kb": "x"}))
    _check("feed_bad_date", svc.client.get(f"/v1/kb/{svc.owner}/feed", headers=svc.reader_hdr,
                                           params={"date_from": "yesterday"}))


# ── who may read ─────────────────────────────────────────────────────────────────

def test_read_refusals(svc):
    path = f"/v1/kb/{svc.owner}/feed"
    _check("read_no_token", svc.client.get(path))
    _check("read_not_bearer", svc.client.get(path, headers={"Authorization": "Basic x"}))
    _check("read_unknown_token", svc.client.get(path, headers=_bearer("not-a-token")))
    _check("read_owner_token", svc.client.get(path, headers=svc.owner_hdr))
    me, reader = _stranger(svc)
    _check("read_other_kb", svc.client.get(f"/v1/kb/{me['owner']}/feed",
                                           headers=svc.reader_hdr), keys=[me["owner"]])
    _check("read_not_arrived", svc.client.post(f"/v1/kb/{me['owner']}/search",
                                               headers=_bearer(reader["token"]),
                                               json={"query": "agent"}), keys=[me["owner"]])


# ── publishing ───────────────────────────────────────────────────────────────────

def test_upload(svc):
    _check("upload", svc.client.post(f"/v1/upload/{svc.owner}", headers=svc.owner_hdr,
                                     content=svc.export.read_bytes()), home=svc.home,
           size=svc.upload["bytes"])
    _check("upload_other_name", svc.client.post("/v1/upload/someone", headers=svc.owner_hdr,
                                                content=b"x"))
    _check("upload_reader_token", svc.client.post(f"/v1/upload/{svc.owner}",
                                                  headers=svc.reader_hdr, content=b"x"))


def test_upload_delta(svc, tmp_path):
    """Changes-only uploads (docs/plans/2026-10-13-shared-copy-deltas.md): a copy uploaded whole
    with its digest stamped, then a delta cut against it, then one cut against another copy."""
    whole = tmp_path / "stamped.db"
    shutil.copyfile(svc.export, whole)
    held = fingerprint.of_copy(whole)
    fingerprint.stamp(whole, held.digest)
    assert svc.client.post(f"/v1/upload/{svc.owner}", headers=svc.owner_hdr,
                           content=whole.read_bytes()).status_code == 200
    path = f"/v1/upload/{svc.owner}/delta"

    cut = tmp_path / "push.delta"
    delta.build(whole, fingerprint.diff(held, held), held.digest, held.digest, cut)
    r = svc.client.post(path, headers=svc.owner_hdr, content=cut.read_bytes())
    _check("upload_delta", r, size=r.json()["bytes"])
    delta.build(whole, fingerprint.diff(held, held), "0" * 64, held.digest, cut)
    _check("upload_delta_stale", svc.client.post(path, headers=svc.owner_hdr,
                                                 content=cut.read_bytes()))
    _check("upload_delta_reader_token", svc.client.post(path, headers=svc.reader_hdr,
                                                        content=b"x"))
    _check("upload_delta_no_token", svc.client.post(path, content=b"x"))


def test_register(svc):
    r = svc.client.post("/v1/register", json={"label": "Leo"})
    _check("register", r, keys=[r.json()["owner"]])
    r = svc.client.post("/v1/register", json={})
    _check("register_no_label", r, keys=[r.json()["owner"]])


def test_grant_and_redeem(svc):
    _check("grant", svc.client.post("/v1/grant", headers=svc.owner_hdr, json={"label": "Leo"}))
    _check("grant_reader_token", svc.client.post("/v1/grant", headers=svc.reader_hdr, json={}))
    code = svc.client.post("/v1/grant", headers=svc.owner_hdr, json={}).json()["code"]
    _check("redeem", svc.client.post("/v1/redeem", json={"code": code, "install_id": "i-9"}))
    _check("redeem_spent", svc.client.post("/v1/redeem", json={"code": code}))


def test_tokens(svc):
    svc.client.post(f"/v1/kb/{svc.owner}/search", headers=svc.reader_hdr,
                    json={"query": "agent"})
    r = svc.client.get("/v1/tokens", headers=svc.owner_hdr)
    body = r.json()
    body["tokens"].sort(key=lambda t: (t["role"], t["label"] or ""))
    _check("tokens", r, body=json.dumps(body, ensure_ascii=False, separators=(",", ":")))
    _check("tokens_reader_token", svc.client.get("/v1/tokens", headers=svc.reader_hdr))


def test_revoke(svc):
    reader = svc.client.get("/v1/tokens", headers=svc.owner_hdr).json()["tokens"]
    sha = next(t["token_sha256"] for t in reader if t["role"] == "reader")
    _check("revoke", svc.client.post("/v1/revoke", headers=svc.owner_hdr,
                                     json={"token_sha256": sha}))
    _check("revoke_again", svc.client.post("/v1/revoke", headers=svc.owner_hdr,
                                           json={"token_sha256": sha}))


def test_unlink(svc):
    """Cutting the Google account off, added with the one `share` (2026-10-13): nothing linked
    answers false, and only the owner's token may ask."""
    _check("unlink_nothing_linked", svc.client.post("/v1/unlink", headers=svc.owner_hdr))
    _check("unlink_reader_token", svc.client.post("/v1/unlink", headers=svc.reader_hdr))


def test_unpublish(svc):
    _check("unpublish", svc.client.post("/v1/unpublish", headers=svc.owner_hdr))
    _check("unpublish_again", svc.client.post("/v1/unpublish", headers=svc.owner_hdr))


# ── the agent key, and search and open in pages ──────────────────────────────────

def test_agent_grant_and_redeem(svc):
    """An agent invite is granted like any other, with `agent` in the body, and its redeem says
    which kind of key came back. `grant` and `redeem` above pin that a reader's did not change."""
    r = svc.client.post("/v1/grant", headers=svc.owner_hdr,
                        json={"label": "Instinct", "agent": True})
    _check("grant_agent", r)
    code = r.json()["code"]
    _check("redeem_agent", svc.client.post("/v1/redeem", json={"code": code, "install_id": "i-9"}))
    _check("redeem_agent_spent", svc.client.post("/v1/redeem", json={"code": code}))


def test_agent_key(svc):
    """An agent key reads its own library and no other, is refused where an owner token is
    needed, and is listed with its role."""
    hdr = _bearer(agent_key(svc))
    _check("read_agent_token", svc.client.get(f"/v1/kb/{svc.owner}/feed", headers=hdr))
    me, _reader = _stranger(svc)
    _check("read_agent_other_kb", svc.client.get(f"/v1/kb/{me['owner']}/feed", headers=hdr),
           keys=[me["owner"]])
    _check("grant_agent_token", svc.client.post("/v1/grant", headers=hdr, json={}))
    _check("upload_agent_token", svc.client.post(f"/v1/upload/{svc.owner}", headers=hdr,
                                                 content=b"x"))
    _check("unpublish_agent_token", svc.client.post("/v1/unpublish", headers=hdr))
    r = svc.client.get("/v1/tokens", headers=svc.owner_hdr)
    body = r.json()
    body["tokens"].sort(key=lambda t: (t["role"], t["label"] or ""))
    _check("tokens_with_agent", r, body=json.dumps(body, ensure_ascii=False,
                                                   separators=(",", ":")))


def test_search_and_open_in_pages(svc):
    """`offset` in the body: a page of hits with where the next one starts, and a part of an
    item's text. `search_bm25` and `open` above pin the answers with no `offset`."""
    search = {"query": "agent framework autonomous tools", "k": 2, "as_kb": "x"}
    for case, offset in (("search_offset", 0), ("search_offset_next", 2),
                         ("search_offset_outside", 250)):
        _check(case, svc.client.post(f"/v1/kb/{svc.owner}/search", headers=svc.reader_hdr,
                                     json={**search, "offset": offset}))
    for case, offset in (("open_offset", 21), ("open_offset_outside", 10_000)):
        _check(case, svc.client.post(f"/v1/kb/{svc.owner}/open", headers=svc.reader_hdr,
                                     json={"atom_id": "github:root/agentkit", "as_kb": "x",
                                           "offset": offset}))


# ── pages of sources ─────────────────────────────────────────────────────────────

def test_pages(svc):
    """The two routes an install calls (`client.make_page`, `client.delete_page`). The page's id
    differs on every run."""
    item = {"atom_id": "x:1", "source_type": "x", "raw": "a post", "payload": {}}
    r = svc.client.post("/v1/pages", headers=svc.owner_hdr,
                        json={"title": "Rollups", "note": None, "items": [item]})
    keys = [r.json()["id"]]
    _check("page_make", r, keys=keys)
    _check("page_make_empty", svc.client.post("/v1/pages", headers=svc.owner_hdr,
                                              json={"title": "x", "items": []}))
    _check("page_make_reader_token", svc.client.post("/v1/pages", headers=svc.reader_hdr,
                                                     json={"items": [item]}))
    delete = f"/v1/pages/{keys[0]}/delete"
    _check("page_delete", svc.client.post(delete, headers=svc.owner_hdr, json={}), keys=keys)
    _check("page_delete_again", svc.client.post(delete, headers=svc.owner_hdr, json={}),
           keys=keys)


# ── the line a Mac holds ─────────────────────────────────────────────────────────

def test_line(svc):
    """The two routes an install's line job calls (docs/plans/2026-10-13-connector-writes.md,
    §14.2): it asks for the next call left for it, and posts its answer. A call's id differs on
    every run."""
    ask = "/v1/line/next"
    _check("line_next_no_writer", svc.client.get(ask, headers=svc.owner_hdr, params={"wait": 0}))
    conn = db.connect()
    conn.execute("INSERT INTO accounts (google_sub, owner) VALUES ('google-123', ?)", (svc.owner,))
    conn.commit()
    conn.close()
    _check("line_next_nothing", svc.client.get(ask, headers=svc.owner_hdr, params={"wait": 0}))
    asyncio.run(line.call(svc.owner, "hopper", {"reference": "https://example.com/a",
                                                "confirm": True}, wait_s=0))
    r = svc.client.get(ask, headers=svc.owner_hdr)
    keys = [r.json()["call"]["id"]]
    _check("line_next_call", r, keys=keys)
    answer = f"/v1/line/{keys[0]}/answer"
    body = {"ok": True, "answer": {"status": "saved"}}
    _check("line_answer", svc.client.post(answer, headers=svc.owner_hdr, json=body), keys=keys)
    _check("line_answer_again", svc.client.post(answer, headers=svc.owner_hdr, json=body),
           keys=keys)
    _check("line_answer_no_ok", svc.client.post(answer, headers=svc.owner_hdr,
                                                json={"answer": {}}), keys=keys)
    _check("line_next_reader_token", svc.client.get(ask, headers=svc.reader_hdr))
    _check("line_next_no_token", svc.client.get(ask))


# ── the owner's agent ────────────────────────────────────────────────────────────

def test_library_routes(svc):
    """The eight routes an owner's agent acts through (docs/plans/2026-10-13-connector-writes.md,
    §13): each tool once with an agent key, with no Mac on the line, and the refusal any other
    key gets. A page's id and a reader's short id differ on every run."""
    hdr = _bearer(agent_key(svc))

    def post(tool: str, body: dict):
        return svc.client.post(f"/v1/library/{tool}", headers=hdr, json=body)

    _check("library_status", svc.client.get("/v1/library/status", headers=hdr))
    _check("library_reader_token", svc.client.get("/v1/library/status", headers=svc.reader_hdr))
    _check("library_owner_token", svc.client.get("/v1/library/status", headers=svc.owner_hdr))
    _check("library_no_token", svc.client.get("/v1/library/status"))
    made = post("reader_view", {"atom_ids": ["x:1"], "title": "Rollups"})
    keys = [made.json()["url"].rsplit("/", 1)[-1]]
    _check("library_reader_view", made, keys=keys)
    _check("library_reader_view_delete", post("reader_view", {"delete": made.json()["url"]}),
           keys=keys)
    _check("library_share_preview", post("share", {}))
    _check("library_share", post("share", {"confirm": True, "for_whom": "Maya"}))
    preview = post("unshare", {"reader": "a reader"})
    _check("library_unshare_preview", preview, keys=[preview.json()["reader_id"]])
    _check("library_unshare", post("unshare", {"reader": "a reader", "confirm": True}))
    _check("library_forget_preview", post("forget", {"atom_id": "x:1"}))
    _check("library_hopper_later", post("hopper", {"reference": "https://example.com/a"}))
    _check("library_forget_later", post("forget", {"atom_id": "x:1", "confirm": True}))
    _check("library_watch_unavailable", post("watch", {"action": "list"}))
    _check("library_add_oracle_later", post("add_oracle", {"reference": "@karpathy"}))


# ── the public numbers ───────────────────────────────────────────────────────────

def test_stats(svc):
    svc.client.post(f"/v1/kb/{svc.owner}/search", headers=svc.reader_hdr,
                    json={"query": "nothing matches zzzz"})
    svc.client.post(f"/v1/kb/{svc.owner}/open", headers=svc.reader_hdr,
                    json={"atom_id": "github:root/agentkit"})
    _check("stats_json", svc.client.get("/v1/stats"), size=svc.upload["bytes"])
    _check("stats_page", svc.client.get("/stats"))


# ── the schema FastAPI publishes ─────────────────────────────────────────────────

def _without_descriptions(node):
    if isinstance(node, dict):
        return {k: _without_descriptions(v) for k, v in node.items() if k != "description"}
    if isinstance(node, list):
        return [_without_descriptions(v) for v in node]
    return node


def test_openapi_schema(svc):
    """`/openapi.json` is served too, and it is built from the route functions: their names, their
    parameters and the request bodies. Moving a dependency or renaming a handler changes it.

    Compared WITHOUT its `description` fields, which are the handlers' and models' docstrings:
    prose that has to stay true as the code around it moves, and that no client reads."""
    r = svc.client.get("/openapi.json")
    _check("openapi", r, body=json.dumps(_without_descriptions(r.json()), ensure_ascii=False,
                                         separators=(",", ":")))
