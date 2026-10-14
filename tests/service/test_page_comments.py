"""Comments on a page of sources: read, added and taken down at `/p/{id}/comments` by anyone
holding the page's link, and deleted wherever their page is.

The service, its database and the routes are the real ones (`conftest.svc`). What a reader sees is
drawn by the page's script, in a browser. Here it is what the service keeps, the rules that refuse
a comment (`comments._checked`, checked with plain values), and the key that deletes one.
"""
from __future__ import annotations

import time

import pytest

from service import db
from service.pages import comments, make, store

POST = {"atom_id": "x:1", "source_type": "x", "source_url": "https://x.com/alice/status/1",
        "raw": "---\nauthor: alice\n---\nrollups are cheap now", "payload": {}}
ON_A_PASSAGE = {"name": "  Maya   R. ", "body": " Is that still true? ", "item": 1,
                "quote": "cheap now", "prefix": "x" * 100 + "rollups are ", "suffix": "." * 100}


def page(svc) -> str:
    return make.create(svc.owner, [POST], "Rollups", None, None, time.time())["path"]


def said(svc, path: str, **fields) -> dict:
    r = svc.client.post(f"{path}/comments", json={"name": "Maya", "body": "Hm.", "item": 1,
                                                  **fields})
    assert r.status_code == 200, r.text
    return r.json()


def listed(svc, path: str) -> list[dict]:
    return svc.client.get(f"{path}/comments").json()["comments"]


def kept() -> int:
    conn = db.connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM page_comments").fetchone()[0]
    finally:
        conn.close()


# ── read and added ────────────────────────────────────────────────────────────────

def test_a_comment_on_a_passage_comes_back_to_whoever_reads_the_page(svc):
    path = page(svc)
    got = svc.client.post(f"{path}/comments", json=ON_A_PASSAGE).json()
    r = svc.client.get(f"{path}/comments")
    [c] = r.json()["comments"]
    assert c == got["comment"] and c["thread"] == c["id"]
    assert (c["name"], c["body"], c["item"], c["quote"]) == ("Maya R.", "Is that still true?",
                                                             1, "cheap now")
    # the text either side is only for finding the passage again: a little of it is kept
    assert c["prefix"] == ("x" * 100 + "rollups are ")[-comments.MAX_CONTEXT:]
    assert c["suffix"] == "." * comments.MAX_CONTEXT
    assert r.json()["limits"] == comments.LIMITS
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-robots-tag"] == "noindex, nofollow"


def test_reading_the_comments_never_gives_out_what_deletes_them(svc):
    path = page(svc)
    key = said(svc, path)["key"]
    text = svc.client.get(f"{path}/comments").text
    assert key not in text and "key" not in text


def test_a_reply_joins_its_thread_and_a_reply_cannot_be_replied_to(svc):
    path = page(svc)
    first = said(svc, path)["comment"]
    reply = said(svc, path, thread=first["id"], item=None)["comment"]
    assert reply["thread"] == first["id"] and reply["item"] is None and reply["quote"] is None
    for thread in (reply["id"], "no-such-comment"):
        r = svc.client.post(f"{path}/comments",
                            json={"name": "Tom", "body": "Me too.", "thread": thread})
        assert r.status_code == 400 and "gone" in r.json()["detail"]
    assert [c["id"] for c in listed(svc, path)] == [first["id"], reply["id"]]


def test_a_page_that_is_not_there_has_no_comments_to_read_or_add(svc):
    for path in ("/p/AAAAAAAAAAAAAAAAAAAAAA", "/p/x"):
        assert svc.client.get(f"{path}/comments").status_code == 404
        r = svc.client.post(f"{path}/comments", json={"name": "Maya", "body": "Hm.", "item": 1})
        assert r.status_code == 404
    assert kept() == 0


# ── what is refused ───────────────────────────────────────────────────────────────

def _there(n: int) -> list[store.Comment]:
    return [store.Comment(f"c{i}", f"c{i}", 1, None, None, None, "Maya", "Hm.", 0.0)
            for i in range(n)]


@pytest.mark.parametrize("fields, there, says", [
    ({"name": "  ", "body": "Hm.", "item": 1}, [], "name"),
    ({"name": "Maya", "body": " \n ", "item": 1}, [], "words"),
    ({"name": "M" * (comments.MAX_NAME + 1), "body": "Hm.", "item": 1}, [], "name"),
    ({"name": "Maya", "body": "y" * (comments.MAX_BODY + 1), "item": 1}, [], "2,000"),
    ({"name": "Maya", "body": "Hm."}, [], "sources"),
    ({"name": "Maya", "body": "Hm.", "item": 0}, [], "sources"),
    ({"name": "Maya", "body": "Hm.", "item": 3}, [], "sources"),
    ({"name": "Maya", "body": "Hm.", "item": 1, "quote": "q" * (comments.MAX_QUOTE + 1)}, [],
     "Select"),
    ({"name": "Maya", "body": "Hm.", "item": 1}, _there(comments.MAX_PER_PAGE), "500"),
])
def test_a_comment_that_cannot_be_kept_is_refused_with_a_sentence(fields, there, says):
    with pytest.raises(comments.Refused, match=says):
        comments._checked(fields, sources=2, there=there, comment_id="new", now=1.0)


def test_a_refused_comment_keeps_nothing(svc):
    path = page(svc)
    r = svc.client.post(f"{path}/comments", json={"name": "Maya", "body": "Hm.", "item": 2})
    assert r.status_code == 400 and r.json()["detail"].endswith(".") and kept() == 0


# ── taking one down ───────────────────────────────────────────────────────────────

def test_only_the_key_a_comment_came_back_with_deletes_it(svc):
    path, other = page(svc), page(svc)
    got = said(svc, path)
    delete = f"/comments/{got['comment']['id']}/delete"
    for where, key in ((path, "a-guess"), (other, got["key"])):
        r = svc.client.post(where + delete, json={"key": key})
        assert r.json() == {"deleted": False}
    assert len(listed(svc, path)) == 1
    assert svc.client.post(path + delete, json={"key": got["key"]}).json() == {"deleted": True}
    assert listed(svc, path) == []


def test_deleting_the_comment_that_started_a_thread_deletes_its_replies(svc):
    path = page(svc)
    first, apart = said(svc, path), said(svc, path)
    reply = said(svc, path, thread=first["comment"]["id"])
    r = svc.client.post(f"{path}/comments/{reply['comment']['id']}/delete",
                        json={"key": reply["key"]})
    assert r.json() == {"deleted": True} and len(listed(svc, path)) == 2
    said(svc, path, thread=first["comment"]["id"])
    svc.client.post(f"{path}/comments/{first['comment']['id']}/delete",
                    json={"key": first["key"]})
    assert [c["id"] for c in listed(svc, path)] == [apart["comment"]["id"]]


def test_a_pages_comments_go_wherever_the_page_goes(svc):
    now = time.time()
    stays = page(svc)
    said(svc, stays)
    # taken down by whoever made it
    down = page(svc)
    said(svc, down)
    svc.client.post(f"/v1/pages/{down.rsplit('/', 1)[-1]}/delete", headers=svc.owner_hdr, json={})
    assert kept() == 1
    # expired, and asked for
    old = make.create(svc.owner, [POST], "Old", None, None, now - store.LIFETIME - 60)
    store.add_comment(old["id"], _there(1)[0], "hash")
    assert kept() == 2 and svc.client.get(old["path"]).status_code == 404 and kept() == 1
    # expired, and swept at night
    older = make.create(svc.owner, [POST], "Older", None, None, now - store.LIFETIME - 60)
    store.add_comment(older["id"], _there(2)[1], "hash")
    assert store.purge(now) == 1 and kept() == 1
    assert len(listed(svc, stays)) == 1
    # and when the library that made them is deleted
    assert store.delete_all(svc.owner) == 1 and kept() == 0
