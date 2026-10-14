"""The connector: a Google account reads its own library's served copy, counted, and asks for
changes to the library, which its Mac makes.

The export, the upload and the reads are the real ones (`conftest.svc`). Google sign-in is the
one stand-in: `connector._caller_sub` is what `GoogleProvider` would have verified. For the four
library writes the Mac is a thread that takes each call through the real routes
(`conftest.Mac`) and answers in the local tools' own shapes.
"""
from __future__ import annotations

import asyncio
import re
import time
from types import SimpleNamespace

import pytest
from fastmcp import Client

from pipeline.sharing import agent_invite, desktop_only, invite, page_words
from pipeline.topics import create
from service import (connector, db, grants, library_writes, line, publications, reading, tokens,
                     usage)
from tests.service.conftest import LABEL, Mac

SUB = "google-123"


def call(tool: str, **args) -> dict:
    async def run():
        async with Client(connector.build()) as client:
            return (await client.call_tool(tool, args)).structured_content
    return asyncio.run(run())


def link(sub: str, owner: str) -> None:
    conn = db.connect()
    conn.execute("INSERT INTO accounts (google_sub, owner) VALUES (?, ?)", (sub, owner))
    conn.commit()
    conn.close()


def reads(owner: str) -> int:
    conn = db.connect()
    try:
        return usage.total_reads(conn, owner)
    finally:
        conn.close()


@pytest.fixture()
def signed_in(monkeypatch):
    monkeypatch.setattr(connector, "_caller_sub", lambda: SUB)


def test_an_account_with_no_library_is_told_where_to_set_one_up(svc, signed_in):
    for tool, args in (("search", {"query": "agent"}), ("status", {})):
        got = call(tool, **args)
        assert got["status"] == "no_library" and connector.SETUP_URL in got["message"]


def test_a_linked_library_not_uploaded_yet_says_so(svc, signed_in):
    link(SUB, "not-uploaded")
    assert call("search", query="agent")["status"] == "not_arrived"


def test_search_is_keyword_only_over_the_linked_library(svc, signed_in):
    link(SUB, svc.owner)
    got = call("search", query="autonomous")
    assert got["hits"]
    assert all(h["kb"] == svc.owner for h in got["hits"])


def test_every_read_is_counted_and_makes_the_builder_owe_a_push(svc, signed_in):
    link(SUB, svc.owner)
    before = publications.publish_demand(svc.owner)["reads_since_last_upload"]
    hit = call("search", query="autonomous")["hits"][0]
    call("open", atom_id=hit["atom_id"])
    call("aggregate")
    call("latest")
    assert reads(svc.owner) == 4
    assert publications.publish_demand(svc.owner)["reads_since_last_upload"] == before + 4


def test_own_answers_with_items_say_how_to_make_something_readable(svc, signed_in):
    link(SUB, svc.owner)
    hit = call("search", query="autonomous")
    assert hit["readable"] == page_words.READABLE
    assert call("open", atom_id=hit["hits"][0]["atom_id"])["readable"] == page_words.READABLE
    assert "readable" not in call("search", query="zzzunmatched")


def test_status_is_not_a_read(svc, signed_in):
    link(SUB, svc.owner)
    got = call("status")
    assert got["status"] == "ready" and got["items"] > 0 and got["search"] == "keyword"
    assert reads(svc.owner) == 0


def test_latest_pages_with_the_cursor_it_hands_back(svc, signed_in):
    link(SUB, svc.owner)
    first = call("latest", limit=2)
    assert len(first["items"]) == 2
    nxt = call("latest", since=first["next"]["since"], since_id=first["next"]["since_id"])
    assert {i["atom_id"] for i in nxt["items"]}.isdisjoint(i["atom_id"] for i in first["items"])


def test_another_account_never_reads_this_library(svc, monkeypatch):
    link(SUB, svc.owner)
    monkeypatch.setattr(connector, "_caller_sub", lambda: "someone-else")
    assert call("search", query="autonomous")["status"] == "no_library"


def test_an_unsigned_call_reads_nothing(svc):
    link(SUB, svc.owner)
    assert call("aggregate")["status"] == "no_library"


def test_the_reads_are_declared_reads_and_only_unshare_and_forget_are_destructive():
    """Pinned as sets, so a tool that changes sides, or a new one, is a decision made here."""
    async def listed():
        async with Client(connector.build()) as client:
            return await client.list_tools()
    tools = asyncio.run(listed())
    writes = {"reader_view", "share", "unshare", "hopper", "forget", "watch", "add_oracle",
              "accept"}
    assert {t.name for t in tools} == {"search", "open", "aggregate", "latest", "status"} | writes
    assert len(tools) == 13
    for t in tools:
        # By alias: the field names differ between the MCP SDK here and the service box's.
        hints = t.annotations.model_dump(by_alias=True)
        assert t.annotations.title, t.name
        assert hints["readOnlyHint"] is (t.name not in writes), t.name
        assert hints["destructiveHint"] is (t.name in {"unshare", "forget"}), t.name
    titles = {t.name: t.annotations.title for t in tools}
    assert (titles["hopper"], titles["forget"], titles["watch"], titles["add_oracle"]) == (
        "Save a link", "Forget an item or Oracle", "Follow topics", "Add an Oracle")
    # A link is read, a person looked up and a subject searched outside the library.
    open_world = {t.name for t in tools if t.annotations.model_dump(by_alias=True)["openWorldHint"]}
    assert {"hopper", "watch", "add_oracle"} <= open_world and "forget" not in open_world


def test_the_no_library_answer_names_every_app_opyt_installs_into(svc, signed_in):
    message = call("status")["message"]
    for app in ("Claude Desktop", "Claude Code", "ChatGPT Desktop", "Codex"):
        assert app in message
    assert "connect Opyt to Google" in message and "same Google account" in message
    assert connector.SETUP_URL == "https://useopyt.com/connector.html"


def test_each_account_reads_only_the_library_linked_to_it(svc, monkeypatch):
    """Two libraries on the service, two Google accounts: every hit, open and count stays with
    the caller's own."""
    other = tokens.mint_token("other", "owner")
    svc.client.post("/v1/upload/other", content=svc.export.read_bytes(),
                    headers={"Authorization": f"Bearer {other}"})
    link(SUB, svc.owner)
    link("google-other", "other")
    for sub, mine in ((SUB, svc.owner), ("google-other", "other")):
        monkeypatch.setattr(connector, "_caller_sub", lambda sub=sub: sub)
        hits = call("search", query="autonomous")["hits"]
        assert hits and {h["kb"] for h in hits} == {mine}
        assert call("open", atom_id=hits[0]["atom_id"])["kb"] == mine
        assert call("latest")["kb"] == mine and call("aggregate")["kb"] == mine
    assert reads(svc.owner) == 4 and reads("other") == 4


def test_search_notices_read_as_the_owners_own_library(svc, signed_in):
    """The served-copy notice ("not your own; pass kb to open") is false here, and an unknown
    author is stated, not answered with an instruction to search the web."""
    link(SUB, svc.owner)
    plain = call("search", query="autonomous")
    assert all(n["code"] != "foreign_kb" for n in plain["notices"])

    unknown = call("search", query="autonomous", who="nobody-we-follow")
    said = {n["code"]: n["message"] for n in unknown["notices"]}
    assert "nobody-we-follow" in said["who_unresolved"]
    assert "web" not in said["who_unresolved"] and "kb=" not in " ".join(said.values())


def test_search_narrows_to_one_person_by_id(svc, signed_in):
    link(SUB, svc.owner)
    everyone = call("search", query="agent", k=20)
    assert len({h["who_id"] for h in everyone["hits"]}) > 1
    got = call("search", query="agent", who_id="github:root", k=20)
    assert got["hits"] and {h["who_id"] for h in got["hits"]} == {"github:root"}


def test_nothing_the_connector_says_names_muse_or_scripts_the_model():
    """§3.2/§4.3: descriptions and fixed answers are factual, and the connector is not presented
    as part of any one client."""
    async def listed():
        async with Client(connector.build()) as client:
            return await client.list_tools()
    texts = [t.description or "" for t in asyncio.run(listed())]
    texts += [connector._INSTRUCTIONS, connector._NO_LIBRARY, connector._NOT_ARRIVED]
    for text in texts:
        assert "Muse" not in text and "SAY" not in text, text


def test_the_server_names_itself_as_opyt():
    """What FastMCP's consent page shows before Google's: the name, linked to the site, and the
    icon."""
    mcp = connector.build()
    assert mcp.name == "Opyt" and mcp.website_url == connector.SITE_URL
    assert [i.src for i in mcp.icons] == [connector.ICON_URL]


def test_a_long_item_is_read_in_pages(svc, signed_in, monkeypatch):
    """Claude takes about 150,000 characters per tool answer, and a library holds items longer
    than that, so `open` pages the text instead of cutting it off."""
    link(SUB, svc.owner)
    monkeypatch.setattr(reading, "OPEN_CHARS_MAX", 20)
    atom = call("search", query="autonomous")["hits"][0]["atom_id"]
    first = call("open", atom_id=atom)
    assert len(first["raw"]) == 20 and first["next_offset"] == 20 and first["raw_chars"] > 20

    text, page = first["raw"], first
    while page["next_offset"] is not None:
        page = call("open", atom_id=atom, offset=page["next_offset"])
        text += page["raw"]
    assert len(text) == first["raw_chars"] and page["body_state"] == "partial"


def test_an_offset_past_the_text_says_so_and_is_not_a_read(svc, signed_in):
    link(SUB, svc.owner)
    atom = call("search", query="autonomous")["hits"][0]["atom_id"]
    before = reads(svc.owner)
    got = call("open", atom_id=atom, offset=10_000_000)
    assert "outside this item's text" in got["error"] and "next_offset" in got["error"]
    assert reads(svc.owner) == before


def test_search_pages_continue_one_ranking(svc, signed_in, monkeypatch):
    """More results than one answer holds come in pages: the pages, joined, are the ranking a
    single deep search returns, and the notice says how to get the next page."""
    link(SUB, svc.owner)
    monkeypatch.setattr(reading, "K_MAX", 2)
    whole = [h["atom_id"] for h in reading.reads.run_kb_search(
        "agent framework autonomous tools", k=50, kb=svc.owner)["hits"]]
    assert len(whole) > 2, "the fixture must match more than one page"

    seen, offset, first = [], 0, None
    while offset is not None:
        got = call("search", query="agent framework autonomous tools", k=2, offset=offset)
        first = first or got
        seen += [h["atom_id"] for h in got["hits"]]
        offset = got["next_offset"]
    assert seen == whole
    cut = next(n for n in first["notices"] if n["code"] == "results_truncated")
    assert "Pass offset=2" in cut["message"] and "raise k" not in cut["message"]


def test_a_search_offset_past_the_depth_says_so_and_is_not_a_read(svc, signed_in):
    link(SUB, svc.owner)
    before = reads(svc.owner)
    got = call("search", query="agent", offset=reading.SEARCH_DEPTH_MAX)
    assert "outside" in got["error"] and "next_offset" in got["error"]
    assert reads(svc.owner) == before

# ── reader_view: the one write ───────────────────────────────────────────────────────────

def test_a_page_of_the_linked_librarys_items_is_made_and_read_at_its_link(svc, signed_in,
                                                                         monkeypatch):
    monkeypatch.setenv(connector.BASE_URL_ENV, "https://api.test")
    link(SUB, svc.owner)
    got = call("reader_view", atom_ids=["x:1", "nope:1", "substack:post1"], title="Rollups",
               answer="Rollups got **cheap**.")
    assert got["status"] == "made" and got["missing"] == ["nope:1"]
    assert "do not also make an artifact" in got["message"]
    assert got["url"].startswith("https://api.test/p/") and "1 of the ids" in got["message"]
    assert "deleted 90 days after it was last opened" in got["message"]
    html = svc.client.get(got["url"].removeprefix("https://api.test")).text
    assert "<h1>Rollups</h1>" in html and "2 sources" in html
    assert "<strong>cheap</strong>" in html
    # Read through the served copy, so each item counts as a read.
    assert reads(svc.owner) == 3


def test_a_page_of_nothing_in_the_library_is_not_made(svc, signed_in):
    link(SUB, svc.owner)
    got = call("reader_view", atom_ids=["nope:1"], title="x")
    assert got["status"] == "not_found" and got["missing"] == ["nope:1"]


def test_only_the_account_whose_library_made_a_page_takes_it_down(svc, signed_in, monkeypatch):
    link(SUB, svc.owner)
    path = call("reader_view", atom_ids=["x:1"], title="x")["url"]
    link("someone-else", "another-library")
    monkeypatch.setattr(connector, "_caller_sub", lambda: "someone-else")
    monkeypatch.setattr(connector.serving, "is_served", lambda owner: True)
    assert call("reader_view", delete=path)["status"] == "not_found"
    monkeypatch.setattr(connector, "_caller_sub", lambda: SUB)
    assert call("reader_view", delete=path)["status"] == "deleted"
    assert svc.client.get(path).status_code == 404


def test_a_page_needs_a_library(svc, signed_in):
    assert call("reader_view", atom_ids=["x:1"])["status"] == "no_library"


def test_a_page_shows_a_long_item_whole_not_one_open_page(svc, signed_in, monkeypatch):
    """`open` answers in 60,000-character pages; a reader-view page is a document, so it carries
    the item's whole text."""
    monkeypatch.setenv(connector.BASE_URL_ENV, "https://api.test")
    monkeypatch.setattr(reading, "OPEN_CHARS_MAX", 20)
    link(SUB, svc.owner)
    whole = reading.reads.kb_open("substack:post1", kb=svc.owner)["raw"]
    assert len(whole) > 20, "the fixture item must be longer than one open page"
    url = call("reader_view", atom_ids=["substack:post1"], title="Long")["url"]
    html = svc.client.get(url.removeprefix("https://api.test")).text
    # The words that run across the 20-character cut, which a one-page read would split.
    across = whole[12:]
    assert len(whole) > 20 and across in html, across


# ── share and unshare: invites and reader tokens, on this service ─────────────────────────

def grant_count() -> int:
    conn = db.connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM grant_codes").fetchone()[0]
    finally:
        conn.close()


def reader_labels(owner: str) -> list[str]:
    return sorted(t["label"] for t in tokens.list_tokens(owner) if t["role"] == "reader")


def redeem(answer: dict) -> tuple[str, str]:
    """Redeem a `share` answer's invite, as the reader's own Opyt would: `(owner, token)`."""
    return grants.redeem_grant(invite.parse(answer["invite"], "https://unused").code)[:2]


def test_a_share_preview_mints_nothing_and_says_what_an_invite_gives(svc, signed_in):
    """The fixture's library already has one reader, "a reader"."""
    link(SUB, svc.owner)
    before = grant_count()
    got = call("share", for_whom="Maya")
    assert got["status"] == "preview" and got["items"] > 0 and got["readers"] == 1
    assert f"all {got['items']} items" in got["consent"][0]
    assert grant_count() == before and reads(svc.owner) == 0


def test_a_confirmed_share_is_an_invite_that_redeems_into_a_reader_of_this_library(svc,
                                                                                   signed_in):
    link(SUB, svc.owner)
    got = call("share", confirm=True, for_whom="Maya")
    assert got["status"] == "shared" and "Maya" in got["message"]
    owner, token = redeem(got)
    assert owner == svc.owner
    assert tokens.resolve_token(token)["label"] == "Maya"


def test_unshare_cuts_off_the_named_reader_only(svc, signed_in):
    link(SUB, svc.owner)
    redeem(call("share", confirm=True, for_whom="Maya"))
    _, ben = redeem(call("share", confirm=True, for_whom="Ben"))
    preview = call("unshare", reader="maya")
    assert preview["status"] == "preview" and preview["readers_remaining"] == 2
    assert reader_labels(svc.owner) == ["Ben", "Maya", "a reader"], "a preview revokes nothing"

    got = call("unshare", reader="Maya", confirm=True)
    assert got["status"] == "reader_revoked" and got["readers_remaining"] == 2
    assert reader_labels(svc.owner) == ["Ben", "a reader"]
    assert tokens.resolve_token(ben) is not None
    assert call("unshare", reader="Maya", confirm=True)["status"] == "no_such_reader"


def test_unshare_refuses_a_name_two_readers_share(svc, signed_in):
    link(SUB, svc.owner)
    for _ in range(2):
        redeem(call("share", confirm=True, for_whom="Sam"))
    got = call("unshare", reader="Sam", confirm=True)
    assert got["status"] == "ambiguous_reader" and len(got["readers"]) == 2
    assert reader_labels(svc.owner) == ["Sam", "Sam", "a reader"]


def handed_out(owner: str) -> list[tuple[str, str]]:
    """Every key but the owner's, as `(role, label)`."""
    return sorted((t["role"], t["label"]) for t in tokens.list_tokens(owner)
                  if t["role"] != "owner")


def test_an_agent_share_preview_mints_nothing_and_says_what_an_agent_may_do(svc, signed_in):
    link(SUB, svc.owner)
    before = grant_count()
    got = call("share", for_whom="Instinct", to="agent")
    said = " ".join(got["consent"])
    assert got["status"] == "preview" and got["items"] > 0
    for may in ("acts as the user", "save links", "remove items", "add people to follow",
                "invite or cut off readers", "until it is cut off with `unshare`"):
        assert may in said, may
    assert grant_count() == before and reads(svc.owner) == 0
    assert "acts as the user" not in " ".join(call("share", for_whom="Maya")["consent"])


def test_a_confirmed_agent_share_redeems_once_into_an_agent_key_of_this_library(svc, signed_in,
                                                                                monkeypatch):
    monkeypatch.setenv(connector.BASE_URL_ENV, "https://api.test")
    link(SUB, svc.owner)
    got = call("share", confirm=True, for_whom="Instinct", to="agent")
    assert got["status"] == "shared"
    assert "for their agent" in got["message"]
    assert "POST https://api.test/v1/redeem" in got["message"]

    code = invite.parse(got["invite"], "https://unused").code
    redeemed = svc.client.post("/v1/redeem", json={"code": code}).json()
    assert redeemed["role"] == "agent" and redeemed["owner"] == svc.owner
    assert tokens.resolve_token(redeemed["token"])["label"] == "Instinct"
    assert svc.client.post("/v1/redeem", json={"code": code}).status_code == 404
    assert handed_out(svc.owner) == [("agent", "Instinct"), ("reader", "a reader")]


def test_unshare_lists_an_agent_marked_and_cuts_it_off_by_name(svc, signed_in):
    link(SUB, svc.owner)
    _, maya = redeem(call("share", confirm=True, for_whom="Maya"))
    _, agent = redeem(call("share", confirm=True, for_whom="Instinct", to="agent"))

    listed = call("unshare", reader="nobody by this name")
    assert listed["status"] == "no_such_reader"
    assert {r["label"]: r.get("agent", False) for r in listed["readers"]} == {
        "a reader": False, "Maya": False, "Instinct": True}

    preview = call("unshare", reader="instinct")
    assert preview["status"] == "preview" and preview["agent"] is True
    assert preview["readers_remaining"] == 2
    assert "an agent that acts as the user" in preview["consent"][0]
    assert "can no longer search this knowledge base or change it" in preview["consent"][1]
    assert tokens.resolve_token(agent) is not None, "a preview revokes nothing"

    got = call("unshare", reader="Instinct", confirm=True)
    assert got["status"] == "reader_revoked" and got["agent"] is True
    assert got["readers_remaining"] == 2 and "change it" in got["message"]
    assert tokens.resolve_token(agent) is None and tokens.resolve_token(maya) is not None
    assert call("unshare", reader="Instinct", confirm=True)["status"] == "no_such_reader"


def test_unshare_refuses_a_name_a_reader_and_an_agent_share(svc, signed_in):
    """An agent that read with a reader's key and is then given an agent's key under the same
    name holds two: the name alone picks neither, and the marked `id` picks the agent's."""
    link(SUB, svc.owner)
    redeem(call("share", confirm=True, for_whom="Instinct"))
    redeem(call("share", confirm=True, for_whom="Instinct", to="agent"))

    got = call("unshare", reader="Instinct", confirm=True)
    assert got["status"] == "ambiguous_reader"
    assert sorted(r.get("agent", False) for r in got["readers"]) == [False, True]
    assert handed_out(svc.owner) == [("agent", "Instinct"), ("reader", "Instinct"),
                                     ("reader", "a reader")]

    agents = next(r["id"] for r in got["readers"] if r.get("agent"))
    assert call("unshare", reader=agents, confirm=True)["status"] == "reader_revoked"
    assert handed_out(svc.owner) == [("reader", "Instinct"), ("reader", "a reader")]


def test_share_and_unshare_touch_only_the_callers_own_library(svc, monkeypatch):
    """Another account's `unshare` cannot see, name or revoke this library's readers, and its
    invites open its own library."""
    link(SUB, svc.owner)
    monkeypatch.setattr(connector, "_caller_sub", lambda: SUB)
    redeem(call("share", confirm=True, for_whom="Maya"))
    other = tokens.mint_token("other", "owner")
    svc.client.post("/v1/upload/other", content=svc.export.read_bytes(),
                    headers={"Authorization": f"Bearer {other}"})
    link("google-other", "other")
    monkeypatch.setattr(connector, "_caller_sub", lambda: "google-other")
    assert call("unshare", reader="Maya", confirm=True)["status"] == "no_such_reader"
    assert redeem(call("share", confirm=True))[0] == "other"
    assert reader_labels(svc.owner) == ["Maya", "a reader"]


def test_share_and_unshare_need_a_library(svc, signed_in):
    for tool, args in (("share", {"confirm": True}), ("unshare", {"reader": "Maya"})):
        assert call(tool, **args)["status"] == "no_library"


# ── the four library writes: left for the Mac, answered with what happened ────────────────

URL = "https://arxiv.org/abs/2401.00001"

# What the local tools answer, in their own shapes, with the lines they write for the Mac's host.
SAVED = {"status": "saved", "kind": "paper", "atom_id": "paper:2401.00001",
         "entry_mode": "user-saved", "reference": URL}
FORGOTTEN = {
    "status": "forgotten", "scope": "atom", "atom_id": "x:1", "description": "x:1 card",
    "who_id": "x:user:1", "when_ts": "2025-01-15", "source_url": "https://example/x:1",
    "references": {"atoms": 1, "chunks": 1, "frontier_candidates": 0}, "snapshot": True,
    "shared": True,
    "consent": ["This removes one atom: x:1 card.",
                "If this X post is still in your bookmarks, it will return on an hourly bookmark "
                "pass. Unbookmark it on x.com to keep it removed.",
                "This install is configured for sharing, so a served copy may contain this atom. "
                "It disappears from that copy after the next successful push with reader demand; "
                "a reader can still receive the previous copy before then."]}
FOLLOWED = {
    "status": "ok", "topic_id": "t_1a2b3c4d", "name": "Boltz-2",
    "watches": [{"name": "Boltz-2", "spellings": ["Boltz-2"], "sources": ["arxiv", "github"]}],
    "words": [], "people": [], "waiting": 0,
    "first": "The first search of each watch runs in the background and looks back 14 days; new "
             "finds and posts show up in latest().",
    "next": create.DAILY}
ADDED = {
    "added": {"canonical_id": "x:user:9", "name": "Andrej Karpathy", "source": "freeform",
              "was_already_oracle": False},
    "picks": [{"name": "Andrej Karpathy", "canonical_id": "x:user:9", "newest_posts": False,
               "archives": False, "reach": None, "asked": {"x": None, "web": "all", "papers": "all"},
               "stuck": {"cause": "x_session", "step": "onboard(source='x')"}}],
    "message": "Andrej Karpathy's writing is coming in on its own, and nothing waits on this "
               "conversation. Keep your Mac open and awake while it runs: with the lid closed on "
               "battery it pauses, and picks up again when the Mac wakes. "
               "`oracle(action='progress')` shows where each stands."}
LOCAL = {"hopper": SAVED, "forget": FORGOTTEN, "watch": FOLLOWED, "add_oracle": ADDED}

WRITES = (("hopper", {"reference": URL}),
          ("forget", {"atom_id": "x:1", "confirm": True}),
          ("watch", {"action": "create", "name": "Boltz-2", "watches": ["Boltz-2"]}),
          ("add_oracle", {"reference": "@karpathy", "x_lookback": "1yr"}))

# How a write is done, which no answer or description says: the Mac, the calls left for it, and
# the local tools the web does not have (`oracle`, `onboard`, `look`; `forget` has an `oracle`
# argument, so the tool is matched by its call).
HOW = re.compile(r"\bMacs?\b|queue|\bthe line\b|check[- ]in|onboard|(?<![\w`])oracle\("
                 r"|\blook\(|`look`|worker|background")


def as_the_local_tool(call: dict) -> dict:
    return LOCAL[call["tool"]]


def calls() -> list[dict]:
    conn = db.connect()
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM line_calls ORDER BY asked")]
    finally:
        conn.close()


def until(done, what: str) -> None:
    deadline = time.monotonic() + 5
    while not done():
        assert time.monotonic() < deadline, what
        time.sleep(0.01)


def texts(value) -> list[str]:
    """Every string in an answer."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in texts(v)]
    if isinstance(value, list):
        return [t for v in value for t in texts(v)]
    return []


def wait_for(monkeypatch, seconds: float) -> None:
    """How long a web call waits for its answer here, in place of 25 seconds."""
    monkeypatch.setitem(line.call.__kwdefaults__, "wait_s", seconds)


@pytest.fixture()
def lib(svc, signed_in, monkeypatch):
    """The signed-in account's library, with the line quickened: the table is read every 20 ms,
    and a web call waits at most 5 seconds for an answer the Mac here gives at once."""
    link(SUB, svc.owner)
    monkeypatch.setattr(line, "POLL_S", 0.02)
    wait_for(monkeypatch, 5)
    return svc


def test_each_write_reaches_the_mac_as_the_local_tools_own_call_and_answers_what_happened(lib):
    with Mac(lib, as_the_local_tool) as mac:
        saved, gone, topic, added = (call(tool, **args) for tool, args in WRITES)

    assert [(c["tool"], c["args"]) for c in mac.taken] == [
        ("hopper", {"reference": URL, "confirm": True}),
        ("forget", {"atom_id": "x:1", "confirm": True}),
        ("watch", {"action": "create", "name": "Boltz-2", "watches": ["Boltz-2"]}),
        ("add_oracle", {"reference": "@karpathy", "confirm": True, "x_lookback": "1yr"})]
    assert saved == SAVED
    assert gone == {"status": "forgotten", "scope": "atom", "atom_id": "x:1",
                    "description": "x:1 card", "who_id": "x:user:1", "when_ts": "2025-01-15",
                    "source_url": "https://example/x:1", "consent": FORGOTTEN["consent"][:2]}
    assert topic == {"status": "ok", "topic_id": "t_1a2b3c4d", "name": "Boltz-2",
                     "watches": FOLLOWED["watches"], "words": [], "people": [], "waiting": 0,
                     "message": library_writes._FOLLOWED}
    assert added == {"status": "added", "name": "Andrej Karpathy", "canonical_id": "x:user:9",
                     "was_already_oracle": False}
    for answer in (saved, gone, topic, added):
        for text in texts(answer):
            assert not HOW.search(text) and "latest" not in text, text
            assert "computer" not in text and "install" not in text, text


def test_with_nothing_on_the_line_a_write_is_left_for_later_in_one_plain_sentence(lib):
    said = [call(tool, **args) for tool, args in WRITES]
    assert said == [
        {"status": "later", "message": "It'll be in your library when your computer is next on."},
        {"status": "later",
         "message": "It'll be removed from your library when your computer is next on."},
        {"status": "later", "message": "Your topics will be changed when your computer is next on."},
        {"status": "later",
         "message": "They'll be added to your library when your computer is next on."}]
    assert [(c["tool"], c["state"]) for c in calls()] == [
        ("hopper", "waiting"), ("forget", "waiting"), ("watch", "waiting"),
        ("add_oracle", "waiting")]


@pytest.mark.parametrize("action", ["list", "preview"])
def test_topics_cannot_be_shown_with_nothing_on_the_line_and_no_call_is_left(lib, action):
    """A list or a preview has nothing to do later, so it is not left to run when nobody waits."""
    got = call("watch", action=action, name="Boltz-2", watches=["Boltz-2"])
    assert got == {"status": "unavailable", "message": "Topics can't be shown right now."}
    assert calls() == []


def test_a_write_taken_and_not_finished_within_the_wait_is_being_done(lib, monkeypatch):
    wait_for(monkeypatch, 0.3)
    with Mac(lib):
        assert call("hopper", reference=URL) == {
            "status": "working", "message": "It's being saved now. `status` says how it ends."}
        assert call("add_oracle", reference="@karpathy")["status"] == "working"
    assert [c["state"] for c in calls()] == ["running", "running"]


def test_a_call_the_mac_could_not_run_fails_with_its_reason_or_the_tools_own_sentence(lib):
    with Mac(lib, lambda c: {"message": "That link cannot be read."}, ok=False):
        assert call("hopper", reference=URL) == {"status": "failed",
                                                 "message": "That link cannot be read."}
    with Mac(lib, lambda c: {}, ok=False):
        assert call("forget", atom_id="x:1", confirm=True) == {
            "status": "failed", "message": "It could not be removed."}


def test_an_outcome_that_came_late_is_told_once_on_the_next_answer_of_any_tool(lib):
    assert call("hopper", reference=URL)["status"] == "later"
    assert "earlier" not in call("search", query="autonomous"), "nothing has ended yet"

    with Mac(lib, as_the_local_tool):
        until(lambda: calls()[0]["state"] == "done", "the Mac never saved the link")

    found = call("search", query="autonomous")
    assert found["earlier"] == [f"{URL} is saved in the library."] and found["hits"]
    assert "earlier" not in call("search", query="autonomous")
    assert "earlier" not in call("status")
    (change,) = call("status")["changes"]
    assert (change["tool"], change["args"], change["answer"]) == (
        "hopper", {"reference": URL}, SAVED)
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", change["asked"])


def test_a_preview_that_outlasts_the_wait_is_read_back_with_status(lib, monkeypatch):
    wait_for(monkeypatch, 0.3)
    preview = {"status": "preview", "name": "Boltz-2",
               "watches": [{"name": "Boltz-2", "spellings": ["Boltz-2"],
                            "by_source": {"arxiv": 4, "x": "unavailable"}, "found_last_month": 4,
                            "examples": []}],
               "words": [], "people": [], "posts_last_month": 0, "posts_per_day": 0.0,
               "next": "… say you will check its finds before showing any …"}
    with Mac(lib) as mac:
        got = call("watch", action="preview", name="Boltz-2", watches=["Boltz-2"])
        assert got == {"status": "working", "message": "It is still being put together. "
                                                       "`status` returns it once it is done."}
        lib.client.post(f"/v1/line/{mac.taken[0]['id']}/answer", headers=lib.owner_hdr,
                        json={"ok": True, "answer": preview})

    assert call("latest")["earlier"] == ["The preview of Boltz-2 is ready: `status` returns it."]
    (change,) = call("status")["changes"]
    assert change["args"]["action"] == "preview"
    assert change["answer"] == {**{k: v for k, v in preview.items() if k != "next"},
                                "message": library_writes._PREVIEWED}


def test_another_account_never_sees_or_is_handed_this_librarys_calls(svc, monkeypatch):
    other = SimpleNamespace(owner="other", owner_hdr={
        "Authorization": f"Bearer {tokens.mint_token('other', 'owner')}"})
    svc.client.post("/v1/upload/other", content=svc.export.read_bytes(), headers=other.owner_hdr)
    link(SUB, svc.owner)
    link("google-other", "other")
    monkeypatch.setattr(line, "POLL_S", 0.02)

    monkeypatch.setattr(connector, "_caller_sub", lambda: SUB)
    assert call("hopper", reference=URL)["status"] == "later"

    monkeypatch.setattr(connector, "_caller_sub", lambda: "google-other")
    with Mac(other, as_the_local_tool) as theirs:
        assert call("status")["changes"] == []
        time.sleep(0.3)
    assert theirs.taken == [], "another library's Mac is never handed the call"

    with Mac(svc, as_the_local_tool):
        until(lambda: calls()[0]["state"] == "done", "the Mac never saved the link")
    assert "earlier" not in call("search", query="autonomous"), "nor told how it ended"
    assert call("status")["changes"] == []
    monkeypatch.setattr(connector, "_caller_sub", lambda: SUB)
    assert call("search", query="autonomous")["earlier"] == [f"{URL} is saved in the library."]


def test_forget_previews_from_the_served_copy_and_leaves_no_call(lib):
    """The consent is the local tool's own: a saved X post can come back from the bookmarks. The
    sentences about the install's sharing and about what only the Mac holds are not in it."""
    one = call("forget", atom_id="x:1")
    assert one["status"] == "preview"
    assert one["atoms"] == [{"atom_id": "x:1", "description": "x:1 card", "who_id": "x:user:1",
                             "when_ts": "2025-01-15", "source_url": "https://example/x:1"}]
    assert one["consent"][0] == "This removes one atom: x:1 card."
    assert any("bookmarks" in s for s in one["consent"])

    several = call("forget", atom_ids=["x:1", "blog:home", "x:1"])
    assert [a["atom_id"] for a in several["atoms"]] == ["x:1", "blog:home"]
    assert several["consent"][0] == "This removes these 2 atoms."

    missing = call("forget", atom_ids=["x:1", "nope:1"])
    assert missing["status"] == "not_found" and missing["unknown"] == ["nope:1"]

    oracle = call("forget", oracle="@alice")
    assert oracle == {
        "status": "preview", "scope": "oracle", "canonical_id": "x:user:1", "name": "Alice",
        "atoms_kept": 1,
        "consent": ["This stops tracking one Oracle: Alice (x:user:1).",
                    "Their 1 existing atoms, identity records and trust roots remain.",
                    "Future scheduled refreshes will not select this person. Removal takes "
                    "effect at once: whatever a running refresh brings in after it is not kept. "
                    "Bookmarks and separately saved links can still arrive.",
                    "Adding this Oracle again starts a new subscription."]}
    assert call("forget", oracle="nobody we follow")["status"] == "not_found"

    for answer in (one, several, missing, oracle):
        for text in texts(answer):
            assert not HOW.search(text), text
            assert not re.search(r"install|served copy|delete_atoms|registrations", text), text
    assert calls() == [] and reads(lib.owner) == 0


@pytest.mark.parametrize("args", [{}, {"atom_id": "x:1", "oracle": "@alice"},
                                  {"atom_ids": []}, {"oracle": "  "},
                                  {"atom_id": "x:1", "atom_ids": ["x:2"]}])
def test_forget_names_exactly_one_thing_or_nothing_is_asked(lib, args):
    got = call("forget", confirm=True, **args)
    assert got == {"status": "error",
                   "message": "Give exactly one of atom_id, atom_ids or oracle."}
    assert calls() == []


def test_watch_offers_five_actions_and_asks_nothing_for_any_other(lib):
    got = call("watch", action="checked")
    assert got["status"] == "error" and "preview, create, edit, drop, list" in got["reason"]
    assert calls() == []

    async def schema():
        async with Client(connector.build()) as client:
            tool = next(t for t in await client.list_tools() if t.name == "watch")
            return tool.model_dump(by_alias=True)["inputSchema"]
    assert set(asyncio.run(schema())["properties"]) == {
        "action", "name", "watches", "words", "people", "topic_id", "drop_watches"}


def test_adding_nobody_is_refused_before_anything_is_asked(lib):
    assert call("add_oracle", reference="  ")["status"] == "not_added"
    assert calls() == []


def test_the_library_writes_need_a_library(svc, signed_in):
    for tool, args in WRITES:
        assert call(tool, **args)["status"] == "no_library"
    assert calls() == []


def test_status_lists_no_changes_when_none_were_asked(lib):
    assert call("status")["changes"] == []


def test_a_failure_to_tell_an_earlier_outcome_never_breaks_a_tool_call(lib, monkeypatch):
    def broken(owner):
        raise RuntimeError("an answer this version cannot read")
    monkeypatch.setattr(library_writes, "earlier", broken)
    assert call("search", query="autonomous")["hits"]


def test_nothing_the_writes_say_tells_how_they_are_done():
    """Results, not plumbing: no description and no fixed sentence names the Mac, the calls left
    for it, or a tool the web does not have. The one mention of the computer is the sentence for a
    write that waits for it."""
    async def listed():
        async with Client(connector.build()) as client:
            return await client.list_tools()
    described = {t.name: t.description for t in asyncio.run(listed())
                 if t.name in dict(WRITES)}
    assert set(described) == set(dict(WRITES))
    said = [s for kind in (*library_writes._WORDING.values(), library_writes._SHOWING)
            for s in (kind.working, kind.later, kind.failed) if s]
    fixed = [*said, library_writes._CANT_SHOW, library_writes._PREVIEWED,
             library_writes._FOLLOWED]
    for text in (*described.values(), connector._INSTRUCTIONS, *fixed):
        assert not HOW.search(text), HOW.search(text)
    for text in (*described.values(), connector._INSTRUCTIONS):
        assert "computer" not in text, text
    with_computer = [s for s in fixed if "computer" in s]
    assert with_computer == [kind.later for kind in library_writes._WORDING.values()]
    assert all(s.endswith("when your computer is next on.") for s in with_computer)
    # `latest` here carries no topics, so nothing about topics sends the host to it.
    assert "latest" not in described["watch"] + library_writes._FOLLOWED + library_writes._PREVIEWED


# ── a library someone shared with the caller ──────────────────────────────────────
#
# The friend has no Opyt and no library: they sign in to the connector with their own Google
# account and paste the invite. Everything they read is the owner's served copy.

FRIEND = "google-friend"


@pytest.fixture()
def friend(monkeypatch):
    monkeypatch.setattr(connector, "_caller_sub", lambda: FRIEND)


def invite_for(svc, *, label: str = "Leo", agent: bool = False, hdr: dict | None = None) -> str:
    """An invite link the way an owner gets one, for `svc`'s library or the one `hdr` owns."""
    code = svc.client.post("/v1/grant", json={"label": label, "agent": agent},
                           headers=hdr or svc.owner_hdr).json()["code"]
    return invite.link(code)


def test_an_account_with_no_library_accepts_an_invite_and_reads_it(svc, friend):
    got = call("accept", invite=invite_for(svc))
    assert (got["status"], got["kb"], got["from"]) == ("accepted", LABEL, LABEL)
    # Where a library of their own comes from is said with the acceptance.
    assert connector.SETUP_URL in got["message"]

    found = call("search", query="autonomous", kb=LABEL)
    hits = found["hits"]
    # A page holds only the user's own items, so a shared library's answers do not offer one.
    assert "readable" not in found
    assert hits and all(h["kb"] == LABEL for h in hits)
    assert call("open", atom_id=hits[0]["atom_id"], kb=LABEL)["raw"]
    assert call("aggregate", kb=LABEL)["total"] and call("latest", kb=LABEL)["items"]
    # One library to read and none of their own: they need not name it.
    assert call("search", query="autonomous")["hits"]
    # Counted, and the owner sees this reader under the name they invited.
    assert reads(svc.owner) == 5
    assert "Leo" in [t["label"] for t in tokens.list_tokens(svc.owner)]


def test_a_shared_library_is_read_and_nothing_else(svc, friend):
    call("accept", invite=invite_for(svc))
    for tool, args in (("hopper", {"reference": "https://example.com/a"}), ("share", {}),
                       ("reader_view", {"atom_ids": ["x:1"]}), ("status", {})):
        got = call(tool, **args)
        assert got["status"] == "no_library" and got["shared_with_you"] == [LABEL], tool
        assert connector.SETUP_URL in got["message"]


def test_the_owner_cutting_a_reader_off_ends_their_reads_here(svc, friend):
    call("accept", invite=invite_for(svc))
    held = next(t for t in tokens.list_tokens(svc.owner) if t["label"] == "Leo")
    tokens.revoke(svc.owner, held["token_sha256"])
    assert call("search", query="autonomous", kb=LABEL)["status"] == "access_ended"
    # Said once: nothing is kept about a share that has ended.
    assert call("search", query="autonomous", kb=LABEL)["status"] == "no_such_library"
    assert call("search", query="autonomous")["status"] == "no_library"


def test_an_agent_invite_is_refused_and_left_for_the_agent(svc, friend):
    link_ = invite_for(svc, agent=True)
    assert call("accept", invite=link_)["status"] == "agent_invite"
    redeemed = svc.client.post("/v1/redeem", json={"code": link_.rsplit("#", 1)[1]})
    assert redeemed.json()["role"] == "agent"


def test_a_used_invite_and_a_paste_with_no_code_say_so(svc, friend):
    link_ = invite_for(svc)
    assert call("accept", invite=link_)["status"] == "accepted"
    assert call("accept", invite=link_)["status"] == "code_unavailable"
    assert call("accept", invite="here you go")["status"] == "not_an_invite"


def test_an_account_reads_its_own_library_unless_kb_names_a_shared_one(svc, signed_in):
    mine = tokens.mint_token("mine", "owner", label="Mine")
    svc.client.post("/v1/upload/mine", content=svc.export.read_bytes(),
                    headers={"Authorization": f"Bearer {mine}"})
    link(SUB, "mine")
    call("accept", invite=invite_for(svc))
    assert all(h["kb"] == "mine" for h in call("search", query="autonomous")["hits"])
    assert all(h["kb"] == LABEL for h in call("search", query="autonomous", kb=LABEL)["hits"])
    assert call("status")["shared_with_you"] == [LABEL]
    assert call("search", query="autonomous", kb="nobody")["status"] == "no_such_library"


def test_two_libraries_of_one_name_are_told_apart_and_a_second_invite_keeps_the_name(svc, friend):
    other = tokens.mint_token("other", "owner", label=LABEL)
    other_hdr = {"Authorization": f"Bearer {other}"}
    assert call("accept", invite=invite_for(svc))["kb"] == LABEL
    second = call("accept", invite=invite_for(svc, hdr=other_hdr))
    assert (second["kb"], second["renamed_from"]) == (f"{LABEL}-2", LABEL)
    again = call("accept", invite=invite_for(svc, label="Leo again"))
    assert again["kb"] == LABEL and "renamed_from" not in again



# ── what only the desktop app does, and who an invite is for ──────────────────────

def test_the_connector_says_what_only_the_desktop_app_does(svc, signed_in):
    """One sentence, the same the user hears when they connect: in the server's instructions,
    and in `status`'s description for a client that shows a model only the tools."""
    async def listed():
        async with Client(connector.build()) as client:
            return await client.list_tools()
    described = {t.name: t.description for t in asyncio.run(listed())}
    assert desktop_only.SENTENCE in connector._INSTRUCTIONS
    assert desktop_only.SENTENCE in described["status"]


def test_share_takes_who_it_is_for_and_refuses_another_word(svc, signed_in):
    link(SUB, svc.owner)
    assert call("share", to="agent")["consent"] == list(agent_invite.CONSENT)
    assert call("share", confirm=True, to="connector")["status"] == "error"
