"""The line: a tool call the web leaves for the owner's Mac, taken and answered over the two
routes the Mac calls.

The decisions are checked with plain values; the rest runs through the real routes against the
real `service.db` (`conftest.svc`). Waits are real time, kept short: the table is read every 20 ms
here instead of every second, except where the one-second promise itself is measured.
"""
from __future__ import annotations

import asyncio
import re
import threading
import time

import pytest
from fastapi.testclient import TestClient

from service import db, deletion, line, tokens
from service.app import app
from tests.service.conftest import Mac

SUB = "google-123"
LINK = {"reference": "https://example.com/a", "confirm": True}


def link(sub: str, owner: str) -> None:
    conn = db.connect()
    conn.execute("INSERT INTO accounts (google_sub, owner) VALUES (?, ?)", (sub, owner))
    conn.commit()
    conn.close()


def ask(owner: str, args: dict = LINK, **how) -> line.Result:
    """A web tool's call for `owner`'s Mac."""
    return asyncio.run(line.call(owner, "hopper", args, **how))


def nxt(svc, wait: float = 0, headers: dict | None = None) -> dict:
    """One ask of the Mac's."""
    r = svc.client.get("/v1/line/next", params={"wait": wait}, headers=headers or svc.owner_hdr)
    assert r.status_code == 200, r.text
    return r.json()


def answer(svc, call_id: str, said: dict, *, ok: bool = True, headers: dict | None = None) -> bool:
    r = svc.client.post(f"/v1/line/{call_id}/answer", json={"ok": ok, "answer": said},
                        headers=headers or svc.owner_hdr)
    assert r.status_code == 200, r.text
    return r.json()["recorded"]


def rows(table: str = "line_calls") -> list[dict]:
    conn = db.connect()
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
    finally:
        conn.close()


@pytest.fixture()
def quick(monkeypatch):
    monkeypatch.setattr(line, "POLL_S", 0.02)


@pytest.fixture()
def lib(svc, quick):
    """The fixture library with a Google account linked, so something can ask its Mac."""
    link(SUB, svc.owner)
    return svc


# ── the decisions, with plain values ───────────────────────────────────────────────────

def test_a_call_is_offered_until_taken_and_again_once_it_has_run_too_long():
    assert line.offered(line.WAITING, None, 100.0)
    assert not line.offered(line.RUNNING, 100.0, 100.0 + line.RERUN_AFTER_S)
    assert line.offered(line.RUNNING, 100.0, 100.0 + line.RERUN_AFTER_S + 1)
    assert line.RERUN_AFTER_S == 600
    for ended in line.ENDED:
        assert not line.offered(ended, 100.0, 1e12)


def test_the_macs_answer_ends_a_call_done_or_failed():
    assert (line.ended_as(True), line.ended_as(False)) == (line.DONE, line.FAILED)


def test_a_line_is_held_for_a_little_longer_than_one_ask():
    assert not line.is_held(None, 100.0)
    assert line.is_held(100.0, 100.0 + line.HELD_S)
    assert not line.is_held(100.0, 100.0 + line.HELD_S + 1)
    assert line.HELD_S > line.NEXT_WAIT_MAX_S == 50


# ── a call's life, through the routes ──────────────────────────────────────────────────

def test_a_call_waits_is_taken_and_ends_with_the_macs_answer(lib):
    assert ask(lib.owner) == line.Result(line.WAITING, None), "nobody is on the line yet"
    assert [r["state"] for r in rows()] == [line.WAITING]

    call = nxt(lib)["call"]
    assert re.fullmatch(r"[0-9a-f]{32}", call["id"])
    assert (call["tool"], call["args"]) == ("hopper", LINK)
    assert [r["state"] for r in rows()] == [line.RUNNING]
    assert nxt(lib) == {"call": None}, "a running call is not handed over twice"

    assert answer(lib, call["id"], {"status": "saved"}) is True
    (row,) = rows()
    assert (row["state"], row["answer"]) == (line.DONE, '{"status": "saved"}')


def test_a_call_the_mac_could_not_run_ends_failed(lib):
    ask(lib.owner)
    call = nxt(lib)["call"]
    assert answer(lib, call["id"], {"message": "that tool is not one this Mac runs"}, ok=False)
    assert rows()[0]["state"] == line.FAILED


def test_the_oldest_call_is_handed_over_first(lib):
    for n in (1, 2, 3):
        ask(lib.owner, {"reference": f"https://example.com/{n}"})
    assert [nxt(lib)["call"]["args"]["reference"][-1] for _ in range(3)] == ["1", "2", "3"]
    assert nxt(lib) == {"call": None}


def test_next_answers_at_once_when_a_call_is_waiting(lib, monkeypatch):
    monkeypatch.setattr(line, "POLL_S", 1.0)
    ask(lib.owner)
    started = time.monotonic()
    assert nxt(lib, wait=10)["call"]["tool"] == "hopper"
    assert time.monotonic() - started < 0.5


def test_next_answers_within_about_a_second_of_a_call_arriving_while_it_waits(lib, monkeypatch):
    """Measured with the real one-second look at the table."""
    monkeypatch.setattr(line, "POLL_S", 1.0)
    got: dict = {}

    def hold():
        r = TestClient(app).get("/v1/line/next", params={"wait": 10}, headers=lib.owner_hdr)
        got.update(r.json(), at=time.monotonic())

    mac = threading.Thread(target=hold)
    mac.start()
    time.sleep(0.3)
    assert not got, "nothing is waiting, so the ask is still open"
    asked = time.monotonic()
    ask(lib.owner, wait_s=0)
    mac.join(timeout=5)
    assert got["call"]["args"] == LINK
    assert got["at"] - asked < 2.0


def test_next_answers_no_call_when_none_arrived_within_the_wait(lib):
    started = time.monotonic()
    assert nxt(lib, wait=0.3) == {"call": None}
    assert 0.3 <= time.monotonic() - started < 1.5


def test_the_wait_is_never_longer_than_fifty_seconds(lib, monkeypatch):
    monkeypatch.setattr(line, "NEXT_WAIT_MAX_S", 0.2)
    started = time.monotonic()
    assert nxt(lib, wait=3600) == {"call": None}
    assert time.monotonic() - started < 1.5


def test_a_library_nothing_can_write_to_is_told_to_come_back_later(svc, quick):
    assert nxt(svc, wait=10) == {"call": None, "retry_after": 3600}


def test_an_agent_key_counts_as_a_writer_as_a_google_link_does(svc, quick):
    """The role is another part's to mint; here a token row with it is enough."""
    assert not line.has_writer(svc.owner)
    conn = db.connect()
    conn.execute("INSERT INTO tokens (token_sha256, owner, role) VALUES ('an-agent', ?, 'agent')",
                 (svc.owner,))
    conn.commit()
    conn.close()
    assert line.has_writer(svc.owner) and nxt(svc) == {"call": None}
    link(SUB, "another-library")
    assert not line.has_writer("no-such-library")


def test_an_agent_invite_nobody_redeemed_yet_counts_as_a_writer(svc, quick):
    """So the line is already held when the agent's first call comes, not an hour later."""
    assert "retry_after" in nxt(svc)
    svc.client.post("/v1/grant", json={"label": "Instinct", "agent": True}, headers=svc.owner_hdr)
    assert nxt(svc) == {"call": None}


def test_a_readers_invite_is_no_writer(svc, quick):
    svc.client.post("/v1/grant", json={"label": "Leo"}, headers=svc.owner_hdr)
    assert "retry_after" in nxt(svc)


def test_a_mac_running_a_call_is_on_the_line_though_it_asks_for_nothing(lib, monkeypatch):
    """A Mac runs one call at a time and asks for nothing while it does. A long call must not
    make the next one read as left for a computer that is off."""
    ask(lib.owner, wait_s=0)
    nxt(lib)                                            # the Mac takes it, and goes quiet
    monkeypatch.setattr(line, "HELD_S", 0.0)            # its last ask is old news
    assert line.held(lib.owner)
    assert ask(lib.owner, {"reference": "https://example.com/b", "confirm": True},
               wait_s=0.1).state == line.RUNNING


def test_a_call_no_mac_is_there_for_is_waiting(lib):
    assert not line.held(lib.owner)
    assert ask(lib.owner, wait_s=0.1).state == line.WAITING


def test_each_ask_records_that_the_line_is_held(lib):
    assert not line.held(lib.owner)
    nxt(lib)
    assert line.held(lib.owner)
    assert [r["owner"] for r in rows("line_seen")] == [lib.owner]
    nxt(lib)
    assert len(rows("line_seen")) == 1, "one row per library, moved on each ask"


def test_a_call_running_too_long_is_offered_again_and_answered_once(lib, monkeypatch):
    """The Mac stopped mid-call, so its next ask is handed the call again. Whichever run answers
    first ends it; the other's answer is not recorded."""
    monkeypatch.setattr(line, "RERUN_AFTER_S", 0.05)
    ask(lib.owner)
    first = nxt(lib)["call"]
    time.sleep(0.1)
    again = nxt(lib)["call"]
    assert again == first

    assert answer(lib, first["id"], {"status": "saved"}) is True
    assert answer(lib, first["id"], {"status": "already_present"}) is False
    (row,) = rows()
    assert (row["state"], row["answer"]) == (line.DONE, '{"status": "saved"}')
    time.sleep(0.1)
    assert nxt(lib) == {"call": None}, "a call that ended is never offered again"


def test_an_answer_is_recorded_only_for_a_running_call_of_the_same_library(lib):
    ask(lib.owner)
    waiting = rows()[0]["id"]
    assert answer(lib, waiting, {"status": "saved"}) is False, "nobody took it yet"
    assert answer(lib, "0" * 32, {"status": "saved"}) is False

    other = {"Authorization": f"Bearer {tokens.mint_token('other', 'owner')}"}
    link("google-other", "other")
    assert nxt(lib, headers=other) == {"call": None}, "another library's Mac is handed nothing"
    call = nxt(lib)["call"]
    assert answer(lib, call["id"], {"status": "saved"}, headers=other) is False
    assert rows()[0]["state"] == line.RUNNING
    assert answer(lib, call["id"], {"status": "saved"}) is True


def test_only_an_owner_token_holds_the_line(lib):
    for method, path, body in (("get", "/v1/line/next", None),
                               ("post", f"/v1/line/{'0' * 32}/answer",
                                {"ok": True, "answer": {}})):
        send = getattr(lib.client, method)
        extra = {"json": body} if body is not None else {}
        assert send(path, **extra).status_code == 401
        assert send(path, headers=lib.reader_hdr, **extra).status_code == 403


# ── the web's end ──────────────────────────────────────────────────────────────────────

def test_a_call_gets_the_macs_answer_when_it_comes_within_the_wait(lib):
    with Mac(lib, lambda call: {"status": "saved", "reference": call["args"]["reference"]}) as mac:
        got = ask(lib.owner, wait_s=5)
    assert got == line.Result(line.DONE, {"status": "saved", "reference": LINK["reference"]})
    assert [c["args"] for c in mac.taken] == [LINK]
    assert line.untold(lib.owner) == [], "its asker was given the outcome"


def test_a_call_taken_and_not_finished_within_the_wait_is_running(lib):
    with Mac(lib):
        assert ask(lib.owner, wait_s=0.3) == line.Result(line.RUNNING, None)


def test_an_outcome_nobody_waited_for_is_handed_to_the_next_answer_once(lib):
    assert ask(lib.owner).state == line.WAITING
    assert line.untold(lib.owner) == [], "it has not ended"
    call = nxt(lib)["call"]
    answer(lib, call["id"], {"status": "saved"})

    assert line.untold("another-library") == []
    assert line.untold(lib.owner) == [{"tool": "hopper", "args": LINK, "state": line.DONE,
                                       "answer": {"status": "saved"}}]
    assert line.untold(lib.owner) == []
    (shown,) = line.recent(lib.owner)
    assert shown["state"] == line.DONE and shown["answer"] == {"status": "saved"}, \
        "telling it does not remove it"


def test_a_call_with_nothing_to_do_later_is_not_left_for_a_mac_that_is_not_there(lib):
    assert ask(lib.owner, later=False) == line.Result(line.WAITING, None)
    assert rows() == []


def test_a_call_with_nothing_to_do_later_is_withdrawn_when_no_mac_took_it(lib):
    nxt(lib)                                   # the line is held, but this Mac asks no more
    assert ask(lib.owner, later=False, wait_s=0.2) == line.Result(line.WAITING, None)
    assert rows() == []


# ── what is kept, and for how long ─────────────────────────────────────────────────────

def test_a_call_is_deleted_a_week_after_it_was_asked_whatever_its_state(lib, monkeypatch):
    assert line.KEEP_S == 7 * 24 * 3600
    monkeypatch.setattr(line, "KEEP_S", 0.2)
    ask(lib.owner, {"reference": "https://example.com/done"})
    ask(lib.owner, {"reference": "https://example.com/waiting"})
    answer(lib, nxt(lib)["call"]["id"], {"status": "saved"})
    assert sorted(r["state"] for r in rows()) == [line.DONE, line.WAITING]
    time.sleep(0.3)

    ask(lib.owner, {"reference": "https://example.com/new"}, wait_s=0)
    assert [r["args"] for r in rows()] == ['{"reference": "https://example.com/new"}']
    time.sleep(0.3)
    assert nxt(lib) == {"call": None}, "a call past its week is deleted, not handed over"
    assert rows() == []


def test_deleting_a_library_deletes_its_calls_and_leaves_another_librarys(lib):
    ask(lib.owner)
    nxt(lib)
    ask("another-library")
    assert rows("line_seen")

    deletion.delete_library(lib.owner)

    assert [r["owner"] for r in rows()] == ["another-library"]
    assert rows("line_seen") == []
