"""
service/line.py — the line: tool calls waiting for the owner's Mac, and the Mac's answers.

A web tool that changes the library cannot write the served copy, which only the Mac's uploads
write. So it leaves the local tool's own arguments here as a call (`call`) and waits. The Mac
holds the line: it asks for the next call (`next_call`, `GET /v1/line/next`), an ask this service
keeps open until one arrives, runs the call through its own tool, and posts the answer (`answer`).
The service never calls the Mac.

A call's life is listed here and nowhere else:

    waiting ──► running ──► done | failed

  waiting → running   the Mac took it (`offered`);
  running → running   offered again after `RERUN_AFTER_S`, because the Mac stopped mid-call. A
                      call run twice is harmless: the local tools treat a repeat as a no-op;
  running → done      the Mac ran it and answered; `failed` when it could not (`ended_as`).

A call is deleted `KEEP_S` after it was asked, whatever its state, and with its library
(`deletion.py`).

Both ends wait by reading the table every `POLL_S`, so any uvicorn worker can hold either end and
nothing passes between processes but the rows. The functions that decide (`offered`, `ended_as`,
`is_held`) take plain values; the rest read and write `line_calls` and `line_seen`.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import time
from typing import NamedTuple

from service import accounts, db, grants, tokens

WAITING, RUNNING, DONE, FAILED = "waiting", "running", "done", "failed"
ENDED = (DONE, FAILED)

# How often a waiting end reads the table.
POLL_S = 1.0
# The longest one ask of the Mac's is kept open.
NEXT_WAIT_MAX_S = 50.0
# The Mac asks again as soon as an ask ends, so its line is held while its last ask is this
# young: one whole ask, and time to come back.
HELD_S = 70.0
# A call running this long is offered again.
RERUN_AFTER_S = 600.0
# A Mac that took a call this recently is at work on it, and so on the line, though it asks for
# nothing while it runs one. Longer than a call takes; far shorter than `RERUN_AFTER_S`, so a
# call handed to a Mac that fell asleep with its ask open reads as held only this long.
BUSY_S = 120.0
# How long a call is kept after it was asked.
KEEP_S = 7 * 24 * 3600
# What a Mac is told to wait before asking again when nothing could ask it for a call.
NO_WRITER_RETRY_S = 3600


class Result(NamedTuple):
    """How a call stood when its asker stopped waiting."""
    state: str             # done | failed | running (the Mac is at work) | waiting (no Mac)
    answer: dict | None    # the Mac's own answer for done and failed, else None


# ── deciding ─────────────────────────────────────────────────────────────────────

def offered(state: str, taken: float | None, now: float) -> bool:
    """Whether the Mac's next ask is handed this call: one nobody has taken, or one taken so
    long ago that the Mac must have stopped mid-call."""
    return state == WAITING or (state == RUNNING and now - taken > RERUN_AFTER_S)


def ended_as(ok: bool) -> str:
    """The state the Mac's answer ends a running call in."""
    return DONE if ok else FAILED


def is_held(seen: float | None, now: float) -> bool:
    """Whether a Mac whose last ask was at `seen` is still on the line."""
    return seen is not None and now - seen <= HELD_S


# ── the web's end ────────────────────────────────────────────────────────────────

async def call(owner: str, tool: str, args: dict, *, wait_s: float = 25.0,
               later: bool = True) -> Result:
    """Ask `owner`'s Mac to run `tool` with `args`, and wait up to `wait_s` for its answer.

    The wait is kept only while the line is held: with no Mac on it the call is left for the next
    one, and this answers `waiting` at once. A call that ends after its asker stopped waiting is
    handed to the next answer instead (`untold`).

    `later=False` is for a call with nothing to do once its asker has gone (a list, a preview):
    it is not left for a Mac that is not on the line, and it is withdrawn when none took it
    within the wait."""
    now = time.time()
    on_line = held(owner)
    if not (on_line or later):
        return Result(WAITING, None)
    until = now + wait_s if on_line else now
    call_id = _ask(owner, tool, args, now, until)
    while True:
        state, said = _read(call_id)
        if state in ENDED or time.time() >= until:
            break
        await asyncio.sleep(min(POLL_S, until - time.time()))
    if state in ENDED:
        _mark_told(call_id)
        return Result(state, said)
    if state == WAITING and not later and not _withdraw(call_id):
        state = RUNNING            # taken between the last read and the withdrawal
    if state == WAITING and later and held(owner):
        # The Mac is on the line and has not reached this call: it runs one at a time. To the
        # asker that is a call being worked on, not one waiting for a computer to come on.
        state = RUNNING
    return Result(state, None)


def untold(owner: str) -> list[dict]:
    """The calls of `owner` that ended after their asker stopped waiting, oldest first, each as
    `{tool, args, state, answer}`. Marks them told, so each is returned once."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT id, tool, args, state, answer FROM line_calls WHERE owner = ? AND told = 0 "
            "AND state IN (?, ?) AND waited_until < ? ORDER BY asked, id",
            (owner, DONE, FAILED, time.time())).fetchall()
        # One conditional UPDATE each: of two answers built at once, only one carries a call.
        mine = [r for r in rows if conn.execute(
            "UPDATE line_calls SET told = 1 WHERE id = ? AND told = 0", (r["id"],)).rowcount]
        conn.commit()
    finally:
        conn.close()
    return [_shown(r) for r in mine]


def recent(owner: str, limit: int = 10) -> list[dict]:
    """`owner`'s latest calls, newest first, each as `{tool, args, state, answer, asked}`. Marks
    nothing told."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT tool, args, state, answer, asked FROM line_calls WHERE owner = ? "
            "ORDER BY asked DESC, id LIMIT ?", (owner, limit)).fetchall()
    finally:
        conn.close()
    return [{**_shown(r), "asked": r["asked"]} for r in rows]


def held(owner: str) -> bool:
    """Whether `owner`'s Mac is on the line: it asked within about the last `HELD_S` seconds, or
    it took a call within the last `BUSY_S` and is running it, which is when it asks for
    nothing."""
    now = time.time()
    conn = db.connect()
    try:
        row = conn.execute("SELECT seen FROM line_seen WHERE owner = ?", (owner,)).fetchone()
        running = conn.execute(
            "SELECT 1 FROM line_calls WHERE owner = ? AND state = ? AND taken > ? LIMIT 1",
            (owner, RUNNING, now - BUSY_S)).fetchone()
    finally:
        conn.close()
    return running is not None or is_held(row["seen"] if row is not None else None, now)


# ── the Mac's end ────────────────────────────────────────────────────────────────

def has_writer(owner: str) -> bool:
    """Whether anything could ask `owner`'s Mac for a call: a Google account linked to the
    library (the connector), an agent key, or an agent invite not yet redeemed, so the line is
    already held when the agent's first call comes."""
    return (accounts.linked(owner) or grants.agent_invite_open(owner)
            or any(t["role"] == "agent" for t in tokens.list_tokens(owner)))


async def next_call(owner: str, wait_s: float) -> dict:
    """One ask of the Mac's, as `GET /v1/line/next` answers it:

      `{"call": {id, tool, args}}`           handed over; the call is now running
      `{"call": None}`                       nothing arrived within the wait; ask again at once
      `{"call": None, "retry_after": s}`     nothing could ask this library for a call; ask again
                                             after that many seconds

    The wait is `wait_s`, at most `NEXT_WAIT_MAX_S`. Each ask records that the line is held."""
    _hold(owner, time.time())
    if not has_writer(owner):
        return {"call": None, "retry_after": NO_WRITER_RETRY_S}
    until = time.monotonic() + min(wait_s, NEXT_WAIT_MAX_S)
    while True:
        handed = _take(owner, time.time())
        if handed is not None or time.monotonic() >= until:
            return {"call": handed}
        await asyncio.sleep(min(POLL_S, until - time.monotonic()))


def answer(owner: str, call_id: str, ok: bool, said: dict) -> bool:
    """Record the Mac's answer to a call it was handed. False when no running call of `owner`
    has that id (answered already, or deleted), and the Mac drops it."""
    conn = db.connect()
    try:
        ended = conn.execute(
            "UPDATE line_calls SET state = ?, answer = ? WHERE id = ? AND owner = ? AND state = ?",
            (ended_as(ok), json.dumps(said), call_id, owner, RUNNING)).rowcount
        conn.commit()
    finally:
        conn.close()
    return ended == 1


# ── the rows ─────────────────────────────────────────────────────────────────────

def _shown(row) -> dict:
    return {"tool": row["tool"], "args": json.loads(row["args"]), "state": row["state"],
            "answer": json.loads(row["answer"]) if row["answer"] is not None else None}


def _drop_expired(conn, now: float) -> None:
    """Delete every library's calls past their week. Run by both writes that start an exchange
    (a call asked, a Mac asking), so nothing is handed over or kept past it."""
    conn.execute("DELETE FROM line_calls WHERE asked <= ?", (now - KEEP_S,))


def _ask(owner: str, tool: str, args: dict, now: float, waited_until: float) -> str:
    """Store a new call, waiting, and return its id."""
    call_id = secrets.token_hex(16)
    conn = db.connect()
    try:
        _drop_expired(conn, now)
        conn.execute("INSERT INTO line_calls (id, owner, tool, args, state, asked, waited_until) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (call_id, owner, tool, json.dumps(args), WAITING, now, waited_until))
        conn.commit()
    finally:
        conn.close()
    return call_id


def _read(call_id: str) -> tuple[str, dict | None]:
    """A call's state and, once it has one, the Mac's answer."""
    conn = db.connect()
    try:
        row = conn.execute("SELECT state, answer FROM line_calls WHERE id = ?",
                           (call_id,)).fetchone()
    finally:
        conn.close()
    return row["state"], json.loads(row["answer"]) if row["answer"] is not None else None


def _mark_told(call_id: str) -> None:
    conn = db.connect()
    try:
        conn.execute("UPDATE line_calls SET told = 1 WHERE id = ?", (call_id,))
        conn.commit()
    finally:
        conn.close()


def _withdraw(call_id: str) -> bool:
    """Delete a call nobody took. False when the Mac took it first."""
    conn = db.connect()
    try:
        gone = conn.execute("DELETE FROM line_calls WHERE id = ? AND state = ?",
                            (call_id, WAITING)).rowcount
        conn.commit()
    finally:
        conn.close()
    return gone == 1


def _hold(owner: str, now: float) -> None:
    """Record that `owner`'s Mac asked at `now`."""
    conn = db.connect()
    try:
        _drop_expired(conn, now)
        conn.execute("INSERT INTO line_seen (owner, seen) VALUES (?, ?) "
                     "ON CONFLICT(owner) DO UPDATE SET seen = excluded.seen", (owner, now))
        conn.commit()
    finally:
        conn.close()


def _take(owner: str, now: float) -> dict | None:
    """Hand over the oldest call `owner`'s Mac is owed, as `{id, tool, args}`, and mark it
    running; None when there is none.

    The move is one conditional UPDATE on the state the call was read in, so of two asks at once
    the loser updates nothing and finds the call on its next read, if it is still owed."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT id, tool, args, state, taken FROM line_calls WHERE owner = ? "
            "AND state IN (?, ?) ORDER BY asked, id", (owner, WAITING, RUNNING)).fetchall()
        due = next((r for r in rows if offered(r["state"], r["taken"], now)), None)
        if due is None:
            return None
        took = conn.execute(
            "UPDATE line_calls SET state = ?, taken = ? WHERE id = ? AND state = ? AND taken IS ?",
            (RUNNING, now, due["id"], due["state"], due["taken"])).rowcount
        conn.commit()
    finally:
        conn.close()
    if not took:
        return None
    return {"id": due["id"], "tool": due["tool"], "args": json.loads(due["args"])}
