"""
mcp_server/line.py — the `line` job: hold this install's line to the sharing service, and run the
tool calls it hands over.

The service never calls an install. A tool call made on the web or by an agent waits on the
service until this job asks for it (`client.line_next`, which the service holds open), runs it
here and posts the tool's own answer (`client.line_answer`). That makes this another way in to the local tools, as
`cli.py` is, with two differences:

  • It runs only what `ALLOWED` lists. Until this job existed the service could not make an
    install do anything, so the list is this install's own guard, whatever the service sends.
  • It runs the server's registered tool without the rider flow (`rider_hook.py`): a web call
    must not spend the notices and offers meant for this computer's host.

It is a worker job (`registry.LINE`) that runs beside the scrapers. Unlike them it stays up for as
long as the worker does, and it never holds off the computer's sleep: a sleeping computer holds no
line, and the service answers for it.

How long it stays is the service's answer. A library with a web or agent writer holds the line
until the worker's restart stops it, and then asks to run again at once. A library with none is
told to come back later, and the job's hourly cadence does. An install with no service token asks
nothing.

The decisions are plain functions: `refusal`, `changed_library`, `next_step`, `gives_up` and
`pause_s`.
"""
from __future__ import annotations

import asyncio
import functools
import time
from typing import Callable

from pipeline.ingestion.utils import log
from pipeline.sharing import account, client, push_gate
from pipeline.worker import job_queue
from pipeline.worker.registry import LINE, SCRAPERS

# How long the service holds one ask open: the most it takes.
WAIT_S = 50
# The waits after an ask the service did not answer, in order; the last one repeats.
PAUSES_S = (5.0, 15.0, 60.0)

# Every tool this install runs for the service, and the arguments each may be given.
ALLOWED: dict[str, frozenset[str]] = {
    "hopper": frozenset({"reference", "kind_hint", "confirm"}),
    # Not `source` (an Oracle's sources are not in the shared copy, so the web cannot show one)
    # and not `delete_atoms`.
    "forget": frozenset({"atom_id", "atom_ids", "oracle", "confirm"}),
    "watch": frozenset({"action", "name", "watches", "words", "people", "topic_id",
                        "drop_watches"}),
    # Not `force`.
    "add_oracle": frozenset({"reference", "confirm", "x_lookback", "web_lookback",
                             "scholar_lookback", "scholar_topics"}),
}
# Not `checked`: a topic's finds are checked where `latest` hands them over, on this computer.
# A tuple, so any value the service sends can be looked up in it.
WATCH_ACTIONS = ("preview", "create", "edit", "drop", "list")

_NO_WRITER = ("this library has no web or agent writer, so there is no line to hold; the job's "
              "cadence asks again")


def refusal(tool: object, args: object) -> str | None:
    """Why this install will not run a call, or None when it will. The one judge of what the
    service may ask for: a tool, an argument or a `watch` action that is not listed is refused,
    whatever its value."""
    if not isinstance(tool, str) or tool not in ALLOWED:
        return f"Opyt does not run {tool!r} from the web."
    if not isinstance(args, dict):
        return f"Opyt needs {tool}'s arguments as an object."
    if extra := sorted(set(args) - ALLOWED[tool]):
        return f"Opyt does not take {', '.join(extra)} for {tool} from the web."
    if tool == "watch" and args.get("action") not in WATCH_ACTIONS:
        return f"Opyt does not run watch's {args.get('action')!r} action from the web."
    return None


def changed_library(tool: str, args: dict, answer: dict) -> bool:
    """Did this answer change the library, so the shared copy is owed a push? One row per tool,
    in its own words: a preview, a refusal and a repeat (`already_present`, `not_found`) change
    nothing."""
    status = answer.get("status")
    return (tool == "hopper" and status == "saved"
            or tool == "forget" and status == "forgotten"
            or tool == "watch" and status == "ok" and args["action"] in ("create", "edit", "drop")
            or tool == "add_oracle" and "added" in answer)


def next_step(got: dict) -> str:
    """What the service's answer to an ask calls for: `run` the call it hands over, `leave`
    when the library has no web or agent writer, else `ask` again at once."""
    if got["call"] is not None:
        return "run"
    return "leave" if "retry_after" in got else "ask"


def gives_up(status: int | None) -> bool:
    """Is a failed exchange one that asking again will not fix? A 401 is: the library is gone
    and `client` has dropped the token. So is any other 4xx, which is a service that does not
    know the line. No answer at all, a 5xx and a proxy's page are the service being away."""
    return status is not None and 400 <= status < 500


def pause_s(unanswered: int) -> float:
    """How long to wait before asking again, after `unanswered` asks in a row got no answer."""
    return PAUSES_S[min(unanswered, len(PAUSES_S)) - 1]


def run() -> dict:
    """The `line` job. It returns when this install has no line to hold. A held line ends with
    the worker's SIGTERM, which `child` raises here as a `SystemExit`: the job then asks to run
    again at once, as the scrapers' process does, so the next worker lifetime brings the line
    straight back.

    A stop in the middle of a call posts no answer. The tool's body runs on its own thread, so
    it is not cut: the process exits when it returns, and the service offers the call again
    later, where a repeat is harmless (the plan's §14.3 and constraint 9)."""
    try:
        return hold()
    except SystemExit:
        job_queue.request_now(LINE)
        return {"status": "stopped"}


def hold(sleep: Callable[[float], None] = time.sleep) -> dict:
    """Ask, run what is handed over, answer, and ask again, until there is no line to hold. The
    account is read before every ask, so the token sent is the one this install holds now, and
    the job ends when it holds none. A token the service no longer knows ends it at the 401."""
    unanswered = 0
    while True:
        acct = account.load()
        if not acct.token:
            return {"status": "not_shared"}
        try:
            got = client.line_next(acct, WAIT_S)
            match next_step(got):
                case "leave":
                    return {"status": "nothing_owed", "message": _NO_WRITER}
                case "run":
                    serve(acct, got["call"])
        except client.Failed as e:
            if gives_up(e.status):
                return {"status": e.kind, "message": str(e)}
            unanswered += 1
            if unanswered == 1:
                log(f"[line] the service did not answer, asking again: {e}")
            sleep(pause_s(unanswered))
            continue
        if unanswered:
            log("[line] the service answers again")
            unanswered = 0


def serve(acct: account.Account, call: dict) -> None:
    """Run one call and post what it came to. A call that changed the library owes the shared
    copy a push first and wakes the scrapers, whose publishing pass sends it, so the copy follows
    within about a minute whether or not the answer reaches the service."""
    tool, args = call["tool"], call["args"]
    ok, answer = outcome(tool, args)
    log(f"[line] {tool}: {answer.get('status', 'answered') if ok else answer['message']}")
    if ok and changed_library(tool, args, answer):
        push_gate.owe()
        job_queue.request_now(SCRAPERS)
    client.line_answer(acct, call["id"], ok, answer)


def outcome(tool: object, args: object) -> tuple[bool, dict]:
    """What a call comes to, as `(ok, answer)`: the tool's own answer, or why it was not run or
    did not finish. A call that is refused is not run, and one that raised is not run again
    here."""
    if (why := refusal(tool, args)) is not None:
        return False, {"status": "refused", "message": why}
    try:
        return True, asyncio.run(_tool_answer(tool, args))
    except Exception as exc:
        return False, {"status": "error", "message": f"{type(exc).__name__}: {exc}"}


async def _tool_answer(tool: str, args: dict) -> dict:
    """The answer of the server's own registered tool: its validation and its body, which is the
    one code path a host's call takes, without the middleware that attaches riders."""
    registered = await _server().get_tool(tool)
    return (await registered.run(args)).structured_content


@functools.cache
def _server():
    """The built server, kept for the life of the process."""
    # Imported on the first call, not at the top: building the server loads FastMCP and every
    # tool module, and a line that has run no call should stay small (47 MB against 112).
    from mcp_server import build
    return build.server()
