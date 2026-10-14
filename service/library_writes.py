"""
service/library_writes.py — the four tools that change the library, as the web offers them: save
a link (`hopper`), `forget`, follow topics (`watch`) and `add_oracle`.

The library lives on the owner's Mac, so nothing here writes it. Each flow builds the arguments
of the local tool's own call, asks the Mac to run it (`line.call`), and gives the answer back.
The connector's tools are thin entries to these flows, and nothing a local tool does is done a
second time here.

An answer says what happened and never how (`answered`): the Mac's own answer when it came within
the wait, without the parts written for the Mac's own host; otherwise one sentence, that it is
being done or that it will be. An outcome that came after its call stopped waiting is told once,
as a line on the next answer of any tool (`earlier`), and `status` shows the latest (`recent`).

Where the host can check for itself, the web skips the local preview: a link is saved and a
person added in one step. `forget` keeps its preview, built from the served copy with no call: an
Oracle by the local preview itself, which runs on the copy, and items from their rows there with
the local consent sentences, because the local item preview reads tables the copy does not carry.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import NamedTuple

from pipeline.forget import atoms as removal
from pipeline.forget import oracle as forget_oracle
from pipeline.retrieval import peers
from service import line

WATCH_ACTIONS = ("preview", "create", "edit", "drop", "list")
# The two that change nothing, so there is nothing to do for them once their asker has gone.
_SHOWS = ("preview", "list")


# ── the flows ────────────────────────────────────────────────────────────────────

async def hopper(owner: str, reference: str, kind_hint: str | None = None) -> dict:
    """Save one link into `owner`'s library: the local `hopper`, confirmed. The host has read
    the link itself, so there is no preview; an x.com link is saved as given."""
    return await _asked(owner, "hopper", {"reference": reference, "kind_hint": kind_hint,
                                          "confirm": True})


async def forget(owner: str, *, atom_id: str | None = None, atom_ids: list[str] | None = None,
                 oracle: str | None = None, confirm: bool = False) -> dict:
    """Remove items from `owner`'s library, or stop following one Oracle. Without `confirm`, a
    preview from the served copy, with no call; with it, the local `forget`, confirmed."""
    target = _target(atom_id, atom_ids, oracle)
    if target is None:
        return {"status": "error",
                "message": "Give exactly one of atom_id, atom_ids or oracle."}
    if not confirm:
        return _preview(owner, target)
    return await _asked(owner, "forget", {**target, "confirm": True})


async def watch(owner: str, action: str, *, name: str | None = None, watches: list | None = None,
                words: list[str] | None = None, people: list[str] | None = None,
                topic_id: str | None = None, drop_watches: list[str] | None = None) -> dict:
    """One action of the local `watch` on `owner`'s topics. Every action is a call, the list and
    the preview too: the served copy carries no topics."""
    if action not in WATCH_ACTIONS:
        return {"status": "error",
                "reason": f"action is one of {', '.join(WATCH_ACTIONS)} (got {action!r})."}
    return await _asked(owner, "watch", {"action": action, "name": name, "watches": watches,
                                         "words": words, "people": people, "topic_id": topic_id,
                                         "drop_watches": drop_watches},
                        later=action not in _SHOWS)


async def add_oracle(owner: str, reference: str, *, x_lookback: str | None = None,
                     web_lookback: str | None = None, scholar_lookback: str | None = None,
                     scholar_topics: list[str] | None = None) -> dict:
    """Add one person to `owner`'s Oracles: the local `add_oracle`, confirmed. The host has
    checked who they are with the user, so there is no preview."""
    if not reference.strip():
        # Refused here: the local refusal names a tool the web does not have.
        return {"status": "not_added", "message": "Give the person's X @handle, or the URL of "
                                                  "their Substack, blog or site."}
    return await _asked(owner, "add_oracle", {
        "reference": reference, "confirm": True, "x_lookback": x_lookback,
        "web_lookback": web_lookback, "scholar_lookback": scholar_lookback,
        "scholar_topics": scholar_topics})


async def _asked(owner: str, tool: str, args: dict, *, later: bool = True) -> dict:
    """Ask the Mac for one call of `tool`, and give the web's answer for how it stands."""
    args = {k: v for k, v in args.items() if v is not None}
    return answered(tool, args, await line.call(owner, tool, args, later=later))


# ── what the web answers ─────────────────────────────────────────────────────────

def answered(tool: str, args: dict, result: line.Result) -> dict:
    """What the web says of a call, by how it stands:

      done      the Mac's answer, as this tool gives it here (`_Wording.done`)
      failed    `{status: "failed", message}`: the Mac's reason, else the tool's own sentence
      running   `{status: "working", message}`: it is being done now
      waiting   `{status: "later", message}`: it will be done; `unavailable` for a call with
                nothing to do later"""
    said = _wording(tool, args)
    if result.state == line.DONE:
        return said.done(result.answer)
    if result.state == line.FAILED:
        return {"status": "failed", "message": result.answer.get("message") or said.failed}
    if result.state == line.RUNNING:
        return {"status": "working", "message": said.working}
    if said.later is None:
        return {"status": "unavailable", "message": _CANT_SHOW}
    return {"status": "later", "message": said.later}


def earlier(owner: str) -> list[str]:
    """One line for each of `owner`'s calls that ended after its asker stopped waiting. Each is
    given once: the next answer of any tool carries it."""
    return [_wording(c["tool"], c["args"]).line(c["args"], _answer(c)) for c in line.untold(owner)]


def with_earlier(owner: str, answer: dict) -> dict:
    """`answer` with those lines under `earlier`, when there are any: what every answer to
    `owner` passes, on the connector and on the agent's routes. It is a courtesy on another
    tool's answer: if it fails, the answer goes out as it was."""
    try:
        lines = earlier(owner)
    except Exception:
        return answer
    return {**answer, "earlier": lines} if lines else answer


def recent(owner: str) -> list[dict]:
    """What was asked to change in `owner`'s library lately and how each stands, newest first, as
    `{tool, asked, args, answer}`: what `status` shows, so an answer that came after its call
    stopped waiting can be read whole."""
    return [{"tool": c["tool"],
             "asked": datetime.fromtimestamp(c["asked"], timezone.utc).strftime(
                 "%Y-%m-%d %H:%M:%S"),
             "args": {k: v for k, v in c["args"].items() if k != "confirm"},
             "answer": _answer(c)}
            for c in line.recent(owner)]


def _answer(c: dict) -> dict:
    return answered(c["tool"], c["args"], line.Result(c["state"], c["answer"]))


# ── forget: the preview, from the served copy ────────────────────────────────────

def _target(atom_id: str | None, atom_ids: list[str] | None, oracle: str | None) -> dict | None:
    """The one thing a forget names, as the local tool's own argument; None when the call names
    nothing, or more than one."""
    named = {"atom_id": (atom_id or "").strip(), "atom_ids": atom_ids or [],
             "oracle": (oracle or "").strip()}
    named = {k: v for k, v in named.items() if v}
    return named if len(named) == 1 else None


def _preview(owner: str, target: dict) -> dict:
    """What a forget would remove, read from `owner`'s served copy."""
    conn, _label = peers.open_peer(owner)
    try:
        if "oracle" in target:
            return _oracle(forget_oracle.forget(conn, target["oracle"]))
        ids = list(dict.fromkeys(target.get("atom_ids") or [target["atom_id"]]))
        rows = [conn.execute("SELECT * FROM atoms WHERE atom_id = ?", (i,)).fetchone()
                for i in ids]
    finally:
        conn.close()
    return _items_preview(ids, rows)


def _items_preview(ids: list[str], rows: list) -> dict:
    """The preview of removing the items `ids`, from their rows in the served copy (None for an
    id it lacks). An unknown id refuses the whole set, as the local tool does. The consent is
    the local one, without its sentence about a served copy."""
    unknown = [i for i, row in zip(ids, rows) if row is None]
    if unknown:
        return {"status": "not_found", "unknown": unknown,
                "message": f"Not in the library: {', '.join(unknown)}. Nothing is removed "
                           "until every id given is one of its items."}
    return {"status": "preview",
            "atoms": [{key: row[key] for key in ("atom_id", *removal.IDENTITY)} for row in rows],
            "consent": removal.consent(rows, shared=False)}


_ORACLE_KEYS = ("status", "scope", "canonical_id", "name", "atoms_kept")


def _oracle(found: dict) -> dict:
    """The local answer about one Oracle as the web gives it, a preview read from the served copy
    or the Mac's removal. Its consent becomes the local sentences that are true said from here;
    the ones that count refresh registrations, which the copy does not carry, or offer
    `delete_atoms`, which the web does not, are left out with the counts behind them."""
    if "consent" not in found:
        return found            # nobody matches, or several do: the candidates to choose from
    return {**{key: found[key] for key in _ORACLE_KEYS},
            "consent": [forget_oracle.stops_tracking(found["name"], found["canonical_id"]),
                        forget_oracle.atoms_remain(found["atoms_kept"]),
                        *forget_oracle.AFTERWARDS]}


# ── each tool's answer, as given here ────────────────────────────────────────────

def _why(said: dict) -> str:
    """The reason an answer gives, under the key its tool keeps one in; else its status."""
    return next((said[key] for key in ("message", "detail", "error", "reason") if said.get(key)),
                said["status"])


def _saved(got: dict) -> dict:
    """The Mac's hopper answer as it is: a confirmed save names nothing the web lacks."""
    return got


def _saved_line(args: dict, said: dict) -> str:
    link = args["reference"]
    if said["status"] == "saved":
        return f"{link} is saved in the library."
    if said["status"] == "already_present":
        return f"{link} was already in the library."
    return f"{link} was not saved: {_why(said)}"


# What only the Mac's own host can use of a forget answer: its counts of internal rows.
_FORGET_INTERNAL = ("references", "snapshot", "shared")


def _forgotten(got: dict) -> dict:
    """The Mac's forget answer without its internal counts. An Oracle's goes through `_oracle`.
    For items, the local consent ends with a sentence about the served copy and its uploads when
    the install shares (`removal.consent`, `shared`); that one is left out."""
    if got.get("scope") == "oracle":
        return _oracle(got)
    out = {key: value for key, value in got.items() if key not in _FORGET_INTERNAL}
    if got.get("shared"):
        out["consent"] = got["consent"][:-1]
    return out


def _forgotten_line(args: dict, said: dict) -> str:
    if said["status"] == "forgotten":
        if said["scope"] == "oracle":
            return f"{said['name']} is no longer followed."
        if said["scope"] == "atoms":
            return f"{len(said['atoms'])} items are removed from the library."
        return f"{said['description'] or said['atom_id']} is removed from the library."
    what = args.get("oracle") or args.get("atom_id") or f"{len(args['atom_ids'])} items"
    if said["status"] == "not_found":
        return f"{what} was not found in the library, so nothing was removed."
    return f"{what} was not removed: {_why(said)}"


_PREVIEWED = (
    "Show the user each watch with how many finds it would have brought last month and an "
    "example or two, each with its `cite` link, then how many of their people's posts a day the "
    "words and people would have matched. A watch that found nothing names something quiet, "
    "which is fine; one whose examples are about something else needs a narrower name or a "
    "`context` word. When they agree, call watch(action=\"create\") with the same lists.")
_FOLLOWED = (
    "The topic is followed: its watches are searched every day from now on, the first time over "
    "the last 14 days. New finds are not listed here.")


def _watched(got: dict) -> dict:
    """The Mac's watch answer. A preview and a new topic each end with lines for the Mac's own
    host, about checking finds and reading them from `latest`; neither happens here, so those
    lines are replaced with what is true here."""
    if got.get("status") == "preview":
        return {**{k: v for k, v in got.items() if k != "next"}, "message": _PREVIEWED}
    if "first" in got:
        return {**{k: v for k, v in got.items() if k not in ("first", "next")},
                "message": _FOLLOWED}
    return got


def _watched_line(args: dict, said: dict) -> str:
    action = args["action"]
    if action in _SHOWS:
        what = (f"The preview of {args.get('name') or 'the topic'}" if action == "preview"
                else "The list of topics")
        if said["status"] in ("ok", "preview"):
            return f"{what} is ready: `status` returns it."
        return f"{what} did not come: {_why(said)}"
    if said["status"] != "ok":
        return f"The topic was not changed: {_why(said)}"
    if action == "drop":
        return f"The topic {said['dropped']} is no longer followed."
    return f"The topic {said['name']} is {'followed' if action == 'create' else 'changed'}."


def _added(got: dict) -> dict:
    """The Mac's add_oracle answer as the web gives it: who was added, or why nobody was. Its
    `message` and `picks` say where the pull stands, in terms of the Mac's own tools and of the
    Mac itself, so they are left out; the web has no tool that follows a pull."""
    if "added" not in got:
        return {"status": "not_added", "message": got["error"]}
    who = got["added"]
    return {"status": "added", "name": who["name"] or who["canonical_id"],
            "canonical_id": who["canonical_id"], "was_already_oracle": who["was_already_oracle"]}


def _added_line(args: dict, said: dict) -> str:
    if said["status"] == "added":
        return f"{said['name']} is added as an Oracle."
    return f"{args['reference']} was not added: {_why(said)}"


# ── what is said of each kind of call ────────────────────────────────────────────

class _Wording(NamedTuple):
    """What the web says about one kind of call."""
    done: Callable[[dict], dict]         # the Mac's answer, as given here
    line: Callable[[dict, dict], str]    # an outcome nobody waited for, in one line: from the
                                         # call's arguments and its answer here
    working: str                         # taken, not finished
    later: str | None                    # not taken; None when there is nothing to do later
    failed: str                          # the Mac could not run it and gave no reason


_WORDING = {
    "hopper": _Wording(
        _saved, _saved_line,
        "It's being saved now. `status` says how it ends.",
        "It'll be in your library when your computer is next on.",
        "The link could not be saved."),
    "forget": _Wording(
        _forgotten, _forgotten_line,
        "It's being removed now. `status` says how it ends.",
        "It'll be removed from your library when your computer is next on.",
        "It could not be removed."),
    "watch": _Wording(
        _watched, _watched_line,
        "The topic is being changed now. `status` says how it ends.",
        "Your topics will be changed when your computer is next on.",
        "The topic could not be changed."),
    "add_oracle": _Wording(
        _added, _added_line,
        "They're being added now. `status` says how it ends.",
        "They'll be added to your library when your computer is next on.",
        "They could not be added."),
}
# `watch`'s list and preview.
_SHOWING = _Wording(
    _watched, _watched_line,
    "It is still being put together. `status` returns it once it is done.",
    None,
    "Topics could not be read.")
_CANT_SHOW = "Topics can't be shown right now."


def _wording(tool: str, args: dict) -> _Wording:
    return _SHOWING if tool == "watch" and args["action"] in _SHOWS else _WORDING[tool]
