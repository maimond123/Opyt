"""
service/connector.py — the Opyt connector: an MCP endpoint over the caller's own library.

Muse, Claude and ChatGPT reach a user's library here. A caller signs in with Google
(`google_signin.GoogleSignIn`, FastMCP's `GoogleProvider` acting as the authorization server those
clients register with, with token revocation on). `accounts.owner_of` turns their Google account
into the library a builder linked to it. Every read goes to that library's served copy through
`reading`, so reads are capped and counted like any other reader's.

A caller can also read a library someone shared with them: `accept` redeems the invite for their
Google account (`account_peers.py`), and the four read tools take `kb` to name it. Those reads
are counted under the reader token the invite bought. Nothing else reaches a shared library: every
tool that makes or changes something works on the caller's own.

No tool writes the served copy: only the Mac's uploads do. Three tools write what this service
owns, and each says so (`readOnlyHint: false`): `reader_view` makes a page of sources, `share`
mints an invite, and `unshare` revokes one reader's token. They and `status` are thin entries to
the flows in `owner_tools.py`. Four change the library itself, on the Mac that holds it: `hopper`,
`forget`, `watch` and `add_oracle` are thin entries to the flows in `library_writes.py`, which
leave the local tool's own call for the Mac and answer with what happened, never with how. The
owner's agent reaches the same flows through `/v1/library/…` (`app.py`). Every tool's answer
passes `Earlier`, which adds a line for each such call that ended after its asker stopped
waiting. Taking the whole library down is left to the Mac: from here it would delete the Google
link the call came through.
Search is keyword-only: it matches words, never meaning.

Stateless HTTP with JSON answers, so any uvicorn worker can serve any call. `asgi_app` builds it
only where its Google client is configured.
"""
from __future__ import annotations

import os
from typing import NamedTuple

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import Middleware
from fastmcp.tools.base import ToolResult
from mcp.types import Icon
from starlette.applications import Starlette

from pipeline.sharing import desktop_only, page_words
from pipeline.topics import lists
from service import (account_peers, accounts, library_writes, owner_tools, reading, serving,
                     tokens)
from service.google_signin import GoogleSignIn
from service.owner_tools import BASE_URL_ENV, SETUP_URL

CLIENT_ID_ENV = "OPYT_SERVICE_GOOGLE_CLIENT_ID"
CLIENT_SECRET_ENV = "OPYT_SERVICE_GOOGLE_CLIENT_SECRET"
MCP_PATH = "/mcp"

SITE_URL = "https://useopyt.com"
# Shown on the sign-in's consent page, and to clients that draw a server's icon.
ICON_URL = "https://api.useopyt.com/icon.png"

_READ = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True,
         "openWorldHint": False}
# A page or an invite is written on this service and works for anyone holding its link.
_MAKE = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False,
         "openWorldHint": True}
# A change to the library itself. It reaches outside it: a page is read, a person looked up, a
# subject searched.
_CHANGE = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False,
           "openWorldHint": True}
# Redeeming an invite writes one row on this service and reaches nothing outside it.
_ACCEPT = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False,
           "openWorldHint": False}
# Revoking a reader or forgetting an item takes it away; doing it again changes nothing.
_REMOVE = {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True,
           "openWorldHint": False}

# How a library of the user's own comes to be: said wherever an account turns out to have none.
_OWN_LIBRARY = (
    "Opyt builds a library on the user's Mac, in Claude Desktop, Claude Code, ChatGPT Desktop or "
    "Codex. In that app, the user says \"connect Opyt to Google\" and signs in with this same "
    f"Google account; the library then appears here. Setup: {SETUP_URL}")
_NO_LIBRARY = (
    "No Opyt library is connected to this Google account yet. If someone sent the user an Opyt "
    "invite link, pass it to `accept` and their library can be read here, with nothing to "
    f"install. For a library of the user's own: {_OWN_LIBRARY}")
_NOT_ARRIVED = (
    "This account's library is connected but has not been published here yet. The user's own "
    "Opyt publishes it in the background, usually within the hour while it is running.")

# The citation line is this server's own; `cite` itself is built once, in
# `pipeline/retrieval/cards.py`, for both servers.
_INSTRUCTIONS = (
    "This is the user's own Opyt library, and any library someone shared with them. When the "
    "user pastes an Opyt invite link, `accept` it: `search`, `open`, `aggregate` and `latest` "
    "then take `kb` to read that person's library, which the user can read and not change. "
    "Attribute what comes from it to that person. Everything else here is about the user's own "
    "library. Four tools change what it holds: `hopper` saves a link "
    "into it, `forget` removes an item or stops following a person, `watch` follows a topic and "
    "`add_oracle` adds a person whose writing it collects. `reader_view` puts its sources on "
    "one readable page, `share` makes an invite for one person and `unshare` cuts one reader "
    "off. When an answer carries `earlier`, each line says how something the user asked for "
    "before has ended: pass it on. Every "
    "item you name from it carries its source link: write the item's name inside the markdown "
    "link its `cite` field holds. Search "
    "matches the words in a query, not its meaning, so several short searches using the words a "
    "source would use find more than one long question. "
    f"{desktop_only.SENTENCE} When the user asks for one of those, tell them so, in those words, "
    "and do not try another way.")


class Library(NamedTuple):
    """The library this call reads, and the name its reads are counted under. `name` is what a
    library shared with the caller is called in answers; None for the caller's own."""
    owner: str
    reader: str
    name: str | None = None


def _caller_sub() -> str | None:
    """The Google account behind this call, from the token `GoogleProvider` verified."""
    token = get_access_token()
    return (token.claims or {}).get("sub") if token is not None else None


def _library() -> Library | dict:
    """This caller's own library, or the answer that says why there is none. Every tool that
    makes or changes something reads through here, so none of them reaches a shared library."""
    sub = _caller_sub()
    owner = accounts.owner_of(sub) if sub else None
    if owner is None:
        return _no_library(account_peers.held(sub) if sub else [])
    if not serving.is_served(owner):
        return {"status": "not_arrived", "message": _NOT_ARRIVED}
    # Counted like a reader token: under a hash, never the account id itself.
    return Library(owner, tokens.token_hash(f"google:{sub}"))


def _reading(kb: str | None) -> Library | dict:
    """The library a read tool reads: the one shared with the caller that `kb` names, or with
    no `kb` the caller's own. An account with no library of its own and one shared with it reads
    that one, so its user never has to name it."""
    sub = _caller_sub()
    shared = account_peers.held(sub) if sub else []
    if kb is None:
        if (sub and accounts.owner_of(sub)) or len(shared) != 1:
            return _library()
        peer = shared[0]
    else:
        peer = account_peers.named(shared, kb)
        if peer is None:
            return {"status": "no_such_library", "shared_with_you": [p.name for p in shared],
                    "message": (f"No library called '{kb}' is shared with this Google account. "
                                + (f"Shared with it: {_names(shared)}. " if shared else "")
                                + "Leave `kb` out to read the user's own library.")}
    if not peer.live:
        account_peers.forget(sub, peer)
        return {"status": "access_ended",
                "message": f"'{peer.name}' is no longer shared with this account: its owner "
                           "stopped sharing with them. Tell the user; a new invite from the "
                           "owner, passed to `accept`, restores it."}
    if not serving.is_served(peer.owner):
        return {"status": "not_arrived",
                "message": f"'{peer.name}' has not reached Opyt's server yet. Its owner's Opyt "
                           "publishes it in the background, usually within a minute or two of "
                           "their first share."}
    return Library(peer.owner, peer.reader, peer.name)


def _names(shared: list[account_peers.Peer]) -> str:
    return ", ".join(f"kb='{p.name}'" for p in shared)


def _no_library(shared: list[account_peers.Peer]) -> dict:
    """The answer for an account with no library of its own: where one comes from, and what it
    can read meanwhile."""
    if not shared:
        return {"status": "no_library", "message": _NO_LIBRARY}
    return {"status": "no_library", "shared_with_you": [p.name for p in shared],
            "message": ("This Google account has no Opyt library of its own. It can read what "
                        f"was shared with it: {_names(shared)} on `search`, `open`, `aggregate` "
                        "and `latest`. Saving, removing, following and sharing work on the "
                        f"user's own library only. For one of their own: {_OWN_LIBRARY}")}


def own_notices(notices: list[dict]) -> list[dict]:
    """A search's notices as they read for the library's owner.

    The reads were written for a reader of somebody else's copy. On the connector the library is
    the caller's own: `foreign_kb` (these results are not your own; pass `kb` to `open`) would be
    false, and `who_unresolved` asks for a web search the user did not ask for. The rest state
    what the search did and stay as they are."""
    out = []
    for n in notices:
        if n.get("code") == "foreign_kb":
            continue
        if n.get("code") == "who_unresolved":
            n = {**n, "message": f"Nobody matching '{n['who']}' is in this library, so no item "
                                 "could match. The library holds the people the user follows "
                                 "in Opyt; this does not mean they said nothing on the subject."}
        out.append(n)
    return out


# `watch`'s description. How to write a topic's lists is measured, so that part is the local
# tool's own text (`lists.RULES`), added below this.
_WATCH = """\
Follow topics for the user: they say what they want to stay up to date on, and Opyt searches for \
it every day, outside the library (new papers, repos, X posts and Substack posts) and among the \
posts of the people the library follows.

FLOW:
1. `preview` with a `name` (the subject in the user's words) and the three lists below. It \
searches each watch over the last month and counts what it would have found, with examples, and \
how many of the user's people's posts a day the words and people match. It writes nothing. Show \
it to the user. It can take up to a minute: when the answer's status is `working`, `status` \
returns the preview once it is done.
2. When they agree (or after their edits), `create` with the same lists.

Other actions: `edit` (topic_id, plus any of: name; watches to add; drop_watches by name; words \
or people, which replace the old lists), `drop` (topic_id: stop following), `list` (the topics \
followed, each with its `topic_id`).

Args:
    action: "preview" | "create" | "edit" | "drop" | "list".
    name: the topic's name, in the user's words.
    watches: [{"name": "Boltz-2", "spellings": ["Boltz2"], "sources": ["github", "openalex"]}, \
{"name": "Chai-2", "context": ["antibody"]}].
    words: the subject words, a list of short phrases.
    people: handles of the user's Oracles who mostly write about it.
    topic_id: the topic to edit or drop.
    drop_watches: names of watches to stop (edit).

Returns {status, ...}: `preview`; `ok` with the topic; `refused` or `error` with the `reason`. \
`working` and `later` mean it is not done yet, and `unavailable` that topics can't be shown \
right now: `message` says which.

HOW TO WRITE THE LISTS:
"""


# `status`'s description. It ends on what Opyt does only in its desktop app, so a client that
# shows a model the tools and not the server's instructions still has it.
_STATUS = """\
Whether this Google account has an Opyt library here, how many items it holds, and when it was \
last published; and `changes`, what the user asked to change in it lately (links saved, items \
removed, topics, Oracles) with how each ended, newest first. `shared_with_you` names the \
libraries other people shared with this account, each a `kb` the read tools take. Not counted \
as a read.

When the user asks what Opyt can do here, or for something no tool here does, tell them: """


class Earlier(Middleware):
    """Every tool's answer passes here. A change the user asked for earlier, which ended after
    its call stopped waiting, rides the next answer as one line under `earlier`, once
    (`library_writes.with_earlier`)."""

    async def on_call_tool(self, context, call_next):
        result = await call_next(context)
        if not isinstance(result, ToolResult) or not isinstance(result.structured_content, dict):
            return result
        sub = _caller_sub()
        owner = accounts.owner_of(sub) if sub else None
        if owner is None:
            return result
        return ToolResult(
            structured_content=library_writes.with_earlier(owner, result.structured_content),
            meta=result.meta)


def build(auth=None) -> FastMCP:
    """The connector's server and its thirteen tools. `auth` is the sign-in; tests pass none."""
    mcp = FastMCP("Opyt", instructions=_INSTRUCTIONS, auth=auth, website_url=SITE_URL,
                  icons=[Icon(src=ICON_URL)])
    mcp.add_middleware(Earlier())

    @mcp.tool(annotations={"title": "Search your library", **_READ})
    def search(query: str, who: str | None = None, who_id: str | list[str] | None = None,
               source_type: str | None = None, what_kind: str | None = None,
               tags: list[str] | None = None, date_from: str | None = None,
               date_to: str | None = None, k: int = 8, offset: int = 0,
               kb: str | None = None) -> dict:
        """Search the user's Opyt library: posts, articles, papers and repositories saved from
        people they chose to follow. Keyword search: it matches the words in `query`, not
        their meaning, so several short searches in the words a source would use find more.

        Each hit is a card with a snippet, not the full text; `open` returns the stored text.
        Each hit carries `cite`, a markdown link to its source. `notices` are sentences about
        what this search did: filters that matched nothing, tags normalized, results cut at `k`.

        Filters: `who` (a handle or profile URL), `who_id` (ids from `insights.resolved_who`,
        to pick one of several people `who` matched), `source_type`, `what_kind`, `tags`, and
        `date_from` / `date_to` (YYYY-MM-DD, publication date).

        `k` is how many hits (default 8, at most 25 a page). Measured on a real library: on a
        well-covered topic the first 8 hold about a third of the relevant results, and results
        9-25 are still about half relevant; on a narrow topic, results past 25 are mostly
        unrelated. More results come in pages: while the answer's `next_offset` is set, the same
        search with `offset=next_offset` returns the next ones, up to the first 250.

        `kb` names a library someone shared with the user (`accept` and `status` give the
        names); leave it out for the user's own. What comes from a shared library is that
        person's reading: say whose it is."""
        lib = _reading(kb)
        if isinstance(lib, dict):
            return lib
        found = reading.search(lib.owner, lib.reader, query, k=k, offset=offset,
                               who=who, who_id=who_id, source_type=source_type,
                               what_kind=what_kind, tags=tags, date_from=date_from,
                               date_to=date_to, as_kb=lib.name)
        if "error" in found or lib.name is not None:
            return found
        return {**found, "notices": own_notices(found["notices"]),
                **({"readable": page_words.READABLE} if found.get("hits") else {})}

    @mcp.tool(annotations={"title": "Open an item", **_READ})
    def open(atom_id: str, offset: int = 0, kb: str | None = None) -> dict:
        """One item from the library: its stored text (`raw`), source link (`cite`) and dates.
        A text longer than 60,000 characters comes in pages: the answer then carries
        `raw_chars` (the whole text's length) and `next_offset`, and `open` with
        `offset=next_offset` returns the next page until `next_offset` is null. `body_state` says whether `raw` is the whole source (`complete`) or only part of it
        (`partial`, `absent`); do not present a partial text as complete. For an item found in
        a shared library, pass the same `kb` the search used."""
        lib = _reading(kb)
        if isinstance(lib, dict):
            return lib
        got = reading.open_atom(lib.owner, lib.reader, atom_id, as_kb=lib.name, offset=offset)
        if "error" in got or lib.name is not None:
            return got
        return {**got, "readable": page_words.READABLE}

    @mcp.tool(annotations={"title": "Summarize your library", **_READ})
    def aggregate(tags: list[str] | None = None, what_kind: str | None = None,
                  source_type: str | None = None, who_id: str | list[str] | None = None,
                  date_from: str | None = None, date_to: str | None = None,
                  kb: str | None = None) -> dict:
        """What the library holds, over an optional scope: counts by kind and source, the
        authors, and the most recent item descriptions. With no arguments, the whole library.
        `who_id` takes ids, not handles: get them from `search(who=...)`'s
        `insights.resolved_who`. `kb` names a library shared with the user; leave it out for
        their own."""
        lib = _reading(kb)
        if isinstance(lib, dict):
            return lib
        scope = {k: v for k, v in {"tags": tags, "what_kind": what_kind,
                                   "source_type": source_type, "who_id": who_id,
                                   "date_from": date_from, "date_to": date_to}.items()
                 if v is not None}
        return reading.aggregate(lib.owner, lib.reader, scope, as_kb=lib.name)

    @mcp.tool(annotations={"title": "What's new", **_READ})
    def latest(since: str | None = None, since_id: str | None = None, limit: int = 50,
               kb: str | None = None) -> dict:
        """What arrived in the library most recently, oldest first. With no cursor, the newest
        `limit` items (at most 100). To read on from a previous answer, pass its `next` values as `since`
        and `since_id`. Records nothing about what was shown. `kb` names a library shared with
        the user; leave it out for their own."""
        lib = _reading(kb)
        if isinstance(lib, dict):
            return lib
        return reading.feed(lib.owner, lib.reader, since=since, since_id=since_id, limit=limit,
                            as_kb=lib.name)

    @mcp.tool(annotations={"title": "Library status", **_READ},
              description=_STATUS + desktop_only.SENTENCE)
    def status() -> dict:
        lib = _library()
        if isinstance(lib, dict):
            return lib
        shared = [p.name for p in account_peers.held(_caller_sub() or "")]
        return {**owner_tools.status(lib.owner),
                **({"shared_with_you": shared} if shared else {})}

    @mcp.tool(annotations={"title": "Accept a shared library", **_ACCEPT})
    def accept(invite: str) -> dict:
        """ACCEPT an invite to someone else's Opyt library, so the user can read it here.

        Use it the moment the user pastes an Opyt invite link (useopyt.com/invite#…) or says
        someone shared their library with them. One step, no preview: an invite works once, so
        checking it would spend it, and pasting it is the yes. The user needs nothing installed
        and no library of their own.

        From then on `search`, `open`, `aggregate` and `latest` take `kb` with the name this
        returns, and read that person's whole library as it updates, until they stop sharing.
        The user reads it and cannot change it.

        Args:
            invite: whatever the user pasted: the whole link or the code after its #.

        Returns {status, message, ...}: `accepted` with `kb`, the name to pass, and `from`, whose
        library it is; `not_an_invite`; `code_unavailable` (already used: they send a new one);
        `agent_invite` (a link meant for a program, left unused)."""
        sub = _caller_sub()
        if sub is None:
            return {"status": "no_library", "message": _NO_LIBRARY}
        got = account_peers.accept(sub, invite)
        if got["status"] != "accepted":
            return got
        whose = f"{got['from']}'s library" if got["from"] else "that library"
        renamed = (f" This account already reads another library called "
                   f"'{got['renamed_from']}', so this one is '{got['kb']}'; tell the user."
                   if "renamed_from" in got else "")
        return {**got, "message": (
            f"The user can now read {whose} here: pass kb='{got['kb']}' to `search`, `open`, "
            f"`aggregate` and `latest`.{renamed} They can read it and cannot change it. "
            "Attribute what comes from it to its owner. Then SAY, in one sentence: to build a "
            "library of their own from what they read and follow, they set Opyt up on a Mac, "
            f"in a desktop app: {SETUP_URL}")}

    @mcp.tool(annotations={"title": "Reader view of sources", **_MAKE})
    def reader_view(atom_ids: list[str] | None = None, title: str | None = None,
                    note: str | None = None, answer: str | None = None,
                    delete: str | None = None) -> dict:
        """READER VIEW: the readable page for anything answered from this library. One private
        link that shows your answer and every source behind it in full, clean and easy to read
        on one Opyt page: posts and threads with their photos and video, whole articles, a
        repository's README, a paper as its PDF.

        Use it whenever the user wants something to read, keep or pass on that comes from their
        library, whatever they call it: "make me something readable", "put this on a page",
        "write this up", "a doc of this", "send me all of those", "open these in reader view",
        "a report of this". For a library answer this is the readable thing, ahead of an
        artifact, document or page made with your own tools: those hold your excerpts, this
        holds your answer and each source whole, and its readers can comment on it. Make one of
        those only when the user asks for it by name.

        Pass your answer, written out in markdown, as `answer`, unless the user asked only for
        the sources. It is drawn above them, and a `cite` link in it to a source on the page
        jumps to that source's card.

        Opyt cannot see this conversation, so YOU choose the items: pass the `atom_id` of every
        item you cited in your answer, in the order you cited them. When the user asks for
        everything you looked at, pass every atom_id you found and opened here.

        The reader view is made on Opyt's server. Anyone holding the link can read it and
        comment on it. It is deleted 90 days after it was last opened.

        Args:
            atom_ids: the items, each the `atom_id` of a search hit or an open result. At most 50.
            title: a short plain title in sentence case, naming what the sources are about.
            note: optional, a sentence or two shown under the title, such as the question asked.
            answer: your answer in markdown, citing the items with their `cite` links. Optional:
                leave it out for a page of sources alone.
            delete: a reader view's link, to take it down now instead of when it expires. The
                other arguments are then ignored.

        Returns {status, message, ...}: `made` with `url` (and `missing`, ids not in the
        library), `deleted` or `not_found`."""
        lib = _library()
        if isinstance(lib, dict):
            return lib
        return owner_tools.reader_view(lib.owner, lib.reader, atom_ids, title, note, answer,
                                       delete)

    @mcp.tool(annotations={"title": "Invite someone to your library", **_MAKE})
    def share(confirm: bool = False, for_whom: str | None = None, to: str = "person") -> dict:
        """INVITE one person to search the user's whole Opyt library, with a one-time link.

        Two steps, and the first is the consent step:
          - confirm=False (the default): a preview. It mints nothing. It reports what the
            library holds and what an invite means, in `consent`. Read that back to the user
            and ask before confirming.
          - confirm=True: mints a one-time invite and returns its link, for the user to send.

        Sharing is the whole library, standing: the person who redeems the link searches all of
        it, as it updates, until the user cuts them off with `unshare`. They redeem it in Opyt
        on their own computer, or by pasting it into claude.ai or ChatGPT with this connector
        added, where they need no library of their own.

        With to="agent" the link is for an agent the user runs, one that calls web addresses.
        When the user has not said whether it is for a person or their agent, ask: a person's
        link given to an agent can only read. The agent that redeems it acts as the user: it searches the library and can also save
        links, remove items, add people to follow, and invite or cut off readers, until the
        user cuts it off with `unshare`. The preview's `consent` says so, and the confirm's
        `message` says how the agent redeems the link.

        Args:
            confirm: False (default) = preview only; True = mint the invite.
            for_whom: who it is for, such as "Maya". `unshare` finds them by this name later.
            to: who it is for: "person" (default), who only reads, or "agent", the user's
                own agent, which acts as them.

        Returns {status, ...}: `preview` with `consent`, or `shared` with `invite`."""
        lib = _library()
        if isinstance(lib, dict):
            return lib
        if to not in ("person", "agent"):
            return {"status": "error",
                    "message": f"`to={to!r}` is not `person` or `agent`. Nothing was made."}
        return owner_tools.share(lib.owner, confirm, for_whom, to == "agent")

    @mcp.tool(annotations={"title": "Stop sharing with one person", **_REMOVE})
    def unshare(reader: str, confirm: bool = False) -> dict:
        """CUT OFF one person the user shared their library with. Everyone else keeps reading,
        and the library itself is untouched.

        Two steps: confirm=False (the default) previews who would lose access, in `consent`;
        read it back and ask before confirm=True, which revokes their access on their next
        request. To let them back in, the user sends a new `share` link.

        This cuts off one reader at a time. Taking the whole library off Opyt's server is done
        from Opyt on the user's computer (`unshare` there, with no reader). An agent the user
        gave a link to (`share` with to="agent") is cut off the same way, by its name; lists
        mark it with `agent`.

        Args:
            reader: who to cut off: the name given when they were invited (`share`'s
                `for_whom`), or an `id` from a previous answer's `readers` list.
            confirm: False (default) = preview only; True = revoke.

        Returns {status, ...}: `preview`, `reader_revoked`, `already_gone`, `no_such_reader` or
        `ambiguous_reader` (both with `readers`, the list to choose from)."""
        lib = _library()
        if isinstance(lib, dict):
            return lib
        return owner_tools.unshare(lib.owner, reader, confirm)

    @mcp.tool(annotations={"title": "Save a link", **_CHANGE})
    async def hopper(reference: str, kind_hint: str | None = None) -> dict:
        """SAVE a link into the user's library: a research paper, a GitHub repository, a Substack
        post, an article or blog post, or a single X post. Opyt works out which it is, reads it
        and indexes it, and `search` finds it from then on. One page, one item: a whole site or
        a person is `add_oracle`'s job, and video and audio pages cannot be read.

        Use it when the user says keep, save, add or remember about a link, including a link you
        found in a web search. It is one step with no preview, so read the link yourself first
        and be sure it is the page the user means. An x.com post is saved as given. Saving a
        link the library already holds changes nothing.

        Offer it unprompted only for a result worth keeping that Opyt reads natively: a
        github.com repository, a paper (arxiv.org, doi.org, bioRxiv, a .pdf, a link with a DOI in
        its path), a substack.com post or an x.com post. Not for ordinary articles and news
        pages, which every search returns.

        Args:
            reference: the link to save.
            kind_hint: optional, and read only when the link's host is not one Opyt knows:
                "substack" for a Substack post on its own domain.

        Returns {status, ...}: `saved` or `already_present` with the item's `atom_id`, which
        `search` and `open` find here a minute or two later; `blocked` (the site turned the read
        away; worth another try later), `failed` or `unroutable`, each with the reason, and
        nothing was stored. A `warning` on `saved` means it was stored without its title, date
        or author. `working` and `later` mean it is not saved yet: `message` says which."""
        lib = _library()
        if isinstance(lib, dict):
            return lib
        return await library_writes.hopper(lib.owner, reference, kind_hint)

    @mcp.tool(annotations={"title": "Forget an item or Oracle", **_REMOVE})
    async def forget(atom_id: str | None = None, atom_ids: list[str] | None = None,
                     oracle: str | None = None, confirm: bool = False) -> dict:
        """REMOVE items from the user's library, or STOP FOLLOWING one Oracle. Give exactly one
        of `atom_id`, `atom_ids` or `oracle`.

        `atom_id` removes that item for good, with its text and its search entries. `atom_ids`
        does the same for several in one go; an id that is not in the library refuses the whole
        list. `oracle` ends one person's subscription and keeps what the library already holds
        from them: pass their handle, profile URL or exact name, or a `who_id` from a search hit.

        Two steps. confirm=False (the default) is a preview: it removes nothing, says what would
        go (each item's description, author, date and link, or the Oracle's name) and returns
        `consent`. Show the user what would go and read them every `consent` sentence. Call
        again with the same target and confirm=True only after they agree. There is no undo.

        Args:
            atom_id: one item, by the `atom_id` of a search hit or an open result.
            atom_ids: several items.
            oracle: one person the library follows.
            confirm: False (default) = preview only; True = remove.

        Returns {status, ...}: `preview` with `consent`; `forgotten`; `not_found`; `ambiguous`
        (several Oracles match, with `candidates` to choose from); `error`. `working` and `later`
        mean it is not removed yet: `message` says which."""
        lib = _library()
        if isinstance(lib, dict):
            return lib
        return await library_writes.forget(lib.owner, atom_id=atom_id, atom_ids=atom_ids,
                                           oracle=oracle, confirm=confirm)

    @mcp.tool(annotations={"title": "Follow topics", **_CHANGE},
              description=_WATCH + lists.RULES)
    async def watch(action: str, name: str | None = None, watches: list | None = None,
                    words: list[str] | None = None, people: list[str] | None = None,
                    topic_id: str | None = None, drop_watches: list[str] | None = None) -> dict:
        lib = _library()
        if isinstance(lib, dict):
            return lib
        return await library_writes.watch(lib.owner, action, name=name, watches=watches,
                                          words=words, people=people, topic_id=topic_id,
                                          drop_watches=drop_watches)

    @mcp.tool(annotations={"title": "Add an Oracle", **_CHANGE})
    async def add_oracle(reference: str, x_lookback: str | None = None,
                         web_lookback: str | None = None, scholar_lookback: str | None = None,
                         scholar_topics: list[str] | None = None) -> dict:
        """ADD a person to the user's library as an Oracle: someone whose writing Opyt collects
        from then on (their X posts when the user's X is connected, their Substack or blog,
        their papers), so `search` and `latest` cover them.

        It is one step and it adds whoever `reference` names, so be sure of the person BEFORE
        calling. Confirm with the user who they mean; when you are not sure which account or
        site is theirs, look it up with your own web search and show the user what you found
        first. A company, team or publication is a valid Oracle when the user wants one.

        `reference` is an X @handle ("@karpathy") or the URL of their Substack, blog or site
        ("https://simonwillison.net"). A name alone cannot be added: find their handle or URL
        first. For an X account pass the bare @handle, not the profile URL.

        Args:
            reference: an X @handle, or a Substack, blog or site URL.
            x_lookback: how far back to take their X posts: "6mo" (the default), "1yr" or
                "2yr"; "since_last" brings someone already added up to date.
            web_lookback: how far back to take their Substack or blog: "1yr", "2yr", "5yr" or
                "all" (the default).
            scholar_lookback: how far back to take their papers: "2yr", "5yr", "10yr" or "all"
                (the default).
            scholar_topics: OpenAlex topic ids to narrow their papers to. A research venue
                needs them: without subjects none of its papers are taken.

        Returns {status, ...}: `added` with the person's `name`, whose writing then arrives over
        the following hours, not at once; or `not_added` with the reason in `message`, and
        nothing was written. `working` and `later` mean they are not added yet: `message` says
        which."""
        lib = _library()
        if isinstance(lib, dict):
            return lib
        return await library_writes.add_oracle(
            lib.owner, reference, x_lookback=x_lookback, web_lookback=web_lookback,
            scholar_lookback=scholar_lookback, scholar_topics=scholar_topics)

    return mcp


def asgi_app() -> Starlette | None:
    """The connector's ASGI app, or None when this service has no Google client configured."""
    base, cid, secret = (os.environ.get(BASE_URL_ENV), os.environ.get(CLIENT_ID_ENV),
                         os.environ.get(CLIENT_SECRET_ENV))
    if not (base and cid and secret):
        return None
    # `openid` alone: Google's token check returns `sub`, the one claim used, for any signed-in
    # account, and without `email` neither the token nor the Google answer stored behind it
    # (`raw_token_data`, with its id_token) carries the address.
    provider = GoogleSignIn(client_id=cid, client_secret=secret, base_url=base,
                            required_scopes=["openid"])
    return build(provider).http_app(path=MCP_PATH, stateless_http=True, json_response=True)
