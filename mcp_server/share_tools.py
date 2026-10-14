"""
mcp_server/share_tools.py — the three tools that make knowledge-base sharing exist for a person.

The constraint is frictionless UX: no shell on either side, no credential a human transcribes,
and one yes per person, covering everything after it.

  • `share`   — two-phase: the preview is the consent step (`pipeline/sharing/share.py`). For a
                person, for the user's own agent, or for their own claude.ai or ChatGPT.
  • `accept`  — single-phase: pasting the invite is the yes (`pipeline/sharing/accept.py`).
  • `unshare` — two-phase, two scopes under one verb (`pipeline/sharing/unshare.py`): one
                reader, or everything Opyt's server holds for this library, deleted for good.

Every failure is a sentence in the return value, never a raise.
"""
from __future__ import annotations

from pipeline.sharing import accept as accept_flow
from pipeline.sharing import share as share_flow
from pipeline.sharing import unshare as unshare_flow


def register_share_tools(mcp) -> None:

    @mcp.tool(annotations={"title": "Share the knowledge base", "readOnlyHint": False,
                           "destructiveHint": False, "openWorldHint": True})
    def share(confirm: bool = False, as_name: str | None = None,
              for_whom: str | None = None, refresh: bool = False, to: str = "person") -> dict:
        """SHARE this knowledge base: with someone else, with an agent the user runs, or with
        the user's own claude.ai or ChatGPT. This is the one tool for every way of reaching the
        library from outside this computer.

        WHO IT IS FOR decides what happens, so find out before confirming (`to`):
          • to="person" (the default) → a link for SOMEONE ELSE, who can read the library and
            change nothing. Reach for it when the user says share my KB / send my knowledge base
            to X / let X search what I've read. The person pastes the link into Opyt on their
            own computer, or into claude.ai or ChatGPT with the Opyt connector added, where they
            need nothing installed.
          • to="agent" → a link for an AGENT THE USER RUNS, one that calls web addresses and not
            Opyt's tools. It acts as the user.
          • to="connector" → the user's OWN claude.ai or ChatGPT (also what "connect Opyt to
            Google" means). No link: a sign-in on Google's page.
        When the user has not said which, ASK. A reader's link given to an agent can only read.

        TWO-PHASE, and the preview is the consent step, so run it first:
          • confirm=False (the default) → a PREVIEW. It publishes nothing, mints nothing and
            hands nobody anything; on an install that has already shared it asks the service
            whether an export has landed, and that is the only thing it sends. It
            reports WHAT would be shared — how many atoms, from which sources and authors, over
            what date span — and what sharing means. READ THAT BACK to the user before confirming.
            Sharing is the WHOLE knowledge base, standing, not a slice and not a snapshot: there
            is no way to share part of it, anyone given a link keeps access until it is revoked,
            and the copy updates itself every hour while this Mac is awake.
          • confirm=True → registers this install with the service if it is not registered yet,
            starts the first upload, mints a one-time invite, and returns the link.

        Skip straight to confirm=True only when the user has already said yes to sharing THIS
        knowledge base, having been told what is in it.

        The first share uploads the whole corpus, which runs in the background and usually takes a
        minute or two. The link works immediately: a reader who accepts inside that window is
        registered normally, and it is their first SEARCH that comes back saying the copy is
        still arriving. The next one works. Later shares are instant.

        UPDATE NOW: refresh=True uploads the library to the shared copy at once, instead of at
        the next hourly pass. Use it when the user wants the people they share with, or the
        Opyt connector on claude.ai / chatgpt.com, to see what just changed. No invite, no
        preview; an unchanged library uploads nothing and says so.

        THE USER'S OWN AGENT (to="agent"): the agent that redeems the link acts as the user: it
        can search the library, save links, remove items, add people to follow, and invite or
        cut off readers, until `unshare` cuts it off. Same two phases; the preview's `consent`
        says what the agent may do, and the confirm's `message` says how the agent redeems the
        link. Use it only when the user asks for a link for their agent, never for a person.

        THE USER'S OWN CLAUDE.AI OR CHATGPT (to="connector"): it ties this library to the
        user's Google account and uploads the whole library to useopyt.com, where it stays
        current. The preview's `consent` says what that gives and what still only works here;
        read it back. confirm=True then returns a code to enter on Google's page: give the user
        the page and the code as `message` says, WAIT until they say they approved, and call
        `share(to="connector", confirm=True)` again to finish. Nothing is connected until then.
        `as_name` and `for_whom` are not used.

        WHAT THIS IS NOT. It does not publish anything on the open web — the service serves only
        people holding a link, one reader token per link, revocable. It does not send anything to
        the person for you: hand the user the link and let them send it.

        Args:
            confirm: False (default) = preview only, nothing written or sent; True = register,
                publish and mint the invite.
            as_name: what to call this knowledge base's owner. This is what every recipient's
                install suggests as the name they search under, so use the user's own name or
                handle. Required on the FIRST share; ignored afterwards, because the name is
                already registered.
            refresh: True = update the shared copy now; every other argument is ignored.
            for_whom: an optional label for who this particular invite is for — a first name
                or handle. It is what makes that person nameable afterwards:
                `unshare(reader="<that label>")` cuts off exactly them. Without it they can
                only be named by a token id.
            to: who it is for: "person" (default), "agent" or "connector".

        Returns {status, ...}. A preview carries `atoms`, `by_source_type`, `top_entities`,
        `date_span`, `consent` and `already_shared`, and for a person `who_is_it_for`, the
        question to put. A confirm carries `invite` — the link — plus `owner` and `queued`
        (whether the background push was queued); for the connector it is `waiting` with
        `open_this_url`, then `ok`. A refresh is `refreshing` (with `queued`), `no_change`,
        `not_published` or `not_shared`.
        """
        if refresh:
            return share_flow.refresh()
        return share_flow.run(confirm, as_name, for_whom, to)

    @mcp.tool(annotations={"title": "Accept a shared knowledge base", "readOnlyHint": False,
                           "destructiveHint": False, "openWorldHint": True})
    def accept(invite: str, name: str | None = None) -> dict:
        """ACCEPT an invitation to search somebody else's knowledge base.

        Reach for this the moment the user pastes an Opyt invite link or a grant code, or says
        someone shared their knowledge base with them. One call and it is done: from then on the
        search / open / aggregate tools take `kb='<name>'` and read that person's corpus, with
        every result attributed to them.

        SINGLE-PHASE on purpose — there is no preview. A grant code buys exactly one reader token
        and then dies, so checking it would spend it, and pasting an invite is already the yes.

        Pass whatever the user gave you: the whole link, the fragment, or the bare code. The code
        is found inside any of them.

        Args:
            invite: the invite link or grant code.
            name: OPTIONAL — what `kb=` should call this knowledge base locally. Leave it out and
                the owner's own name is used. Pass one only if the user asks for a specific name.

        Returns {status, kb, owner, message}. `kb` is the name to pass as `kb=` — use THAT string,
        not the owner's, because a name already taken on this install gets a suffix.
        """
        return accept_flow.run(invite, name)

    @mcp.tool(annotations={"title": "Stop sharing", "readOnlyHint": False,
                           "destructiveHint": True, "openWorldHint": True})
    def unshare(confirm: bool = False, reader: str | None = None) -> dict:
        """STOP sharing — either with ONE person, or with everybody. `reader` picks which.

        READ THIS BEFORE CALLING. The two scopes are not the same act and only one of them is
        expensive to undo:
          • `reader="<name>"` → that one person loses access. Everyone else keeps reading,
            the served copy stays, and you can invite them again with `share` whenever you
            like. An agent given a link with `share(to="agent")` is cut off the same way, by
            its name; lists mark it with `agent`. `reader="google"` disconnects the user's own
            claude.ai and ChatGPT (their Google account) and nobody else.
          • `reader` omitted → DELETES, for good, everything Opyt's server holds for this
            knowledge base: the served copy, every reader and every invite ever sent, the count
            of its reads, and its link to the user's Google account, so claude.ai and ChatGPT
            can no longer read it. Sharing or connecting Google again starts over. The knowledge
            base on this computer is not touched.

        So when the user names a person — "stop sharing with <name>", "cut <name> off" —
        `reader` is REQUIRED. Omitting it there does something much larger than what was asked,
        and the preview is where you catch that: it always says which of the two scopes it is
        about, in its first sentence.

        Reach for the whole-knowledge-base form when the user says stop sharing / unshare / take
        my knowledge base down / revoke everyone / delete my library from Opyt's server. That
        form is one act and not two, because a person who says "stop sharing my KB" means both
        halves and will not say it twice.

        TWO-PHASE either way:
          • confirm=False (the default) → a PREVIEW of exactly what that scope would do. Nothing
            is changed.
          • confirm=True → does it. Access ends on that reader's very next request; there is no
            refresh cycle and no window. For the whole knowledge base, call it only after the
            user has said yes to the preview's `consent`, read back to them: it deletes their
            data on Opyt's server and cannot be undone.

        Args:
            confirm: False (default) = preview only; True = do it.
            reader: WHO to cut off, matched against the label you gave when you shared
                (`share`'s `for_whom`), case-insensitively. A token hash from a previous preview
                also works, which is how you name a reader you never labelled. Omit ONLY when the
                user means everybody.

        Returns {status, ...}. A `reader` call carries `reader_label`; a whole-knowledge-base
        confirm carries `readers_revoked` and `export_deleted`. `scope` is on every preview and
        says which of the two you are looking at.
        """
        return unshare_flow.run(confirm, reader)
