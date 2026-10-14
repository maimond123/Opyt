"""
mcp_server/page_tools.py — `reader_view`: an answer and its sources, on one private page.

The flow is `pipeline/sharing/page.py`; the service draws the page (`service/pages/`). The
connector offers the same tool over the caller's served copy (`service/connector.py`).
"""
from __future__ import annotations

from pipeline.sharing import page as page_flow


def register_page_tools(mcp) -> None:

    @mcp.tool(annotations={"title": "Reader view of sources", "readOnlyHint": False,
                           "destructiveHint": False, "openWorldHint": True})
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
        "a report of this", or a yes when it is offered. For a library answer this is the
        readable thing, ahead of an artifact, document or page made with your own tools: those
        hold your excerpts, this holds your answer and each source whole, and its readers can
        comment on it. Make one of those only when the user asks for it by name.

        Pass your answer, written out in markdown, as `answer`, unless the user asked only for
        the sources. It is drawn above them, and a `cite` link in it to a source on the page
        jumps to that source's card.

        Opyt cannot see this conversation, so YOU choose the items: pass the `atom_id` of every
        item you cited in your answer, in the order you cited them. When the user asks for
        everything you looked at, pass every atom_id you found and opened here. Only items from
        the user's own library; an id from someone else's shared library is left out.

        Making a reader view uploads those items, and nothing else of the library, to Opyt's
        server. Anyone holding the link can read it and comment on it. It is deleted 90 days
        after it was last opened.

        Args:
            atom_ids: the items, each the `atom_id` of a search hit or an open result. At most 50.
            title: a short plain title in sentence case, naming what the sources are about.
            note: optional, a sentence or two shown under the title, such as the question asked.
            answer: your answer in markdown, citing the items with their `cite` links. Optional:
                leave it out for a page of sources alone.
            delete: a reader view's link, to take it down now instead of when it expires. The
                other arguments are then ignored.

        Returns {status, message, ...}: `made` with `url` (and `missing`, ids not in the
        library), `deleted`, `not_found` or `failed`.
        """
        if delete:
            return page_flow.delete(delete)
        return page_flow.make(atom_ids or [], title, note, answer)
