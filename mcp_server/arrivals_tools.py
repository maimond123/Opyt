"""
mcp_server/arrivals_tools.py — the `latest` tool: what is new for the user since they last looked,
by the topics they follow. The work lives in `pipeline/arrivals/show.py`.
"""
from __future__ import annotations

from contextlib import closing

from pipeline.arrivals import show
from pipeline.kb import schema
from pipeline.timeparse import utc_now


def register_arrivals_tools(mcp) -> None:
    @mcp.tool(annotations={"title": "What's new", "readOnlyHint": True,
                           "destructiveHint": False, "openWorldHint": False})
    def latest(topic_id: str | None = None, limit: int = 50) -> dict:
        """What is new for the user since they last looked. Use it when they ask what's new or
        what came in, and when a scheduled task asks about their topics.

        Per topic they follow (`watch`): `finds`, new papers and repos its watches found, which
        wait outside the library for YOUR check; and `posts`, their own people's new posts on it.
        Then `unfiled`: what their Oracles posted, what they saved and pages their Oracles linked,
        outside every topic. Each post says `why` it is here.

        CHECK BEFORE YOU SHOW, as `next` says: keep only what is genuinely about its topic and
        worth the user's time, show those with their `cite` links, record your check with
        watch(action="checked"), and never mention what you passed over. Present the rest as it
        is: do not judge it or point out any of it as junk, broken or not worth keeping.

        New means recently published. Older items that arrived at the same time, such as an
        Oracle's archive, are not listed: `older` counts them in finished lines, one per Oracle or
        kind. Mention them in a sentence; do not list them.

        EVERY ITEM YOU NAME CARRIES ITS LINK: write its name inside its `cite`, as with `search`.

        Called with no `topic_id`, it marks the posts seen, so the next call starts from now; with
        `topic_id` it shows that topic alone and marks nothing. Finds are never marked seen: they
        wait until checked. `limit` (default 50) caps the posts listed, shared among the groups.
        """
        with closing(schema.connect()) as conn:
            return show.show(conn, topic_id=topic_id, limit=limit, now=utc_now())
