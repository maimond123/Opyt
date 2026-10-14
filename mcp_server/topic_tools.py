"""
mcp_server/topic_tools.py — `watch`: the topics the user follows.

Thin: each action calls one flow in `pipeline/topics/`. Opyt runs no model for a topic: the daily
pulls search its watches in the background, and the host checks what `latest` hands over. How to
write a topic's lists is measured, so it rides this tool's description (`lists.RULES`).
"""
from __future__ import annotations

from contextlib import closing

from pipeline.kb import schema
from pipeline.timeparse import utc_now
from pipeline.topics import checked, create, lists, preview

_WATCH = """\
Follow topics for the user: they say what they want to stay up to date on, and they are told when \
something new on it appears, from outside (new papers, repos, X posts and Substack posts) and from \
their own people (their Oracles' posts and what they saved).

FLOW:
1. `preview` with a `name` (the subject in the user's words) and the three lists below. It \
searches each watch over the last month and counts what it would have found, with examples, and \
how many of the user's people's posts a day the words and people match. It writes nothing. Show \
it to the user.
2. When they agree (or after their edits), `create` with the same lists. Its answer says what to \
ask them next.
3. New finds wait, outside the library, for your check: `latest` hands them over with the \
user's people's new posts on the topic. Check each against the topic and show only what is worth \
the user's time.
4. `checked` with every find you checked: `keep` and `passed`, by candidate_id. Kept finds join \
the library; passed ones never come back. Never tell the user about the ones you passed.

Other actions: `edit` (topic_id, plus any of: name; watches to add; drop_watches by name; words \
or people, which replace the old lists), `drop` (topic_id: stop following), `list`.

Args:
    action: "preview" | "create" | "edit" | "drop" | "list" | "checked".
    name: the topic's name, in the user's words.
    watches: [{"name": "Boltz-2", "spellings": ["Boltz2"], "sources": ["github", "openalex"]}, \
{"name": "Chai-2", "context": ["antibody"]}].
    words: the subject words, a list of short phrases.
    people: handles of the user's Oracles who mostly write about it.
    topic_id: the topic to edit or drop.
    drop_watches: names of watches to stop (edit).
    keep, passed: candidate_ids of the finds you checked (checked).

HOW TO WRITE THE LISTS:
"""


def _watch_description() -> str:
    return _WATCH + lists.RULES


def register_topic_tools(mcp) -> None:
    @mcp.tool(annotations={"title": "Follow topics", "readOnlyHint": False,
                           "destructiveHint": False, "openWorldHint": True},
              description=_watch_description())
    def watch(action: str, name: str | None = None, watches: list | None = None,
              words: list[str] | None = None, people: list[str] | None = None,
              topic_id: str | None = None, drop_watches: list[str] | None = None,
              keep: list[str] | None = None, passed: list[str] | None = None) -> dict:
        with closing(schema.connect()) as conn:
            match action:
                case "preview":
                    return preview.run(conn, name=name, watches=watches, words=words,
                                       people=people, now=utc_now())
                case "create":
                    return create.create(conn, name=name, watches=watches, words=words,
                                         people=people)
                case "edit":
                    return create.edit(conn, topic_id=topic_id, name=name, watches=watches,
                                       drop_watches=drop_watches, words=words, people=people)
                case "drop":
                    return create.drop(conn, topic_id=topic_id)
                case "list":
                    return create.listing(conn)
                case "checked":
                    return checked.run(conn, keep=keep, passed=passed)
        return {"status": "error",
                "reason": ("action is \"preview\", \"create\", \"edit\", \"drop\", \"list\" or "
                           f"\"checked\" (got {action!r})")}
