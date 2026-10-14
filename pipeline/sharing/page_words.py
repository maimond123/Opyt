"""
pipeline/sharing/page_words.py — what a reader view tells the host, worded once for the Mac's
`reader_view` and the connector's: how to make something readable of an answer, and what to do
with a page just made.
"""
from __future__ import annotations

# Rides every search and open answer on the user's own library, on both servers: the host decides
# how to make something readable when the user asks, which can be long after any offer, and a
# tool's description may not be in its context then (clients that load tools on demand).
READABLE = (
    "When the user wants any of this as something to read, keep or send, or written up (a page, "
    "a doc, a report, a summary), make it with `reader_view`: the atom_id of each item you cite, "
    "and your write-up as `answer`. It shows the answer and every source whole on one page, so "
    "do not make an artifact or a document of it with your own tools unless the user names one.")


def made_message(days: int, missing: list[str]) -> str:
    """What the host does with a page just made, on either server."""
    left_out = (f" {len(missing)} of the ids are not in the user's library and are not on it."
                if missing else "")
    return (f"Give the user the link as a markdown link whose text is the page's title, and say "
            f"in the same line that anyone with the link can read it and comment on it, and that "
            f"it is deleted {days} days after it was last opened. The page is what they asked "
            f"for: do not also make an artifact, document or page of the same answer with your "
            f"own tools.{left_out}")
