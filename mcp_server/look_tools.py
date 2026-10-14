"""
mcp_server/look_tools.py — `look`: search outside the library now for a few named things.

Thin: it reads the arguments and calls one flow, `pipeline/frontier/look.py`, which reads nothing
of the store and writes nothing.
"""
from __future__ import annotations

from pipeline.frontier import look as look_flow
from pipeline.timeparse import utc_now

_LOOK = """\
Search OUTSIDE the user's library, right now, for one to three specific named things: what X, \
Substack, arXiv, OpenAlex and GitHub have on each over the last days. Use it when the user asks \
what is new or what people are saying about a named thing: a model, drug, trial, benchmark, \
dataset, protocol, a lab's program ("what are people saying about Boltz-2 this week?"). It \
writes nothing, and takes up to a minute.

Each name is searched as a phrase, and a result is kept only if it names it, as a topic's watch \
does: give the thing's name, never a subject ("AI drug discovery") or a description. When the \
user asks about a subject, pull the one to three NAMES out of it yourself — the compound, model, \
dataset, benchmark or program they are really asking about — and look for those.

`days` can reach back years, which is what a literature survey wants: past 90 days OpenAlex \
answers with the most-cited work rather than the newest. X and Substack cannot be searched \
further back than a month and say so in `searched_less` when you ask for more.

What comes back is not the user's own: say so, and never present it as their people's view. \
Show only what is worth their time, each with its `cite` link. Each result carries the `save` \
arguments for `hopper`, if the user wants to keep it.

Args:
    names: 1 to 3 things, each a name or {"name": "Chai-2", "spellings": ["Chai 2"], \
"context": ["antibody"]}. `context` is for X only: one or two words any on-topic post would \
carry, for a name that is also an idiom, a common word or another thing ("Chai-2" without \
"antibody" finds football).
    sources: any of "x", "substack", "arxiv", "openalex", "github"; all of them when left out.
    days: how far back, a whole number of days, 1 or more (default 7).
"""


def register_look_tools(mcp) -> None:
    @mcp.tool(annotations={"title": "Look outside the library", "readOnlyHint": True,
                           "destructiveHint": False, "openWorldHint": True},
              description=_LOOK)
    def look(names: list, sources: list[str] | None = None, days: int | None = None) -> dict:
        return look_flow.run(names=names, sources=sources, days=days, now=utc_now())
