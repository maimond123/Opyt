"""
mcp_server/instructions.py — what the server says at connection setup: the standing citation rule.

`instructions` is optional on InitializeResult and a client may drop it, so nothing may depend on
it: routing policy lives in tool descriptions. The citation rule survives being dropped, because it
is reinforcement: `search`'s description and every hit's `cite` still carry it.
"""
from __future__ import annotations

# One paragraph: text here competes with the user's own instructions for the same budget.
CITATION_RULE = (
    "CITING OPYT RESULTS: every knowledge-base item you name in an answer must carry its source "
    "link. Each `search` hit ships a ready-made markdown link in `cite` — write the item's name "
    "inside it, so the title you were already going to write IS the link, costing no extra width. "
    "Never hand back a list of results with no links in it, and never make the user ask where "
    "something came from: a result they cannot click back to is a claim they cannot check."
)
