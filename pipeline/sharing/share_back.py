"""
pipeline/sharing/share_back.py — the offer to share back, made once per peer and never repeated.

Sharing is one-directional; the second direction is an offer, made when a peer's answer found
something (the server's rider flow, `mcp_server/rider_rules.py`), never after empty hits.
"""
from __future__ import annotations

import re
from pathlib import Path

from opyt_core.paths import opyt_path


def _marker(kb_name: str) -> Path:
    """Where the offer for one normalized peer name is latched. Resolved at call time so it
    honors `$OPYT_HOME`; names that normalize alike share one latch on purpose."""
    slug = re.sub(r"[^a-z0-9]+", "-", kb_name.lower()).strip("-") or "peer"
    return opyt_path("share_offers", slug)


def offer(kb_name: str) -> dict | None:
    """The offer to share back after a read of `kb_name`, latched as it is handed over — or None
    once it has been made.

    The latch is on disk, not in the session, so a new session does not re-offer. A marker that
    cannot be written leaves the offer unlatched, to be made again next time."""
    marker = _marker(kb_name)
    if marker.exists():
        return None
    notice = {
        "code": "reciprocal_offer", "kb": kb_name,
        "message": f"That answer came from {kb_name}'s knowledge base. If the user has one of "
                   f"their own, ask whether they want to share it back — the `share` tool "
                   f"returns a link they can send. Ask once; do not raise it again."}
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except Exception:
        pass
    return notice
