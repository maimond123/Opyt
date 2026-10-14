"""
pipeline/retrieval/cards.py — a hit as the host reads it, and its paste-ready `cite` link.
"""
from __future__ import annotations


# `cite` is a ready-made markdown link whose label is the text the host would write anyway, so
# linking costs it no width. `source_url` stays alongside it for callers that program against a hit.
def _md_link(label: str | None, source_url: str | None) -> str | None:
    """`[label](url)`, or None when there is no URL (never a dead `[label]()`).

    The label loses brackets and a target with spaces or parens is wrapped in angle brackets, so
    user-controlled text cannot end either half early.
    """
    if not source_url:
        return None
    text = (label or "source").replace("[", "").replace("]", "").strip() or "source"
    target = source_url.strip()
    if any(c in target for c in " ()<>"):
        target = f"<{target.replace('<', '').replace('>', '')}>"
    return f"[{text}]({target})"


def cite(source_url: str | None, who_name: str | None, source_type: str | None,
          when_ts: str | None) -> str | None:
    """The paste-ready link, labelled with author and day so it still attributes the source."""
    label = (who_name or source_type or "source")
    day = (when_ts or "")[:10]
    return _md_link(f"{label} · {day}" if day else label, source_url)


def hit_card(h, kb_name: str) -> dict:
    return {
        "citation_id": h.atom_id,
        "atom_id": h.atom_id,
        # On every hit, local ones too ("me"), so a card quoted alone still carries its attribution.
        "kb": kb_name,
        "source_type": h.source_type,
        "what_kind": h.what_kind,
        "who_id": h.who_id,
        "who_name": h.who_name,     # display label; `who_id` stays the identity
        "when_ts": h.when_ts,
        "when_precision": h.when_precision,
        "description": h.description,
        "snippet": h.snippet,
        # Where `snippet` sits in the text `open` returns, as [start, end) offsets.
        "snippet_span": list(h.snippet_span) if h.snippet_span else None,
        "source_url": h.source_url,
        "cite": cite(h.source_url, h.who_name, h.source_type, h.when_ts),
        "score": round(h.score, 4),
        # On the card so a card read alone still says which section it belongs to.
        "entry_mode": h.entry_mode,
        "body_state": h.body_state,   # "complete"/"partial"/"absent"/"pending" — qualifies `snippet`
        "body_basis": h.body_basis,
        "payload": h.payload,         # source-shaped extras, verbatim (no allowlist)
    }
