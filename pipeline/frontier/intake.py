"""
pipeline/frontier/intake.py — which of one page's candidates are kept: those that name what the
watch watches.

A watch names one thing (a model, a trial, a benchmark) and every source searches it as a phrase,
but a source still returns finds that only brush it: a citation in passing, a near-identical name.
A find is kept when the name appears in its title or its abstract (a repo's name or description),
which cut off-topic finds from 44% to 13% on 403 labelled results (measured). One rule for every source, applied where each page passes.

A decision: a page of candidates and the watch's name in, the kept ones and the dropped count out.
`mention` is the same reading of a longer text: the passage that names the watch, for a source
whose results carry no body (Substack) and show a find by where it names it.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")


def _word(token: str) -> str:
    """A plural's final `s` dropped, so "Virtual Cell Challenges" matches "Virtual Cell
    Challenge"."""
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _tokens(text: str) -> list[str]:
    """Lowercase words and numbers, so "Boltz-2" reads as `boltz 2`."""
    return [_word(t) for t in _TOKEN.findall((text or "").lower())]


def _at(name: str, text: str) -> tuple[int, int] | None:
    """Where `text` first holds `name` as a phrase (its words, in order, side by side), as a
    character span, or None."""
    want = _tokens(name)
    if not want:
        return None
    have = [(_word(m.group()), m.start(), m.end())
            for m in _TOKEN.finditer((text or "").lower())]
    for i in range(len(have) - len(want) + 1):
        if [w for w, _, _ in have[i:i + len(want)]] == want:
            return have[i][1], have[i + len(want) - 1][2]
    return None


def names(name: str, text: str) -> bool:
    """Whether `text` holds `name` as a phrase: its words, in order, side by side."""
    return _at(name, text) is not None


def mention(name: str, text: str, *, width: int = 200) -> str | None:
    """The passage around the first place `text` names `name`, about `width` characters each
    side, or None when it never does: what a find whose title does not name the watch is shown
    by."""
    span = _at(name, text)
    if span is None:
        return None
    start, end = max(0, span[0] - width), min(len(text), span[1] + width)
    # Whole words only: a cut word at either edge is dropped.
    if start and (cut := text.find(" ", start, span[0])) != -1:
        start = cut + 1
    if end < len(text) and (cut := text.rfind(" ", span[1], end)) != -1:
        end = cut
    passage = " ".join(text[start:end].split())
    return f"{'…' if start else ''}{passage}{'…' if end < len(text) else ''}"


def named(found: list, name: str) -> tuple[list, int]:
    """The candidates whose title or summary holds `name`: `(kept, dropped_count)`."""
    kept = [c for c in found if names(name, f"{c.title or ''}\n{c.summary or ''}")]
    return kept, len(found) - len(kept)
