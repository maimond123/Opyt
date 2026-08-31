"""
pipeline/kb/chunk.py — split a raw snapshot into overlapping windows for embedding.

Lifted from `gui/indexer/build_db.py::split_passages`, but BODY-ONLY: the note indexer
glued title+summary onto passage 0 (a retrieval head for its notes). Here the snapshot
IS the routing text, so we window it uniformly — no synthetic head. Each window keeps
its `(char_start, char_end)` into the snapshot, so the semantic arm's argmax chunk can
point the host at the exact span it matched (`chunk_span`), not just the atom.

Why windows at all, now that Qwen3's 32k context means chunk size is no longer
model-capped? SPAN-LOCALIZED routing: max-pooling per-chunk cosine lets one strong
passage float a long document, and the matched chunk names WHERE the signal is. So
chunk size is now a free tuning knob (`CHUNK_CHARS`), not a model ceiling — kept at
1600 for granular routing; enlarge if a coarser card ever wins on eval.
"""

from __future__ import annotations

CHUNK_CHARS = 1600     # window width; a free knob under Qwen3's 32k context
CHUNK_OVERLAP = 200    # chars carried back between adjacent windows (context bleed)
# Hard cap so a pathological giant snapshot can't explode the store; the overflow guard
# below LOGS a true giant rather than silently dropping its tail.
MAX_CHUNKS = 250


def strip_frontmatter(md: str) -> tuple[str, int]:
    """A snapshot's YAML frontmatter (`---\\n…\\n---`) is provenance, NOT content — the
    plan's "body-only" rule. Return `(body, offset)` where `offset` is the char length of
    the stripped prefix, so a caller can keep chunk spans SNAPSHOT-absolute (add it back)
    while embedding/indexing only the body. No frontmatter → `(md, 0)` unchanged.
    """
    if not md.startswith("---"):
        return md, 0
    # Find the closing fence that ends the leading block (the second `---` on its own line).
    end = md.find("\n---", 3)
    if end == -1:
        return md, 0
    nl = md.find("\n", end + 1)     # first newline AFTER the closing fence
    if nl == -1:
        return md, 0
    offset = nl + 1
    while offset < len(md) and md[offset] == "\n":   # skip the blank line(s) before the body
        offset += 1
    return md[offset:], offset


def split_text(text: str) -> list[tuple[int, str, int, int]]:
    """`text` → `[(seq, chunk_text, char_start, char_end), ...]`, ≥1 chunk always.

    A short text collapses to a single chunk spanning the whole thing. `char_start`/
    `char_end` index into `text`. Windows overlap by `CHUNK_OVERLAP` so a sentence
    straddling a boundary still lands whole in at least one chunk.

    Invariant: a chunk shorter than `CHUNK_OVERLAP` is ALWAYS its atom's only chunk (every
    other window is exactly `CHUNK_CHARS`, and the tail window spans more than `CHUNK_OVERLAP`
    whenever it exists). Callers like `sitting_vectors._atom_chunk_vectors` rely on this to tell
    "short atom" from "scrap of a long atom" by length alone — lowering `CHUNK_OVERLAP` breaks it.
    """
    text = text or ""
    if len(text) <= CHUNK_CHARS:
        # Short-circuit: one chunk covering everything (empty text → one empty chunk,
        # so every atom has ≥1 embeddable row — the "every atom ≥1 vector" sanity check).
        return [(0, text, 0, len(text))]

    chunks: list[tuple[int, str, int, int]] = []
    pos, seq = 0, 0
    while pos < len(text) and len(chunks) < MAX_CHUNKS:
        end = min(pos + CHUNK_CHARS, len(text))
        chunks.append((seq, text[pos:end], pos, end))
        if end >= len(text):
            break
        pos = end - CHUNK_OVERLAP   # step forward, carrying overlap back
        seq += 1
    else:
        # Loop hit MAX_CHUNKS with text still uncovered → the tail is DROPPED. Never silent:
        # a document this long is either genuinely huge or a scrape gone wrong — say so.
        if pos < len(text):
            try:
                from pipeline.ingestion.utils import log
                log(f"[chunk] snapshot exceeds MAX_CHUNKS={MAX_CHUNKS} "
                    f"({len(text)} chars) — tail from char {pos} DROPPED (not embedded).")
            except Exception:
                pass
    return chunks


def stitch(rows) -> str:
    """Reassemble an atom's text from its stored chunks, REMOVING the embedding overlap.

    The inverse of `split_text`. Adjacent windows share `CHUNK_OVERLAP` characters so a sentence
    is never cut in half for the embedder; a naive join would re-emit that overlap at every
    boundary. `char_start`/`char_end` index the original snapshot, so `prev_end - char_start` is
    exactly the shared span; a NULL or non-contiguous range degrades to a plain join rather than
    guessing, since a wrong trim would delete real text.
    """
    out: list[str] = []
    prev_end: int | None = None
    for r in rows:
        text = r["text"] or ""
        cs, ce = r["char_start"], r["char_end"]
        if prev_end is not None and cs is not None and prev_end > cs:
            skip = prev_end - cs
            text = text[skip:] if skip < len(text) else ""
        out.append(text)
        if ce is not None:
            prev_end = ce
    return "".join(out)
