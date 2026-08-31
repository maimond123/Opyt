"""
pipeline/kb/sitting_render.py — render a sitting (or the unread mass) as the document a reader sees.

Reads a sitting that already exists (`sitting_store.get_sitting`) or the atoms no sitting has
covered (`sitting_store.unread_atom_ids`) and turns it into text, always chronological order.
Decides no membership and writes nothing back except the on-disk export in `write_artifacts`.

Depends on `sitting_store` and `sitting_vectors` — never eagerly on `sitting_builder` or
`sitting_zoom`, so this module imports standalone (a render-only CLI or test) without the build
loop or fracture logic. `_stored_projection` reaches for `sitting_builder.SeedError` lazily, inside
the function, because the arrow now points the other way: `sitting_builder` imports THIS module for
`projection`, which is the one place that decides what a long atom costs and what it shows. Billing
and rendering share that function precisely so they cannot drift.
"""
from __future__ import annotations

import json
from pathlib import Path

from opyt_core.paths import opyt_path

import numpy as np

from . import chunk as chunk_mod
from . import sitting_store as sst
from . import sitting_vectors as sv

# ── The tiered projection: what a long atom actually costs ──────────────────────
# RULED 2026-08-24. A paper atom is full-document text whenever a PDF mirror was reachable, so its
# size is an accident of mirror luck: ~8-10 full-text papers is the entire 120k budget and a
# paper-heavy stretch renders as mostly methodology sections. The sitting is a SURVEY read, so the
# fix is at render, never at membership — a paper that qualifies only through one buried section
# still qualifies, and storage is untouched. The projection is head + the sections that cleared the
# floor against this region's seed: "what it claims" plus "why it is here".
#
# 2k tokens ≈ a long-ish blog post. Source-agnostic on purpose: no host list, because the property
# that matters is length, and a paper with an abstract-only body is short and needs no projection.
LONG_ATOM_TOKENS = 2_000
CHARS_PER_TOKEN = 4          # the estimator this rail has always used; one spelling, one place


def _tokens(chars: int) -> int:
    return int(chars // CHARS_PER_TOKEN)


def _spans(conn, atom_ids) -> dict:
    """`{atom_id: [(seq, char_start, char_end), ...]}` in seq order — LENGTHS, never text.

    The billing half of this module runs over a whole region (thousands of atoms), so it may
    measure text but must never load it. `_atom_bodies` is the one place text enters RAM.
    """
    out: dict[str, list] = {a: [] for a in dict.fromkeys(atom_ids)}
    ids = list(out)
    for i in range(0, len(ids), sv.SQL_VARS):
        part = ids[i:i + sv.SQL_VARS]
        for r in conn.execute(
                f"SELECT atom_id, seq, char_start, char_end FROM chunks "
                f" WHERE atom_id IN ({sv._in_clause(len(part))}) ORDER BY atom_id, seq", part):
            out[r["atom_id"]].append((r["seq"], r["char_start"] or 0, r["char_end"] or 0))
    return out


def whole_tokens(spans: dict) -> dict:
    """`{atom_id: tokens the atom costs rendered in FULL}` over a `_spans` result.

    `MIN(char_start)..MAX(char_end)`, never a sum of chunk lengths, because adjacent chunks overlap
    by 200 chars and summing inflates every multi-chunk atom. An atom with no chunks costs 0.

    Seed-INDEPENDENT, which is what makes it worth having as a map: `projection` needs it to decide
    whether an atom is long enough to cut, and `sitting_zoom.sweep_k` needs it to bill every short
    atom once instead of once per candidate k.
    """
    out = {}
    for a, rows in spans.items():
        span = (max(ce for _s, _cs, ce in rows) - min(cs for _s, cs, _ce in rows)) if rows else 0
        out[a] = _tokens(span if span > 0 else sum(max(0, ce - cs) for _s, cs, ce in rows))
    return out


def projection(conn, atom_ids, *, vectors: dict, seed_vector, floor: float,
               spans: dict | None = None) -> dict:
    """`{atom_id: {"seqs": [...] | None, "tokens": int, "hidden": int, "hidden_tokens": int}}`.

    THE ONE PLACE that decides what a long atom costs and what it shows, so billing and rendering
    cannot drift. That is the whole point of the function existing: the budget bills RENDERED
    tokens, and a builder that measured stored size while the renderer projected would cut parts at
    the wrong boundary — permanently, since a closed part is read, claims-extracted and lens-cached
    and never repartitioned.

    `seqs is None` means "render whole", which is every atom under `LONG_ATOM_TOKENS` and every
    atom whose chunks are missing or unembedded (fail-safe: an atom we cannot score renders in
    full rather than silently losing its middle).

    Selection for a long atom: seq 0 always — the head is the author's own summary, and an
    abstract IS a claims table — plus the floor-clearing sections in descending cosine until the
    projection itself reaches `LONG_ATOM_TOKENS`. One threshold does both jobs, so a tiered atom
    can never cost more than the bar that made it tiered.

    The token estimate for a projection sums chunk lengths rather than taking a span, so it
    double-counts the ~200-char embedding overlap between any two ADJACENT kept sections. Left
    uncorrected: it over-estimates by at most a few hundred characters on an atom already being
    cut, and over-estimating is the safe direction for a budget.

    `spans` lets a caller that already fetched them hand them in — the chunk-span query is
    seed-INDEPENDENT, so a sweep scoring one atom set against many centroids would otherwise
    re-run it once per centroid. It must cover every id in `atom_ids`; a caller that supplies a
    short dict is a bug and gets a `KeyError` rather than a silently under-billed region.
    """
    seed = np.asarray(seed_vector, dtype=np.float32).reshape(-1)
    seed = seed / (np.linalg.norm(seed) + 1e-9)
    spans = (_spans(conn, atom_ids) if spans is None
             else {a: spans[a] for a in dict.fromkeys(atom_ids)})
    whole_map = whole_tokens(spans)
    out: dict[str, dict] = {}
    for a, rows in spans.items():
        whole = whole_map[a]
        V = vectors.get(a)
        if whole <= LONG_ATOM_TOKENS or V is None or len(V) != len(rows):
            out[a] = {"seqs": None, "tokens": whole, "hidden": 0, "hidden_tokens": 0}
            continue
        cos = V @ seed
        lens = [max(0, ce - cs) for _seq, cs, ce in rows]
        keep = {0}
        used = _tokens(lens[0])
        for i in sorted(range(1, len(rows)), key=lambda j: -float(cos[j])):
            if float(cos[i]) < floor:
                break                                  # sorted descending: nothing after clears it
            if used + _tokens(lens[i]) > LONG_ATOM_TOKENS:
                break
            keep.add(i)
            used += _tokens(lens[i])
        hidden = [i for i in range(len(rows)) if i not in keep]
        out[a] = {"seqs": sorted(rows[i][0] for i in keep), "tokens": used,
                  "hidden": len(hidden), "hidden_tokens": _tokens(sum(lens[i] for i in hidden))}
    return out


def _atom_block(atom_id: str, meta: dict, text: dict) -> str:
    """One atom as the document spells it — `### {date} — {who}  ({atom_id})`, then the body.

    A PARSE CONTRACT, not a layout choice: both reader prompts (`sitting_reader._SYSTEM` and
    `sitting_claims._SYSTEM`) tell the model this is the shape it will see and cite atoms by the
    id in the parentheses. Spelled once so a document and a digest can never disagree about it.
    """
    who, date = meta.get(atom_id, ("?", "?"))
    return f"### {date or '?'} — {who or '?'}  ({atom_id})\n\n{text.get(atom_id, '')}\n"


# ── Render ──────────────────────────────────────────────────────────────────────
def render_sitting(conn, sitting_id: str) -> str:
    """The sitting as one markdown document, chronological.

    Time order is load-bearing: it is what surfaces a position reversal across a region that
    score-order would hide. Admission order is in the manifest for auditing, not for reading.
    The header carries an author-concentration line so a reader can steer away from
    self-referential queries in single-author-heavy sittings.
    """
    s = sst.get_sitting(conn, sitting_id)
    if s is None:
        raise KeyError(f"no sitting {sitting_id!r}")
    ids = [a["atom_id"] for a in s["admissions"]]
    meta, text = _atom_bodies(conn, ids, proj=_stored_projection(conn, s, ids))
    order = _chronological(ids, meta)
    who = [meta.get(a, ("?", ""))[0] for a in ids]
    top_share, top_who = _concentration(who)

    head = [
        f"# Sitting: {s['seed_ref']} ({s['seed_kind']} seed)",
        "",
        f"built {s['built_at']} · sitting_id `{sitting_id}`",
        f"seeds: {', '.join(s['seed_atom_ids']) or '(centroid)'}",
        f"dials: floor={s['floor']} ceiling={s['ceiling']} "
        f"budget={s['budget_tokens']:,} (corpus noise ceiling {s['calibrated_floor']})",
        f"still admissible at floor: {s['region_atoms']} atoms / ~{s['region_tokens']:,} tok",
        f"this sitting: {s['atoms']} atoms / ~{s['tokens']:,} tok · stop={s['stop']} · "
        f"near-duplicates skipped={s['skipped_dupes']}",
        f"author concentration: {top_who} wrote {top_share:.0%} of these atoms",
    ]
    if s.get("continues"):
        # Named explicitly: a fresh agent hasn't seen earlier parts and would otherwise
        # mistake a mid-conversation slice for the whole region. "Continuation" stopped being a
        # mechanism on 2026-08-24 and became a description — every read after the first is the
        # same operation, "read the unread part holding the notebook" — so this says PART, and
        # points at the claims table that carries what those earlier parts established.
        lo, hi = part_span(conn, sitting_id)
        head.append(
            f"Part {_part_index(conn, sitting_id)} of this region, covering {lo}–{hi}, "
            f"following `{s['continues']}` · {s['prior_atoms']} atoms were read in earlier parts "
            f"and are NOT below — they appear as claims above, not as text")
    head += ["", "## Context (chronological)", ""]
    body = [_atom_block(a, meta, text) for a in order]
    return "\n".join(head + body)


# The sprouts digest is a lens, not a rail: it emits no standing queries and writes nothing, so
# it needs no place in the seed/build/read loop, only a renderer.
SPROUTS_DIGEST_MAX_CHARS = 400_000  # Unbounded reading window, not a metered call — sized to
# stop one digest from making a tool response unusably large; not calibrated to real growth yet.


def render_sprouts_digest(conn) -> dict:
    """`{document, atoms, truncated}` — every human-attested atom no READ sitting has covered, one
    document, chronological.

    Uses `sitting_store.unread_atom_ids` unfiltered and unranked — no ranking has been measured to
    order this material correctly, so the lens gets the whole unread set and is expected to throw
    out grab-bags itself. Not a sitting: no seed, no floor, no `region_key`, and the material spans
    every orphaned conversation, not one. Writes and stores nothing — no `sitting_id` exists to key
    a receipt on.
    """
    ids = sorted(sst.unread_atom_ids(conn))
    if not ids:
        return {"document": "", "atoms": 0, "truncated": False}
    meta, text = _atom_bodies(conn, ids)
    order = _chronological(ids, meta)
    head = [
        "# Sprouts digest — everything no sitting has read",
        "",
        f"{len(ids)} atoms, none covered by a read sitting: true orphans (matched no seed's "
        "floor), built-but-unread regions, and fracture leftovers too small to stand alone.",
        "This is NOT one conversation. It is grab-bag material by construction — throw out "
        "whatever does not cohere rather than forcing an arc across all of it.",
        "", "## Context (chronological)", "",
    ]
    body, used, truncated = [], sum(len(h) for h in head), False
    for a in order:
        chunk = _atom_block(a, meta, text)
        if used + len(chunk) > SPROUTS_DIGEST_MAX_CHARS:
            truncated = True
            break
        body.append(chunk)
        used += len(chunk)
    if truncated:
        body.append(f"\n[TRUNCATED — {len(ids) - len(body)} more unread atoms not shown]")
    return {"document": "\n".join(head + body), "atoms": len(ids), "truncated": truncated}


def _stored_projection(conn, s: dict, ids: list) -> dict | None:
    """`projection()` re-derived from what the sitting STORED — its own anchor and its own floor.

    Deterministic, so it reproduces exactly the selection the build billed: the anchor is a stored
    blob (never re-resolved, which is non-deterministic on the hosted embedder) and the floor is a
    stored dial. Nothing is persisted per atom because nothing needs to be.

    Fail-safe: a sitting whose anchor cannot be rebuilt renders in FULL. Showing more than was
    billed is the harmless direction — it costs the reader nothing to see extra, while silently
    dropping a section nobody chose to drop is the failure this whole path exists to avoid.
    """
    from . import sitting_builder as sb           # lazy: sitting_builder imports this module
    try:
        anchor = sst.ensure_seed_vector(conn, s["sitting_id"])
    except (KeyError, sb.SeedError):
        return None
    return projection(conn, ids, vectors=sv._atom_chunk_vectors(conn, ids),
                      seed_vector=anchor, floor=s["floor"])


def _chronological(ids, meta) -> list:
    """The reading order, and the one place it is spelled.

    Undated atoms sort last, not first (an empty date string would otherwise sort before every
    real one and open the document). Shared with the positional-coverage metric so both callers
    use the exact same sequence — a second, subtly different sort would mis-attribute citations.
    """
    return sorted(ids, key=lambda a: (not (meta.get(a, ("?", ""))[1] or ""),
                                      meta.get(a, ("?", ""))[1] or "", a))


def chronological_ids(conn, atom_ids) -> list:
    """`atom_ids` in reading order — the public door onto `_chronological` for callers that hold a
    plain id list rather than a sitting. Reads who/when only; no atom text enters RAM.

    `sitting_builder`'s admission loop cuts a PART with this. Amendment 3 permits exactly that and
    no more: time may SELECT and ORDER a sitting, and the budget cut may slice by time — what stays
    forbidden is a date BOUNDING membership (which pool an atom is in) and material being dropped
    by AGE. The pool is built before this is called and time never touches it.
    """
    meta, _ = _atom_bodies(conn, atom_ids, with_text=False)
    return _chronological(atom_ids, meta)


def chronological_order(conn, sitting_id: str) -> dict:
    """`{order, undated}` — the exact sequence `render_sitting` lays this sitting out in.

    `undated` names the trailing no-date block separately so a coverage histogram can exclude or
    bucket it instead of reading it as recency bias. Reads no atom text — a metric input, not a render.
    """
    s = sst.get_sitting(conn, sitting_id)
    if s is None:
        raise KeyError(f"no sitting {sitting_id!r}")
    ids = [a["atom_id"] for a in s["admissions"]]
    meta, _ = _atom_bodies(conn, ids, with_text=False)
    return {"order": _chronological(ids, meta),
            "undated": [a for a in ids if not (meta.get(a, ("?", ""))[1] or "")]}


def part_span(conn, sitting_id: str) -> tuple[str, str]:
    """`(first date, last date)` of the stretch this part covers — `('?', '?')` when nothing is
    dated. Display only: the part header's "covering A–B".

    SEED ATOMS ARE EXCLUDED. They are re-admitted into every part so a fresh agent reading part N
    standalone can see what the region is anchored to — they are not part of the stretch it covers.
    Counted, a seed dated after the whole part stretches its range to the seed's own date, and two
    consecutive parts then report overlapping ranges for a cut that is exactly contiguous.
    (Measured 2026-08-25 on the live store: part 1 read 2025-10-27..2026-03-18 and part 2
    2026-03-19..2026-06-25, while both headers claimed to end on the seed's 2026-06-25.)
    """
    ids = [a["atom_id"] for a in sst.get_sitting(conn, sitting_id)["admissions"]
           if not a["is_seed"]]
    meta, _ = _atom_bodies(conn, ids, with_text=False)
    dates = sorted(d for a in ids if (d := meta.get(a, ("?", ""))[1]))
    return (dates[0], dates[-1]) if dates else ("?", "?")


def _atom_bodies(conn, atom_ids, *, with_text: bool = True, proj: dict | None = None
                 ) -> tuple[dict, dict]:
    """`({atom_id: (who, date)}, {atom_id: rendered text})` — the only place text enters RAM,
    and only for the atoms of one sitting.

    `with_text=False` returns metadata only, no text: `sitting_surface`'s scope report needs
    who/when for a region but not the bodies, and stitching that text would defeat the RAM bound.

    `proj` is `projection()`'s answer. Where it names a seq list, only those sections are stitched
    and a pointer line replaces the rest — the SAME selection the budget was billed against, which
    is why it is passed in rather than recomputed here. Absent (or naming no cut), every atom
    renders whole, which is what the sprouts digest and any pre-projection caller get.
    """
    ids, meta, text = list(atom_ids), {}, {}
    for i in range(0, len(ids), sv.SQL_VARS):
        part = ids[i:i + sv.SQL_VARS]
        for r in conn.execute(
                f"SELECT atom_id, who_id, when_ts FROM atoms "
                f"WHERE atom_id IN ({sv._in_clause(len(part))})", part):
            meta[r["atom_id"]] = (r["who_id"] or "?", (r["when_ts"] or "")[:10])
    if not with_text:
        return meta, text
    for a in ids:
        rows = conn.execute(
            "SELECT seq, text, char_start, char_end FROM chunks WHERE atom_id = ? ORDER BY seq",
            (a,)).fetchall()
        pa = (proj or {}).get(a) or {}
        seqs = pa.get("seqs")
        if seqs is None:
            text[a] = chunk_mod.stitch(rows)
        else:
            keep = set(seqs)
            text[a] = (chunk_mod.stitch([r for r in rows if r["seq"] in keep])
                       + f"\n*[{pa['hidden']} further section(s), ~{pa['hidden_tokens']:,} tokens, "
                         f"not shown — open this atom to read the full text.]*")
    return meta, text


def _part_index(conn, sitting_id: str) -> int:
    """1 for a fresh sitting, N for the Nth part of a chain. Display only."""
    return len(sst.ancestors(conn, sitting_id)) + 1


def _concentration(authors: list) -> tuple[float, str]:
    if not authors:
        return 0.0, "?"
    counts: dict = {}
    for w in authors:
        counts[w] = counts.get(w, 0) + 1
    top = max(counts, key=lambda k: counts[k])
    return counts[top] / len(authors), top


def manifest(conn, sitting_id: str) -> dict:
    s = sst.get_sitting(conn, sitting_id)
    if s is None:
        raise KeyError(f"no sitting {sitting_id!r}")
    return s


def write_artifacts(conn, sitting_id: str, out_dir: Path | str | None = None) -> dict:
    """Render the sitting and its manifest to disk; return the two paths.

    Defaults under `$OPYT_HOME`, never a repo path. These files are an export — the DB is the
    record, so a deleted artifact loses no state.
    """
    s = sst.get_sitting(conn, sitting_id)
    if s is None:
        raise KeyError(f"no sitting {sitting_id!r}")
    out = Path(out_dir) if out_dir else Path(opyt_path("sittings"))
    out.mkdir(parents=True, exist_ok=True)
    slug = _slug(s["seed_ref"]) + "-" + sitting_id[:8]
    md, mf = out / f"{slug}.md", out / f"{slug}.manifest.json"
    md.write_text(render_sitting(conn, sitting_id))
    # seed_vector is dropped, not serialized: default=str would abbreviate the 4096-dim array via
    # numpy's repr, silently exporting numbers that look like the vector but aren't.
    mf.write_text(json.dumps({k: v for k, v in s.items() if k != "seed_vector"},
                             indent=1, default=str))
    return {"markdown": str(md), "manifest": str(mf)}


def _slug(ref: str | None) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in (ref or "sitting").lower()]
    return ("".join(keep).strip("-") or "sitting")[:60]
