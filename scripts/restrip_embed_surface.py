#!/usr/bin/env python3
"""
scripts/restrip_embed_surface.py — recompute `chunks.embed_text` and RE-EMBED every chunk whose
stored vector was built from a different surface than this build produces.

Why it exists: `embed_surface.STRIP_VERSION` is an input that produced every vector in the store,
exactly as the model is. Change a pattern and every existing vector is stale — built from text this
build would no longer feed the embedder. `embed.ensure_kb_meta` REFUSES to write a second
generation into the same store (that would leave a corpus whose vectors are comparable to each
other only by accident); this script is the escape hatch that makes the refusal actionable.

SAFETY — DRY-RUN by default; it measures and touches nothing:
  --apply     recompute + re-embed the stale rows, then stamp kb_meta. PAID (embeds).
  --limit N   only the first N stale chunks (a smoke test; --apply with --limit never stamps).

Resumable and idempotent, by construction. A row stops being stale the moment it is written, and
`kb_meta.strip_version` is stamped ONLY after a full pass with nothing left over. So an interrupted
run leaves the store honestly labelled as the OLD generation, and a rerun picks up exactly the
remainder rather than re-paying for what already landed. `char_start`/`char_end` and `chunks_fts`
are never touched — this rewrites what the embedder SAW, never what a reader is shown or what a
`chunk_span` points at.

Sandbox first. `$OPYT_HOME` selects the store, so a measurement run never has to go near `~/.opyt`:

    OPYT_HOME=/path/to/a/scratch/opyt-home \\
    PYTHONPATH=$(pwd) python3 scripts/restrip_embed_surface.py --apply

Point it at a COPY, never at a store you still want. The example here used to name a specific
`opyt-baselines/` directory; that whole fleet of parallel stores was retired on 2026-08-12 when
the merged KB was installed at `~/.opyt`, so the path no longer exists and naming any single
store invites re-creating the exact ambiguity the merge removed.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.kb import probe_store, schema                     # noqa: E402
from pipeline.kb.embed_surface import (  # noqa: E402
    DEFAULT_PROFILE,
    _PROFILES,
    _strip_body,
    strip_for_embedding,
    strip_version,
)

# Rows written per embed call + commit. Bounds two things at once: peak RAM (a 4096-dim float32
# vector is 16KB, so 256 rows in flight is ~4MB, not the ~350MB an all-at-once pass would hold) and
# the spend at risk — a crash loses at most this many chunks' worth of embedding, because each
# group commits before the next is requested. The embedder re-splits this into its own 64-wide
# HTTP slices internally, so this number is about durability, not about call count.
GROUP = 256

# ~4 chars/token, $0.01 per 1M input tokens (the Qwen3 embedding rate). Only ever printed.
_COST_PER_CHAR = 0.01 / 1_000_000 / 4


def _rows(conn):
    """Every chunk with its atom's source_type, in a stable order."""
    return conn.execute(
        "SELECT c.chunk_id, c.atom_id, c.text, c.embed_text, a.source_type "
        "FROM chunks c JOIN atoms a USING(atom_id) ORDER BY c.chunk_id"
    ).fetchall()


def survey(conn, profile: str = "full") -> dict:
    """What WOULD change, per source type. Pure measurement — no writes, no spend, no embedder.

    A row is stale when the strip this build computes differs from what is stored. NULL `embed_text`
    is stale by definition (it predates the column), which is what makes the very first run a full
    corpus pass and every later run a delta.
    """
    per = collections.defaultdict(lambda: {"chunks": 0, "stale": 0, "orig": 0, "kept": 0})
    stale_ids: list[int] = []
    fallbacks = 0
    for r in _rows(conn):
        st = r["source_type"] or ""
        want = strip_for_embedding(r["text"] or "", st, profile)
        p = per[st]
        p["chunks"] += 1
        p["orig"] += len(r["text"] or "")
        p["kept"] += len(want)
        if not _strip_body(r["text"] or "", st, profile):
            fallbacks += 1                      # all-scaffolding chunk → kept whole (fail-safe)
        if r["embed_text"] != want:
            p["stale"] += 1
            stale_ids.append(r["chunk_id"])
    return {"per": dict(per), "stale_ids": stale_ids, "fallbacks": fallbacks}


def _stamp(conn, version: str) -> None:
    """Record the strip that produced the store's vectors. Written DIRECTLY rather than through
    `ensure_kb_meta`, which exists to REFUSE exactly this transition — the whole job of this script
    is to be the one place allowed to make it, after the vectors actually agree."""
    if conn.execute("SELECT COUNT(*) FROM kb_meta WHERE id = 1").fetchone()[0]:
        conn.execute("UPDATE kb_meta SET strip_version = ? WHERE id = 1", (version,))
        conn.commit()


def restrip(conn, embedder, stale_ids: list[int], *, apply: bool = False,
            profile: str = "full") -> dict:
    """Re-embed each stale chunk from its stripped surface. Returns counts.

    Fail-safe per group (CLAUDE.md): a group whose embed call fails is SKIPPED whole — no write, no
    partial vectors, and its rows stay stale so the next run retries them. A partially-written group
    is the one outcome worse than doing nothing, because those rows would look current while
    holding vectors from the old surface."""
    from pipeline.kb.embed import EmbedError

    s = {"embedded": 0, "failed": 0, "groups": 0}
    if not apply:
        return s
    by_id = {r["chunk_id"]: r for r in _rows(conn)}
    for i in range(0, len(stale_ids), GROUP):
        group = stale_ids[i:i + GROUP]
        rows = [by_id[cid] for cid in group]
        texts = [strip_for_embedding(r["text"] or "", r["source_type"] or "", profile)
                 for r in rows]
        try:
            vecs = embedder.embed(texts, role="document")
        except EmbedError as e:
            print(f"  [group {s['groups']}] embed FAILED — {len(group)} chunks left stale: {e}")
            s["failed"] += len(group)
            s["groups"] += 1
            continue
        import numpy as np
        from pipeline.kb.embed import CHUNK_STORAGE_DTYPE
        dt = np.dtype(CHUNK_STORAGE_DTYPE)
        for r, t, v in zip(rows, texts, vecs):
            conn.execute(
                "UPDATE chunks SET embed_text = ?, vector = ? WHERE chunk_id = ?",
                (t, np.asarray(v, dtype=dt).tobytes(), r["chunk_id"]),
            )
        conn.commit()                    # per group: bounds what an interrupt can cost
        s["embedded"] += len(group)
        s["groups"] += 1
        print(f"  [group {s['groups']}] {s['embedded']}/{len(stale_ids)} chunks re-embedded")
    return s


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Recompute chunks.embed_text and re-embed stale vectors.")
    ap.add_argument("--apply", action="store_true",
                    help="Write changes (recompute + re-embed + stamp). Without this it's a DRY-RUN.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only the first N stale chunks (0 = all). Never stamps kb_meta.")
    # Defaults to the profile the INGEST path writes with, not to "full". A store is only ever
    # built by that path, so "full" as a default meant the no-flag run measured the corpus against
    # a surface no store has — and, with --apply, re-embedded every chunk onto it and stamped an
    # identity `assert_strip_version` then refuses. The guard's own error message points here, so
    # the default this lands on is the one an operator following it gets.
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(_PROFILES),
                    help=f"Which rule set runs (default: {DEFAULT_PROFILE}, what ingest writes). "
                         f"'full' = every rule, including urls the AUTHOR wrote — moves the "
                         f"geometry and restates sitting_builder's floors. 'scaffolding' = "
                         f"machine-emitted render template only.")
    args = ap.parse_args(argv)

    home = os.environ.get("OPYT_HOME", "~/.opyt (default)")
    conn = schema.connect()
    try:
        # Through `read_kb_meta`, not a raw SELECT: kb_meta's DDL and its additive `strip_version`
        # column live in embed.py, and a store predating either has neither until that runs.
        from pipeline.kb.embed import read_kb_meta
        meta = read_kb_meta(conn)
        stored = (meta or {}).get("strip_version") or "(unstripped)"
        print(f"[restrip] store={home}")
        target = strip_version(args.profile)
        print(f"[restrip] stored strip_version={stored}  this build={target} (profile={args.profile})")

        r = survey(conn, args.profile)
        per, stale_ids = r["per"], r["stale_ids"]
        tot_o = sum(p["orig"] for p in per.values())
        tot_k = sum(p["kept"] for p in per.values())
        print(f"\n{'source':10} {'chunks':>7} {'stale':>7} {'orig chars':>12} "
              f"{'kept chars':>12} {'removed':>9}")
        for st, p in sorted(per.items(), key=lambda x: -x[1]["orig"]):
            pct = 100 * (p["orig"] - p["kept"]) / max(p["orig"], 1)
            print(f"{st:10} {p['chunks']:7d} {p['stale']:7d} {p['orig']:12,} "
                  f"{p['kept']:12,} {pct:8.1f}%")
        pct = 100 * (tot_o - tot_k) / max(tot_o, 1)
        print(f"{'TOTAL':10} {sum(p['chunks'] for p in per.values()):7d} {len(stale_ids):7d} "
              f"{tot_o:12,} {tot_k:12,} {pct:8.1f}%")
        # The fail-safe counter, printed every run: a chunk that stripped to nothing keeps its
        # original text rather than embedding an empty string. A number that climbs means a pattern
        # got greedy.
        print(f"\n  all-scaffolding chunks kept whole (fail-safe): {r['fallbacks']}")

        if args.limit:
            stale_ids = stale_ids[:args.limit]

        # The second population. `survey`/`_rows` walk `chunks JOIN atoms`, so they cannot see the
        # untrusted probe store — and `kb_meta.strip_version` is a claim about every vector in the
        # file, not about the ones this script happens to enumerate. Surveyed here (free: no embed
        # under `apply=False`) so the DRY-RUN reports it too; a hidden population is exactly what
        # makes a maintenance run look complete when it isn't.
        p = probe_store.restrip_probe_rows(conn, None, profile=args.profile, apply=False)
        if p["stale"]:
            print(f"  probe chunks stale (untrusted store): {p['stale']:,}")

        if not stale_ids and not p["stale"]:
            print("\n  nothing stale — every vector already matches this build's surface.")
            if args.apply:
                _stamp(conn, target)
            return 0

        stale_chars = sum(
            len(strip_for_embedding(r["text"] or "", r["source_type"] or "", args.profile))
            for r in _rows(conn) if r["chunk_id"] in set(stale_ids))
        print(f"  stale chunks to re-embed: {len(stale_ids):,}  "
              f"({stale_chars:,} chars ≈ ${stale_chars * _COST_PER_CHAR:.3f})")

        if not args.apply:
            print("\n  (dry-run: rerun with --apply to recompute + re-embed + stamp kb_meta)")
            return 0

        from pipeline.kb.embed import get_kb_embedder
        embedder = get_kb_embedder()
        print(f"\n[restrip] APPLY — embedder model={embedder.model} provider={embedder.provider}")
        s = restrip(conn, embedder, stale_ids, apply=True, profile=args.profile)
        print(f"\n  embedded={s['embedded']:,}  failed={s['failed']:,}  groups={s['groups']}")
        # Probe rows go through `probe_store`, never inline SQL here: the `.guards.py` trust
        # boundary says those table names belong to that module, and allowlisting this script would
        # widen the boundary to buy convenience. A function call costs nothing.
        p = probe_store.restrip_probe_rows(conn, embedder, profile=args.profile, apply=True)
        if p["stale"]:
            print(f"  probe: embedded={p['embedded']:,}  failed={p['failed']:,}  "
                  f"groups={p['groups']}")

        # ONE gate, over BOTH passes. Stamping is the script's assertion that every vector in the
        # file now agrees with `target`; gating it on the trusted pass alone would relocate the lie
        # one table over instead of removing it — and the probe pass is the half with no guard
        # downstream to catch it.
        if args.limit:
            print("  --limit set → kb_meta NOT stamped (the store is only partly converted).")
        elif s["failed"] or p["failed"]:
            which = " + ".join(n for n, f in (("trusted", s["failed"]), ("probe", p["failed"])) if f)
            print(f"  some groups failed ({which}) → kb_meta NOT stamped. "
                  f"Rerun to pick up the remainder.")
        else:
            _stamp(conn, target)
            print(f"  kb_meta.strip_version stamped {target!r}.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
