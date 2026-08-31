"""
pipeline/kb/rechunk.py — re-clean + re-chunk EXISTING atoms from their raw snapshots.

Why it exists: the content gate runs at ingest, but `raw_hash` is computed from the FULL page,
so improving the gate later does NOT re-trigger ingest (unchanged raw → dedup skip). To apply a
new/changed gate to already-stored atoms we read each atom's preserved raw snapshot (`kb_raw/…`,
the immutable safety net), re-run the gate, and replace its chunk index. The chunks are a derived,
rebuildable VIEW over raw, so this is reversible.

Safety — dry-run by default; it reports what WOULD change and touches nothing:
  --apply            re-chunk the KEEPERS (embed the cleaned text → replace_chunks). PAID (embeds).
  --delete-rejected  additionally DELETE atoms the gate now whole-page-rejects (raw is preserved).
A DEGRADED verdict (gate model unavailable → keep-all) is SKIPPED, never mutated — we don't
re-embed a full page because the gate was down, and never delete on a degraded verdict.

Scope: blog + substack (the gated sources). GitHub/papers/X are not gated (SCOPE decision).
"""

from __future__ import annotations

import sqlite3

from pipeline.ingestion.utils import log

from . import content_gate, schema
from .embed import assert_strip_version
from .ingest_common import embed_chunks
from .raw_store import read_snapshot

_GATED_SOURCES = ("blog", "substack")


def _atoms(conn: sqlite3.Connection, sources: tuple[str, ...]) -> list[tuple[str, str, str]]:
    q = ("SELECT atom_id, raw_ref, source_type FROM atoms WHERE source_type IN (%s) "
         "ORDER BY atom_id" % ",".join("?" for _ in sources))
    return [(r[0], r[1], r[2]) for r in conn.execute(q, sources).fetchall()]


def _delete_atom(conn: sqlite3.Connection, atom_id: str) -> None:
    """Remove a now-rejected atom: clear its chunks + FTS (via replace_chunks), then the atom row.
    Raw snapshot stays on disk, so a later gate change can re-ingest it."""
    schema.replace_chunks(conn, atom_id, [])               # deletes chunks + chunks_fts rows
    conn.execute("DELETE FROM atoms WHERE atom_id = ?", (atom_id,))
    conn.commit()


def rechunk_from_raw(conn, embedder, *, sources: tuple[str, ...] = _GATED_SOURCES,
                     apply: bool = False, delete_rejected: bool = False, limit: int = 0) -> dict:
    """Re-run the gate over stored atoms' raw snapshots. Returns a summary of counts. With
    `apply=False` (default) NOTHING is written — it only measures the change the gate would make."""
    if apply:
        # This function writes vectors without going through `AtomSink`/`_write_atom`, so it needs
        # its own strip-version guard to avoid mixing vectors from two different strip surfaces.
        # Checked before the loop and only under `apply`, so a dry run still measures freely.
        assert_strip_version(conn)
    atoms = _atoms(conn, sources)
    if limit:
        atoms = atoms[:limit]
    s = {"scanned": 0, "no_raw": 0, "degraded": 0, "unchanged": 0,
         "rechunked": 0, "rejected": 0, "errors": 0, "units_dropped": 0}
    for atom_id, raw_ref, source_type in atoms:
        s["scanned"] += 1
        md = read_snapshot(raw_ref)
        if md is None:                                     # snapshot gone → can't re-derive
            s["no_raw"] += 1
            continue
        try:
            v = content_gate.classify_page(md)
        except Exception as e:                             # fail-safe: one bad atom never sinks the run
            log(f"[rechunk] gate error on {atom_id} (skip): {type(e).__name__}: {e}")
            s["errors"] += 1
            continue
        if v.degraded:                                     # gate was down → do NOT mutate
            s["degraded"] += 1
            continue
        if v.kept_text is None:                            # whole-page reject (wrong source)
            s["rejected"] += 1
            if apply and delete_rejected:
                _delete_atom(conn, atom_id)
        elif v.n_dropped == 0:                             # gate keeps everything → no change
            s["unchanged"] += 1
        else:
            s["rechunked"] += 1
            s["units_dropped"] += v.n_dropped
            if apply:
                schema.replace_chunks(
                    conn, atom_id, embed_chunks(embedder, v.kept_text, source_type))
    return s


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Re-clean + re-chunk existing atoms from raw snapshots.")
    ap.add_argument("--apply", action="store_true",
                    help="Write changes (re-chunk keepers). Without this it's a DRY-RUN.")
    ap.add_argument("--delete-rejected", action="store_true",
                    help="With --apply, also DELETE atoms the gate now whole-page-rejects.")
    ap.add_argument("--limit", type=int, default=0, help="Only the first N atoms (0 = all).")
    ap.add_argument("--sources", default=",".join(_GATED_SOURCES),
                    help="Comma-separated source types to rechunk (default: blog,substack).")
    args = ap.parse_args(argv)
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())

    conn = schema.connect()
    try:
        embedder = None
        if args.apply:
            from .embed import get_kb_embedder
            embedder = get_kb_embedder()
            print(f"[rechunk] APPLY mode — embedder model={embedder.model}"
                  f"{'  + delete-rejected' if args.delete_rejected else ''}")
        else:
            print("[rechunk] DRY-RUN — no changes will be written.")
        s = rechunk_from_raw(conn, embedder, sources=sources, apply=args.apply,
                             delete_rejected=args.delete_rejected, limit=args.limit)
    finally:
        conn.close()

    print(f"\n  scanned={s['scanned']}  no_raw={s['no_raw']}  degraded={s['degraded']}")
    print(f"  unchanged={s['unchanged']}  rechunked={s['rechunked']} "
          f"(units_dropped={s['units_dropped']})  rejected={s['rejected']}  errors={s['errors']}")
    if not args.apply:
        print("\n  (dry-run: rerun with --apply to re-chunk keepers, "
              "add --delete-rejected to remove the rejects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
