"""
pipeline/kb/sitting_store.py — persist a built sitting, its read stamps, and the coverage ledger.

Owns the SQL for a sitting's lifecycle after `build_sitting` decides membership: the INSERT, the
read stamps (`mark_read`/`mark_lens_read` — a build covers nothing, only a read does), and the
coverage ledger. `ensure_seed_vector` imports `sitting_builder.SeedError` lazily, inside the
function, to avoid a cycle with `sitting_builder`'s eager import of this module.
"""
from __future__ import annotations

import json
from datetime import datetime
from pipeline.timeparse import utc_iso, utc_now

import numpy as np

from . import schema
from . import sitting_vectors as sv

# ── Store ───────────────────────────────────────────────────────────────────────
def _encode_vector(conn, v) -> bytes | None:
    """Encode a seed centroid to a blob at the same width `chunks.vector` uses (from `kb_meta`,
    never assumed) so it stays comparable to chunk vectors on every membership run."""
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    return (a / (np.linalg.norm(a) + 1e-9)).astype(sv._dtype(conn)).tobytes()


def record_sitting(conn, rec: dict) -> str:
    """Write the sitting and its membership. Idempotent on `sitting_id`.

    The read stamp is read before the write and carried across explicitly, because `INSERT OR
    REPLACE` deletes the conflicting row first and would otherwise lose it. `region_key` is derived
    here (not taken from `rec`) so there is exactly one place that decides what a region is;
    `seed_vector` defaults to NULL so a hand-assembled record is writable without one.

    `skipped` stores the ceiling-skip list the builder produced, not just its count: the ruling
    that KEPT the near-duplicate skip rests on those skips being auditable atom by atom.
    """
    prior = conn.execute("SELECT read_at, read_status FROM sittings WHERE sitting_id = ?",
                         (rec["sitting_id"],)).fetchone()
    conn.execute(
        "INSERT OR REPLACE INTO sittings (sitting_id, built_at, seed_kind, seed_ref, "
        " seed_atom_ids, floor, calibrated_floor, ceiling, budget_tokens, region_atoms, "
        " region_tokens, atoms, tokens, stop, skipped_dupes, skipped, continues, prior_atoms, "
        " parent_sitting_id, region_key, seed_vector, read_at, read_status) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (rec["sitting_id"], rec["built_at"], rec["seed_kind"], rec["seed_ref"],
         json.dumps(rec["seed_atom_ids"]), rec["floor"], rec["calibrated_floor"],
         rec["ceiling"], rec["budget_tokens"], rec["region_atoms"], rec["region_tokens"],
         rec["atoms"], rec["tokens"], rec["stop"], rec["skipped_dupes"],
         json.dumps(rec.get("skipped", [])),
         rec.get("continues"), rec.get("prior_atoms", 0), rec.get("parent_sitting_id"),
         schema.region_key(rec["seed_kind"], rec["seed_ref"], rec["floor"],
                           rec["ceiling"], rec["budget_tokens"]),
         _encode_vector(conn, rec.get("seed_vector")),
         prior["read_at"] if prior else None, prior["read_status"] if prior else None))
    conn.execute("DELETE FROM sitting_atoms WHERE sitting_id = ?", (rec["sitting_id"],))
    conn.executemany(
        "INSERT INTO sitting_atoms (sitting_id, atom_id, rank, is_seed, rel, red, tokens) "
        "VALUES (?,?,?,?,?,?,?)",
        [(rec["sitting_id"], a["atom_id"], a["rank"], a["is_seed"], a["rel"], a["red"],
          a["tokens"]) for a in rec["admissions"]])
    conn.commit()
    return rec["sitting_id"]


def ancestors(conn, sitting_id: str) -> list[str]:
    """This part's ancestors along `continues`, ROOT FIRST — so index 0 is part 1.

    Excludes `sitting_id` itself: part N is the text about to be read, not part of its own memory.
    THE one place the notebook chain is walked — the claims carry, the lens map loop and the
    render's part number all read it here, so "which parts came before this one" has a single
    answer. It lives in this module because it is the bottom of the sitting import graph: reader,
    render and claims all import it, and render cannot import reader.

    Deliberately NOT merged with `sitting_builder.chain_atom_ids`, which walks the same links: that
    one RAISES on a missing part (a build over a chain it cannot see would silently under-read),
    this one degrades to what was reachable (a display number and a notebook are worth having
    partial). Same links, opposite failure contracts.

    Bounded by a `seen` set for the same reason `chain_atom_ids` is — only a corrupted store can
    loop, and hanging on it is worse than returning what was reachable.
    """
    out, cur, seen = [], sitting_id, set()
    while cur and cur not in seen:
        seen.add(cur)
        row = conn.execute("SELECT continues FROM sittings WHERE sitting_id = ?", (cur,)).fetchone()
        if row is None:
            break
        cur = row["continues"]
        if cur:
            out.append(cur)
    out.reverse()
    return out


def mark_read(conn, sitting_id: str, *, status: str = "ok", at: datetime | None = None) -> None:
    """Stamp a sitting as read — the only thing that moves an atom out of never-read mass.
    Called by the reader, never by the builder: a built-but-unread region stays unread."""
    conn.execute("UPDATE sittings SET read_at = ?, read_status = ? WHERE sitting_id = ?",
                 (utc_iso(at or utc_now()), status, sitting_id))
    conn.commit()


def mark_lens_read(conn, sitting_id: str, lens: str, *, status: str = "ok",
                   at: datetime | None = None) -> None:
    """Stamp `sitting_id` as read under `lens` — the same re-read guard `mark_read` gives `queries`,
    given independently to every other API lens (`claims` today).

    Writes to `sitting_reads(sitting_id, lens, ...)`, a separate child table from
    `sittings.read_at`, so one lens's read state never blocks or satisfies another's.
    `sitting_scheduler` stays scoped to `sittings.read_at` alone.
    """
    ts = utc_iso(at or utc_now())
    conn.execute(
        "INSERT INTO sitting_reads (sitting_id, lens, read_at, read_status) VALUES (?,?,?,?) "
        "ON CONFLICT(sitting_id, lens) DO UPDATE SET read_at=excluded.read_at, "
        "read_status=excluded.read_status",
        (sitting_id, lens, ts, status))
    conn.commit()


def lens_read_state(conn, sitting_id: str, lens: str) -> dict | None:
    """`{read_at, read_status}` for `sitting_id` under `lens`, or `None` if that lens has never read
    it. The idempotency check every non-`queries` API lens's reader guards its own re-read with."""
    row = conn.execute(
        "SELECT read_at, read_status FROM sitting_reads WHERE sitting_id = ? AND lens = ?",
        (sitting_id, lens)).fetchone()
    return dict(row) if row else None


def get_lens_output(conn, sitting_id: str, lens: str) -> dict | None:
    """One part's cached map output under `lens`, or None. See the `sitting_lens_outputs` DDL for
    why this cache needs no invalidation rule."""
    row = conn.execute(
        "SELECT output, model, in_tokens, out_tokens, cost_usd, created_at "
        "  FROM sitting_lens_outputs WHERE sitting_id = ? AND lens = ?",
        (sitting_id, lens)).fetchone()
    return dict(row) if row else None


def record_lens_output(conn, sitting_id: str, lens: str, output: str, *, usage: dict | None = None,
                       at: datetime | None = None) -> None:
    """Cache one part's map output. Upsert rather than insert-only: a `force` re-map of the open
    tail replaces its row instead of raising, and a closed part is never re-mapped at all."""
    u = usage or {}
    conn.execute(
        "INSERT INTO sitting_lens_outputs "
        "  (sitting_id, lens, output, model, in_tokens, out_tokens, cost_usd, created_at) "
        "VALUES (?,?,?,?,?,?,?,?) "
        "ON CONFLICT(sitting_id, lens) DO UPDATE SET output=excluded.output, "
        "  model=excluded.model, in_tokens=excluded.in_tokens, out_tokens=excluded.out_tokens, "
        "  cost_usd=excluded.cost_usd, created_at=excluded.created_at",
        (sitting_id, lens, output, u.get("model"), u.get("in_tokens"), u.get("out_tokens"),
         u.get("cost_usd"), utc_iso(at or utc_now())))
    conn.commit()


def record_claims(conn, sitting_id: str, claims: list[dict], *, at: datetime | None = None) -> int:
    """Write claims from one `claims`-lens read. Returns how many were written.
    Each claim carries `falsified_by` (the observation that would disprove it) for a human reader
    to judge; nothing automatically re-checks a stored claim."""
    ts = utc_iso(at or utc_now())
    conn.executemany(
        "INSERT INTO sitting_claims (sitting_id, claim, falsified_by, atom_ids, created_at) "
        "VALUES (?,?,?,?,?)",
        [(sitting_id, c["claim"], c["falsified_by"], json.dumps(c["atom_ids"]), ts)
         for c in claims])
    conn.commit()
    return len(claims)


def get_claims(conn, sitting_id: str) -> list[dict]:
    """Every stored claim for one sitting, oldest first, with `atom_ids` parsed back to a list."""
    return [dict(r, atom_ids=json.loads(r["atom_ids"])) for r in conn.execute(
        "SELECT claim_id, claim, falsified_by, atom_ids, created_at FROM sitting_claims "
        " WHERE sitting_id = ? ORDER BY claim_id", (sitting_id,))]


def get_sitting(conn, sitting_id: str) -> dict | None:
    """The stored row, with `seed_atom_ids` parsed and `seed_vector` decoded to a unit array.

    `seed_vector` is `None` when the column is NULL. A caller needing the anchor must handle that
    case rather than substitute a zero vector, since every cosine against zero is 0.0. Use
    `ensure_seed_vector` to repair a NULL column.
    """
    row = conn.execute("SELECT * FROM sittings WHERE sitting_id = ?", (sitting_id,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    out["seed_atom_ids"] = json.loads(out.get("seed_atom_ids") or "[]")
    blob = out.get("seed_vector")
    out["seed_vector"] = sv._decode([blob], sv._dtype(conn))[0] if blob else None
    out["admissions"] = [dict(r) for r in conn.execute(
        "SELECT atom_id, rank, is_seed, rel, red, tokens FROM sitting_atoms "
        "WHERE sitting_id = ? ORDER BY rank", (sitting_id,))]
    return out


def ensure_seed_vector(conn, sitting_id: str) -> np.ndarray:
    """The stored anchor for `sitting_id`, rebuilding and persisting it once if the column is NULL.

    Repairs a row written before `seed_vector` existed, by rebuilding the centroid from
    `seed_atom_ids` (the exact atoms the original build used). Raises for a `vector`-kind seed,
    whose `seed_atom_ids` is empty by construction and whose centroid was never stored — there is
    nothing to rebuild from, so the caller must build a fresh region instead.
    """
    # LAZY: sitting_builder imports sitting_store eagerly (for record_sitting), so a
    # module-level import here the other way would cycle. Resolved at call time only.
    from . import sitting_builder as sb
    s = get_sitting(conn, sitting_id)
    if s is None:
        raise KeyError(f"no sitting {sitting_id!r}")
    if s["seed_vector"] is not None:
        return s["seed_vector"]
    if not s["seed_atom_ids"]:
        raise sb.SeedError(
            f"sitting {sitting_id!r} has no stored seed vector and no seed atoms to rebuild one "
            f"from (seed_kind={s['seed_kind']!r}) — build a fresh region instead")
    vecs = sv._atom_chunk_vectors(conn, s["seed_atom_ids"])
    stack = [m for a in s["seed_atom_ids"] if (m := vecs.get(a)) is not None and len(m)]
    if not stack:
        raise sb.SeedError(f"sitting {sitting_id!r}: its seed atoms have no embedded chunks")
    c = np.vstack(stack).mean(axis=0)
    c = c / (np.linalg.norm(c) + 1e-9)
    conn.execute("UPDATE sittings SET seed_vector = ? WHERE sitting_id = ?",
                 (_encode_vector(conn, c), sitting_id))
    conn.commit()
    return c


# A sitting counts as read if any API lens read it: `queries` via `sittings.read_at`, or another
# lens via a `sitting_reads` row. UNION (not UNION ALL) dedupes a sitting read by both.
_READ_SITTING_IDS = (
    "SELECT sitting_id FROM sittings WHERE read_at IS NOT NULL "
    "UNION SELECT sitting_id FROM sitting_reads WHERE read_status = 'ok'")


# ── The coverage ledger ─────────────────────────────────────────────────────────
def coverage(conn) -> dict:
    """`{total, read, never_read, pct_read, built_unread, sittings_read}`.

    Measures how much of the human-attested corpus has been covered by a read sitting. Nothing here
    pays down the unread debt automatically — coverage only advances when something is read.

    STAYS NARROW after the union widened membership (RULED 2026-08-24). This is a read-DEBT ledger
    over David's own material: "how much of what I saved have I actually read". Frontier's finds
    are unbounded and arrive nightly, so counting them makes the denominator grow faster than any
    read can move it, `never_read` becomes a number nobody can act on, and the sprouts digest —
    which is fed from this — floods with machine pulls. The atoms still enter regions and still get
    read; they just aren't debt.
    """
    total = conn.execute(
        f"SELECT COUNT(*) FROM atoms a WHERE {sv._human_clause()}", sv.HUMAN_ATTESTED).fetchone()[0]
    read = conn.execute(
        f"SELECT COUNT(DISTINCT sa.atom_id) FROM sitting_atoms sa "
        f"  JOIN atoms a ON a.atom_id = sa.atom_id "
        f" WHERE sa.sitting_id IN ({_READ_SITTING_IDS}) AND {sv._human_clause()}",
        sv.HUMAN_ATTESTED).fetchone()[0]
    built_unread = conn.execute(
        f"SELECT COUNT(DISTINCT sa.atom_id) FROM sitting_atoms sa "
        f"  JOIN atoms a ON a.atom_id = sa.atom_id "
        f" WHERE sa.sitting_id NOT IN ({_READ_SITTING_IDS}) AND {sv._human_clause()} "
        f"   AND sa.atom_id NOT IN (SELECT sa2.atom_id FROM sitting_atoms sa2 "
        f"        WHERE sa2.sitting_id IN ({_READ_SITTING_IDS}))", sv.HUMAN_ATTESTED).fetchone()[0]
    n_read = conn.execute(f"SELECT COUNT(*) FROM ({_READ_SITTING_IDS})").fetchone()[0]
    return {"total": total, "read": read, "never_read": total - read,
            "pct_read": round(100.0 * read / total, 1) if total else 0.0,
            "built_unread": built_unread, "sittings_read": n_read}


def unread_atom_ids(conn) -> set:
    """Human-attested atoms no READ sitting has covered, under any API lens.

    Narrow for the same reason `coverage` is — this is the same ledger, itemized."""
    rows = conn.execute(
        f"SELECT a.atom_id FROM atoms a WHERE {sv._human_clause()} AND a.atom_id NOT IN ("
        f"  SELECT sa.atom_id FROM sitting_atoms sa "
        f"   WHERE sa.sitting_id IN ({_READ_SITTING_IDS}))",
        sv.HUMAN_ATTESTED).fetchall()
    return {r["atom_id"] for r in rows}


# `coverage` and `unread_atom_ids` measure unread debt only; nothing here proposes a seed to pay
# it down. Before ranking unread mass by density, read
# docs/Old-Investigations/2026-08-13-density-is-not-importance-in-the-unread-ranking.md.
