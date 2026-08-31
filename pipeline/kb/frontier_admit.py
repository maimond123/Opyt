"""
pipeline/kb/frontier_admit.py — Frontier stage 3: ADMIT.

Drives the existing mint helpers (`ingest_papers.atomize_paper`,
`ingest_github.github_atom_from_url`, both called with `entry_mode="frontier"`) to materialize
staged candidates from stage 2 into `atoms`. This is the only transition in the rail that writes
`atoms`.

Stage 3 never judges quality — `rejected` means mechanical failure only, never "not good enough";
a boring artifact is materialized and ranked down by stage 4 instead. It always checks atom
presence before and after each ingest attempt (never trusts the helper's return value), because
the two mint helpers have inverted dedup contracts and a github content refresh can silently
rewrite an atom's entry_mode.
hazard writeup.

Never raises. Every outcome is a status, a reason slug, and a row.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess  # unused directly — tests patch `fa.subprocess.Popen` to intercept spawn_rail()'s Popen call
from datetime import datetime
from pipeline.timeparse import utc_iso, utc_now

from pipeline.kb.rail_runtime import COALESCE_DEFAULT, models_unroutable, spawn_rail
from pipeline.ingestion.utils import log

from . import schema
from .ingest_common import FETCH_UNDETERMINED

# The mode this rail writes. Never `user-saved` — that's approval, not discovery. (It was also
# never `crawled`, which the v1 artifacts sweep once owned; that sweep and that mode are both
# retired — see docs/plans/2026-08-25-rename-github-crawled-to-oracle-footprint.md.)
ENTRY_MODE = "frontier"

# Forces `rejected` after this many failed attempts, regardless of classification — guarantees a
# candidate can't retry forever.
ADMIT_MAX_ATTEMPTS = int(os.environ.get("OPYT_ADMIT_MAX_ATTEMPTS", 5))

# Caps RAM/spend per pass (materializing fetches + embeds full text). Oldest-first so a backlog
# drains steadily. Logged when it binds, since a silent truncation reads as "all admitted".
ADMIT_MAX_PER_RUN = int(os.environ.get("OPYT_ADMIT_MAX_PER_RUN", 25))


# ── Identity, computed OFFLINE ──────────────────────────────────────────────────
def target_atom_id(kind: str, url: str | None) -> tuple[str | None, bool]:
    """(atom_id, match_case_insensitively) for a candidate, WITHOUT touching the network.

    Dispatches on `Candidate.kind` — the ATOM KIND — and never on `source`, the finder. Those two
    coincided while arxiv and github were the only adapters, and the coincidence is not a fact
    about the rail: a third paper source would otherwise be a third arm here doing byte-identically
    what the first already does. An unknown or missing kind returns None, which `admit_one` turns
    into `rejected`/`no_atom_id` — the same fail-closed answer as an unparseable url.

    Staying offline keeps the presence pre-check free of network cost.
    """
    from .ingest_github import _github_owner_repo
    from .ingest_papers import paper_atom_id, paper_from_url

    k = (kind or "").strip().lower()
    if k == "paper":
        paper = paper_from_url(url or "", enrich=False)      # offline — no S2 call
        return (paper_atom_id(paper) if paper else None), False
    if k == "repo":
        owner_repo = _github_owner_repo(url or "")
        if owner_repo is None:
            return None, True
        # Case-insensitive: github_atom_from_url keys on the API's canonical owner.login, which
        # this offline parse can't know, so an exact-case check could miss an already-stored atom.
        return f"github:{owner_repo[0]}/{owner_repo[1]}", True
    return None, False


def _present(conn: sqlite3.Connection, atom_id: str, *, nocase: bool) -> str | None:
    """The STORED atom_id if this artifact is already an atom, else None.

    Returns the stored id (not a bool) so a caller can log which spelling matched. `nocase` costs
    a table scan instead of a primary-key lookup — acceptable at current table size.
    """
    sql = ("SELECT atom_id FROM atoms WHERE atom_id = ? COLLATE NOCASE" if nocase
           else "SELECT atom_id FROM atoms WHERE atom_id = ?")
    row = conn.execute(sql, (atom_id,)).fetchone()
    return row[0] if row else None


# ── The attempt ─────────────────────────────────────────────────────────────────
def _known_metadata(row) -> dict:
    """The metadata the FINDER already supplied, in the field names `paper_from_url(known=…)`
    wants. Read off the candidate row, so nothing is re-fetched that stage 2 already has.

    This is not an optimization. Semantic Scholar resolved 1 of 15 OpenAlex DOIs on 2026-08-26 —
    the rest 404 (Zenodo, institutional repositories) or 429 — and a 404 is `FETCH_ABSENT`, which
    `atomize_paper` does not skip. Without this, every one of those would freeze a title-less,
    abstract-less atom into the store permanently, since papers are immutable under Policy B.

    Source-agnostic by construction: every paper adapter already writes its title into `title`,
    its abstract into `summary`, its author names into `payload["authors"]` and an open PDF url,
    when it has one, into `payload["pdf_url"]` — so there is no per-source branch to write here
    and none to forget when the next paper source lands.
    """
    try:
        payload = json.loads(row["payload"] or "{}")
    except (TypeError, ValueError):
        payload = {}
    names = [n for n in (payload.get("authors") or []) if isinstance(n, str) and n.strip()]
    pdf_url = payload.get("pdf_url")
    return {"title": row["title"], "abstract": row["summary"],
            # No `authorId`: an OpenAlex author id is not a Semantic Scholar one, and
            # `derive_paper` reads that key to mint `who_id = scholar:{id}`. A name-only author
            # falls to the honest `paper-authors:{paper_id}` placeholder instead of asserting a
            # scholar identity that does not exist. S2's own authors still win when it answers.
            "authors": [{"name": n} for n in names],
            "publicationDate": row["published"],
            # The finder's open PDF, in S2's field name, so `_fulltext_pdf_urls` picks it up with
            # NO change of its own — it already reads `openAccessPdf.url`, and `_merge_paper`
            # already refuses to let an S2 null erase it. That is what turns an abstract-only
            # atom into a full-document one for the works S2 has never heard of: measured
            # 2026-08-26, 25 of 39 non-arXiv OpenAlex results carry a PDF that this reaches and
            # nothing else does.
            **({"openAccessPdf": {"url": pdf_url}} if isinstance(pdf_url, str) and pdf_url
               else {})}


def _ingest(conn, embedder, row) -> str | None:
    """Drive the right mint helper. Returns a reason-slug HINT, never a decision.

    The hint is advisory only — `admit_one` overrules it with its own presence check either way.
    Takes the whole candidate row rather than `(source, url)`: the minter needs the metadata stage
    2 already collected, and threading it field by field would grow a parameter per source.
    """
    from . import ingest_papers as ip
    from .ingest_github import github_atom_from_url

    kind = (row["kind"] or "").strip().lower()
    url = row["url"]
    if kind == "paper":
        paper = ip.paper_from_url(url, enrich=True, known=_known_metadata(row))
        if paper is None:
            return "no_atom_id"
        ip.atomize_paper(conn, embedder, paper, entry_mode=ENTRY_MODE)
        # Return value discarded on purpose (see module docstring); `_s2_verdict` distinguishes a
        # throttled S2 fetch (retryable) from a generic decline.
        if paper.get(ip._S2_VERDICT) == FETCH_UNDETERMINED:
            return "blocked_metadata"
        return None
    if kind == "repo":
        github_atom_from_url(conn, embedder, url, entry_mode=ENTRY_MODE)
        return None                                   # ditto — presence is the only question


def admit_one(conn, embedder_for, row, *, log_fn=log) -> tuple[str, str | None]:
    """One candidate → (status, error_slug). Never raises, never leaves partial state.

    `embedder_for` is a zero-arg callable rather than an embedder, so a run whose candidates are
    all already present never constructs one (which needs an API key and a live network path).
    """
    atom_id, nocase = target_atom_id(row["kind"], row["url"])
    if atom_id is None:
        return "rejected", "no_atom_id"

    # Presence check before the attempt: avoids re-ingesting (and rewriting entry_mode on) an
    # already-materialized atom, and skips the fetch/embed/spend entirely.
    stored = _present(conn, atom_id, nocase=nocase)
    if stored is not None:
        return "materialized", None

    hint: str | None = None
    try:
        hint = _ingest(conn, embedder_for(), row)
    except Exception as e:                            # fail-safe: SKIP, never a partial write
        hint = f"error_{type(e).__name__.lower()}"
        log_fn(f"[frontier-admit] {row['candidate_id']} raised: {type(e).__name__}: {e}")

    # Decide on atom presence, never on the helper's return value (see module docstring).
    if _present(conn, atom_id, nocase=nocase) is not None:
        return "materialized", None
    return "new", hint or "ingest_declined"


# ── The run ─────────────────────────────────────────────────────────────────────
def due_candidates(conn, *, limit: int) -> list[sqlite3.Row]:
    """The oldest-first slice of `new` candidates this pass will try.

    Selects the descriptive columns too, not just the identifying ones: `_known_metadata` mints
    from what stage 2 already collected rather than re-fetching it.
    """
    return list(conn.execute(
        "SELECT candidate_id, source, kind, url, title, summary, payload, published, attempts "
        "FROM frontier_candidates "
        "WHERE status = 'new' ORDER BY first_seen_at ASC, candidate_id ASC LIMIT ?", (limit,)))


def run_frontier_admit(conn=None, *, dry_run: bool = False, limit: int | None = None,
                       embedder=None, now: datetime | None = None) -> dict:
    """One admission pass. Never raises."""
    if (reason := models_unroutable("frontier-admit")) is not None:
        return {"status": "models_unroutable", "reason": reason}
    ref = now or utc_now()
    own = conn is None
    if own:
        conn = schema.connect()
    try:
        return _run(conn, dry_run=dry_run, limit=limit, embedder=embedder, ref=ref)
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log(f"[frontier-admit] run errored: {detail}")
        return {"status": "failed", "reason": detail}
    finally:
        if own:
            conn.close()


def _run(conn, *, dry_run: bool, limit: int | None, embedder, ref: datetime) -> dict:
    cap = ADMIT_MAX_PER_RUN if limit is None else limit
    backlog = conn.execute(
        "SELECT COUNT(*) FROM frontier_candidates WHERE status = 'new'").fetchone()[0]
    due = due_candidates(conn, limit=cap)
    if not due:
        return {"status": "ok", "considered": 0, "materialized": 0, "retried": 0, "rejected": 0}

    if backlog > len(due):
        # Log when the cap binds so a truncated run doesn't read as a complete one.
        log(f"[frontier-admit] admission cap bound — {len(due)} of {backlog} pending candidates "
            f"this pass (ADMIT_MAX_PER_RUN={cap})")

    held: list = [embedder]

    def embedder_for():
        if held[0] is None:
            from .embed import get_kb_embedder
            held[0] = get_kb_embedder()
        return held[0]

    counts = {"materialized": 0, "retried": 0, "rejected": 0}
    reasons: dict[str, int] = {}
    for row in due:
        if dry_run:
            continue
        status, slug = admit_one(conn, embedder_for, row)
        attempts = (row["attempts"] or 0)

        if status == "new":
            attempts += 1
            if attempts >= ADMIT_MAX_ATTEMPTS:
                status = "rejected"
                log(f"[frontier-admit] {row['candidate_id']} rejected at the attempt cap "
                    f"({attempts}/{ADMIT_MAX_ATTEMPTS}), last_error={slug}")
        elif status == "rejected":
            attempts += 1
        # A success never increments attempts — it counts failures only.

        conn.execute(
            "UPDATE frontier_candidates SET status=?, attempts=?, last_error=?, last_attempt_at=? "
            "WHERE candidate_id=?",
            (status, attempts, slug, utc_iso(ref), row["candidate_id"]))
        conn.commit()

        counts["materialized" if status == "materialized"
               else "rejected" if status == "rejected" else "retried"] += 1
        if slug:
            reasons[slug] = reasons.get(slug, 0) + 1

    out = {"status": "dry-run" if dry_run else "ok", "considered": len(due),
           "backlog": backlog, **counts}
    return {**out, "reasons": reasons} if reasons else out


def requeue_rejected(conn, *, last_error: str | None = None) -> int:
    """Reset `rejected` rows to `new` with a cleared attempt count. Returns the row count.

    Operator-only, never automatic — an automatic reset would just be a higher attempt cap in
    disguise. `last_error` narrows the reset to rows with that specific failure reason.
    """
    sql = "UPDATE frontier_candidates SET status='new', attempts=0, last_error=NULL " \
          "WHERE status='rejected'"
    params: tuple = ()
    if last_error is not None:
        sql += " AND last_error = ?"
        params = (last_error,)
    n = conn.execute(sql, params).rowcount
    conn.commit()
    return n


# ── The detached spawn ──────────────────────────────────────────────────────────
def spawn_frontier_admit(force: bool = False, coalesce_window: float = COALESCE_DEFAULT) -> bool:
    """Fire one admission pass as a detached, non-blocking child and return immediately.

    This rail owns its spawner rather than riding as a tail of stage 2, so the two can be disabled
    independently (they fail on different things: upstream index vs. fetch/embed).

    Cheap to fire often: a pass with nothing pending exits after one SELECT, and the per-run cap
    bounds the expensive case regardless of trigger rate.
    """
    return spawn_rail("pipeline.kb.frontier_admit", slug="frontier_admit",
                      force=force, coalesce=coalesce_window)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Frontier stage 3 — admit candidates into atoms")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--force", action="store_true", help="accepted for spawn parity; no TTL here")
    ap.add_argument("--dry-run", action="store_true", help="report what would be tried, write none")
    ap.add_argument("--limit", type=int, default=None, help="override ADMIT_MAX_PER_RUN")
    ap.add_argument("--requeue-rejected", nargs="?", const="", default=None, metavar="SLUG",
                    help="reset rejected rows to new (optionally only those with this last_error)")
    args = ap.parse_args(argv)

    if args.requeue_rejected is not None:
        conn = schema.connect()
        try:
            n = requeue_rejected(conn, last_error=args.requeue_rejected or None)
            print(f"requeued {n} rejected candidate(s)")
        finally:
            conn.close()
        return 0

    print(run_frontier_admit(dry_run=args.dry_run, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
