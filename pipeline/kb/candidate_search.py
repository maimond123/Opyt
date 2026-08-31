"""pipeline/kb/candidate_search.py — answer "which candidate writes about X" from BOTH stores.

This module owns one question — *which of the people I have curated is worth promoting to an
Oracle, and on what evidence* — and it is the only place that composes the two stores which can
answer it:

  • `probe_search`  → `probe_chunks`, a candidate's own timeline that the probe rail sampled.
                      UNVETTED. Nobody vouched for it. ~25 posts per probed account.
  • `retrieve`      → `atoms`/`chunks`, the TRUSTED corpus. A candidate appears here when the
                      user bookmarked one of their posts. Usually exactly one atom.

WHY THIS MODULE EXISTS AT ALL, rather than a second arm inside `probe_search`: that module's
header carries a human rule — *never join to `atoms` here* — because it is the read side of a
trust boundary and stays single-store on purpose. Composing two stores is a different job, so it
gets a different module. `probe_search` keeps its rule; this file is the one that knows both.

THE ARMS ARE NEVER SUMMED, and that is the load-bearing decision. `probe_search` scores an account
by an UNCAPPED RRF sum over its matching passages. A probed account has ~25 posts; a saved account
usually has 1. Add those numbers together and the probed accounts win mechanically, regardless of
relevance — which would bury the ~506 people this module exists to make visible. So each arm ranks
inside itself, and the merge INTERLEAVES by within-arm rank. No cross-arm arithmetic anywhere.

EVERY ROW NAMES ITS BASIS, and that is not cosmetic either. A single bookmarked post is selection
on the dependent variable: it matched the query partly BECAUSE the user kept it for that reason.
That is bad evidence for "characterize this account" and good evidence for "is this person worth
reading more of" — which is the decision this payload actually serves. `basis` + the passage count
makes the difference visible instead of leaving a reader to assume a sample.

Design record: docs/plans/2026-08-23-candidate-search-atom-arm.md
"""

from __future__ import annotations

import sqlite3

from . import probe_search, probe_store, schema

# Passages shown per account, per arm. Same value as `probe_search.DEFAULT_EVIDENCE` by intent, not
# by import: raising one arm's evidence budget should not silently raise the other's.
DEFAULT_EVIDENCE = 3

# Candidates returned by default. Matches `oracle(action='screen')`'s top_n=30 neighbourhood, so
# the two halves of the promotion decision are the same size list.
DEFAULT_K = 15

# Fraction of candidates that must have NO local material before the payload adds a coverage note.
# See `_coverage_note` for why the denominator changed when the atom arm landed.
COVERAGE_NOTE_FRACTION = 0.10

# What each basis means, stated IN the payload rather than only in this docstring: the reader is a
# model that will otherwise treat any retrieved passage as corpus knowledge.
_PROVENANCE = {
    "probed": ("UNVETTED — sampled from this candidate's own timeline for a promotion decision. "
               "NOT part of the knowledge base; must not be cited as such."),
    "saved":  ("TRUSTED — this is a knowledge-base atom the user saved. Citable, but open it "
               "first: nothing asserts what a source says without opening its raw."),
}


def _atom_evidence(hit, *, limit: int = 240) -> dict:
    """One `AtomHit` → an evidence item shaped like `probe_search`'s, so a reader does not have to
    learn two record layouts for the same idea."""
    text = " ".join(((hit.snippet or hit.description) or "").split())
    if len(text) > limit:
        text = text[:text.rfind(" ", 0, limit) or limit].rstrip() + "…"
    return {"when": hit.when_ts, "snippet": text, "score": round(hit.score, 3),
            "url": hit.source_url, "atom_id": hit.atom_id,
            "arm": "semantic" if hit.sem_rank is not None else "lexical",
            "provenance": _PROVENANCE["saved"]}


def search_saved_atoms(conn: sqlite3.Connection, query: str, embedder, *,
                       who_ids: set[str], k: int = DEFAULT_K,
                       evidence: int = DEFAULT_EVIDENCE) -> list[dict]:
    """Candidates whose SAVED atoms best match `query`, each with its strongest passages.

    The trusted-corpus mirror of `probe_search.search_candidates`: same hybrid shape (BM25 +
    vector), same per-ACCOUNT fusion, same RRF constant — so the two arms produce comparable
    WITHIN-arm rankings even though their scores are never added.

    An empty query returns accounts by recency of saved content, matching the probe arm's
    "show me who's here" opening call.
    """
    from . import retrieve

    if not who_ids:
        return []
    # `who_id=` is an explicit author set: an empty list matches nothing, never everything.
    pool = retrieve.candidate_atom_ids(conn, None, None, None, who_id=sorted(who_ids))
    if not pool:
        return []

    if not (query or "").strip():
        rows = conn.execute(
            "SELECT who_id, MAX(when_ts) AS last_ts, COUNT(*) AS n FROM atoms "
            "WHERE atom_id IN (%s) GROUP BY who_id" % ",".join("?" * len(pool)),
            sorted(pool)).fetchall()
        ranked = sorted(rows, key=lambda r: (r["last_ts"] or ""), reverse=True)[:k]
        return [{"who_id": r["who_id"], "atoms": r["n"], "last_ts": r["last_ts"],
                 "match_score": 0.0, "evidence": []} for r in ranked]

    # Widened `k` on each arm: these are ATOM ranks, and one account can own several of the top
    # atoms, so an account-level top-k needs more atom-level room than k to fill from.
    span = max(k * evidence * 2, 40)
    scored: dict[str, float] = {}
    passages: dict[str, list[dict]] = {}
    for arm in (retrieve.atom_bm25_search(conn, query, pool, span),
                retrieve.atom_semantic_search(conn, query, embedder, pool, span)):
        for rank, hit in enumerate(arm):
            w = hit.who_id
            if not w:
                continue
            # RRF over the ACCOUNT, accumulating across its atoms rather than pinning it to its
            # single best one — the same rule the probe arm uses, so the two ranks mean the same.
            scored[w] = scored.get(w, 0.0) + 1.0 / (60 + rank)
            passages.setdefault(w, []).append(_atom_evidence(hit))

    counts = dict(conn.execute(
        "SELECT who_id, COUNT(*) FROM atoms WHERE atom_id IN (%s) GROUP BY who_id"
        % ",".join("?" * len(pool)), sorted(pool)).fetchall())
    out = []
    for w in sorted(scored, key=lambda x: scored[x], reverse=True)[:k]:
        seen, ev = set(), []
        for p in passages.get(w, []):
            if p["atom_id"] in seen:          # the two arms routinely surface the same atom
                continue
            seen.add(p["atom_id"])
            ev.append(p)
            if len(ev) >= evidence:
                break
        out.append({"who_id": w, "atoms": counts.get(w, 0),
                    "match_score": round(scored[w], 5), "evidence": ev})
    return out


def _interleave(probed: list[dict], saved: list[dict], k: int) -> list[dict]:
    """Merge two WITHIN-arm rankings by position, not by score.

    Rank 1 of each arm, then rank 2 of each, and so on. An account present in both arms is kept
    on the PROBED side only — a sampled timeline is strictly deeper evidence than the one post
    that was saved, so showing it twice would be the same person competing with themselves.

    The probe arm goes first at equal rank, for the same reason.
    """
    seen = {r["who_id"] for r in probed}
    saved = [r for r in saved if r["who_id"] not in seen]
    out = []
    for i in range(max(len(probed), len(saved))):
        for row, basis in ((probed[i] if i < len(probed) else None, "probed"),
                           (saved[i] if i < len(saved) else None, "saved")):
            if row is not None:
                out.append({**row, "basis": basis})
            if len(out) >= k:
                return out
    return out


def _coverage_note(no_material: int, total: int) -> str | None:
    """The residue line. Its DENOMINATOR changed when the atom arm landed and the change matters:
    it used to count "not probed" (762 people), but a candidate whose saved post is searchable is
    no longer invisible. What is left is candidates with no local material of ANY kind — measured
    at 49 on the live store — and those must stay VISIBLE as a count rather than silently absent,
    because a candidate missing for lack of material looks exactly like one who did not match.
    """
    if not no_material or no_material < COVERAGE_NOTE_FRACTION * max(total, 1):
        return None
    return (f"PARTIAL COVERAGE — {no_material} of {total} candidates have no local material yet "
            f"(neither a sampled timeline nor a post you saved), so they cannot match any query. "
            f"A thin or empty result here means thin coverage, NOT that nobody matches. The "
            f"background probe rail fills them in over the following days; nothing needs to be "
            f"run by hand.")


def candidates_payload(conn: sqlite3.Connection, query: str, embedder, *,
                       k: int = DEFAULT_K, evidence: int = DEFAULT_EVIDENCE,
                       min_signals: int = 1, ttl_days: float | None = None) -> dict:
    """The `oracle(action='candidates')` payload: candidates ranked by what they write.

    Joins the two halves of a promotion decision: how hard the user vouched for a person (curation
    signals from `screen`) and what that person actually writes — from EITHER store that has them,
    with the source of every row named in `basis`.

    Counts are reported rather than hidden: a candidate missing because nobody pulled them yet
    looks identical, in a ranked list, to one whose content simply didn't match.
    """
    from . import candidate_probe, screen

    # Identity for every candidate, whether or not probed, sourced from the SCREEN ranking the
    # user has already seen.
    ident: dict[str, dict] = {}
    # Retired count is surfaced, never silently dropped, so a list that shrank is distinguishable
    # from one that never had those candidates.
    ranked = screen.rank_candidates(conn, include_retired=True)
    retired = sum(1 for c in ranked if c.retired)
    for cand in ranked:
        if cand.retired:
            continue
        uid = candidate_probe._x_user_id(cand.members)
        if not uid:
            continue
        ident[f"x:user:{uid}"] = {
            "canonical_id": cand.canonical_id, "name": cand.name, "handle": cand.handle,
            "distinct_signals": cand.distinct_signals,
            "is_oracle": schema.is_oracle(conn, cand.canonical_id)}

    eligible = {w for w, i in ident.items()
                if i["distinct_signals"] >= min_signals and not i["is_oracle"]}

    # Both arms are asked for a FULL k. The interleave decides the final cut, so neither arm may
    # pre-truncate itself on a guess about how many slots it will win.
    probed_hits = probe_search.search_candidates(conn, query, embedder, k=k, evidence=evidence,
                                                 who_ids=eligible)
    saved_hits = search_saved_atoms(conn, query, embedder, who_ids=eligible, k=k,
                                    evidence=evidence)
    for h in probed_hits:
        for p in h.get("evidence", []):
            p["provenance"] = _PROVENANCE["probed"]

    states = probe_store.pull_states(conn)
    rows = []
    for h in _interleave(probed_hits, saved_hits, k):
        w = h["who_id"]
        st = states.get(w, {})
        rows.append({**ident.get(w, {}), **h,
                     "probed_at": st.get("pulled_at"), "pull_status": st.get("status")})

    probed = set(probe_store.probed_who_ids(conn))
    fresh = probe_store.fresh_who_ids(
        conn, ttl_days=ttl_days if ttl_days is not None else candidate_probe.DEFAULT_TTL_DAYS)
    with_atoms = {r[0] for r in conn.execute("SELECT DISTINCT who_id FROM atoms")}
    no_material = eligible - probed - with_atoms
    note = _coverage_note(len(no_material), len(eligible))
    return {
        "query": query,
        "candidates": rows,
        "probed": len(eligible & probed),
        "searchable_by_saved_post": len((eligible & with_atoms) - probed),
        "no_local_material": len(no_material),
        "stale": len(eligible & probed - fresh),
        "basis_key": {"probed": "timeline we sampled (unvetted, ~25 posts)",
                      "saved": "a post YOU saved (trusted, usually 1 — a positive example, "
                               "not a sample of their output)"},
        **({"coverage_note": note} if note else {}),
        # Same rule as coverage_note: reported only when non-zero, so a shrunk list stays visible.
        **({"retired": retired,
            "retired_note": (f"{retired} candidate(s) hidden — you no longer follow them "
                             f"(their signals and saved posts are kept; add_oracle still "
                             f"takes them by name).")} if retired else {}),
    }
