"""pipeline/oracles/candidates.py — answer "which candidate writes about X" from three arms.

Composes the unvetted probe timelines (`probe_search`), the trusted saved atoms (`retrieve`) and
the people the Oracles' own posts engage with (`mentions`); `probe_search` itself stays
single-store. The arms are never summed: each ranks inside itself and the merge interleaves by
within-arm rank, since a probed account's ~25 posts would otherwise beat a saved account's one.
Every row names its `basis`.
"""

from __future__ import annotations

import sqlite3

from pipeline.curation import clock
from pipeline.curation import collectors
from pipeline.curation import papers
from pipeline.curation import reconcile
from pipeline.oracles import mentions, probe_search, probe_store, ranking
from pipeline.retrieval import engine
from pipeline.retrieval import filters
from pipeline.retrieval.rank import rrf_term
from pipeline.worker import liveness
from pipeline.oracles.sources.timeline import samples


# Passages shown per account, per arm. Same value as `probe_search.DEFAULT_EVIDENCE` by intent, not
# by import: raising one arm's evidence budget should not silently raise the other's.
DEFAULT_EVIDENCE = 3

# Candidates returned by default. Matches `oracle(action='screen')`'s top_n=30 neighbourhood, so
# the two halves of the promotion decision are the same size list.
DEFAULT_K = 15

# Fraction of candidates with no local material at which the payload adds a coverage note.
COVERAGE_NOTE_FRACTION = 0.10

# What each basis means, stated in the payload rather than only in this docstring: the reader is a
# model that will otherwise treat any retrieved passage as corpus knowledge.
# The identities a candidate's probe sample is kept under: X timelines and Substack titles.
_PROBE_PREFIXES = ("x:user:", "substack:")

_PROVENANCE = {
    "probed": "Sampled from this candidate's own posts; not part of the knowledge base.",
    "saved":  ("TRUSTED — this is a knowledge-base atom the user saved. Citable, but open it "
               "first: nothing asserts what a source says without opening its raw."),
    "mentioned": mentions.PROVENANCE,
}


# A passage's length in a candidate's evidence, the same budget `probe_search` trims to.
_EVIDENCE_CHARS = probe_search.EVIDENCE_CHARS


def _atom_evidence(hit, *, limit: int = _EVIDENCE_CHARS) -> dict:
    """One `AtomHit` → an evidence item shaped like `probe_search`'s, so a reader does not have to
    learn two record layouts for the same idea."""
    text = probe_search.snippet(hit.snippet or hit.description, limit)
    return {"when": hit.when_ts, "snippet": text, "score": round(hit.score, 3),
            "url": hit.source_url, "atom_id": hit.atom_id,
            "provenance": _PROVENANCE["saved"]}


def search_saved_atoms(conn: sqlite3.Connection, query: str, *,
                       who_ids: set[str], k: int = DEFAULT_K,
                       evidence: int = DEFAULT_EVIDENCE,
                       person_by_who: dict[str, str] | None = None) -> list[dict]:
    """Candidates whose saved atoms best match `query`, each with its strongest passages.

    The trusted-corpus mirror of `probe_search.search_candidates`: same keyword arm, same
    per-account sum, same RRF constant, so the two arms produce comparable within-arm rankings
    even though their scores are never added.

    `person_by_who` collapses already-resolved platform identities into one person before ranking.
    An empty query returns those people by recency of saved content, matching the probe arm's
    "show me who's here" opening call.
    """
    if not who_ids:
        return []
    person_by_who = person_by_who or {}

    def person(who_id: str) -> str:
        return person_by_who.get(who_id, who_id)

    # `who_id=` is an explicit author set: an empty list matches nothing, never everything.
    pool = filters.candidate_atom_ids(conn, None, None, None, who_id=sorted(who_ids))
    if not pool:
        return []

    if not (query or "").strip():
        rows = conn.execute(
            "SELECT who_id, MAX(when_ts) AS last_ts, COUNT(*) AS n FROM atoms "
            "WHERE atom_id IN (%s) GROUP BY who_id" % ",".join("?" * len(pool)),
            sorted(pool)).fetchall()
        grouped: dict[str, dict] = {}
        for r in rows:
            entry = grouped.setdefault(person(r["who_id"]),
                                       {"atoms": 0, "last_ts": None})
            entry["atoms"] += r["n"]
            if (r["last_ts"] or "") > (entry["last_ts"] or ""):
                entry["last_ts"] = r["last_ts"]
        ranked = sorted(grouped.items(), key=lambda item: item[1]["last_ts"] or "",
                        reverse=True)[:k]
        return [{"who_id": who_id, **row, "match_score": 0.0, "evidence": []}
                for who_id, row in ranked]

    # Widened `k`: these are atom ranks, and one account can own several of the top atoms, so an
    # account-level top-k needs more atom-level room than k to fill from.
    span = max(k * evidence * 2, 40)
    scored: dict[str, float] = {}
    passages: dict[str, list[dict]] = {}
    hits = engine.atom_bm25_search(conn, query, pool, span)
    engine.fill_excerpts(conn, query, hits, width=_EVIDENCE_CHARS)
    for rank, hit in enumerate(hits):
        if not hit.who_id:
            continue
        w = person(hit.who_id)
        # RRF over the account, accumulating across its atoms rather than pinning it to its single
        # best one — the same rule the probe arm uses, so the two ranks mean the same.
        scored[w] = scored.get(w, 0.0) + rrf_term(rank)
        passages.setdefault(w, []).append(_atom_evidence(hit))

    counts: dict[str, int] = {}
    for row in conn.execute(
        "SELECT who_id, COUNT(*) AS n FROM atoms WHERE atom_id IN (%s) GROUP BY who_id"
        % ",".join("?" * len(pool)), sorted(pool)):
        w = person(row["who_id"])
        counts[w] = counts.get(w, 0) + row["n"]
    out = []
    for w in sorted(scored, key=lambda x: scored[x], reverse=True)[:k]:
        seen, ev = set(), []
        for p in passages.get(w, []):
            if p["atom_id"] in seen:
                continue
            seen.add(p["atom_id"])
            ev.append(p)
            if len(ev) >= evidence:
                break
        out.append({"who_id": w, "atoms": counts.get(w, 0),
                    "match_score": round(scored[w], 5), "evidence": ev})
    return out


def _interleave(arms: list[tuple[str, list[dict]]], k: int) -> list[dict]:
    """Merge within-arm rankings by position, not by score.

    Rank 1 of each arm, then rank 2 of each, and so on, in the order `arms` lists them (probed,
    saved, mentioned). A person present in several arms is kept in the first one only — a sampled
    timeline is strictly deeper evidence than the one post that was saved, which is deeper than an
    Oracle naming them, so showing them twice would be the same person competing with themselves.
    """
    seen: set[str] = set()
    kept = []
    for basis, rows in arms:
        kept.append((basis, [r for r in rows if r["who_id"] not in seen]))
        seen |= {r["who_id"] for r in rows}
    out = []
    for i in range(max((len(rows) for _, rows in kept), default=0)):
        for basis, rows in kept:
            if i < len(rows):
                out.append({**rows[i], "basis": basis})
            if len(out) >= k:
                return out
    return out


def _coverage_note(no_material: int, total: int) -> str | None:
    """The residue line: candidates with no local material of any kind, kept visible as a count
    because a candidate missing for lack of material looks exactly like one who did not match."""
    if not no_material or no_material < COVERAGE_NOTE_FRACTION * max(total, 1):
        return None
    head = (f"PARTIAL COVERAGE — {no_material} of {total} candidates have no local material yet "
            f"(neither a sampled timeline nor a post you saved), so they cannot match any query. "
            f"A thin or empty result here means thin coverage, NOT that nobody matches. ")
    # The fill-in promise is dropped only when the worker check says no; unreadable keeps it.
    try:
        report = liveness.repair_or_report_cached()
    except Exception:
        report = {}
    if report.get("ok") is False:
        return head + ("Opyt's background process is not running on this machine right now, "
                       "so they will not fill in on their own: "
                       f"{(report.get('cause') or 'nothing is collecting them').rstrip('.')}."
                       + (f" {report['step']}" if report.get("step") else ""))
    return head + ("Background scouting fills them in over the following days; nothing needs "
                   "to be run by hand.")


def candidates_payload(conn: sqlite3.Connection, query: str, *,
                       k: int = DEFAULT_K, evidence: int = DEFAULT_EVIDENCE,
                       min_signals: int = 1, oracle: str | None = None) -> dict:
    """The `oracle(action='candidates')` payload: candidates ranked by what they write.

    Joins the two halves of a promotion decision: how hard the user vouched for a person (curation
    signals from `screen`) and what that person actually writes — from either store that has them,
    with the source of every row named in `basis`.

    Counts are reported rather than hidden: a candidate missing because nobody pulled them yet
    looks identical, in a ranked list, to one whose content simply didn't match.

    `oracle` asks about one Oracle ("who does Imas engage with"): only the mention arm answers,
    reading that Oracle's posts, beside `coauthors`: who they co-wrote papers with
    (`curation.papers.coauthors_of`). The mention arm runs only at `min_signals` <= 1: the people
    Oracles name usually carry no curation signal of the user's. With no query it runs only for a
    named Oracle, so the plain "who is here" listing stays the candidate list.
    """
    wants_mentions = min_signals <= 1 and (bool((query or "").strip()) or bool(oracle))
    named = mentions.oracle_authors(conn, oracle) if (wants_mentions or oracle) else None
    if named and "error" in named:
        return {"query": query, "oracle": oracle, **named}

    # Identity for every candidate, whether or not probed, sourced from the screen ranking the
    # user has already seen. Saved writing is keyed on every resolved platform identity; probing
    # on each identity a sample is kept under (X, and Substack titles).
    ident: dict[str, dict] = {}
    members_by_person: dict[str, list[str]] = {}
    probe_person: dict[str, str] = {}
    saved_person: dict[str, str] = {}
    # Retired count is surfaced, never silently dropped, so a list that shrank is distinguishable
    # from one that never had those candidates.
    ranked = ranking.rank_candidates(conn, include_retired=True)
    retired = sum(1 for c in ranked if c.retired)
    for cand in ranked:
        if cand.retired:
            continue
        who_id = f"x:user:{uid}" if (uid := samples.x_user_id(cand.members)) \
            else cand.canonical_id
        ident[cand.canonical_id] = {
            "who_id": who_id,
            "canonical_id": cand.canonical_id, "name": cand.name, "handle": cand.handle,
            "distinct_signals": cand.distinct_signals}
        members_by_person[cand.canonical_id] = cand.members

    eligible = {who_id for who_id, i in ident.items()
                if i["distinct_signals"] >= min_signals}
    for who_id in eligible:
        for member in members_by_person[who_id]:
            saved_person[member] = who_id
            if member.startswith(_PROBE_PREFIXES):
                probe_person[member] = who_id

    # A named person who is already a candidate joins them; anyone else stands as a new name.
    mentioned_hits = []
    if named and wants_mentions:
        for h in mentions.search_mentions(conn, query, authors=named["authors"],
                                          exclude=named["exclude"], k=k, evidence=evidence):
            person = probe_person.get(h["who_id"])
            if person is None:
                ident.setdefault(h["who_id"], {"who_id": h["who_id"], "canonical_id": None,
                                               "name": h["name"], "handle": h["handle"],
                                               "distinct_signals": 0})
            mentioned_hits.append({**h, "who_id": person or h["who_id"]})

    states = probe_store.pull_states(conn)
    states_by_person: dict[str, dict] = {}
    for who_id, person in sorted(probe_person.items(), key=lambda kv: not kv[0].startswith("x:")):
        if who_id in states:
            states_by_person.setdefault(person, states[who_id])   # the X sample's, when both

    def listed(arms: list[tuple[str, list[dict]]]) -> list[dict]:
        rows = []
        for h in _interleave(arms, k):
            st = states_by_person.get(h["who_id"], {})
            rows.append({**h, **ident[h["who_id"]],
                         "probed_at": st.get("pulled_at"), "pull_status": st.get("status")})
        return rows

    # One Oracle asked about: only the people their posts name answer. The probed and saved arms
    # cannot be narrowed to one Oracle, and merged in they filled 20 of 30 rows with people that
    # Oracle never named (a16zcrypto).
    if oracle:
        return {"query": query, "oracle": oracle,
                "coauthors": papers.coauthors_of(conn, named["head"]),
                "candidates": listed([("mentioned", mentioned_hits)])}

    # Both arms are asked for a full k. The interleave decides the final cut, so neither arm may
    # pre-truncate itself on a guess about how many slots it will win.
    probed_hits = probe_search.search_candidates(conn, query, k=k, evidence=evidence,
                                                 who_ids=set(probe_person))
    # A person sampled on X and on Substack shows once, at their better-ranked sample.
    by_person: dict[str, dict] = {}
    for hit in probed_hits:
        person = probe_person.get(hit["who_id"])
        if person is not None and person not in by_person:
            by_person[person] = {**hit, "who_id": person}
    probed_hits = list(by_person.values())
    saved_hits = search_saved_atoms(conn, query, who_ids=set(saved_person), k=k,
                                    evidence=evidence, person_by_who=saved_person)
    for h in probed_hits:
        for p in h.get("evidence", []):
            p["provenance"] = _PROVENANCE["probed"]
    rows = listed([("probed", probed_hits), ("saved", saved_hits),
                   ("mentioned", mentioned_hits)])

    probed = {probe_person[who_id] for who_id in probe_store.probed_who_ids(conn)
              if who_id in probe_person}
    with_atoms = {saved_person[row[0]] for row in conn.execute("SELECT DISTINCT who_id FROM atoms")
                  if row[0] in saved_person}
    no_material = eligible - probed - with_atoms
    note = _coverage_note(len(no_material), len(eligible))
    return {
        "query": query,
        "candidates": rows,
        "probed": len(eligible & probed),
        "searchable_by_saved_post": len((eligible & with_atoms) - probed),
        "no_local_material": len(no_material),
        "basis_key": {"probed": "their own posts we sampled (~25 X posts, or ~5 Substack "
                                 "titles)",
                      "saved": "a post YOU saved (trusted, usually 1 — a positive example, "
                               "not a sample of their output)",
                      "mentioned": "an Oracle's own post that quotes, mentions or links them "
                                   "(trusted, about them not by them; read the Oracle's words "
                                   "for stance). canonical_id null = not a candidate yet"},
        **({"coverage_note": note} if note else {}),
        # Same rule as coverage_note: reported only when non-zero, so a shrunk list stays visible.
        **({"retired": retired,
            "retired_note": (f"{retired} candidate(s) hidden — you no longer follow them "
                             f"(their signals and saved posts are kept; add_oracle still "
                             f"takes them by name).")} if retired else {}),
    }


def candidates(conn, *, query: str, top_n: int, min_signals: int,
               oracle: str | None = None) -> dict:
    """action='candidates' — the same people ranked by what they write. See `oracle()`."""

    # Reconcile the `save` signal from stored atoms before ranking (pure SQL, no network) so a
    # bookmark the catch-up landed is a candidate on the next read. Fail-safe: a reconcile failure
    # degrades to ranking whatever signals are already stored.
    try:
        reconciled = reconcile.reconcile_saved_signals(conn)
    except Exception as e:
        reconciled = {"error": f"{type(e).__name__}: {e}"}

    out = candidates_payload(conn, query or "", k=top_n, min_signals=min_signals, oracle=oracle)
    # Reported only when it changed something or failed — printed on every call, it trains the
    # reader to skip the field.
    if reconciled.get("inserted") or reconciled.get("orphans") or reconciled.get("error"):
        out["signal_reconcile"] = reconciled

    # The other 4 signals (list/follow/like/subscribe) leave no atom to reconcile from, so their
    # only freshness record is when each collector's walk last completed; this reports it, same
    # surface-only-when-stale rule as `signal_reconcile`. Fail-safe: a state-read failure degrades
    # to the payload without freshness.
    try:
        freshness = clock.status_summary(conn, collectors.COLLECTORS)
        if freshness.get("needs_attention"):
            out["list_freshness"] = freshness
    except Exception:
        pass
    return out
