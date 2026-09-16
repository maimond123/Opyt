"""
pipeline/kb/scholar_probe.py — the light paper pull over the SCHOLAR candidate list.

Same job as `candidate_probe` does for X, for a person who has no timeline: pull a shallow sample
of a candidate researcher's own recent papers, so `oracle(action='candidates')` can answer "what
does this person actually work on" from their titles and abstracts rather than from a name.

Two things make this a separate module rather than an arm of `candidate_probe`. Almost nothing is
shared: that module builds one X cookie session up front, paces at 22 s against a bucket measured
off `x-rate-limit-limit`, and returns early on `SyncAuthError` — a scholar pull has no session, no
auth to lose, and a different host with a different allowance. And what IS shared is already
factored: the write target (`probe_store`) and the rank order (`screen.rank_candidates`).

THE HOST BUDGET IS SHARED, AND SO IS THE TRANSPORT. `frontier_sources.OpenAlexWorksAdapter`
carries `breaker_host = "api.openalex.org"`, the same string the Frontier works adapter uses, and
`CircuitBreaker` is keyed by it in a persisted table — so every path against that host trips and
recovers as ONE breaker. Two budgets against one host would hammer it, and OpenAlex's anonymous
allowance is per IP and resets only at midnight UTC, so a burned day stays burned. The adapter
lives in `frontier_sources` rather than here because `ingest_scholar_footprint` needs it too, and
that module writes `atoms` while this one writes the untrusted store — opposite sides of a trust
boundary, which must not import each other.

Only candidates carrying an `openalex:` entity id are reachable here. A `scholar:`-only candidate
(a Semantic Scholar author id, no OpenAlex one) is ABSENT from this queue, not failed — the same
way `candidate_probe` treats a candidate with no X identity. Stage 6's ORCID merge is what gives
those candidates an `openalex:` member.
"""

from __future__ import annotations

import sqlite3
import time

from . import probe_store, schema
from .embed import assert_model
from .frontier_sources import OpenAlexWorksAdapter, SourceError
from .ingest_common import BASIS_OBSERVED, BODY_PARTIAL, AtomSink, StageTimer, body_fields, \
    snapshot_and_hash

# How stale a scholar snapshot may be before a re-pull. Longer than the X probe's 30 days on
# purpose: this sample answers "what field is this person in", and a researcher's field moves on
# the scale of a publication cycle, not a news cycle. Their median gap between works is measured
# in months.
DEFAULT_TTL_DAYS = 90.0

# One page, and the page IS the sample. 25 recent works characterize a researcher's field as well
# as 100 would and costs a quarter as much of an allowance that resets only at midnight UTC.
DEFAULT_WORKS = 25

# Seconds between candidates. Politeness, not the budget — the budget is the shared breaker plus
# the anonymous daily allowance. Matches `OpenAlexWorksAdapter.min_interval_s`, which is itself an
# order of magnitude inside OpenAlex's documented 10 requests/second.
_PACE_SECONDS = 1.0

_ATOM_PREFIX = "oaprobe"
_SNAPSHOT_SOURCE = "probe"      # shares kb_raw/probe/ with the X probe — one candidate store


# ── the queue ─────────────────────────────────────────────────────────────────

def _openalex_author_id(members: list[str]) -> str | None:
    """The OpenAlex author id in a resolved cluster, or None."""
    for m in members:
        if (m or "").startswith("openalex:"):
            return m.split(":", 1)[1]
    return None


def candidate_queue(conn: sqlite3.Connection, *, min_signals: int = 1,
                    ttl_days: float = DEFAULT_TTL_DAYS) -> list[dict]:
    """Scholar candidates due for a pull, in the SCREEN's own rank order.

    Reusing `screen.rank_candidates` rather than inventing an ordering is the same decision
    `candidate_probe.candidate_queue` makes: it is already the answer to "who has the user vouched
    for hardest", and a second ranking here would be a second opinion with no evidence behind it.
    Ordering picks who goes first under a bounded budget; it never picks membership.

    Excluded: confirmed Oracles (their real footprint is already in `atoms`), candidates with no
    OpenAlex author id, candidates below `min_signals`, and candidates still fresh.
    """
    from . import screen

    fresh = probe_store.fresh_who_ids(conn, ttl_days=ttl_days)
    out: list[dict] = []
    for cand in screen.rank_candidates(conn):
        if cand.distinct_signals < min_signals or not cand.is_scholar:
            continue
        if schema.is_oracle(conn, cand.canonical_id):
            continue
        author_id = _openalex_author_id(cand.members)
        if not author_id:
            continue
        who_id = f"openalex:{author_id}"
        if who_id in fresh:
            continue
        out.append({"who_id": who_id, "author_id": author_id,
                    "canonical_id": cand.canonical_id, "name": cand.name,
                    "distinct_signals": cand.distinct_signals})
    return out


# ── the description ───────────────────────────────────────────────────────────

def papers_for_run(conn: sqlite3.Connection) -> dict[str, list]:
    """`{entity_id: [saved atom_id, …]}` for the whole run. One walk, reused per candidate."""
    from . import paper_authors
    return paper_authors.papers_by_author(conn, "user-saved")


def saved_titles(conn: sqlite3.Connection, atom_ids: list[str], *, limit: int = 3) -> list[str]:
    """Titles of the papers THE USER SAVED that this person wrote — their own evidence, first.

    Takes the atom ids from `paper_authors.papers_by_author`, which is the only correct way to ask
    "which papers did this person author": `atoms.who_id` names the FIRST author and nothing else,
    and 67 of 82 live paper atoms do not even have that — they carry the
    `paper-authors:{paper_id}` placeholder because Semantic Scholar never resolved the work. A
    `WHERE who_id = …` query answered a different, much smaller question and returned [] for
    every candidate measured on the live store.

    Titles come off `atoms.description`, which `derive_paper` builds mechanically as
    `author · title · venue · year`, so the title is its second field. Nothing is re-fetched:
    every one of these rows is on disk already, because saving it is what made this person a
    candidate.
    """
    if not atom_ids:
        return []
    rows = conn.execute(
        "SELECT description FROM atoms WHERE atom_id IN "
        f"({','.join('?' for _ in atom_ids)}) ORDER BY when_ts DESC", tuple(atom_ids)).fetchall()
    out = []
    for (desc,) in rows:
        parts = [p.strip() for p in (desc or "").split(" · ")]
        # `Untitled` is `derive_paper`'s placeholder for a paper whose metadata carried no title
        # (2 of 11 saved papers on the live store). Rendering "you saved: Untitled" on a card is
        # worse than rendering nothing — it spends the line saying we do not know.
        if len(parts) >= 2 and parts[1] and parts[1] != "Untitled":
            out.append(parts[1])
        if len(out) >= limit:
            break
    return out


def describe(author: dict | None, *, saved: list[str], recent: list[str]) -> str:
    """A researcher's card line. MECHANICAL — no LLM, nothing interpretive, in the order a user
    who does not recognize the name needs it.

    ⚠️ OpenAlex `topics` and `affiliations` are DELIBERATELY NOT USED. Measured 2026-09-08:
    Frances Arnold's top topic comes back "Atmospheric chemistry and aerosols" — she won a Nobel
    for the directed evolution of enzymes — and her affiliation history lists Pasadena City
    College. Those derived fields would misdescribe real people on their own cards.
    `last_known_institutions` is a stated fact, not a derived one, so it stays.
    """
    bits: list[str] = []
    if saved:
        bits.append("you saved: " + "; ".join(saved))
    a = author or {}
    inst = ((a.get("last_known_institutions") or [{}])[0] or {}).get("display_name")
    stats = a.get("summary_stats") or {}
    facts = [x for x in (inst,
                         f"{a['works_count']} works" if a.get("works_count") else None,
                         f"{a['cited_by_count']} citations" if a.get("cited_by_count") else None,
                         f"h-index {stats['h_index']}" if stats.get("h_index") else None) if x]
    if facts:
        bits.append(" · ".join(facts))
    if recent:
        bits.append("recent: " + "; ".join(recent))
    return " — ".join(bits)


# ── one candidate ─────────────────────────────────────────────────────────────

def _abstract(work: dict) -> str:
    from .frontier_sources import _abstract_from_inverted
    return _abstract_from_inverted(work.get("abstract_inverted_index"))[:2000]


def render_works(works: list[dict], *, who_id: str, name: str | None) -> list[dict]:
    """Works → write-ready probe atoms. PURE (no DB, no network, no embedder).

    `body_state` is PARTIAL on every one of them, and OBSERVED: the abstract is what the works
    response carries, and no PDF is fetched on this path. Saying COMPLETE would claim we read the
    paper. A work with neither title nor abstract carries nothing to read and is dropped.
    """
    out: list[dict] = []
    for w in works:
        work_id = str(w.get("id") or "").rsplit("/", 1)[-1]
        title = " ".join((w.get("title") or "").split())
        abstract = _abstract(w)
        if not work_id or not (title or abstract):
            continue
        loc = w.get("primary_location") or {}
        venue = ((loc.get("source") or {}).get("display_name") or "").strip()
        published = (w.get("publication_date") or "")[:10]
        md = (f"---\nsource: paper\nurl: {w.get('doi') or loc.get('landing_page_url') or ''}\n"
              f"date: {published}\ntype: paper\n---\n\n"
              f"# {title or 'Untitled'}\n\n**Authors:** {name or '—'}\n\n"
              + (f"**Venue:** {venue}\n\n" if venue else "")
              + (f"## Abstract\n\n{abstract}\n\n" if abstract else ""))
        out.append({
            "atom_id": f"{_ATOM_PREFIX}:{work_id}",
            "source_type": "paper",
            "who_id": who_id,
            "when_ts": published,
            "when_precision": "day" if published else "",
            "source_url": w.get("doi") or loc.get("landing_page_url") or "",
            "description": " · ".join(p for p in (name, title, venue) if p),
            "payload": {"venue": venue, "type": w.get("type"),
                        "cited_by_count": w.get("cited_by_count", 0),
                        **body_fields(BODY_PARTIAL, BASIS_OBSERVED)},
            "_markdown": md,
        })
    return out


def probe_candidate(conn, embedder, adapter: OpenAlexWorksAdapter, cand: dict,
                    saved_papers: dict[str, list] | None = None) -> dict:
    """Pull, render and store ONE scholar candidate's recent works, and cache their description.

    Raises nothing for a per-candidate problem — a failure records `failed` (always due again next
    run) and returns. `SourceError` from an open breaker propagates: it says the whole HOST is
    backing off, so every remaining candidate would fail identically and the loop must stop rather
    than burn through the queue marking everyone failed.
    """
    from pipeline.ingestion.utils import log

    who_id, author_id = cand["who_id"], cand["author_id"]
    probe_store.record_attempt(conn, who_id)
    try:
        works = adapter.works(author_id, limit=DEFAULT_WORKS)
    except SourceError:
        raise                                   # host-wide — the LOOP decides, not this function
    except Exception as e:
        log(f"[scholar-probe] {who_id} fetch failed — SKIP (retried next run): "
            f"{type(e).__name__}: {e}")
        probe_store.record_pull(conn, who_id, probe_store.STATUS_FAILED,
                                detail=f"{type(e).__name__}: {e}")
        return {"who_id": who_id, "status": probe_store.STATUS_FAILED, "atoms": 0}

    if not works:
        # A real author id with no works in OpenAlex. A FACT about the candidate, recorded so it is
        # never re-fetched every run and never read as "no field".
        probe_store.record_pull(conn, who_id, probe_store.STATUS_EMPTY)
        return {"who_id": who_id, "status": probe_store.STATUS_EMPTY, "atoms": 0, "fetched": 0}

    # The description reuses the works already in hand for the recent titles, so it costs ONE more
    # call (the author record), not two. Fail-safe: a failed profile call still yields a
    # description built from the user's own saved titles, which is the half that matters most.
    try:
        author = adapter.author(author_id)
    except Exception as e:
        log(f"[scholar-probe] {who_id} author record unavailable — description degraded: {e}")
        author = None
    recent = [" ".join((w.get("title") or "").split()) for w in works[:3] if w.get("title")]
    # The map is computed ONCE per run and threaded in — it is one walk of the saved papers, and
    # doing it per candidate would re-walk them 44 times on the live store.
    mine = (saved_papers or papers_for_run(conn)).get(who_id) or []
    desc = describe(author, saved=saved_titles(conn, mine), recent=recent)
    name = (author or {}).get("display_name") or cand.get("name")
    try:
        schema.set_entity_profile(conn, cand["canonical_id"], {"description": desc})
    except Exception as e:                       # a card without a description is not a failed pull
        log(f"[scholar-probe] {who_id} description not cached: {e}")

    atoms = render_works(works, who_id=who_id, name=name)
    seen = probe_store.load_probe_hashes(conn, who_id)
    written = {"n": 0}
    sink = AtomSink(conn, embedder, writer=probe_store.write_probe_atom)
    submitted = skipped = 0
    for atom in atoms:
        md = atom.pop("_markdown")
        decided = snapshot_and_hash(_SNAPSHOT_SOURCE, atom["atom_id"], md, seen)
        if decided is None:                     # unchanged → skip the embed
            skipped += 1
            continue
        atom["raw_ref"], atom["raw_hash"] = decided
        seen[atom["atom_id"]] = atom["raw_hash"]
        submitted += 1
        sink.submit(atom, md, on_written=lambda: written.__setitem__("n", written["n"] + 1))
    sink.close()

    # `ok` REQUIRES everything submitted to have landed — a systemic embed failure would otherwise
    # record `ok` on a silent shortfall and freeze this candidate for a full TTL.
    if submitted and written["n"] < submitted:
        detail = f"{written['n']}/{submitted} atoms stored (embed or write failed)"
        log(f"[scholar-probe] {who_id} — {detail}; recording FAILED so it retries")
        probe_store.record_pull(conn, who_id, probe_store.STATUS_FAILED,
                                atoms=written["n"], detail=detail)
        return {"who_id": who_id, "status": probe_store.STATUS_FAILED, "fetched": len(works),
                "submitted": submitted, "written": written["n"], "unchanged": skipped,
                "atoms": written["n"]}

    probe_store.record_pull(conn, who_id, probe_store.STATUS_OK, atoms=written["n"])
    return {"who_id": who_id, "status": probe_store.STATUS_OK, "fetched": len(works),
            "submitted": submitted, "written": written["n"], "unchanged": skipped,
            "atoms": written["n"], "described": bool(desc)}


# ── the run ───────────────────────────────────────────────────────────────────

def probe_scholars(conn, embedder, *, min_signals: int = 1, max_candidates: int = 0,
                   ttl_days: float = DEFAULT_TTL_DAYS, pace_seconds: float = _PACE_SECONDS,
                   adapter: OpenAlexWorksAdapter | None = None) -> dict:
    """Sample due scholar candidates, highest-ranked first. Never raises.

    Bounded runs are the normal mode: outcome state lives in `probe_pulls`, so stopping and
    resuming costs nothing and re-pulls nobody.
    """
    from pipeline.ingestion.utils import log

    queue = candidate_queue(conn, min_signals=min_signals, ttl_days=ttl_days)
    if max_candidates:
        queue = queue[:max_candidates]
    if not queue:
        return {"source": "scholar-probe", "queued": 0, "note": "no scholar candidate is due"}

    assert_model(conn, embedder)      # guard the store's embedding identity before any work
    try:
        embedder.embed(["probe preflight"], role="document")
    except Exception as e:
        log(f"[scholar-probe] embedder unavailable — nothing pulled (no requests made): {e}")
        return {"source": "scholar-probe", "stopped": "embedder",
                "error": f"{type(e).__name__}: {e}", "requests": 0, "atoms": 0}

    adapter = adapter or OpenAlexWorksAdapter()
    # BEFORE the politeness delay, and before the queue is walked: a host already known to be down
    # must not cost a second per candidate to discover that again.
    if not adapter.available():
        log("[scholar-probe] openalex breaker is open — nothing pulled")
        return {"source": "scholar-probe", "queued": len(queue), "stopped": "breaker",
                "requests": 0, "atoms": 0}

    timer = StageTimer()
    tally = {probe_store.STATUS_OK: 0, probe_store.STATUS_EMPTY: 0,
             probe_store.STATUS_FAILED: 0}
    atoms = requests = 0
    stopped: str | None = None
    log(f"[scholar-probe] {len(queue)} candidate(s) due (min_signals={min_signals}, "
        f"ttl={ttl_days}d) — FREE, paced at {pace_seconds}s/request")

    saved_papers = papers_for_run(conn)
    for i, cand in enumerate(queue):
        if i and pace_seconds > 0:
            time.sleep(pace_seconds)
        try:
            with timer.stage("scholar-probe"):
                res = probe_candidate(conn, embedder, adapter, cand, saved_papers)
        except SourceError as e:
            log(f"[scholar-probe] host backing off after {requests} request(s) — stopping. {e}")
            stopped = "breaker"
            break
        requests += 1
        tally[res["status"]] = tally.get(res["status"], 0) + 1
        atoms += res.get("atoms", 0)

    return {"source": "scholar-probe", "queued": len(queue), "requests": requests,
            "atoms": atoms, "by_status": tally, "stopped": stopped,
            "remaining": len(candidate_queue(conn, min_signals=min_signals, ttl_days=ttl_days)),
            "stage_seconds": timer.totals}
