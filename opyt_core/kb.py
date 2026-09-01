"""
opyt_core/kb.py — the read entry points for the atom-KB MCP tools (search / open / aggregate).

Retrieval + data only, no LLM — the host model reasons over what these return. Search embeds
the query (cheap) so the vector arm can run; that is not synthesis.

The LOCAL store is opened write-capable, not read-only: `assert_model`'s idempotent
`CREATE TABLE IF NOT EXISTS` needs write access, and running it on connect means a
never-ingested store returns an empty result instead of "no such table".

All three take `kb=`, which selects WHICH knowledge base to read: omitted (or "me") is the
local store and behaves exactly as before; any other name is a registered peer
(pipeline/kb/peers.py) — an export file on this disk, opened READ-ONLY, or a knowledge base
served over HTTPS, routed to `opyt_core/kb_remote.py`. Both return the same envelope: the
service is a thin adapter over these same entry points, and an export is the same substrate, so
`retrieve.py` never learns a store can be foreign. Design records:
docs/plans/2026-08-26-foreign-kb-export-builder-phase1.md (Part 2) and
docs/plans/2026-08-27-foreign-kb-phase4-implementation.md (Task B4).
"""
from __future__ import annotations

from collections import Counter

from opyt_core import kb_remote
from pipeline.kb import peers, schema
from pipeline.kb.derive import slugify
from pipeline.kb.embed import (
    SubspaceError,
    assert_model,
    embedder_for_store,
    get_kb_embedder,
)
from pipeline.kb.peers import LOCAL_KB
from pipeline.kb.raw_store import read_body, resolve_ref
from pipeline.kb.retrieve import (
    _date_bound,
    _in_clause,
    _json_obj,
    candidate_atom_ids,
    filter_costs,
    resolve_who,
    search_atoms,
)

_MODES = ("hybrid", "semantic", "bm25")

# ── The frontier block (RULED 2026-08-27) ───────────────────────────────────────
# A default search ranks its full k over HUMAN_ATTESTED atoms and returns keyword-discovered
# ones BESIDE that list, never inside it. No score is compared across provenances at any
# surface: the two populations are ranked separately and labeled, so whichever wins a slot won
# it against its own kind. Rationale, and the alternatives rejected (one blended list with a
# share cap; human-attested only with frontier opt-in):
# docs/plans/2026-08-26-frontier-crowding-in-search.md
FRONTIER_BLOCK_CAP = 8
# A fraction of the frontier list's OWN top score, so the block cannot pad itself out to the cap
# with a tail nobody would read. 0.80 measured 2026-08-27 over 8 probe queries: cosines carry a
# high random-similarity floor, so every ratio-to-own-top in a frontier block lands in 0.61-1.00
# and a fraction near the intake cut's 0.5 would remove nothing. 0.80 prunes the blocks with a
# real step down after the head and leaves a smooth decay whole; 0.85 cut a smooth decay from 8
# hits to 2, which is cutting on noise.
FRONTIER_BLOCK_FLOOR_FRAC = 0.80


# ── which store this call reads ──────────────────────────────────────────────────

def _open_kb(kb: str | None):
    """`(conn, kb_name, kb_label)` — the store a `kb=` argument names.

    `None`/"me" is the local store, opened WRITABLE exactly as before. Any other name is a peer's,
    opened read-only by `peers.open_peer` — which is where invariant I3 lives, so a reader cannot
    write to somebody else's KB even by accident.

    A SERVED peer never reaches here — `_remote_row` routes it to `kb_remote` first — so this
    function only ever opens files.

    Raises `PeerUnavailable` for a name this install cannot read. The three entry points below
    catch it and return their own empty envelope: `kb=` is a string a host model typed, so an
    unreadable one is bad input at a trust boundary, not an exception to propagate (P3)."""
    if kb is None or kb == LOCAL_KB:
        return schema.connect(), LOCAL_KB, None
    conn, label = peers.open_peer(kb)
    return conn, kb, label


def _remote_row(kb: str | None) -> dict | None:
    """The peer row for `kb` IF it names a SERVED (https) peer — the one registry lookup the
    entry points do before opening anything. None for the local store, for a file peer, and for
    an unregistered name (which then fails in `_open_kb` with the sentence that lists the real
    ones). The local-name guard is also what keeps a plain local search from touching the
    registry at all."""
    if kb in (None, LOCAL_KB):
        return None
    row = peers.get(kb)
    if row and peers.is_remote(row["location"]):
        return row
    return None


def _label_as(kb_name: str, as_kb: str | None) -> str:
    """What the envelope CALLS the store that answered — the display half of R4's split.

    `kb=` is the ROUTING key: it picked the store, and by the time this runs that decision is
    made and unchangeable. `as_kb` decides only the string every `kb` field carries back, which
    matters because the envelope crosses the wire verbatim. A served knowledge base is routed by
    an opaque key, so without this the reader reads hit cards saying `kb: "a3f9c2e1"` and a
    `foreign_kb` notice telling them to pass that back to `open()` — while their own install
    knows the peer as `alex`. The service labels the envelope with the reader's own name for it
    instead, so no layer downstream rewrites strings inside prose.

    `LOCAL_KB` is refused rather than honoured. `as_kb` arrives in an HTTP body from a reader, so
    it is untrusted input, and every `kb_name == LOCAL_KB` test downstream reads as "this is my
    own store" — `kb_open`'s `raw_path` would hand a reader a path on the SERVER's filesystem.
    Refusing it here keeps those tests store-truth without threading a second flag through three
    entry points. A real client cannot send it anyway: `peers.add` will not register that name."""
    return as_kb if as_kb and as_kb != LOCAL_KB else kb_name


def _kb_phrase(kb_name: str, kb_label: str | None) -> str:
    """How a notice names the store it is describing. The local store is "this knowledge base" —
    the reader has exactly one and no ambiguity. A peer gets its label, because on a foreign read
    "this" could mean either store and the reader is the one who has to tell them apart."""
    if kb_name == LOCAL_KB:
        return "this knowledge base"
    return kb_label or f"the knowledge base '{kb_name}'"


def _kb_unavailable_notice(kb: str, err: Exception) -> dict:
    """The `kb=` name could not be opened. Names the registered peers, because a host that guessed
    a name has no other way to find the real ones and would otherwise guess again."""
    known = [p["name"] for p in peers.list_peers()]
    tail = (f" Registered knowledge bases: {', '.join(known)}." if known
            else " No other knowledge bases are registered on this install.")
    return {"code": "kb_unknown", "kb": kb, "known": known,
            "message": f"Could not read a knowledge base named '{kb}': {err}{tail} "
                       f"Omit `kb` to search your own."}


def _query_embedder(conn, *, foreign: bool):
    """`(embedder, notice)` — what to embed the query with, for THIS store.

    Locally, `get_kb_embedder()` plus the loud `assert_model`: a local store whose recorded model
    disagrees with this install's config is a misconfiguration, and a query answered from the
    wrong subspace returns confident garbage rather than an error, so it must fail hard.

    On a peer's store the model is a fact its owner recorded, so `embedder_for_store` follows the
    store (see its docstring for why no `assert_model` follows it). The one thing that cannot
    follow is the provider — this install has one endpoint and one key. That mismatch DEGRADES to
    the keyword arm, which needs no vectors at all, rather than raising out of a tool call: half a
    search plus a sentence saying which half is missing beats a stack trace (P3)."""
    if not foreign:
        emb = get_kb_embedder()
        assert_model(conn, emb)   # same-subspace guard; RAISES loudly on local model drift
        return emb, None
    try:
        return embedder_for_store(conn), None
    except SubspaceError as e:
        return None, _vector_arm_notice(e)


def _vector_arm_notice(e: Exception) -> dict:
    """The one degrade sentence for "the vector arm cannot run on this foreign read", emitted by
    BOTH transports — `_query_embedder` for a file peer, `kb_remote._query_vector` for a served
    one — so hosts treat the two identically. The CODE is the contract; the wording may change,
    but only here."""
    return {"code": "vector_arm_unavailable", "reason": str(e),
            "message": f"Only the keyword half of this search ran: {e} Results are "
                       f"BM25-ranked, so a question phrased conceptually rather than "
                       f"with the words the source used may find nothing."}


def _slug_tags(tags):
    """A caller's `tags` → the slug list the pre-filter takes. No tags asked for → None.
    A non-empty `tags` whose every value slugifies away returns `[]`, never None — the filter
    reads `[]` as "match nothing" and None as "no filter", so returning None here would silently
    widen a tag-restricted search to the whole store."""
    if not tags:
        return None
    return [s for s in (slugify(t) for t in tags) if s]


def _author_ids(conn, who: str | None, who_id: str | list[str] | None):
    """Fold a `who` handle and any explicit `who_id`s into ONE author-id list (a union, not an
    intersection). Returns `(ids, resolved)`; `resolved` is None when no `who` was passed.
    If `who` resolves to nobody, returns `[]` (not None) so the filter reads "match nothing"
    rather than silently widening to the whole store. `resolved` is returned so an empty result
    is distinguishable as "unresolvable handle" vs "tracked person, nothing on topic"."""
    ids = ([who_id] if isinstance(who_id, str) else list(who_id or [])) if who_id else []
    if not who:
        return who_id, None
    resolved = resolve_who(conn, who)
    for cand in resolved:
        ids.extend(cand["who_ids"])
    return (sorted(set(i for i in ids if i)) if ids else []), resolved


def _tally(values) -> dict:
    """Counted, highest first, Nones dropped. The `insights` distributions are all this shape."""
    return dict(Counter(v for v in values if v is not None).most_common())


def _tags_in_store(conn, slugs: list[str]) -> set[str]:
    """Which of these slugs ANY atom actually carries in `payload.source_tags` — the same field
    the pre-filter matches on."""
    if not slugs:
        return set()
    ph = ",".join("?" for _ in slugs)
    rows = conn.execute(
        f"SELECT js.value FROM atoms a, "
        f"json_each(json_extract(a.payload, '$.source_tags')) js WHERE js.value IN ({ph})",
        slugs,
    ).fetchall()
    return {r[0] for r in rows}


def _build_insights(conn, hits, *, resolved, filter_cost: dict) -> dict:
    """Facts about the EVIDENCE — distributions a host needs to judge what it is holding.
    VALUES, never sentences: e.g. `authors == {"X": 8}`, not a warning. No thresholds live here;
    the host decides what is alarming."""
    dated = sorted(h.when_ts for h in hits if h.when_ts)
    out = {
        # Keyed on display name, falling back to id — a nameless hit still needs a bucket.
        "authors": _tally(h.who_name or h.who_id for h in hits),
        "sources": _tally(h.source_type for h in hits),
        "topics": _tally(t for h in hits for t in (h.payload.get("source_tags") or [])),
        "date_span": [dated[0], dated[-1]] if dated else None,   # undated hits skipped
        "body_state": _tally(h.body_state for h in hits),
        "saved_vs_crawled": _tally(h.entry_mode for h in hits),
        "corpus_newest": conn.execute("SELECT MAX(ingested_at) FROM atoms").fetchone()[0],
        "filter_cost": filter_cost,
    }
    if resolved is not None:   # only when who= was passed — absent key, never an empty one
        out["resolved_who"] = resolved
    return out


def _build_notices(conn, run, *, tags, slugs, who, resolved, k: int,
                   filter_cost: dict, applied: dict,
                   kb_name: str = LOCAL_KB, kb_label: str | None = None) -> list[dict]:
    """Facts about what the QUERY did, as finished SENTENCES meant to be repeated to the user
    verbatim (unlike `insights`/`trace`, which are bare values). Reports filters, resolution and
    truncation; never judges evidence quality — that's `insights`.

    These sentences speak in the first person about YOUR knowledge base, and on a foreign read
    some of them are simply false — so two are adjusted here. This is scoped to what actually
    misleads, not a rewrite of the layer: `oracles_stale` reads tables an export does not carry
    and already degrades to silence, and the rest describe the query, which is the reader's own
    either way."""
    foreign = kb_name != LOCAL_KB
    whose = _kb_phrase(kb_name, kb_label)
    out: list[dict] = []
    if not conn.execute("SELECT EXISTS(SELECT 1 FROM atoms)").fetchone()[0]:
        if foreign:
            # `onboard` sets up the READER's install and would not put a single atom in somebody
            # else's store, so pointing at it here would send them to fix the wrong thing.
            return [{"code": "store_empty", "kb": kb_name,
                     "message": f"{whose} has no atoms in it, so there was nothing to search. "
                                f"That is a fact about the knowledge base you asked for, not "
                                f"about your own — omit `kb` to search yours."}]
        # Points to `onboard` rather than a bare "ingest something" — that used to name no tool
        # and nothing about the missing API keys that actually block a fresh install.
        return [{"code": "store_empty", "next_tool": "onboard",
                 "message": "The knowledge base has no atoms yet, so there was nothing to "
                            "search. Call `onboard` to set it up — it collects the two API keys "
                            "this needs, then pulls the people you already follow and save."}]

    if foreign:
        # Unconditional on a foreign read, unlike every other notice here, and deliberately so:
        # whose corpus answered is not a fact about this query that might or might not matter, it
        # is the attribution that has to travel with anything the host repeats (X2).
        out.append({"code": "foreign_kb", "kb": kb_name,
                    "message": f"These results come from {whose}, not your own. Attribute "
                               f"anything you repeat from them to it, and pass kb='{kb_name}' "
                               f"back to `open()` to read one — your own store does not hold "
                               f"these atoms."})

    # Gated on `needs_attention` (unlike `oracle(action='screen')`, which always prints
    # freshness) — `search` is high-frequency, and an unconditional block trains the reader to
    # skip it.
    try:
        from pipeline.kb import oracle_refresh
        fresh = oracle_refresh.status_summary(conn)
        if fresh.get("needs_attention"):
            out.append({"code": "oracles_stale", "freshness": fresh,
                        "message": "Some of your Oracles have not been refreshed in a while, so "
                                   "these results may miss what they published recently."})
    except Exception:
        pass          # Fail-safe: a freshness hiccup must never break a search

    # A paused rail otherwise goes quiet with no visible symptom besides stale results; surface
    # it here, gated on a real pause (not unconditional). Skipped on a foreign read: these are the
    # READER's rails and the reader's spend ceiling, and reporting them inside somebody else's
    # results says their corpus is missing material when nothing of the sort is true.
    try:
        from pipeline.kb import rail_budgets
        paused = [] if foreign else rail_budgets.paused_today()
        if paused:
            names = ", ".join(p["label"] for p in paused)
            out.append({"code": "rails_budget_paused", "rails": paused,
                        "message": f"Background collection is paused for the rest of today: "
                                   f"{names} reached its daily spend ceiling. Nothing new is "
                                   f"being brought in from those sources until it resets at UTC "
                                   f"midnight, so these results may be missing recent material."})
    except Exception:
        pass          # Fail-safe: a spend-meter hiccup must never break a search

    if who and not resolved:
        out.append({"code": "who_unresolved", "who": who, "kb": kb_name,
                    "message": f"Nobody matching '{who}' is in {whose}, so no atom "
                               f"could match. This is 'we don't track them', not 'they said "
                               f"nothing about this'. This lookup is local-only and never "
                               f"searches the web, so if '{who}' is a real person whose handle "
                               f"you don't have, web-search for their X/GitHub/Substack handle "
                               f"and retry with who=<handle> before concluding they're untracked."})
    elif resolved and len(resolved) > 1:
        names = ", ".join(f"{c['name'] or c['canonical_id']}" for c in resolved)
        out.append({"code": "who_multiple", "who": who,
                    "canonical_ids": [c["canonical_id"] for c in resolved],
                    "message": f"'{who}' matched {len(resolved)} different people ({names}), so "
                               f"these results MIX them. Narrow with who_id= to pick one."})

    if run.effective_mode == "none" and not (who and not resolved):
        out.append({"code": "filters_matched_nothing", "filter_cost": filter_cost,
                    "message": "No atoms passed your filters, so nothing was ranked."
                               + _cost_sentence(filter_cost, applied)})

    if tags:
        dropped = [t for t in tags if not slugify(t)]
        changed = [(t, slugify(t)) for t in tags if slugify(t) and slugify(t) != t]
        if dropped or changed:
            parts = [f"'{t}' is not a usable tag and was dropped" for t in dropped]
            parts += [f"'{t}' was normalized to '{s}'" for t, s in changed]
            out.append({"code": "tags_normalized", "dropped": dropped,
                        "changed": [list(c) for c in changed],
                        "message": "Tags are matched as slugs: " + "; ".join(parts) + "."})
    if slugs:
        missing = [s for s in slugs if s not in _tags_in_store(conn, slugs)]
        if missing:
            # Only queried on the miss path — a host that guessed a tag needs to know what the
            # store actually uses, and a bare "no such tag" leaves it guessing again.
            live = [r[0] for r in conn.execute(
                "SELECT js.value, COUNT(*) c FROM atoms a, "
                "json_each(json_extract(a.payload, '$.source_tags')) js "
                "GROUP BY js.value ORDER BY c DESC LIMIT 6")]
            tail = f" The store's most common tags are: {', '.join(live)}." if live else ""
            out.append({"code": "tag_not_in_store", "tags": missing, "known": live,
                        "message": f"No atom is tagged {', '.join(repr(m) for m in missing)}."
                                   + tail})

    if run.undated_excluded:
        # Non-zero only when a date bound was active. An undated atom is dropped by the date
        # filter (unlike a year-precision atom, which is kept); say so rather than let it show
        # only as a smaller result.
        n = run.undated_excluded
        out.append({"code": "undated_excluded", "count": n,
                    "message": f"{n} atom{'s' if n != 1 else ''} cleared your other filters but "
                               f"ha{'ve' if n != 1 else 's'} no recorded date, so the date "
                               f"filter could not evaluate {'them' if n != 1 else 'it'}. "
                               f"{'They were' if n != 1 else 'It was'} excluded — drop "
                               f"date_from/date_to to include {'them' if n != 1 else 'it'}."})

    if run.ranked > len(run.hits):
        out.append({"code": "results_truncated", "ranked": run.ranked, "showing": len(run.hits),
                    "message": f"Showing the top {len(run.hits)} of {run.ranked} atoms that "
                               f"scored; raise k to see more."})
    return out


def _cost_sentence(filter_cost: dict, applied: dict) -> str:
    """`{"source_type": 14}` → " Dropping source_type='paper' would return 14." Empty → ""."""
    if not filter_cost:
        return ""
    name, gain = max(filter_cost.items(), key=lambda kv: kv[1])
    # Name the VALUE only when it's non-empty — "Dropping tags=[]" is noise once a filter
    # normalized away to nothing.
    value = applied.get(name)
    shown = f"{name}={value!r}" if value not in (None, [], ()) else name
    return f" Dropping {shown} would return {gain}."


def _frontier_block(conn, query, embedder, *, kb_name, slugs, what_kind, source_type, who_id,
                    d_from, d_to, mode) -> dict:
    """The labeled frontier section of a default search: its OWN ranked list, capped and floored.

    Every filter the caller asked for applies here too — a `source_type` or date bound scopes
    both lists or the two stop describing the same question.

    The floor is a fraction of this block's own top, never of the human-attested list's: the two
    are separate populations and comparing their scores is the thing this whole design refuses to
    do.
    """
    run = search_atoms(conn, query, embedder, tags=slugs, what_kind=what_kind,
                       source_type=source_type, who_id=who_id, date_from=d_from, date_to=d_to,
                       entry_mode="frontier", k=FRONTIER_BLOCK_CAP, mode=mode)
    out = {"hits": [], "candidates": run.candidates}
    if not run.hits:
        return out
    # THE FLOOR NEEDS A RATIO SCALE, and only the vector arm has one. A cosine is a similarity,
    # so "80% of the top" is a statement about how well things matched. Every other scale under
    # this key is POSITIONAL, and a fraction of a position measures nothing — measured
    # 2026-08-27:
    #
    #   reciprocal_rank (1/(1+rank)):  the BM25 relevance value orders the rows and is then
    #     DISCARDED; `score` is computed from the loop counter. Every floor from 0.51 to 1.00
    #     keeps exactly rank 0, whatever it matched and however many good hits sit behind it.
    #     That is `hits[:1]` wearing a quality floor's name.
    #   rrf (sums of w/(60+rank)):  worse than a no-op. Along RANK the c=60 constant compresses
    #     the range — rank 7 is still 0.90 of top, so nothing inside the cap is ever cut. Along
    #     ARM MEMBERSHIP the score is a SUM, so an atom found by one arm scores 0.25-0.50 of one
    #     found by both at the same rank (bm25_weight is 1/2/3, semantic is always 1.0). A 0.80
    #     floor would therefore delete every single-arm hit the moment a dual-arm hit exists —
    #     cutting on which retrieval method found it, which is not a quality judgement at all.
    #
    # So it does not run there, and `applied` SAYS so: "ran and cut nothing" and "could not run"
    # both leave `dropped: 0`, and those are opposite facts.
    absolute = run.score_scale == "cosine"
    top = max(h.score for h in run.hits)
    kept = ([h for h in run.hits if h.score >= FRONTIER_BLOCK_FLOOR_FRAC * top]
            if absolute else run.hits)
    out["hits"] = [_hit_card(h, kb_name) for h in kept]
    # Never a silent cap: a block that dropped its weak tail must not read like one that had none.
    out["floor"] = {"frac": FRONTIER_BLOCK_FLOOR_FRAC, "applied": absolute,
                    "score_scale": run.score_scale, "top": round(top, 4),
                    "dropped": len(run.hits) - len(kept)}
    return out


def run_kb_search(query: str, tags: list[str] | None = None, what_kind: str | None = None,
                  source_type: str | None = None, who_id: str | list[str] | None = None,
                  who: str | None = None, date_from: str | None = None,
                  date_to: str | None = None, entry_mode: str | list[str] | None = None,
                  k: int = 8, mode: str = "hybrid", kb: str | None = None,
                  as_kb: str | None = None, embedder=None) -> dict:
    """Enforced-hybrid retrieval over atoms → `{hits, notices, insights, trace, frontier_atoms}`.

    `hits` are ranked routing cards (matched-chunk snippet + pointer), never a content claim —
    the host `open()`s an atom to assert what it says. `notices` are finished sentences meant to
    be repeated verbatim (filters, resolution, truncation); `insights` are values describing the
    evidence (author/source/topic distributions, date span, filter cost); `trace` is values
    describing what the engine did (arms run, score units, cutoff). Only `notices` is recited.

    `entry_mode` decides the SHAPE of the answer, not just a filter (RULED 2026-08-27):

      • absent (the default) → SECTIONED. `hits` is the full k ranked over HUMAN_ATTESTED atoms
        alone, and `frontier_atoms` carries keyword-discovered ones as their own capped, floored
        list. `notices`/`insights`/`trace` all describe the human-attested run. That pass is
        fail-safe: if it raises, `frontier_atoms.hits` is empty and a `frontier_block_failed`
        notice says so — a block that is an ADDITION to the answer must never break it.
      • given → ONE list, scoped to those modes, exactly as any other filter. This is the "give
        me more frontier" path (`entry_mode="frontier", k=20`); no `frontier_atoms` key comes
        back, because the whole answer is already that population.

    `frontier_atoms`, not `frontier`: the MCP layer already puts Frontier's QUEUE push notice
    (staged candidates nobody has seen) under `frontier`, and its absence there means silence.
    These are atoms already in the KB; that is candidates not yet in it.

    `who` (handle/URL, resolved via `resolve_who`) and `who_id` (exact) both restrict authorship
    and union when both are given; an unresolvable `who` matches nothing rather than widening.

    `date_from`/`date_to` bound the atom's own date, inclusive; a malformed bound raises
    `ValueError` (see `_date_bound`). An undated atom is excluded and counted in
    `undated_excluded`; a year-precision atom is included if its year overlaps the window.

    `payload` is returned verbatim, never filtered on — payload keys are source-specific and
    sparse (e.g. `stars` only exists on GitHub), so there's no single correct predicate rule.

    `kb` picks WHICH knowledge base: omitted (or "me") is your own; any other name is a registered
    peer's, read-only. Every hit carries the `kb` it came from — always, local ones included, so a
    host never has to infer provenance from a missing key.

    `as_kb` renames that store IN THE ENVELOPE and nowhere else — see `_label_as`. Supplied by
    one caller, a reader reading a SERVED knowledge base, so the routing key can be opaque while
    the answer comes back labelled with whatever that reader's install calls it.

    `embedder` overrides how the query is turned into a vector. Omitted — the normal case — this
    function builds one for the store it opened, which is the whole of the local and the
    filesystem-peer paths. It is supplied by exactly one caller: a process SERVING this knowledge
    base to someone else, where the reader embedded their own query and sent the vector
    (`embed.PrecomputedEmbedder`), so the server holds no embedding key and pays nothing per
    query. The parameter is here rather than in `retrieve.py` because choosing what embeds a
    query is this layer's job — `search_atoms` has always just used the one it is handed."""
    mode = mode if mode in _MODES else "hybrid"
    slugs = _slug_tags(tags)
    # Normalized before the connection opens: a malformed bound fails as an argument error.
    d_from = _date_bound(date_from, end=False)
    d_to = _date_bound(date_to, end=True)
    try:
        row = _remote_row(kb)
        if row is not None:
            return kb_remote.search(row, query, tags=tags, what_kind=what_kind,
                                    source_type=source_type, who_id=who_id, who=who,
                                    date_from=date_from, date_to=date_to,
                                    entry_mode=entry_mode, k=k, mode=mode)
        conn, kb_name, kb_label = _open_kb(kb)
    except peers.PeerUnavailable as e:
        # An empty envelope of the normal shape, so a host reads one channel either way — an
        # unreadable `kb=` must not be the one call that returns a different structure. One
        # handler for both transports: a dead service answers the way a missing file does.
        return {"hits": [], "notices": [_kb_unavailable_notice(kb, e)], "insights": {},
                "trace": {"kb": kb, "ran": "none", "why": "knowledge base unavailable"}}
    foreign = kb_name != LOCAL_KB
    kb_name = _label_as(kb_name, as_kb)   # the store is chosen; this is only what to call it
    try:
        who_id, resolved = _author_ids(conn, who, who_id)
        # bm25-only needs no vectors (and no cost); hybrid/semantic embed the query. A
        # caller-supplied embedder already IS the query vector, so it skips `_query_embedder`
        # entirely — there is no model to resolve and no provider that could mismatch.
        if mode not in ("hybrid", "semantic"):
            query_embedder, degraded = None, None
        elif embedder is not None:
            query_embedder, degraded = embedder, None
        else:
            query_embedder, degraded = _query_embedder(conn, foreign=foreign)
        if degraded is not None:
            mode = "bm25"   # the arm that needs no vectors; `degraded` says why the other is gone
        # The default scope is HUMAN_ATTESTED, not "everything" — the frontier population is
        # answered beside this list rather than inside it. Peer exports carry the full `atoms`
        # table, `entry_mode` included, so the sectioned shape holds on a foreign store too.
        scope = list(schema.HUMAN_ATTESTED) if entry_mode is None else entry_mode
        run = search_atoms(conn, query, query_embedder, tags=slugs,
                           what_kind=what_kind, source_type=source_type, entry_mode=scope,
                           who_id=who_id, date_from=d_from, date_to=d_to, k=k, mode=mode)
        hits = run.hits
        block, block_error = None, None
        if entry_mode is None:
            try:
                block = _frontier_block(conn, query, query_embedder, kb_name=kb_name,
                                        slugs=slugs,
                                        what_kind=what_kind, source_type=source_type,
                                        who_id=who_id, d_from=d_from, d_to=d_to, mode=mode)
            except Exception as e:      # FAIL-SAFE: a second list may not cost the first one
                block, block_error = {"hits": [], "candidates": None}, f"{type(e).__name__}: {e}"
        # Filters as APPLIED (slugified tags, expanded who_ids), not as asked for.
        applied = {}
        if slugs is not None:
            applied["tags"] = slugs
        if what_kind:
            applied["what_kind"] = what_kind
        if source_type:
            applied["source_type"] = source_type
        if who_id is not None and who_id != "":
            applied["who_id"] = who_id
        # As applied, i.e. normalized — `date_to="2026"` shows as `2026-12-31`.
        if d_from:
            applied["date_from"] = d_from
        if d_to:
            applied["date_to"] = d_to
        if entry_mode is not None:      # the DEFAULT scope is the response shape, not a filter
            applied["entry_mode"] = entry_mode
        # `scope`, not `entry_mode`: every filter `_filter_clauses` knows about has to be here or
        # the baseline — and so every other filter's cost — is computed over the wrong population.
        costs = filter_costs(conn, tags=slugs, what_kind=what_kind,
                             source_type=source_type, who_id=who_id,
                             date_from=d_from, date_to=d_to, entry_mode=scope)
        if entry_mode is None:
            # What dropping the default scope would return is the frontier block itself, shown in
            # full beside these hits. A sentence pricing it would duplicate that list, worse.
            costs.pop("entry_mode", None)
        insights = _build_insights(conn, hits, resolved=resolved, filter_cost=costs)
        notices = _build_notices(conn, run, tags=tags, slugs=slugs, who=who, resolved=resolved,
                                 k=k, filter_cost=costs, applied=applied,
                                 kb_name=kb_name, kb_label=kb_label)
        if degraded is not None:
            notices.append(degraded)
    finally:
        conn.close()

    scope = (f" ({run.candidates} passed filters)" if run.candidates is not None
             else " (whole store searched)")
    trace = {
        "kb": kb_name,                      # which store answered — "me" or a peer's name
        "ran": run.effective_mode,          # which arms ACTUALLY ran — "hybrid" often runs one
        "why": run.why,
        "score_scale": run.score_scale,     # the units `hits[].score` is in, which VARY by arm
        "candidates": run.candidates,       # None = unfiltered whole store, ≠ 0
        "ranked": run.ranked,
        "showing": f"{len(hits)} of {run.ranked} ranked{scope}",
        "fts_query": run.fts_query,         # the rewritten OR-of-tokens; None if BM25 didn't run
        "filters": applied,
        "pool_saturated": run.pool_saturated,
    }
    if run.cutoff is not None:   # absent, not zeroed, when nothing was cut
        trace["cutoff"] = run.cutoff
    out = {"hits": [_hit_card(h, kb_name) for h in hits], "notices": notices,
           "insights": insights, "trace": trace}
    if block is not None:
        out["frontier_atoms"] = block
        if block_error:
            notices.append({"code": "frontier_block_failed", "error": block_error,
                            "message": "The frontier list could not be built for this query; "
                                       "the results below are the human-attested ones only."})
    return out


def _hit_card(h, kb_name: str = LOCAL_KB) -> dict:
    return {
        "citation_id": h.citation_id,
        "atom_id": h.atom_id,
        # On the HIT, not only the envelope: a host that quotes one card into a document has to
        # carry the attribution with it, and an unlabelled foreign atom is just a bookmark (X2).
        # Present on local hits too, as "me" — provenance a host has to infer from a MISSING key
        # is provenance it will get wrong.
        "kb": kb_name,
        "source_type": h.source_type,
        "what_kind": h.what_kind,
        "who_id": h.who_id,
        "who_name": h.who_name,     # display label; `who_id` stays the identity
        "when_ts": h.when_ts,
        "when_precision": h.when_precision,
        "description": h.description,
        "snippet": h.snippet,
        "chunk_span": list(h.chunk_span) if h.chunk_span else None,
        "source_url": h.source_url,
        "raw_ref": h.raw_ref,
        "score": round(h.score, 4),
        # On the card, not only in `insights`: `entry_mode` is user-facing POLICY since the
        # sectioned response (2026-08-27), and a card read on its own must still say which list
        # it belongs to.
        "entry_mode": h.entry_mode,
        "bm25_rank": h.bm25_rank,
        "sem_rank": h.sem_rank,
        "body_state": h.body_state,   # "complete"/"partial"/"absent"/"pending" — qualifies `snippet`
        "body_basis": h.body_basis,
        "payload": h.payload,         # source-shaped extras, verbatim (no allowlist)
    }


def kb_open(atom_id: str, kb: str | None = None, as_kb: str | None = None) -> dict:
    """Follow an atom's pointer. Returns the REAL raw snapshot text + live `source_url` so the
    host asserts from the source, not a description. Fail-safe: unknown atom / missing snapshot
    returns an error dict. `body_state` matters here more than on a search hit — it's the last
    check before a stub could be quoted as the whole article.

    `kb` must be the value the hit card carried. An atom id is scoped to its store — the same
    tweet ingested by two people is one id in two knowledge bases — so opening a foreign id
    against your own store finds nothing, or finds YOUR copy of the same source.

    `as_kb` renames the store in the returned dict and nowhere else — see `_label_as`."""
    try:
        row = _remote_row(kb)
        if row is not None:
            return kb_remote.open_atom(row, atom_id)
        conn, kb_name, _label = _open_kb(kb)
        kb_name = _label_as(kb_name, as_kb)
    except peers.PeerUnavailable as e:
        # The same sentence `search` gives, including which names DO resolve — a host that
        # guessed one has no other way to find the real ones and would otherwise guess again.
        return {"atom_id": atom_id, "kb": kb, "error": _kb_unavailable_notice(kb, e)["message"]}
    try:
        row = conn.execute(
            "SELECT atom_id, source_type, what_kind, who_id, when_ts, when_precision, "
            "source_url, raw_ref, description, payload FROM atoms WHERE atom_id=?",
            (atom_id,),
        ).fetchone()
        # Inside the connection, not after it: where a body LIVES is a fact about the store, and
        # an export keeps its bodies in a table rather than in files beside it. `read_body` reads
        # the store to find out, so it needs the store still open.
        raw = read_body(conn, row["atom_id"], row["raw_ref"]) if row is not None else None
    finally:
        conn.close()
    if row is None:
        return {"atom_id": atom_id, "kb": kb_name, "error": "not found"}
    payload = _json_obj(row["payload"])   # same decode as a search hit, via the same helper
    body_state = payload.pop("body_state", None)
    body_basis = payload.pop("body_basis", None)
    return {
        "atom_id": row["atom_id"],
        "kb": kb_name,                            # which store this body came from
        "source_type": row["source_type"],
        "what_kind": row["what_kind"],
        "who_id": row["who_id"],
        "when_ts": row["when_ts"],
        "when_precision": row["when_precision"],
        "source_url": row["source_url"],          # the LIVE pointer (re-fetch for the freshest)
        # A path on THIS machine's filesystem, so it is a lie about a foreign atom: `resolve_ref`
        # rehydrates against the reader's own `opyt_home()`, and a peer's snapshots were never
        # written there. `raw` already carries the body either way.
        "raw_path": (str(resolve_ref(row["raw_ref"]))
                     if row["raw_ref"] and kb_name == LOCAL_KB else None),
        "description": row["description"],
        "raw": raw,                               # the stored snapshot — assert from THIS
        "raw_available": raw is not None,
        "body_state": body_state,   # "partial" → do NOT present raw as the whole thing
        "body_basis": body_basis,
        "payload": payload,                       # source-shaped extras, verbatim
    }


def kb_aggregate(scope: dict | None = None, kb: str | None = None,
                 as_kb: str | None = None) -> dict:
    """A pure-SQL state-of-play skeleton for a dossier: counts by kind/source, trust coverage,
    topic/entity distribution, and top atom DESCRIPTIONS. The host drafts from this, then
    `open()`s pivotal atoms to ground each claim in raw text. `scope` filters like search:
    {tags, what_kind, source_type, who_id, date_from, date_to} (same date-bound semantics).

    No handle resolution here — resolve via `search(who=...)`'s `insights.resolved_who[].who_ids`
    and pass `who_id=` here, so resolution has one home rather than two that can drift.

    `kb` picks which knowledge base to summarize — omitted (or "me") is your own; any other name
    is a registered peer's. `scope` stays filters-only: which store to read is not a filter.

    `as_kb` renames that store in the returned dict and nowhere else — see `_label_as`."""
    scope = scope or {}
    tags = _slug_tags(scope.get("tags"))
    what_kind = scope.get("what_kind")
    source_type = scope.get("source_type")
    who_id = scope.get("who_id")
    d_from = _date_bound(scope.get("date_from"), end=False)
    d_to = _date_bound(scope.get("date_to"), end=True)

    try:
        row = _remote_row(kb)
        if row is not None:
            return kb_remote.aggregate(row, scope)
        conn, kb_name, _label = _open_kb(kb)
        kb_name = _label_as(kb_name, as_kb)
    except peers.PeerUnavailable as e:
        return {"scope": scope, "kb": kb, "total": 0,
                "notices": [_kb_unavailable_notice(kb, e)]}
    try:
        cand = candidate_atom_ids(conn, tags, what_kind, source_type, who_id, d_from, d_to)
        if cand is not None and not cand:
            return {"scope": scope, "kb": kb_name, "total": 0,
                    "notices": [{"code": "scope_matched_nothing",
                                 "message": "No atoms match this scope."}]}
        frag, params = _in_clause(cand, "a")
        base = "FROM atoms a WHERE 1=1" + frag

        total = conn.execute(f"SELECT COUNT(*) {base}", params).fetchone()[0]
        by_kind = {(r[0] or "?"): r[1] for r in conn.execute(
            f"SELECT a.what_kind, COUNT(*) {base} GROUP BY a.what_kind", params)}
        by_source = {(r[0] or "?"): r[1] for r in conn.execute(
            f"SELECT a.source_type, COUNT(*) {base} GROUP BY a.source_type", params)}
        # Counted off `oracles`, the record of who the user actually confirmed. This used to join
        # `entity_trust`, a table seeded from those same confirmations and read nowhere else — a
        # mirror of this join that could only ever drift from it. Both paths returned 1532 on the
        # live store when the mirror was dropped (2026-08-23).
        trusted = conn.execute(
            f"SELECT COUNT(*) FROM atoms a "
            f"JOIN entities e ON e.entity_id = a.who_id "
            f"JOIN oracles o ON o.canonical_id = COALESCE(e.canonical_id, e.entity_id) "
            f"WHERE 1=1{frag}", params).fetchone()[0]
        topics = [{"topic": r[0], "count": r[1]} for r in conn.execute(
            f"SELECT js.value, COUNT(*) c FROM atoms a, "
            f"json_each(json_extract(a.payload, '$.source_tags')) js "
            f"WHERE 1=1{frag} GROUP BY js.value ORDER BY c DESC LIMIT 15", params)]
        entities = [{"who_id": r[0], "count": r[1]} for r in conn.execute(
            f"SELECT a.who_id, COUNT(*) c {base} GROUP BY a.who_id ORDER BY c DESC LIMIT 15",
            params)]
        top = [{"atom_id": r[0], "description": r[1], "who_id": r[2], "when_ts": r[3]}
               for r in conn.execute(
                   f"SELECT a.atom_id, a.description, a.who_id, a.when_ts {base} "
                   f"ORDER BY a.when_ts DESC LIMIT 12", params)]
    finally:
        conn.close()

    return {
        "scope": scope,
        "kb": kb_name,                     # which store these counts describe
        "notices": [],                     # same shape as search's, so a host reads one channel
        "total": total,
        "by_what_kind": by_kind,
        "by_source_type": by_source,
        "trusted_atoms": trusted,          # atoms whose author is a confirmed Oracle
        "top_topics": topics,
        "top_entities": entities,
        "recent_descriptions": top,        # mechanical cards; open() to ground any claim
    }
