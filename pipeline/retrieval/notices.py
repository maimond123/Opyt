"""
pipeline/retrieval/notices.py — what a read says about itself.

Search's `notices` are finished sentences about what the query did, meant to be repeated to the
user verbatim; `insights` are bare values about the evidence. Both are decided here from values the
search gathered (`gather_facts` does the reads), so the wording of a notice has one home. The
notice every read can carry lives here too: `kb_unknown` (the `kb=` name could not be read).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from pipeline.kb.derive import slugify
from pipeline.kb.schema import count_atoms
from pipeline.retrieval.peers import LOCAL_KB


def _kb_phrase(kb_name: str, kb_label: str | None) -> str:
    """How a notice names the store: "this knowledge base" locally, the peer's label otherwise."""
    if kb_name == LOCAL_KB:
        return "this knowledge base"
    return kb_label or f"the knowledge base '{kb_name}'"


def kb_unknown(kb: str, err: Exception, known: list[str]) -> dict:
    """The `kb=` name could not be opened. Names the registered peers so a host need not guess."""
    tail = (f" Registered knowledge bases: {', '.join(known)}." if known
            else " No other knowledge bases are registered on this install.")
    return {"code": "kb_unknown", "kb": kb, "known": known,
            "message": f"Could not read a knowledge base named '{kb}': {err}{tail} "
                       f"Omit `kb` to search your own."}



def _tally(values) -> dict:
    """Counted, highest first, Nones dropped. The `insights` distributions are all this shape."""
    return dict(Counter(v for v in values if v is not None).most_common())


def _tags_in_store(conn, slugs: list[str]) -> set[str]:
    """Which of these slugs any atom carries in `payload.source_tags` (the pre-filter's field)."""
    if not slugs:
        return set()
    ph = ",".join("?" for _ in slugs)
    rows = conn.execute(
        f"SELECT js.value FROM atoms a, "
        f"json_each(json_extract(a.payload, '$.source_tags')) js WHERE js.value IN ({ph})",
        slugs,
    ).fetchall()
    return {r[0] for r in rows}


def _common_tags(conn) -> list[str]:
    """The store's six most common `source_tags`, most common first."""
    return [r[0] for r in conn.execute(
        "SELECT js.value, COUNT(*) c FROM atoms a, "
        "json_each(json_extract(a.payload, '$.source_tags')) js "
        "GROUP BY js.value ORDER BY c DESC LIMIT 6")]


@dataclass(frozen=True)
class NoticeFacts:
    """What the notices and insights need to know about the store, read before they are built."""
    store_empty: bool
    missing_tags: list[str]     # asked-for slugs no atom carries
    common_tags: list[str]      # read only on a miss
    corpus_newest: str | None


def gather_facts(conn, slugs: list[str] | None) -> NoticeFacts:
    """Read the store facts `query_notices` and `insights` decide from.

    The common tags are read only when an asked-for tag is missing, so the host can retry."""
    newest = conn.execute("SELECT MAX(ingested_at) FROM atoms").fetchone()[0]
    if count_atoms(conn) == 0:
        return NoticeFacts(True, [], [], newest)
    present = _tags_in_store(conn, slugs) if slugs else set()
    missing = [t for t in (slugs or []) if t not in present]
    return NoticeFacts(False, missing, _common_tags(conn) if missing else [], newest)


def insights(hits, *, resolved, filter_cost: dict, corpus_newest: str | None) -> dict:
    """Distributions about the evidence, as values not sentences (e.g. `authors == {"X": 8}`).
    No thresholds live here; the host decides what is alarming."""
    dated = sorted(h.when_ts for h in hits if h.when_ts)
    out = {
        # Keyed on display name, falling back to id — a nameless hit still needs a bucket.
        "authors": _tally(h.who_name or h.who_id for h in hits),
        "sources": _tally(h.source_type for h in hits),
        "topics": _tally(t for h in hits for t in (h.payload.get("source_tags") or [])),
        "date_span": [dated[0], dated[-1]] if dated else None,   # undated hits skipped
        "body_state": _tally(h.body_state for h in hits),
        "saved_vs_crawled": _tally(h.entry_mode for h in hits),
        "corpus_newest": corpus_newest,
        "filter_cost": filter_cost,
    }
    if resolved is not None:   # only when who= was passed — absent key, never an empty one
        out["resolved_who"] = resolved
    return out


def query_notices(run, facts: NoticeFacts, *, tags, who, resolved,
                   filter_cost: dict, applied: dict,
                   kb_name: str, kb_label: str | None) -> list[dict]:
    """What the query did, as finished sentences meant to be repeated to the user verbatim.
    Reports filters, resolution and truncation; never judges evidence quality (`insights` does).
    On a foreign read the notices that would misattribute the store are adjusted."""
    foreign = kb_name != LOCAL_KB
    whose = _kb_phrase(kb_name, kb_label)
    out: list[dict] = []
    if facts.store_empty:
        if foreign:
            # Not `onboard`: it sets up the reader's install, not somebody else's store.
            return [{"code": "store_empty", "kb": kb_name,
                     "message": f"{whose} has no atoms in it, so there was nothing to search. "
                                f"That is a fact about the knowledge base you asked for, not "
                                f"about your own — omit `kb` to search yours."}]
        return [{"code": "store_empty", "next_tool": "onboard",
                 "message": "The knowledge base has no atoms yet, so there was nothing to "
                            "search. Call `onboard` to set it up — it pulls the people you "
                            "already follow and the posts you save."}]

    if foreign:
        # Unconditional on a foreign read: the attribution must travel with anything repeated (X2).
        out.append({"code": "foreign_kb", "kb": kb_name,
                    "message": f"These results come from {whose}, not your own. Attribute "
                               f"anything you repeat from them to it, and pass kb='{kb_name}' "
                               f"back to `open()` to read one — your own store does not hold "
                               f"these atoms."})

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

    if run.candidates == 0 and not (who and not resolved):
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
    if facts.missing_tags:
        missing, live = facts.missing_tags, facts.common_tags
        tail = f" The store's most common tags are: {', '.join(live)}." if live else ""
        out.append({"code": "tag_not_in_store", "tags": missing, "known": live,
                    "message": f"No atom is tagged {', '.join(repr(m) for m in missing)}."
                               + tail})

    if run.undated_excluded:
        # Non-zero only when a date bound was active; say so rather than show a silently smaller
        # result.
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
    # Name the value only when non-empty ("Dropping tags=[]" is noise).
    value = applied.get(name)
    shown = f"{name}={value!r}" if value not in (None, [], ()) else name
    return f" Dropping {shown} would return {gain}."


def trace(run, applied: dict, kb_name: str) -> dict:
    """Values about what the engine did, for the host's reasoning — never recited."""
    scope = (f" ({run.candidates} passed filters)" if run.candidates is not None
             else " (whole store searched)")
    out = {
        "kb": kb_name,                      # which store answered — "me" or a peer's name
        "candidates": run.candidates,       # None = unfiltered whole store, ≠ 0
        "ranked": run.ranked,
        "showing": f"{len(run.hits)} of {run.ranked} ranked{scope}",
        "fts_query": run.fts_query,         # the rewritten OR-of-tokens; None if nothing ran
        "filters": applied,
        "pool_saturated": run.pool_saturated,
    }
    if run.cutoff is not None:   # absent, not zeroed, when nothing was cut
        out["cutoff"] = run.cutoff
    return out


def frontier_block_failed(error: str) -> dict:
    """The frontier list beside a default search could not be built; the main list stands."""
    return {"code": "frontier_block_failed", "error": error,
            "message": "The frontier list could not be built for this query; "
                       "the results below are the human-attested ones only."}
