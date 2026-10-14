"""
pipeline/oracles/topics.py — a scholar's or venue's subject selection.

Is this Oracle a research venue (`is_venue_id`), what its papers are about and how many per year
and per subject (one free OpenAlex `group_by` call each, fail-safe), and the subject filter the
user chose, stored on the Oracle's OpenAlex pair where the archives pull reads it.
"""

from __future__ import annotations

from datetime import date

from pipeline.kb import schema
from pipeline.oracles import pairs
from pipeline.oracles import registry
from pipeline.oracles import windows
from pipeline.platforms.papers import openalex


def is_venue_id(canonical_id: str) -> bool:
    """Is this cluster an OpenAlex source (a journal or preprint repository) rather than a person?

    The letter is the whole test, the same one `works_filter` routes on: OpenAlex prefixes an
    author id with `A` and a source id with `S`. No network, no store read."""
    return (canonical_id or "").startswith("openalex:S")


def has_topic_filter(conn, canonical_id: str) -> bool:
    """Has the user already narrowed this cluster's subjects? Fail-safe: an unreadable registry
    reads as not narrowed, so the warning fires. Warning twice costs a sentence; staying silent
    costs 97% of a venue."""
    try:
        openalex_id = _openalex_id(conn, canonical_id)
        return bool(openalex_id and registry.topic_filter_for(conn, canonical_id, openalex_id))
    except Exception:
        return False


def scholar_year_counts(conn, canonical_id: str) -> dict | None:
    """The real distribution of an Oracle's papers by year, or None when there is nothing to ask.

    One free `group_by` call, counted through the pair's stored topic filter so the numbers
    describe the pull that will run. Any failure returns None and the caller falls back to the
    blind presets.
    """

    openalex_id = _openalex_id(conn, canonical_id)
    if not openalex_id:
        return None
    topics = registry.topic_filter_for(conn, canonical_id, openalex_id)
    try:
        counts = openalex.OpenAlexClient().year_counts(openalex_id, topics=topics)
    except Exception:                       # fail-safe: the blind presets are the fallback
        return None
    if not counts:
        return None

    this_year = date.today().year
    by_window = {}
    for name, days in windows.SCHOLAR_LOOKBACK_PRESETS.items():
        if days is None:
            continue
        # Whole years, because that is the granularity `group_by=publication_year` answers at.
        # Rounding a 730-day window to 2 years over-counts by at most the current year's partial
        # — an over-count on a window the user is choosing, which errs toward showing more work
        # than the pull will do, never less.
        first = this_year - (days // 365) + 1
        by_window[name] = sum(n for y, n in counts.items() if y >= first)
    return {"total": sum(counts.values()), "by_window": by_window, "years": len(counts),
            "openalex_id": openalex_id, "topics": topics}


# How many topics the count-first ask names one by one. A display bound and nothing else: every
# topic stays selected until the user narrows, and the ones past this are reported as a number.
TOPIC_SHOW = 25


def topic_counts_for_id(openalex_id: str, selected: str | None = None) -> dict | None:
    """The subject distribution behind one OpenAlex id — the count-first topic ask, id-keyed, so a
    just-pasted venue with no cluster yet can be asked too.

    `topics` is capped at `TOPIC_SHOW` for display and `more` states the remainder; nothing is
    dropped from the pull by not being listed. A failure returns None and the ingest runs
    unfiltered, the direction that loses no papers.
    """

    try:
        topics = openalex.OpenAlexClient().topic_counts(openalex_id)
    except Exception:
        return None
    if not topics:
        return None

    shown = topics[:TOPIC_SHOW]
    total = sum(t["count"] for t in topics)
    return {
        "openalex_id": openalex_id,
        "total_works": total,
        "n_topics": len(topics),
        "topics": shown,
        "more": max(0, len(topics) - len(shown)),
        "selected": selected.split("|") if selected else None,
        "ask": (f"{total} papers across {len(topics)} subjects"
                + (f" (top {len(shown)} listed by size)" if len(topics) > len(shown) else "")
                + ". " + ("Currently narrowed to the ids under `selected` — pass "
                          "scholar_topics=[] to go back to everything."
                          if selected else
                          "EVERYTHING is selected by default; pass scholar_topics=[ids] to "
                          "narrow. The choice sticks to every future refresh, not just this "
                          "pull.")),
    }


def scholar_topic_counts(conn, canonical_id: str) -> dict | None:
    """The distribution of an Oracle's papers by subject, or None when there is nothing to ask.

    Unfiltered by the stored selection: this is the list the user picks from. The selection rides
    along under `selected`.
    """

    openalex_id = _openalex_id(conn, canonical_id)
    if not openalex_id:
        return None
    return topic_counts_for_id(openalex_id,
                               registry.topic_filter_for(conn, canonical_id, openalex_id))


def _openalex_id(conn, canonical_id: str) -> str | None:
    """The cluster's OpenAlex id — an author (`A…`) or a venue source (`S…`) — read off the entity
    rows. No network, so a missing count never also costs the pull."""

    for m in (_oracle_members(conn, canonical_id) or []):
        pair = pairs.pair_from_member(m)
        if pair and pair[0] == "openalex":
            return pair[1]
    return None


def _oracle_members(conn, canonical_id: str):
    """The cluster's entity rows, in the shape `pair_from_member` reads. Empty on any failure."""
    try:
        return conn.execute(
            "SELECT entity_id, identity_links, profile FROM entities "
            "WHERE COALESCE(canonical_id, entity_id) = ?",
            (schema.current_canonical(conn, canonical_id),)).fetchall()
    except Exception:
        return []


def store_scholar_topics(conn, oracle_id: str, topics) -> str | None:
    """Persist the user's topic selection onto this Oracle's OpenAlex pair, and return it.

    Called before the pull, so the first backlog and every later refresh read one stored
    decision. Registers the pair first, since `set_topic_filter` only updates an existing row.

    An empty list clears the filter. Not calling this at all leaves it alone — which is what an
    ordinary top-up does, and why a top-up cannot quietly unfilter someone.
    """

    openalex_id = _openalex_id(conn, oracle_id)
    if not openalex_id:
        return None
    registry.upsert_source(conn, registry.SourceRow(oracle_id, "openalex", openalex_id))
    return registry.set_topic_filter(conn, oracle_id, openalex_id, topics)
