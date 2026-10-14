"""
pipeline/retrieval/corpus_census.py — a representative spread of the store, for the host to read,
and the counts an offer cites (`mcp_server/offers.py`).

Answers census questions ("what topics does my corpus cover?") that `search` cannot, since it
needs a query. This is not a topic layer: it computes no grouping, invents no label and
writes nothing. The host names candidate subjects from the spread, then checks each one with
`search` before reporting it.

The spread walks authors round-robin (everyone's first atom before anyone's second), so the
budget goes to breadth rather than to the densest, most repetitive corner of the store. Rows are
bare description strings to keep the envelope under host tool-result caps.
"""

from __future__ import annotations

from pipeline.retrieval.rank import fts_phrase

# Ceiling on one call's spread; larger asks are clamped, not refused.
SAMPLE_MAX = 500

# `ORDER BY rn` first is the round-robin. Within a round, prolific authors lead, so they survive
# a budget cut; `who_id`/`atom_id` make the spread deterministic. Each author's atoms go newest-first.
_SAMPLE_SQL = """
SELECT atom_id, description, who_id, when_ts FROM (
    SELECT a.atom_id, a.description, a.who_id, a.when_ts,
           ROW_NUMBER() OVER (PARTITION BY a.who_id
                              ORDER BY a.when_ts DESC, a.atom_id) rn,
           COUNT(*)     OVER (PARTITION BY a.who_id) n_author
    FROM atoms a WHERE 1=1{frag}
)
ORDER BY rn, n_author DESC, who_id, atom_id
LIMIT ?
"""

# Written as instructions to the host: the spread is a sample, and a name read from it is a guess
# until `search` confirms it.
HOST_NOTE = (
    "`descriptions` is a SAMPLE of atom descriptions, spread across authors — not the whole "
    "store, and "
    "not a topic list. To answer what this corpus is about: read it, name the subjects you see "
    "in your own words, then CHECK each one with `search(query=<subject>)` before you say it. "
    "A subject that is really here comes back with strong hits from several authors; a guess "
    "that is not comes back weak and scattered — drop those silently. Report only the subjects "
    "that survived, each with its atoms linked via `cite`, and say how many you sampled of the "
    "total. Do not present the raw sample as a finding, and do not store the names anywhere: "
    "they are your reading of this corpus today, re-derived next time it is asked."
)


def corpus_sample(conn, n: int, frag: str = "", params: tuple | list = ()) -> dict | None:
    """A breadth-first spread of `n` atom descriptions, or None when n <= 0 or the scope is empty.

    `descriptions` holds strings, not openable records. `frag`/`params` are `kb_aggregate`'s scope
    clause, so a scoped census describes that scope. `authors`/`of_authors` say whether the breadth
    worked: 300 samples from 300 authors and from 4 authors are very different evidence.
    """
    n = min(int(n or 0), SAMPLE_MAX)
    if n <= 0:
        return None

    sql = _SAMPLE_SQL.format(frag=frag)
    rows = conn.execute(sql, (*params, n)).fetchall()
    if not rows:
        return None

    total, of_authors = conn.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT a.who_id) FROM atoms a WHERE 1=1{frag}",
        params).fetchone()

    return {
        "descriptions": [r[1] for r in rows],
        "sampled": len(rows),
        "of": total,
        "authors": len({r[2] for r in rows}),
        "of_authors": of_authors,
        "host_note": HOST_NOTE,
    }


def subject_count(conn, subject: str) -> int:
    """How many atoms hold `subject` as an exact phrase. Counted, not ranked: a text sharing a word
    with the subject is not about it (`rank.fts_phrase`)."""
    return conn.execute("SELECT COUNT(*) FROM bodies_fts WHERE bodies_fts MATCH ?",
                        (fts_phrase(subject),)).fetchone()[0]


def saved_count(conn, who_ids: list[str]) -> int:
    """How many atoms by these identities (one person's) the user saved themselves."""
    if not who_ids:
        return 0
    return conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE entry_mode = 'user-saved' AND who_id IN (%s)"
        % ",".join("?" * len(who_ids)), who_ids).fetchone()[0]
