"""
pipeline/kb/corpus_census.py — a representative spread of the store, for the host to read.

WHAT QUESTION THIS EXISTS FOR. "What topics does my corpus cover?" is a CENSUS-class question:
it wants a distribution over the whole store, not the top-k neighbours of one query. Nothing in
the retrieval stack answers it, and that is structural rather than an oversight — `search` and
`sitting` both need a query, so the index can VERIFY a subject and can never PROPOSE one.

⚠️ THIS IS DELIBERATELY NOT A TOPIC LAYER, AND THE DIFFERENCE IS THE WHOLE DESIGN.
A clustering job over `chunks.vector` that names its clusters and stores the labels was built,
live-validated and reverted on 2026-08-06 (`docs/plans/2026-08-06-stage6-taxonomy-retired.md`).
It is banned by `.guards.py::retired-about-topics-column`, and re-landing it would overturn the
no-precomputed-insight invariant — *the store admits only machine-verifiable records;
interpretations are query-time and disposable.*

So this module computes NO grouping, invents NO label and writes NOTHING. It hands back atoms
the store already holds, and the naming happens in the host's response — where an interpretation
is disposable, re-derived next time, and answerable to whoever reads it. The host's half of the
loop is the part that is not in this file:

    1. `aggregate(sample=N)`  → this spread, plus the store's counts.
    2. the host names candidate subjects from reading it — its own words, in its own answer.
    3. `search(query=<candidate>)` per candidate → real hits, real authors, real scores, each
       one openable. A guessed subject that is not really there retrieves weak scattered hits;
       a real one retrieves tight multi-author ones.
    4. only survivors are reported, each carrying the atoms that back it.

STEP 3 IS WHAT MAKES THIS HONEST rather than Stage 6 with extra steps. Stage 6's labels were
unfalsifiable once written — every consumer surfaced the slug and none the geometry, so
`ai-labs` / `ai-industry-insights` / `artificial-intelligence` looked like three findings. A
name produced here is checked against the index BEFORE the user sees it, and arrives with the
atoms that justify it. Same instinct, opposite side of the invariant.

WHY ROUND-ROBIN OVER AUTHORS AND NOT THE DENSEST MASS. The tempting sample is the busiest part
of the store, and it is the wrong one: the densest region is the most REPETITIVE, which is why
`densest_unread` was deleted (see `docs/plans/2026-08-16-*`, and D13 in
`tests/kb/test_sitting_builder.py`). On the live store 1,452 of 1,801 atoms are from X and the
top voices are prolific, so a straight `LIMIT` or a random draw both return one loud corner and
the host names that corner as the corpus. Walking authors in rounds — everyone's first atom
before anyone's second — spends the budget on BREADTH, which is what a census is for.

The round-robin also needs no per-author cap, because it self-balances against any store shape:

  • 714 authors, n=300 → 300 authors, one atom each. Maximum breadth.
  • 40 authors, n=300  → ~7 atoms each, taken in rounds. Proportionate.
  • 1 author, n=300    → 300 of their atoms. There is no breadth to find.

⚠️ THE ROWS ARE BARE DESCRIPTION STRINGS, AND THE FIRST LIVE RUN IS WHY. Until 2026-09-16 each
row was `{atom_id, description, who_id, when_ts}`. On the first real call — `sample=300` against a
1,308-atom store — that envelope was **88,429 characters** and overflowed the host's tool-result
cap before the model saw any of it. The host recovered by shelling out and running

    jq -r '.corpus_sample.atoms[] | .description'

— descriptions only, in two chunks, and it never read `atom_id`, `who_id` or `when_ts` once. So
the model itself measured which three quarters of each row were dead weight. Dropping them cuts
39% (88,429 → 53,981 at n=300), and nothing was lost: the description already opens with the
author's handle and closes with the date, and every CITE in the answer comes from `search`, not
from here. A census proposes; it is not a set of records to open.

THAT OVERFLOW IS ALSO WHY THE RECOMMENDED SIZE IS 200, NOT 300. The recovery above needed a
SHELL, which is a Claude-Code-harness capability and not an MCP one — on Cursor, Windsurf, or a
plain Desktop install there is nothing to recover with. A default that only survives on one client
violates the Client-agnostic invariant, so the recommended call has to fit without rescue.

IT IS A SAMPLE, AND CALLERS MUST SAY SO. `sampled` and `of` ride the result for exactly that
reason — a spread read as a complete listing is the same failure `top_topics` was deleted for
(a partial view that reads like a finding). What upgrades a guess drawn from the sample into
something checkable is step 3, not a bigger sample.
"""

from __future__ import annotations

# Ceiling on one call's spread. ~137 chars per description on the live store, so 500 is roughly
# 68KB / 17k tokens — already a large share of a context window, and a host that asks for more
# is not reading them, it is drowning in them. Clamped rather than raising: an over-large ask is
# a host misjudging a budget, not an error worth failing a read-only call over.
SAMPLE_MAX = 500

# One query, not a fan-out per author: `ROW_NUMBER` numbers each author's atoms and `COUNT`
# sizes each author, then a single ORDER BY interleaves them.
#
# ORDER BY rn FIRST is the round-robin — it is what makes this a census rather than a top-N.
# Within a round, prolific authors lead (`n_author DESC`): when the budget cuts a round short,
# the voices that make up more of the store are the ones that survive the cut, which is the one
# place density SHOULD win. `who_id`/`atom_id` break remaining ties so the same store returns
# the same spread twice — a census that reshuffles every call cannot be checked against itself.
#
# Each author's own atoms are ordered newest-first. That is a recency bias WITHIN an author, and
# it is real but small: it only bites on stores narrow enough to give one author many rounds,
# which are exactly the stores where the subject is obvious anyway.
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

# Carried on the result rather than left to each surface to phrase, because every caller that
# ships this spread has the same two things to get right: that it is a SAMPLE, and that a name
# read out of it is a guess until the index confirms it. Written as instructions to the host,
# in the imperative, for the same reason `suggest`'s strings are: a model expands what it is
# handed, and a finished sentence about the user's corpus would get repeated verbatim.
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
    """A breadth-first spread of `n` atom descriptions, or None when there is nothing to say.

    `descriptions` is a list of STRINGS, not rows — no `atom_id`, `who_id` or `when_ts`. That is
    a deliberate trim with a measurement behind it (module header), and it is also why the key is
    not called `atoms`: they are descriptions, and a key promising atoms would invite a caller to
    treat them as openable records.

    `frag`/`params` are `kb_aggregate`'s already-built scope clause, threaded through unchanged
    so a scoped census describes the scope the caller asked about rather than the whole store.

    `authors`/`of_authors` are reported next to `sampled`/`of` because they are the number that
    says whether the breadth worked. "300 of 1801, 300 authors of 714" and "300 of 1801, 4
    authors of 714" are the same sample size and completely different evidence, and only the
    second one should make a reader distrust what the host names from it.

    Returns None rather than an empty spread for n <= 0 or an empty scope: the key is absent
    from the envelope entirely, which is a smaller lie than a present-but-empty census.
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
        # DESCRIPTIONS, NOT ROWS — see the module header for the measurement that decided it.
        "descriptions": [r[1] for r in rows],
        "sampled": len(rows),
        "of": total,
        "authors": len({r[2] for r in rows}),
        "of_authors": of_authors,
        "host_note": HOST_NOTE,
    }
