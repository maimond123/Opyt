"""
pipeline/topics/lists.py — the three lists a host writes for a topic, how to write each, and the
checks on what it sends.

  watches  names searched outside the store (arXiv, OpenAlex, GitHub, X, Substack), one specific
           thing each;
  words    phrases that pick out the user's own people's posts on the subject;
  people   the user's Oracles who mostly write about it, every new post of theirs checked.

How to write each is measured, not guessed and rides the
`watch` tool's description, so the host reads it before writing a topic. The checks decide from
values, except `people`, which resolves handles against the store and never reaches the network.
"""
from __future__ import annotations

import re

from pipeline.frontier import queries as fq
from pipeline.retrieval.rank import fts_phrase
from pipeline.retrieval.who import resolve_who

RULES = """\
WATCHES name what to search outside the store, on a schedule: arXiv, OpenAlex, GitHub, X, Substack.
- Each names ONE specific thing: a model, drug, trial, benchmark, dataset, protocol, a lab's \
program ("Boltz-2", "rentosertib", "Virtual Cell Challenge", "x402"). Never a subject or a \
category ("AI drug discovery"), and no aspect words ("Boltz-2 binding affinity" is "Boltz-2"): a \
find is kept only when the name appears in its title or text, so extra words lose real finds \
and a category fills the check with near-misses (measured: 13% off-topic for names, 44% for \
descriptive queries).
- Give its other spellings in `spellings` ("Boltz2"); each is searched too.
- `sources`: arxiv for preprints, openalex for published work in any field (it indexes arXiv \
too, so one is enough for a CS paper), github for code, x for posts on X (news and reactions, \
often days before any paper; searched on the user's own X session, so only while X is \
connected), substack for newsletter posts. Leave it out to search all five.
- `context`, for X only: one or two words any on-topic post would carry. Give it when the name \
is also an idiom, a common word or another thing: "Chai-2" alone finds football and needs \
"antibody"; "Canaries in the Coal Mine" needs "AI" (measured: 7 of 12 off topic without it, 5 of \
5 on topic with it). A name that is busy on X ("x402", "GDPval") needs none: X's Top tab caps \
it at the 20 posts it ranks first each day.
WORDS pick out the user's own people's posts on the subject: 15 to 20 short phrases people \
actually write when they post about it, broad on purpose ("drug discovery", "clinical trial", \
"wet lab", "protein design", "perturbation"). Their posts are few, about 17 a day, and you check \
every match before showing it, so recall matters more than precision: careful lists of 9 or 10 \
missed 29% to 56% of the relevant posts.
PEOPLE are the 2 to 4 of the user's Oracles who mostly write about it, by handle. Every new post \
of theirs goes to your check, because a quarter of the posts on an argued subject share none of \
its words (recall rose from 44% to 94% with them)."""

# A watch names one thing. More words than this is a description, which the name check punishes.
MAX_NAME_WORDS = 6

_WORD = re.compile(r"[A-Za-z0-9]+")


def as_list(value) -> list:
    """A list, from a list or one bare value."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def strings(value) -> list[str]:
    return [v.strip() for v in as_list(value) if isinstance(v, str) and v.strip()]


def watch_rows(raw, every_source) -> tuple[list[dict], list[str]]:
    """`([{name, text, sources, context, query_id}], problems)`: one row per spelling of each
    watch, its name first. A watch searches every source unless it names some."""
    rows, problems = [], []
    for i, w in enumerate(as_list(raw)):
        w = {"name": w} if isinstance(w, str) else w if isinstance(w, dict) else {}
        name = w.get("name").strip() if isinstance(w.get("name"), str) else ""
        chosen = strings(w.get("sources")) or list(every_source)
        if not _WORD.search(name):
            problems.append(f"watches[{i}] needs a name.")
            continue
        if len(_WORD.findall(name)) > MAX_NAME_WORDS:
            problems.append(f"watches[{i}] {name!r} describes rather than names: a watch names one "
                            f"thing in at most {MAX_NAME_WORDS} words.")
        if bad := [s for s in chosen if s not in every_source]:
            problems.append(f"watches[{i}] names unknown sources {', '.join(bad)}; a watch "
                            f"searches {', '.join(every_source)}.")
        context = strings(w.get("context"))
        for text in fq.spellings(name, strings(w.get("spellings"))):
            rows.append({"name": name, "text": text, "sources": chosen, "context": context,
                         "query_id": fq.query_id_for(fq.normalize(text))})
    return rows, problems


def phrases(raw) -> tuple[list[str], list[str]]:
    """`(words, problems)`: the subject words, deduplicated, each holding a word to match."""
    out = list(dict.fromkeys(strings(raw)))
    return out, [f"{w!r} holds no word to match." for w in out if fts_phrase(w) == '""']


def persons(conn, raw) -> tuple[list[dict], list[str]]:
    """`([{name, who_ids}], problems)`: each handle resolved against the store to one person."""
    out, problems = [], []
    for handle in strings(raw):
        found = resolve_who(conn, handle)
        if not found:
            problems.append(f"Nobody in the store matches {handle!r}. Name someone whose posts "
                            "are already here, or leave them out.")
        elif len(found) > 1:
            names = "; ".join(c["name"] or c["canonical_id"] for c in found)
            problems.append(f"{handle!r} matches more than one person ({names}); use their "
                            "exact handle.")
        else:
            out.append({"name": found[0]["name"] or handle, "who_ids": found[0]["who_ids"]})
    return out, problems


def draft(conn, *, name, watches, words, people, every_source) -> tuple[dict, list[str]]:
    """A whole topic as the host sent it, read and checked: `({name, rows, words, people},
    problems)`."""
    name = name.strip() if isinstance(name, str) else ""
    rows, problems = watch_rows(watches, every_source)
    words, more = phrases(words)
    people, still = persons(conn, people)
    problems += more + still
    if not name:
        problems.insert(0, "A topic needs a name: the subject in the user's words.")
    if not (rows or words or people):
        problems.append("A topic needs watches, words or people.")
    return {"name": name, "rows": rows, "words": words, "people": people}, problems
