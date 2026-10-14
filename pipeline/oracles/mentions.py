"""pipeline/oracles/mentions.py — answer "who do my Oracles engage with about X" from their posts.

The candidates' third arm, beside the probe arm and the saved-atom arm (`candidates.py`). It
searches the Oracles' own X posts, already stored, and returns the people those posts quote,
@mention or link (`payload.engages`, written at ingest by `kb/post_people.py`), with the Oracle's
own words as evidence. No network: it adds recall for people nothing of the user's points at, and
it never decides stance. The host reads the words and judges.

Credit is split across the people one post names, so a paper post tagging twelve affiliations
gives each a twelfth; a quote counts more than a bare mention; and being named by several Oracles
boosts a person without being required.
"""

from __future__ import annotations

import json
import sqlite3

from pipeline.kb import derive, schema
from pipeline.oracles import roster
from pipeline.retrieval import engine
from pipeline.retrieval.rank import rrf_term
from pipeline.retrieval.who import resolve_who

KIND_WEIGHT = {"quote": 1.0, "link": 0.75, "mention": 0.5}

# Each further Oracle who named the person adds this fraction of their score.
SEVERAL_ORACLES_BOOST = 0.25

PROVENANCE = ("TRUSTED, but ABOUT this person, not BY them: an Oracle's own post that "
              "quotes / mentions / links them. Read the Oracle's words to judge whether it agrees "
              "or pushes back. Citable as what the Oracle said, never as what this person thinks.")


def credit(base: float, kind: str, n_people: int) -> float:
    """One post's credit to one person it names: the post's own score (`base`), weighted by how
    it named them and split across everyone the post names."""
    return base * KIND_WEIGHT.get(kind, KIND_WEIGHT["mention"]) / max(n_people, 1)


def boost(score: float, n_oracles: int) -> float:
    """A person's summed credit, raised for each further Oracle that named them."""
    return score * (1 + SEVERAL_ORACLES_BOOST * max(n_oracles - 1, 0))


def oracle_authors(conn: sqlite3.Connection, oracle: str | None = None) -> dict:
    """The Oracles' X identities → `{"authors": {x who_id: label}, "exclude": set}`. `authors` is
    narrowed to one Oracle when `oracle` names one (any Oracle, with or without X), whose
    canonical id is then `head`; `exclude` is always every Oracle (their own X ids and handles),
    since an Oracle is never a candidate. An unknown or ambiguous `oracle`
    returns `{"error": ...}` with the choices."""
    everyone: dict[str, tuple[str, str]] = {}          # x who_id → (head, label)
    exclude: set[str] = set()
    heads_by_label: dict[str, str] = {}
    for o in roster.confirmed_oracles(conn):
        rows = schema.entities_for_canonical(conn, o["canonical_id"])
        name, handle = derive.best_name([dict(r) for r in rows])
        label = handle or o["name"] or name or o["canonical_id"]
        heads_by_label[label] = o["canonical_id"]
        for r in rows:
            if r["entity_id"].startswith("x:user:"):
                everyone[r["entity_id"]] = (o["canonical_id"], label)
                exclude.add(r["entity_id"])
        if handle:
            exclude.add(f"x:@{handle.lower()}")
    if not oracle:
        return {"authors": {w: label for w, (_, label) in everyone.items()}, "exclude": exclude}

    wanted = {c["canonical_id"] for c in resolve_who(conn, oracle)}
    heads = set(heads_by_label.values())
    hits = {schema.current_canonical(conn, w) for w in wanted} & heads
    hits |= {head for label, head in heads_by_label.items()
             if label.casefold() == oracle.lstrip("@").casefold()}
    if len(hits) != 1:
        return {"error": (f"no Oracle matches {oracle!r}" if not hits else
                          f"{oracle!r} matches several Oracles; pass one of their handles"),
                "oracles": sorted(heads_by_label)}
    head = hits.pop()
    return {"authors": {w: label for w, (h, label) in everyone.items() if h == head},
            "exclude": exclude, "head": head}


def _pool(conn: sqlite3.Connection, authors: dict[str, str]) -> dict[str, dict]:
    """The Oracles' X posts that name anyone → `{atom_id: {who_id, when, engages}}`."""
    if not authors:
        return {}
    rows = conn.execute(
        "SELECT atom_id, who_id, when_ts, json_extract(payload, '$.engages') AS engages "
        "FROM atoms WHERE entry_mode = 'oracle-footprint' AND source_type = 'x' "
        "AND who_id IN (%s) AND json_array_length(payload, '$.engages') > 0"
        % ",".join("?" * len(authors)), sorted(authors)).fetchall()
    return {r["atom_id"]: {"who_id": r["who_id"], "when": r["when_ts"],
                           "engages": json.loads(r["engages"])} for r in rows}


def search_mentions(conn: sqlite3.Connection, query: str, *,
                    authors: dict[str, str], exclude: set[str], k: int | None,
                    evidence: int) -> list[dict]:
    """People the `authors`' posts engage with, best match to `query` first, each with the
    Oracle's words. An empty query ranks by how much the Oracles engage with them overall.
    `k=None` returns everyone named."""
    pool = _pool(conn, authors)
    if not pool:
        return []
    # A link carries only a handle; the same handle quoted or mentioned elsewhere carries the id.
    id_by_handle = {e["handle"].lower(): e["who"] for post in pool.values()
                    for e in post["engages"] if e.get("who")}

    def key(e: dict) -> str:
        return e.get("who") or id_by_handle.get(e["handle"].lower()) or f"x:@{e['handle'].lower()}"

    if (query or "").strip():
        span = max(k * evidence * 4, 80)
        ranked = [(hit.atom_id, rrf_term(rank)) for rank, hit in
                  enumerate(engine.atom_bm25_search(conn, query, set(pool), span))]
    else:
        ranked = [(aid, 1.0) for aid in sorted(pool, key=lambda a: pool[a]["when"] or "",
                                              reverse=True)]

    scores: dict[str, float] = {}
    named_by: dict[str, set[str]] = {}
    passages: dict[str, list[dict]] = {}
    handles: dict[str, str] = {}
    for atom_id, base in ranked:
        post = pool[atom_id]
        people = {key(e): e for e in post["engages"]}
        label = authors[post["who_id"]]
        for person, e in people.items():
            if person in exclude or f"x:@{e['handle'].lower()}" in exclude:
                continue
            scores[person] = scores.get(person, 0.0) + credit(base, e["kind"], len(people))
            named_by.setdefault(person, set()).add(label)
            handles.setdefault(person, e["handle"])
            passages.setdefault(person, []).append(
                {"oracle": label, "kind": e["kind"], "words": e["words"], "url": e["url"],
                 "when": post["when"], "provenance": PROVENANCE})

    final = {p: boost(s, len(named_by[p])) for p, s in scores.items()}
    top = sorted(final, key=lambda p: final[p], reverse=True)[:k]
    names = _names(conn, [p for p in top if p.startswith("x:user:")])
    out = []
    for p in top:
        seen, ev = set(), []
        for item in passages[p]:
            if item["url"] in seen:
                continue
            seen.add(item["url"])
            ev.append(item)
            if len(ev) >= evidence:
                break
        out.append({"who_id": p, "handle": handles[p], "name": names.get(p) or handles[p],
                    "match_score": round(final[p], 5), "oracles": sorted(named_by[p]),
                    "evidence": ev})
    return out


def named_by_oracles(conn: sqlite3.Connection) -> list[dict]:
    """Everyone the Oracles' posts name, most engaged-with first: the screen's "Named by your
    Oracles". Read live on every screen and never written as curation signals: the probe scan
    queues every ranked candidate, so a signal row would queue each of these people for it."""
    named = oracle_authors(conn)
    return search_mentions(conn, "", authors=named["authors"], exclude=named["exclude"],
                           k=None, evidence=1)


def named_counts(named: list[dict]) -> dict[str, int]:
    """`named_by_oracles` rows → how many people each Oracle's posts name, by Oracle label."""
    counts: dict[str, int] = {}
    for row in named:
        for label in row["oracles"]:
            counts[label] = counts.get(label, 0) + 1
    return counts


def _names(conn: sqlite3.Connection, who_ids: list[str]) -> dict[str, str]:
    if not who_ids:
        return {}
    return {r["entity_id"]: r["name"] for r in conn.execute(
        "SELECT entity_id, name FROM entities WHERE entity_id IN (%s)"
        % ",".join("?" * len(who_ids)), who_ids) if r["name"]}
