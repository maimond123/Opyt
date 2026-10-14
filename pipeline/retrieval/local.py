"""
pipeline/retrieval/local.py — LocalKB: the reads over one SQLite store.

The store is your own (opened writable, so the idempotent DDL can run and a never-ingested store
answers empty rather than "no such table") or a file peer (opened read-only by
`peers.open_peer`). `reads.open_store` picks which and hands this class the open connection; the
served counterpart is `served.ServedKB`, and both meet `reads.KnowledgeBase`.

Retrieval + data only, no model — the host model reasons over what these return. Opening a paper,
or a post that links papers, on your own store reads their full text into the response
(`kb/paper_text.py`); the store is never written.
"""
from __future__ import annotations

from pipeline.kb import paper_text, schema
from pipeline.retrieval import cards, notices
from pipeline.retrieval import corpus_census as census
from pipeline.retrieval.engine import json_obj, search_atoms
from pipeline.retrieval.filters import (
    applied_filters,
    candidate_atom_ids,
    date_bound,
    entry_mode_clause,
    filter_costs,
    in_clause,
    slug_tags,
)
from pipeline.retrieval.peers import LOCAL_KB
from pipeline.retrieval.who import author_ids

# ── The frontier block ──────────────────────────────────────────────────────────
# A default search ranks its full k over HUMAN_ATTESTED atoms and returns keyword-discovered
# ones beside that list, never inside it. No score is compared across provenances.
FRONTIER_BLOCK_CAP = 8


def _frontier_block(conn, query, *, kb_name, slugs, what_kind, source_type, who_id,
                    d_from, d_to) -> dict:
    """The labeled frontier section of a default search: its own ranked list, capped. Every
    caller filter applies here too."""
    run = search_atoms(conn, query, tags=slugs, what_kind=what_kind,
                       source_type=source_type, who_id=who_id, date_from=d_from, date_to=d_to,
                       entry_mode="frontier", k=FRONTIER_BLOCK_CAP)
    return {"hits": [cards.hit_card(h, kb_name) for h in run.hits], "candidates": run.candidates}


class LocalKB:
    """One open SQLite store, and the name the envelope calls it.

    `name` is the label every `kb` field carries back (already `as_kb`-relabelled by
    `reads.open_store`)."""

    def __init__(self, conn, name: str, label: str | None):
        self.conn = conn
        self.name = name
        self.label = label

    def close(self) -> None:
        self.conn.close()

    def search(self, query: str, *, tags=None, what_kind=None, source_type=None, who_id=None,
               who=None, date_from=None, date_to=None, entry_mode=None, k: int = 8) -> dict:
        """The search over this store — see `reads.run_kb_search` for the envelope."""
        conn, kb_name = self.conn, self.name
        slugs = slug_tags(tags)
        d_from = date_bound(date_from, end=False)
        d_to = date_bound(date_to, end=True)
        who_id, resolved = author_ids(conn, who, who_id)
        # The default scope is HUMAN_ATTESTED; the frontier population is answered beside it.
        scope = list(schema.HUMAN_ATTESTED) if entry_mode is None else entry_mode
        run = search_atoms(conn, query, tags=slugs,
                           what_kind=what_kind, source_type=source_type, entry_mode=scope,
                           who_id=who_id, date_from=d_from, date_to=d_to, k=k)
        block, block_error = None, None
        if entry_mode is None:
            try:
                block = _frontier_block(conn, query, kb_name=kb_name, slugs=slugs,
                                        what_kind=what_kind, source_type=source_type,
                                        who_id=who_id, d_from=d_from, d_to=d_to)
            except Exception as e:      # fail-safe: a second list may not cost the first one
                block, block_error = {"hits": [], "candidates": None}, f"{type(e).__name__}: {e}"
        applied = applied_filters(slugs, what_kind, source_type, who_id, d_from, d_to, entry_mode)
        # `scope`, not `entry_mode`, or every cost is computed over the wrong population.
        costs = filter_costs(conn, tags=slugs, what_kind=what_kind,
                             source_type=source_type, who_id=who_id,
                             date_from=d_from, date_to=d_to, entry_mode=scope)
        if entry_mode is None:
            # Dropping the default scope would return the frontier block, already shown in full.
            costs.pop("entry_mode", None)
        facts = notices.gather_facts(conn, slugs)
        insights = notices.insights(run.hits, resolved=resolved, filter_cost=costs,
                                    corpus_newest=facts.corpus_newest)
        told = notices.query_notices(run, facts, tags=tags, who=who, resolved=resolved,
                                     filter_cost=costs, applied=applied,
                                     kb_name=kb_name, kb_label=self.label)
        out = {"hits": [cards.hit_card(h, kb_name) for h in run.hits], "notices": told,
               "insights": insights, "trace": notices.trace(run, applied, kb_name)}
        if block is not None:
            out["frontier_atoms"] = block
            if block_error:
                told.append(notices.frontier_block_failed(block_error))
        return out

    def open_atom(self, atom_id: str) -> dict:
        """Open one atom of this store — see `reads.kb_open`."""
        conn, kb_name = self.conn, self.name
        # LEFT so an atom with no author entity row still opens; the name only labels `cite`.
        sel = ("SELECT a.atom_id, a.source_type, a.what_kind, a.who_id, a.when_ts, a.when_precision, "
               "a.source_url, a.description, a.payload, e.name AS _who_name "
               "FROM atoms a LEFT JOIN entities e ON e.entity_id = a.who_id WHERE a.atom_id=?")
        row = conn.execute(sel, (atom_id,)).fetchone()
        own = kb_name == LOCAL_KB          # full text is read on your own store only
        if row is None and own and (alias := paper_text.record_id(conn, atom_id)):
            row = conn.execute(sel, (alias,)).fetchone()     # a paper's URL or other id
        if row is None:
            try:
                live = paper_text.open_unstored(atom_id) if own else None
            except Exception:
                live = None       # a paper we could not read live is simply not found
            return {**live, "kb": kb_name} if live else {
                "atom_id": atom_id, "kb": kb_name, "error": "not found"}
        raw = schema.read_body(conn, row["atom_id"])
        payload = json_obj(row["payload"])   # same decode as a search hit, via the same helper
        body_state = payload.pop("body_state", None)
        body_basis = payload.pop("body_basis", None)
        papers = None
        try:
            if own and row["source_type"] == "paper":
                # The record holds the abstract; the full text is read now and never stored. An
                # older record may hold the whole text already.
                if raw is not None and body_state != "complete" and (text := paper_text.of_record(row)):
                    block, clipped = paper_text.section(text, paper_text.BUDGET)
                    raw += block
                    body_state = "partial" if clipped else "complete"
            elif own:
                papers = paper_text.linked(conn, row["atom_id"], raw) or None
        except Exception:
            pass          # an open must return the body we hold, whatever the paper read did
        out = {
            "atom_id": row["atom_id"],
            "kb": kb_name,                            # which store this body came from
            "source_type": row["source_type"],
            "what_kind": row["what_kind"],
            "who_id": row["who_id"],
            "when_ts": row["when_ts"],
            "when_precision": row["when_precision"],
            "source_url": row["source_url"],          # the live pointer (re-fetch for the freshest)
            # The same pre-rendered link a hit carries.
            "cite": cards.cite(row["source_url"], row["_who_name"], row["source_type"], row["when_ts"]),
            "description": row["description"],
            "raw": raw,                               # the stored snapshot — assert from this
            "raw_available": raw is not None,
            "body_state": body_state,   # "partial" → do not present raw as the whole thing
            "body_basis": body_basis,
            "payload": payload,                       # source-shaped extras, verbatim
        }
        if papers:
            out["papers"] = papers                    # the post's papers, full text within budget
        return out

    def aggregate(self, scope: dict, sample: int = 0) -> dict:
        """The aggregate over this store — see `reads.kb_aggregate`."""
        conn, kb_name = self.conn, self.name
        tags = slug_tags(scope.get("tags"))
        what_kind = scope.get("what_kind")
        source_type = scope.get("source_type")
        who_id = scope.get("who_id")
        d_from = date_bound(scope.get("date_from"), end=False)
        d_to = date_bound(scope.get("date_to"), end=True)
        cand = candidate_atom_ids(conn, tags, what_kind, source_type, who_id, d_from, d_to)
        if cand is not None and not cand:
            return {"scope": scope, "kb": kb_name, "total": 0,
                    "notices": [{"code": "scope_matched_nothing",
                                 "message": "No atoms match this scope."}]}
        frag, params = in_clause(cand, "a")
        base = "FROM atoms a WHERE 1=1" + frag

        total = conn.execute(f"SELECT COUNT(*) {base}", params).fetchone()[0]
        by_kind = {(r[0] or "?"): r[1] for r in conn.execute(
            f"SELECT a.what_kind, COUNT(*) {base} GROUP BY a.what_kind", params)}
        by_source = {(r[0] or "?"): r[1] for r in conn.execute(
            f"SELECT a.source_type, COUNT(*) {base} GROUP BY a.source_type", params)}
        # Counted off `oracles`, the record of who the user confirmed.
        trusted = conn.execute(
            f"SELECT COUNT(*) FROM atoms a "
            f"JOIN entities e ON e.entity_id = a.who_id "
            f"JOIN (SELECT DISTINCT head FROM ({schema.ROSTER_HEADS_SQL})) o "
            f"ON o.head = COALESCE(e.canonical_id, e.entity_id) "
            f"WHERE 1=1{frag}", params).fetchone()[0]
        entities = [{"who_id": r[0], "count": r[1]} for r in conn.execute(
            f"SELECT a.who_id, COUNT(*) c {base} GROUP BY a.who_id ORDER BY c DESC LIMIT 15",
            params)]
        top = [{"atom_id": r[0], "description": r[1], "who_id": r[2], "when_ts": r[3]}
               for r in conn.execute(
                   f"SELECT a.atom_id, a.description, a.who_id, a.when_ts {base} "
                   f"ORDER BY a.when_ts DESC LIMIT 12", params)]
        # `recent_descriptions` is what is new, not a census; the census is a separate draw.
        spread = census.corpus_sample(conn, sample, frag, params)

        return {
            "scope": scope,
            "kb": kb_name,                     # which store these counts describe
            "notices": [],                     # same shape as search's, so a host reads one channel
            "total": total,
            "by_what_kind": by_kind,
            "by_source_type": by_source,
            "trusted_atoms": trusted,          # atoms whose author is a confirmed Oracle
            "top_entities": entities,
            "recent_descriptions": top,        # mechanical cards; open() to ground any claim
            **({"corpus_sample": spread} if spread else {}),
        }

    def feed(self, *, since=None, since_id=None, limit: int = 50, entry_mode=None,
             date_from=None, date_to=None) -> dict:
        """The arrivals feed of this store — see `reads.kb_feed`."""
        conn, kb_name = self.conn, self.name
        modes = ([entry_mode] if isinstance(entry_mode, str)
                 else list(entry_mode) if entry_mode else list(schema.HUMAN_ATTESTED))
        limit = max(1, limit)
        d_from = date_bound(date_from, end=False)
        d_to = date_bound(date_to, end=True)
        cand = candidate_atom_ids(conn, None, None, None, None, d_from, d_to)
        if cand is not None and not cand:
            return {"kb": kb_name, "notices": [], "items": [],
                    "next": ({"since": since, "since_id": since_id} if since else None),
                    "more": False}
        frag, cparams = in_clause(cand, "a")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(atoms)")}
        seen = ("COALESCE(a.promoted_at, a.first_seen)" if "promoted_at" in cols
                else "a.first_seen")
        where = entry_mode_clause(modes) + frag
        params: list = [*modes, *cparams]
        if since:
            where += f" AND ({seen} > ? OR ({seen} = ? AND a.atom_id > ?))"
            params += [since, since, since_id or ""]
            order, probe = "ASC", limit + 1     # the +1 is the `more` probe
        else:
            order, probe = "DESC", limit        # seed page: newest N, nothing "more" by design
        rows = conn.execute(
            f"SELECT a.atom_id, a.description, a.source_url, a.who_id, e.name, a.when_ts, "
            f"{seen}, a.source_type, a.what_kind, a.entry_mode "
            f"FROM atoms a LEFT JOIN entities e ON e.entity_id = a.who_id "
            f"WHERE {where} ORDER BY {seen} {order}, a.atom_id {order} LIMIT ?",
            [*params, probe]).fetchall()
        more = since is not None and len(rows) > limit
        rows = rows[:limit]
        if not since:
            rows.reverse()                          # one shape: the body is always ascending
        items = [{"atom_id": r[0], "description": r[1], "source_url": r[2], "who_id": r[3],
                  "who": r[4], "when_ts": r[5], "seen_at": r[6], "source_type": r[7],
                  "what_kind": r[8], "entry_mode": r[9]} for r in rows]
        nxt = ({"since": items[-1]["seen_at"], "since_id": items[-1]["atom_id"]} if items
               else {"since": since, "since_id": since_id} if since else None)
        return {"kb": kb_name, "notices": [], "items": items, "next": nxt, "more": more}
