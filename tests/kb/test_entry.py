"""pipeline.retrieval.reads entry points — the read surface the MCP tools wrap. Exercised over the
$OPYT_HOME sandbox with the no-API paths (bm25 search / open / aggregate), so no paid call."""
from __future__ import annotations

import pytest

from pipeline.retrieval import reads as kb_entry
from pipeline.kb import schema
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.ingest_common import snapshot_hash


def _seed(conn):
    for atom_id, who, topics, snap in [
        ("github:root/agentkit", "github:root", ["ai-agents"],
         "an autonomous agent framework with tools"),
        ("github:stranger/x", "github:stranger", ["ai-agents"], "an agent framework library"),
        ("x:1", "x:user:1", ["crypto"], "thoughts on rollup and proof systems"),
    ]:
        raw_hash = snapshot_hash(snap)
        atom = dict(atom_id=atom_id,
                    source_type="github" if atom_id.startswith("github") else "x",
                    what_kind="artifact" if atom_id.startswith("github") else "opinion",
                    who_id=who, when_ts="2024-05-01", when_precision="day",
                    about_entities=[], source_url=f"https://e/{atom_id}",
                    raw_hash=raw_hash, description=f"{atom_id} card",
                    payload={"source_tags": topics}, entry_mode="user-saved")
        store_atom(conn, atom, snap)
    # `trusted_atoms` counts atoms whose author is a CONFIRMED ORACLE. The fixture states
    # that fact directly.
    schema.upsert_entity(conn, "github:root", name="root")
    conn.execute("INSERT OR IGNORE INTO oracles (canonical_id, name, source) VALUES (?,?,?)",
                 ("github:root", "root", "screen"))
    conn.commit()



def test_trusted_atoms_counts_an_oracle_whose_head_moved(kb_home):
    """A merge leaves the roster row on its confirm-time id. The trusted count joins on
    the folded head, or it misses the Oracle (80 against a true 884 on a real store)."""
    conn = schema.connect(); _seed(conn)
    schema.upsert_entity(conn, "blog:root.dev", name="root")
    schema.set_canonical_ids(conn, {"github:root": "blog:root.dev", "blog:root.dev": "blog:root.dev"})
    conn.close()
    assert kb_entry.kb_aggregate({"tags": ["ai-agents"]})["trusted_atoms"] == 1

def test_search_entry_is_keyword_only_and_needs_no_api(kb_home):
    conn = schema.connect(); _seed(conn); conn.close()
    # run_kb_search is one BM25 arm over the bodies index: no model, no API, pure FTS.
    hits = kb_entry.run_kb_search("framework", tags=["ai-agents"], k=8)["hits"]
    ids = {h["atom_id"] for h in hits}
    assert ids == {"github:root/agentkit", "github:stranger/x"}
    assert all("snippet" in h and "source_url" in h for h in hits)


def test_open_returns_real_raw_and_pointer(kb_home):
    conn = schema.connect(); _seed(conn); conn.close()
    got = kb_entry.kb_open("github:root/agentkit")
    assert got["raw_available"] is True
    assert "autonomous agent framework" in got["raw"]       # the REAL snapshot, not the card
    assert got["source_url"] == "https://e/github:root/agentkit"
    # Unknown id fails safe.
    assert kb_entry.kb_open("github:nope/nope")["error"] == "not found"


def test_aggregate_skeleton_counts_and_trust(kb_home):
    conn = schema.connect(); _seed(conn); conn.close()
    agg = kb_entry.kb_aggregate({"tags": ["ai-agents"]})
    assert agg["total"] == 2
    assert agg["by_source_type"] == {"github": 2}
    assert agg["by_what_kind"] == {"artifact": 2}
    assert agg["trusted_atoms"] == 1                          # only the Oracle-authored atom
    assert len(agg["recent_descriptions"]) == 2
    # The scope FILTERED on `tags` above and the result reports no tag space of its own: an
    # atom's declared labels stay a way IN, and are not an answer OUT.
    assert "top_topics" not in agg

    # Empty scope → whole store (all three atoms).
    assert kb_entry.kb_aggregate()["total"] == 3


# ── who= : the handle path end to end ─────────────────────────────────────────

def _cross_platform_person(conn):
    """`github:root` and `x:user:1` are the SAME person. The seeded corpus gives each of them one
    atom, so only the cluster gets both.

    The `blog:` row is not decoration — it is what makes them merge. Stage-3 unions on
    `self ∩ attests`: someone must BE the home URL for the two who LINK to it to join. Two
    entities that merely both link to the same site stay separate, because two different people
    can each link to one company page."""
    home = "https://root.example.com"
    schema.upsert_entity(conn, "blog:root.example.com", name="Root McRepo", identity_links=[home])          # ← the `self` anchor
    schema.upsert_entity(conn, "github:root", name="Root McRepo", identity_links=[home])
    schema.upsert_entity(conn, "x:user:1", name="Root McRepo", identity_links=[home], profile={"handle": "root"})
    from pipeline.kb import resolve
    resolve.resolve_entities(conn)


def test_search_by_handle_spans_every_platform_that_person_publishes_on(kb_home):
    conn = schema.connect(); _seed(conn); _cross_platform_person(conn); conn.close()
    hits = kb_entry.run_kb_search("agent OR rollup", who="@root", k=8)["hits"]
    # Their GitHub repo AND their X post — one handle, both platforms.
    assert {h["atom_id"] for h in hits} == {"github:root/agentkit", "x:1"}


def test_search_by_handle_excludes_everyone_else(kb_home):
    conn = schema.connect(); _seed(conn); _cross_platform_person(conn); conn.close()
    hits = kb_entry.run_kb_search("framework", who="@root", k=8)["hits"]
    assert {h["atom_id"] for h in hits} == {"github:root/agentkit"}   # NOT github:stranger/x


def test_an_unresolvable_handle_returns_nothing_not_the_whole_store(kb_home):
    """The failure that matters. `who` resolving to nobody must NOT widen back to an unfiltered
    search — those hits would be presented to the user as that person's work."""
    conn = schema.connect(); _seed(conn); conn.close()
    assert kb_entry.run_kb_search("framework", who="@nobody", k=8)["hits"] == []


def test_search_reports_the_resolution_so_an_empty_result_is_readable(kb_home):
    """The fact that MOVED. `[]` alone cannot distinguish "we don't track this person" from
    "we track them and they wrote nothing about this" — two different next actions. That used
    to be readable only by making a SECOND call to `aggregate(who=...)`, which you had to
    already suspect the problem to make. It rides along with the search that raised the
    question now."""
    conn = schema.connect(); _seed(conn); _cross_platform_person(conn); conn.close()

    tracked = kb_entry.run_kb_search("framework", who="@root", k=8)
    assert [c["name"] for c in tracked["insights"]["resolved_who"]] == ["Root McRepo"]
    assert set(tracked["insights"]["resolved_who"][0]["who_ids"]) == {
        "blog:root.example.com", "github:root", "x:user:1"}

    untracked = kb_entry.run_kb_search("framework", who="@nobody", k=8)
    assert untracked["hits"] == []
    assert untracked["insights"]["resolved_who"] == []               # ← "we don't have them"
    assert [n["code"] for n in untracked["notices"]] == ["who_unresolved"]
    assert "@nobody" in untracked["notices"][0]["message"]


def test_a_query_with_no_who_carries_no_resolution_key_at_all(kb_home):
    """ABSENT, not empty. An empty `resolved_who` MEANS "that handle matched nobody", so a
    query that never asked about a person must not carry one — a phantom key that reads as a
    failed lookup is worse than no key."""
    conn = schema.connect(); _seed(conn); conn.close()
    out = kb_entry.run_kb_search("framework", k=8)
    assert "resolved_who" not in out["insights"]


def test_aggregate_scopes_by_the_ids_search_resolved(kb_home):
    """The two-call workflow that replaces `aggregate(who=...)`: search resolves the handle,
    aggregate takes the ids. `aggregate` does not resolve handles, so there is exactly
    one resolver reachable from the tool surface."""
    conn = schema.connect(); _seed(conn); _cross_platform_person(conn); conn.close()

    ids = kb_entry.run_kb_search("framework", who="@root",
                                 k=8)["insights"]["resolved_who"][0]["who_ids"]
    scoped = kb_entry.kb_aggregate({"who_id": ids})
    assert scoped["total"] == 2                                      # both platforms' atoms
    assert scoped["notices"] == []

    empty = kb_entry.kb_aggregate({"who_id": ["x:user:nobody"]})
    assert empty["total"] == 0
    assert [n["code"] for n in empty["notices"]] == ["scope_matched_nothing"]


def test_who_and_who_id_union_rather_than_intersect(kb_home):
    # Both mean "restrict to these authors", so a caller holding a handle for one person and an
    # id for another wants both people — an intersection would return nothing.
    conn = schema.connect(); _seed(conn); _cross_platform_person(conn); conn.close()
    hits = kb_entry.run_kb_search("framework", who="@root", who_id="github:stranger", k=8)["hits"]
    assert {h["atom_id"] for h in hits} == {"github:root/agentkit", "github:stranger/x"}


# ── open() reads a paper's full text, and keeps none of it ────────────────────────
# A record holds its abstract;
# opening it reads the full text into that response, cached as a file outside search, never
# indexed or written to the store. A post's open carries the papers it links.

def _seed_abstract_only_paper(conn, monkeypatch):
    from pipeline.kb import ingest_papers as ip
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: None)
    paper = {"paperId": "arXiv:2401.99999", "title": "A Paper Nobody Has Read Yet",
             "abstract": "Two thousand characters of abstract and no body.",
             "authors": [{"authorId": "9", "name": "Ada Scholar"}],
             "year": 2024, "publicationDate": "2024-03-02", "venue": "ICML",
             "citationCount": 3, "url": "https://doi.org/10.1/xyz",
             "externalIds": {"ArXiv": "2401.99999"}}
    assert ip.atomize_paper(conn, paper, entry_mode="oracle-footprint")
    return "paper:arXiv:2401.99999"


_BODY = "FULL DOCUMENT. " + ("the whole paper, section by section. " * 40)


def _store_rows(atom_id):
    conn = schema.connect()
    try:
        return (conn.execute("SELECT version, raw_hash, payload FROM atoms WHERE atom_id=?",
                             (atom_id,)).fetchone()[:],
                schema.read_body(conn, atom_id))
    finally:
        conn.close()


def test_open_reads_a_records_full_text_and_stores_none_of_it(kb_home, monkeypatch):
    from pipeline.kb import ingest_papers as ip
    conn = schema.connect()
    atom_id = _seed_abstract_only_paper(conn, monkeypatch)
    conn.close()
    before = _store_rows(atom_id)
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: _BODY)

    got = kb_entry.kb_open(atom_id)
    assert got["body_state"] == "complete"
    assert "## Full text" in got["raw"] and "FULL DOCUMENT" in got["raw"]
    assert "Two thousand characters of abstract" in got["raw"]
    assert got["source_url"] == "https://doi.org/10.1/xyz"
    assert _store_rows(atom_id) == before                  # not a byte written to the store

    # The second open reads the cached file: nothing downloads again.
    monkeypatch.setattr(ip, "resolve_fulltext",
                        lambda paper: pytest.fail("a paper already read must not download again"))
    assert "FULL DOCUMENT" in kb_entry.kb_open(atom_id)["raw"]


def test_a_long_paper_is_clipped_to_one_opens_budget(kb_home, monkeypatch):
    from pipeline.kb import ingest_papers as ip, paper_text
    conn = schema.connect()
    atom_id = _seed_abstract_only_paper(conn, monkeypatch)
    conn.close()
    monkeypatch.setattr(paper_text, "BUDGET", 100)
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: _BODY)
    got = kb_entry.kb_open(atom_id)
    assert got["body_state"] == "partial"
    assert f"[First 100 of {len(_BODY):,} characters.]" in got["raw"]


def test_open_of_a_paper_with_no_reachable_pdf_still_answers(kb_home, monkeypatch):
    """Fail-safe: a paywalled paper returns its abstract, not an error, and is not asked again
    the same day."""
    from pipeline.kb import ingest_papers as ip
    conn = schema.connect()
    atom_id = _seed_abstract_only_paper(conn, monkeypatch)
    conn.close()
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: None)
    got = kb_entry.kb_open(atom_id)
    assert got["body_state"] == "partial" and got.get("error") is None
    assert "Two thousand characters of abstract" in got["raw"]

    monkeypatch.setattr(ip, "resolve_fulltext",
                        lambda paper: pytest.fail("a paper with no open PDF is asked once a day"))
    assert kb_entry.kb_open(atom_id)["body_state"] == "partial"


def test_open_never_breaks_when_the_read_blows_up(kb_home, monkeypatch):
    """An open() must return the body we already hold, whatever the full-text read did."""
    from pipeline.kb import paper_text
    conn = schema.connect()
    atom_id = _seed_abstract_only_paper(conn, monkeypatch)
    conn.close()

    def _explode(*a, **k):
        raise RuntimeError("the PDF extractor crashed")
    monkeypatch.setattr(paper_text, "read", _explode)
    got = kb_entry.kb_open(atom_id)
    assert got["raw_available"] is True and got["body_state"] == "partial"


def test_open_of_a_post_carries_the_papers_it_links(kb_home, monkeypatch):
    """A post's paper links, records or lines, come back with their full text while the budget
    lasts; a paper past it comes back with the id to open it by."""
    from pipeline.kb import ingest_papers as ip, paper_text
    conn = schema.connect()
    _seed_abstract_only_paper(conn, monkeypatch)       # a record
    md = ("# An essay\n\nSee [ours](https://arxiv.org/abs/2401.99999), "
          "[theirs](https://arxiv.org/abs/2402.00002) and [a third](https://arxiv.org/abs/2403.00003)."
          "\n\n---\n*Blog · [Original post](https://b.com/p)*\n")
    raw_hash = snapshot_hash(md)
    store_atom(conn, {
        "atom_id": "blog:b.com/p", "source_type": "blog", "what_kind": "opinion",
        "who_id": "blog:b.com", "source_url": "https://b.com/p", "raw_hash": raw_hash,
        "entry_mode": "oracle-footprint", "payload": {}}, md)
    conn.close()
    monkeypatch.setattr(paper_text, "BUDGET", len(_BODY) + 10)
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: _BODY)
    monkeypatch.setattr(ip, "paper_from_url", lambda url, **kw: {
        "paperId": "arXiv:" + url.rsplit("/", 1)[1], "title": "Theirs", "url": url,
        "externalIds": {"ArXiv": url.rsplit("/", 1)[1]}})

    papers = kb_entry.kb_open("blog:b.com/p")["papers"]

    assert [p["open_id"] for p in papers] == [
        "paper:arXiv:2401.99999", "paper:arXiv:2402.00002", "paper:arXiv:2403.00003"]
    assert papers[0]["title"] == "A Paper Nobody Has Read Yet"
    assert papers[0]["full_text"] == _BODY                             # the record, read in full
    assert papers[1]["full_text"] is None and "budget" in papers[1]["note"]
    assert papers[2]["full_text"] is None


def test_open_takes_the_id_of_a_paper_with_no_record(kb_home, monkeypatch):
    """A line's paper has no record; open() reads it live by its id or URL and stores nothing."""
    from pipeline.kb import ingest_papers as ip
    conn = schema.connect(); _seed(conn); conn.close()
    monkeypatch.setattr(ip, "paper_from_url", lambda url, **kw: {
        "paperId": "arXiv:2402.00002", "title": "Theirs", "abstract": "What they found.",
        "url": url, "externalIds": {"ArXiv": "2402.00002"}, "authors": [{"name": "Eve"}]})
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: _BODY)

    for ref in ("paper:arXiv:2402.00002", "https://arxiv.org/abs/2402.00002"):
        got = kb_entry.kb_open(ref)
        assert got["stored"] is False and got["body_state"] == "complete"
        assert "What they found." in got["raw"] and "FULL DOCUMENT" in got["raw"]
    conn = schema.connect()
    assert conn.execute("SELECT count(*) FROM atoms WHERE source_type='paper'").fetchone()[0] == 0
    conn.close()
    assert kb_entry.kb_open("x:nope")["error"] == "not found"


def test_open_by_a_records_url_opens_the_record(kb_home, monkeypatch):
    from pipeline.kb import ingest_papers as ip
    conn = schema.connect()
    atom_id = _seed_abstract_only_paper(conn, monkeypatch)
    conn.close()
    monkeypatch.setattr(ip, "resolve_fulltext", lambda paper: None)
    assert kb_entry.kb_open("https://arxiv.org/abs/2401.99999")["atom_id"] == atom_id
