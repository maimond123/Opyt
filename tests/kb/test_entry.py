"""opyt_core.kb entry points — the read surface the MCP tools wrap. Exercised over the
$OPYT_HOME sandbox with the no-API paths (bm25 search / open / aggregate), so no paid call."""
from __future__ import annotations

from opyt_core import kb as kb_entry
from pipeline.kb import schema
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.raw_store import write_snapshot


def _seed(conn, emb):
    for atom_id, who, topics, snap in [
        ("github:root/agentkit", "github:root", ["ai-agents"],
         "an autonomous agent framework with tools"),
        ("github:stranger/x", "github:stranger", ["ai-agents"], "an agent framework library"),
        ("x:1", "x:user:1", ["crypto"], "thoughts on rollup and proof systems"),
    ]:
        raw_ref, raw_hash = write_snapshot("github" if atom_id.startswith("github") else "x",
                                           atom_id, snap)
        atom = dict(atom_id=atom_id,
                    source_type="github" if atom_id.startswith("github") else "x",
                    what_kind="artifact" if atom_id.startswith("github") else "opinion",
                    who_id=who, when_ts="2024-05-01", when_precision="day",
                    about_entities=[], source_url=f"https://e/{atom_id}",
                    raw_ref=raw_ref, raw_hash=raw_hash, description=f"{atom_id} card",
                    payload={"source_tags": topics}, entry_mode="user-saved")
        store_atom(conn, emb, atom=atom, snapshot_text=snap)
    # `trusted_atoms` counts atoms whose author is a CONFIRMED ORACLE. It used to count against
    # `entity_trust`, a mirror of these same confirmations that nothing else read (deleted
    # 2026-08-23), so the fixture now states the fact directly instead of via the mirror.
    schema.upsert_entity(conn, "github:root", name="root")
    conn.execute("INSERT OR IGNORE INTO oracles (canonical_id, name, source) VALUES (?,?,?)",
                 ("github:root", "root", "screen"))
    conn.commit()


def test_bm25_search_entry_needs_no_embedder(kb_home, fake_embedder):
    conn = schema.connect(); _seed(conn, fake_embedder); conn.close()
    # mode=bm25 → run_kb_search builds NO embedder (no API), pure FTS.
    hits = kb_entry.run_kb_search("framework", tags=["ai-agents"], mode="bm25", k=8)["hits"]
    ids = {h["atom_id"] for h in hits}
    assert ids == {"github:root/agentkit", "github:stranger/x"}
    assert all("snippet" in h and "source_url" in h for h in hits)


def test_open_returns_real_raw_and_pointer(kb_home, fake_embedder):
    conn = schema.connect(); _seed(conn, fake_embedder); conn.close()
    got = kb_entry.kb_open("github:root/agentkit")
    assert got["raw_available"] is True
    assert "autonomous agent framework" in got["raw"]       # the REAL snapshot, not the card
    assert got["source_url"] == "https://e/github:root/agentkit"
    # Unknown id fails safe.
    assert kb_entry.kb_open("github:nope/nope")["error"] == "not found"


def test_aggregate_skeleton_counts_and_trust(kb_home, fake_embedder):
    conn = schema.connect(); _seed(conn, fake_embedder); conn.close()
    agg = kb_entry.kb_aggregate({"tags": ["ai-agents"]})
    assert agg["total"] == 2
    assert agg["by_source_type"] == {"github": 2}
    assert agg["by_what_kind"] == {"artifact": 2}
    assert agg["trusted_atoms"] == 1                          # only the Oracle-authored atom
    assert any(t["topic"] == "ai-agents" for t in agg["top_topics"])
    assert len(agg["recent_descriptions"]) == 2

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


def test_search_by_handle_spans_every_platform_that_person_publishes_on(kb_home, fake_embedder):
    conn = schema.connect(); _seed(conn, fake_embedder); _cross_platform_person(conn); conn.close()
    hits = kb_entry.run_kb_search("agent OR rollup", who="@root", mode="bm25", k=8)["hits"]
    # Their GitHub repo AND their X post — one handle, both platforms.
    assert {h["atom_id"] for h in hits} == {"github:root/agentkit", "x:1"}


def test_search_by_handle_excludes_everyone_else(kb_home, fake_embedder):
    conn = schema.connect(); _seed(conn, fake_embedder); _cross_platform_person(conn); conn.close()
    hits = kb_entry.run_kb_search("framework", who="@root", mode="bm25", k=8)["hits"]
    assert {h["atom_id"] for h in hits} == {"github:root/agentkit"}   # NOT github:stranger/x


def test_an_unresolvable_handle_returns_nothing_not_the_whole_store(kb_home, fake_embedder):
    """The failure that matters. `who` resolving to nobody must NOT widen back to an unfiltered
    search — those hits would be presented to the user as that person's work."""
    conn = schema.connect(); _seed(conn, fake_embedder); conn.close()
    assert kb_entry.run_kb_search("framework", who="@nobody", mode="bm25", k=8)["hits"] == []


def test_search_reports_the_resolution_so_an_empty_result_is_readable(kb_home, fake_embedder):
    """The fact that MOVED. `[]` alone cannot distinguish "we don't track this person" from
    "we track them and they wrote nothing about this" — two different next actions. That used
    to be readable only by making a SECOND call to `aggregate(who=...)`, which you had to
    already suspect the problem to make. It rides along with the search that raised the
    question now."""
    conn = schema.connect(); _seed(conn, fake_embedder); _cross_platform_person(conn); conn.close()

    tracked = kb_entry.run_kb_search("framework", who="@root", mode="bm25", k=8)
    assert [c["name"] for c in tracked["insights"]["resolved_who"]] == ["Root McRepo"]
    assert set(tracked["insights"]["resolved_who"][0]["who_ids"]) == {
        "blog:root.example.com", "github:root", "x:user:1"}

    untracked = kb_entry.run_kb_search("framework", who="@nobody", mode="bm25", k=8)
    assert untracked["hits"] == []
    assert untracked["insights"]["resolved_who"] == []               # ← "we don't have them"
    assert [n["code"] for n in untracked["notices"]] == ["who_unresolved"]
    assert "@nobody" in untracked["notices"][0]["message"]


def test_a_query_with_no_who_carries_no_resolution_key_at_all(kb_home, fake_embedder):
    """ABSENT, not empty. An empty `resolved_who` MEANS "that handle matched nobody", so a
    query that never asked about a person must not carry one — a phantom key that reads as a
    failed lookup is worse than no key."""
    conn = schema.connect(); _seed(conn, fake_embedder); conn.close()
    out = kb_entry.run_kb_search("framework", mode="bm25", k=8)
    assert "resolved_who" not in out["insights"]


def test_aggregate_scopes_by_the_ids_search_resolved(kb_home, fake_embedder):
    """The two-call workflow that replaces `aggregate(who=...)`: search resolves the handle,
    aggregate takes the ids. `aggregate` no longer resolves handles at all, so there is exactly
    one resolver reachable from the tool surface."""
    conn = schema.connect(); _seed(conn, fake_embedder); _cross_platform_person(conn); conn.close()

    ids = kb_entry.run_kb_search("framework", who="@root",
                                 mode="bm25", k=8)["insights"]["resolved_who"][0]["who_ids"]
    scoped = kb_entry.kb_aggregate({"who_id": ids})
    assert scoped["total"] == 2                                      # both platforms' atoms
    assert scoped["notices"] == []

    empty = kb_entry.kb_aggregate({"who_id": ["x:user:nobody"]})
    assert empty["total"] == 0
    assert [n["code"] for n in empty["notices"]] == ["scope_matched_nothing"]


def test_who_and_who_id_union_rather_than_intersect(kb_home, fake_embedder):
    # Both mean "restrict to these authors", so a caller holding a handle for one person and an
    # id for another wants both people — an intersection would return nothing.
    conn = schema.connect(); _seed(conn, fake_embedder); _cross_platform_person(conn); conn.close()
    hits = kb_entry.run_kb_search("framework", who="@root", who_id="github:stranger",
                                  mode="bm25", k=8)["hits"]
    assert {h["atom_id"] for h in hits} == {"github:root/agentkit", "github:stranger/x"}
