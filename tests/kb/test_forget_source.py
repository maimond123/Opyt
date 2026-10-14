"""`forget(source=…)`: one source of one Oracle removed for good, and nothing re-adds it."""

import pytest

from pipeline.forget import source as forget_source
from pipeline.kb import resolve, schema
from pipeline.oracles import found, pairs, registry, review_items, roster
from tests.kb.test_export import _add

SUB = "https://alicesub.substack.com"
BLOG = "https://alice.blog"


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    found.init_schema(c)
    registry.init_state_schema(c)
    schema.upsert_entity(c, "x:user:1", name="Alice", profile={"handle": "alice"},
                         identity_links=[BLOG])                    # her X bio names her blog
    schema.upsert_entity(c, "blog:alice.blog", name="Alice", identity_links=[BLOG])
    schema.upsert_entity(c, "substack:alicesub", name="Alice", identity_links=[SUB])
    roster.upsert_oracle(c, "x:user:1", name="Alice")
    # Discovery added her Substack (found in her posts, links back) and owes its first pull.
    with c:
        review_items.write_found(c, "x:user:1", [{"source_type": "substack", "url": SUB,
                                                 "status": review_items.ADDED,
                                                 "reasons": ["in 41 of their posts"]}])
        c.execute("INSERT INTO found_sources (canonical_id, source_type, url, found_at) "
                  "VALUES ('x:user:1', 'substack', ?, '2026-10-10')", (SUB,))
    resolve.resolve_entities(c)
    pairs.seed_from_entities(c)
    _add(c, "substack:1", "substack", "opinion", "substack:alicesub", [], "essay one")
    _add(c, "substack:2", "substack", "opinion", "substack:alicesub", [], "essay two")
    _add(c, "substack:3", "substack", "opinion", "substack:alicesub", [], "saved",
         entry_mode="user-saved")
    _add(c, "x:9", "x", "opinion", "x:user:1", [], "her own post")
    yield c
    c.close()


def _pair_types(conn):
    return {s.source_type for s in registry.list_sources(conn)}


def test_preview_writes_nothing_and_confirm_removes_the_source_for_good(conn):
    assert "substack" in _pair_types(conn)
    before = list(conn.iterdump())
    preview = forget_source.forget(conn, SUB, delete_atoms=True)
    assert preview["status"] == "preview" and preview["name"] == "Alice"
    assert preview["atoms"] == 3 and preview["atoms_deleted"] == 2
    assert any("are deleted, with their text and search entries." in line
               for line in preview["consent"])
    assert not any("chunks" in line for line in preview["consent"])
    assert list(conn.iterdump()) == before

    out = forget_source.forget(conn, SUB, confirm=True, delete_atoms=True)
    assert out["status"] == "forgotten" and out["atoms_deleted"] == 2
    left = {r[0] for r in conn.execute("SELECT atom_id FROM atoms")}
    assert left == {"substack:3", "x:9"}                  # the user's save and her X stay
    assert conn.execute("SELECT COUNT(*) FROM bodies_fts WHERE bodies_fts MATCH 'essay'"
                        ).fetchone()[0] == 0              # the deleted bodies left the index
    assert "substack" not in _pair_types(conn)
    assert found.list_found(conn) == []
    # Nothing brings it back: discovery holds the key, seeding skips it.
    assert ("substack", review_items.source_key(SUB)) in review_items.held_source_keys(
        conn, "x:user:1")
    pairs.seed_from_entities(conn)
    assert "substack" not in _pair_types(conn)


def test_a_site_her_bio_still_names_is_not_registered_again(conn):
    """The real path: her X attests the blog on its own, so it stays in her cluster after the
    removal. Being in the cluster must not mean being pulled."""
    assert "blog" in _pair_types(conn)                   # a starting pair, no review row
    assert forget_source.forget(conn, BLOG, confirm=True)["status"] == "forgotten"
    assert "blog" not in _pair_types(conn)
    resolve.resolve_entities(conn)
    pairs.seed_from_entities(conn)
    assert "blog" not in _pair_types(conn)
    assert schema.current_canonical(conn, "blog:alice.blog") == schema.current_canonical(
        conn, "x:user:1")                               # still hers, just not pulled
    assert "substack" in _pair_types(conn)               # her other sources keep going


def test_the_profile_she_was_added_by_is_refused(conn):
    out = forget_source.forget(conn, "https://x.com/alice", confirm=True)
    assert out["status"] == "refused" and "forget(oracle=" in out["message"]
    assert "x" in _pair_types(conn)


def test_a_source_two_oracles_share_needs_the_oracle(conn):
    schema.upsert_entity(conn, "x:user:2", name="Bob", profile={"handle": "bob"})
    roster.upsert_oracle(conn, "x:user:2", name="Bob")
    with conn:
        review_items.write_found(conn, "x:user:2", [{"source_type": "substack", "url": SUB,
                                                    "status": review_items.ADDED}])
    out = forget_source.forget(conn, SUB, confirm=True)
    assert out["status"] == "ambiguous" and len(out["candidates"]) == 2
    assert "substack" in _pair_types(conn)
    assert forget_source.forget(conn, SUB, oracle="Bob", confirm=True)["status"] == "forgotten"
    assert "substack" in _pair_types(conn)               # Alice's is untouched


def test_unknown_source(conn):
    assert forget_source.forget(conn, "https://nobody.example")["status"] == "not_found"


def test_tool_takes_exactly_one_target(conn, monkeypatch):
    from mcp_server.forget_tools import register_forget_tools
    got = {}

    class MCP:
        def tool(self, **kw):
            def register(fn):
                got[fn.__name__] = fn
                return fn
            return register

    monkeypatch.setattr("pipeline.credentials.get_credential", lambda service: None)
    register_forget_tools(MCP())
    tool = got["forget"]
    assert tool(source=SUB, atom_id="x:9")["status"] == "error"
    assert tool(atom_id="x:9", delete_atoms=True)["status"] == "error"
    assert tool(source=SUB)["status"] == "preview"
    assert tool(source=SUB, confirm=True)["status"] == "forgotten"
    assert "substack" not in _pair_types(conn)


def test_a_declared_orcid_drops_the_papers_pulled_through_it(conn):
    orcid = "https://orcid.org/0000-0002-4027-364X"
    schema.upsert_entity(conn, "openalex:A1", name="Alice", identity_links=[orcid])
    with conn:
        review_items.write_found(conn, "x:user:1", [{"source_type": "orcid", "url": orcid,
                                                    "status": review_items.ADDED}])
    resolve.resolve_entities(conn)
    pairs.seed_from_entities(conn)
    assert "openalex" in _pair_types(conn)
    assert forget_source.forget(conn, orcid, confirm=True)["status"] == "forgotten"
    assert "openalex" not in _pair_types(conn)
    resolve.resolve_entities(conn)
    pairs.seed_from_entities(conn)
    assert "openalex" not in _pair_types(conn)


def test_the_consent_says_what_goes_and_what_stays():
    """The source's consent, from plain values."""
    keep = forget_source.consent("https://a.dev", "Alice", 5, 0, delete_atoms=False, shared=True)
    assert keep[1].startswith("Its 5 existing atoms remain.")
    assert not any("served copy" in said for said in keep)
    drop = forget_source.consent("https://a.dev", "Alice", 5, 3, delete_atoms=True, shared=True)
    assert drop[1].startswith("3 atoms from it are deleted")
    assert drop[2].startswith("2 atoms the user saved themselves")
    assert "served copy" in drop[-1]
