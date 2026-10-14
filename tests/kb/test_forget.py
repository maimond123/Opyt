"""Singular removal: no dangling references, no invisible re-subscription, no preview writes."""
import re
import sqlite3

import pytest

from pipeline.forget import atoms as forget
from pipeline.kb import schema
from tests.kb.test_export import _add
from pipeline.oracles import roster
from pipeline.oracles import pairs
from pipeline.oracles import registry
from pipeline.forget import oracle as forget_oracle
from pipeline.forget.request import name_target

A = "x:1"
B = "x:10"


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    _add(c, A, "x", "opinion", "x:user:1", [],
         "autonomous agents uniqueatomword", entry_mode="user-saved")
    _add(c, B, "x", "opinion", "x:user:2", [], "crypto proof systems")
    yield c
    c.close()


def test_removal_inventory_equals_atom_reference_tables_in_schema():
    ddl = re.sub(r"--[^\n]*", "", schema._DDL)
    tables = re.findall(r"CREATE (?:VIRTUAL )?TABLE IF NOT EXISTS (\w+)", ddl)
    with sqlite3.connect(":memory:") as c:
        c.executescript(schema._DDL)
        holders = {t for t in tables if any(
            re.search(r"(?:^|_)atom_ids?$", col[1]) for col in c.execute(f"PRAGMA table_info({t})"))}
    assert set(forget._ATOM_REFERENCES) == holders


def test_atom_preview_and_complete_removal(conn):
    before = list(conn.iterdump())
    preview = forget.atom(conn, A, shared=True)
    assert preview["status"] == "preview"
    assert preview["description"] == f"{A} card"
    assert preview["who_id"] == "x:user:1"
    assert preview["when_ts"] == "2024-05-01"
    assert preview["source_url"] == f"https://example/{A}"
    assert preview["snapshot"] and preview["shared"]
    assert list(conn.iterdump()) == before

    assert forget.atom(conn, A, confirm=True)["status"] == "forgotten"
    for table, where in forget._ATOM_REFERENCES.items():
        assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}",
                            {"atom_id": A}).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM bodies_fts WHERE bodies_fts MATCH 'uniqueatomword'"
                        ).fetchone()[0] == 0
    conn.execute("INSERT INTO bodies_fts(bodies_fts, rank) VALUES ('integrity-check', 1)")
    assert conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_id=?", (B,)).fetchone()[0] == 1
    assert forget.atom(conn, A, confirm=True)["status"] == "not_found"


def test_an_atom_with_no_body_still_forgets(conn):
    conn.execute("DELETE FROM bodies WHERE atom_id=?", (A,))
    conn.commit()
    assert forget.atom(conn, A)["snapshot"] is False
    assert forget.atom(conn, A, confirm=True)["status"] == "forgotten"


def test_materialized_frontier_record_survives_without_its_atom(conn):
    """The row stays `materialized`, so the candidate is never admitted again; only its pointer to
    the forgotten atom goes."""
    conn.execute("UPDATE atoms SET entry_mode='frontier' WHERE atom_id=?", (A,))
    conn.execute("INSERT INTO frontier_candidates (candidate_id,source,kind,title,url,status,"
                 "first_seen_at,last_seen_at,atom_id) VALUES ('c','arxiv','paper','a',"
                 "'https://example','materialized','a','a',?)", (A,))
    conn.commit()
    assert forget.atom(conn, A)["references"]["frontier_candidates"] == 1
    forget.atom(conn, A, confirm=True)
    assert tuple(conn.execute("SELECT status, atom_id FROM frontier_candidates "
                              "WHERE candidate_id='c'").fetchone()) == ("materialized", None)


def test_forgetting_oracle_stops_refresh_even_after_cluster_head_drift(conn):
    schema.upsert_entity(conn, "x:user:1", name="Alice", profile={"handle": "alice"})
    schema.upsert_entity(conn, "blog:alice.test", name="Alice")
    roster.upsert_oracle(conn, "x:user:1", name="Alice")
    pairs.seed_from_entities(conn)
    conn.execute("UPDATE entities SET canonical_id='blog:alice.test'")
    conn.commit()
    roster.upsert_oracle(conn, "blog:alice.test", name="Alice")
    pairs.seed_from_entities(conn)
    before = list(conn.iterdump())
    preview = forget_oracle.forget(conn, "@alice")
    assert preview["status"] == "preview" and preview["canonical_id"] == "blog:alice.test"
    assert preview["sources_removed"] >= 2
    assert preview["atoms_kept"] == 1
    assert list(conn.iterdump()) == before
    assert forget_oracle.forget(conn, "@alice", confirm=True)["status"] == "forgotten"
    assert roster.list_oracles(conn) == []
    assert registry.list_sources(conn) == []
    assert pairs.seed_from_entities(conn)["pairs"] == 0      # nothing registers them again
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 2
    assert schema.get_entity(conn, "x:user:1") is not None
    assert forget_oracle.forget(conn, "@alice", confirm=True)["status"] == "not_found"


def test_ambiguous_oracle_requires_singular_choice_and_keeps_other_subscription(conn):
    for i in (1, 2):
        schema.upsert_entity(conn, f"x:user:{i}", name="Alex", profile={"handle": f"alex{i}"})
        roster.upsert_oracle(conn, f"x:user:{i}", name="Alex")
    pairs.seed_from_entities(conn)
    assert forget_oracle.forget(conn, "Alex", confirm=True)["status"] == "ambiguous"
    assert len(roster.list_oracles(conn)) == 2
    assert forget_oracle.forget(conn, "x:user:1", confirm=True)["status"] == "forgotten"
    assert [o["canonical_id"] for o in roster.list_oracles(conn)] == ["x:user:2"]
    assert {s.canonical_id for s in registry.list_sources(conn)} == {"x:user:2"}


def test_oracle_without_refresh_registry_can_be_forgotten(conn):
    schema.upsert_entity(conn, "x:user:1", name="Alice")
    roster.upsert_oracle(conn, "x:user:1", name="Alice")
    assert forget_oracle.forget(conn, "Alice", confirm=True)["status"] == "forgotten"


def test_oracle_removal_waits_on_no_lease(conn):
    """The `oracle-refresh` lease and the `busy` answer do not exist (the refresh rail is gone).
    A scraper unit still running for the Oracle checks for itself (`roster.on_roster`), so a
    held lease of that name stops nothing."""
    from pipeline.sync_lock import CatchupLock

    schema.upsert_entity(conn, "x:user:1", name="Alice", profile={"handle": "alice"})
    roster.upsert_oracle(conn, "x:user:1", name="Alice")
    pairs.seed_from_entities(conn)
    with CatchupLock("oracle-refresh") as held:
        assert held.acquired
        assert forget_oracle.forget(conn, "Alice", confirm=True)["status"] == "forgotten"
    assert pairs.seed_from_entities(conn)["pairs"] == 0
    assert registry.list_sources(conn) == []


def _forget_tool(monkeypatch):
    """The `forget` tool as registered, with sharing off."""
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
    return got["forget"]


def test_tool_is_singular_and_previews_before_removing(conn, monkeypatch):
    tool = _forget_tool(monkeypatch)
    # What the agent is told it removes matches what goes: the atom, its text and index entry.
    assert "deletes that atom, its text, search entry and references" in " ".join(tool.__doc__.split())
    assert tool()["status"] == "error"
    assert tool(atom_id=A, oracle="Alice", confirm=True)["status"] == "error"
    assert tool(atom_id=" ")["status"] == "error"
    assert tool(atom_id=A)["status"] == "preview"
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 2
    assert tool(atom_id=A, confirm=True)["status"] == "forgotten"
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 1


def test_many_atoms_preview_once_and_say_each_consequence_once(conn):
    _add(conn, "x:11", "x", "opinion", "x:user:3", [], "another saved post",
         entry_mode="user-saved")
    ids = [A, B, "x:11"]
    before = list(conn.iterdump())
    preview = forget.atoms(conn, ids, shared=True)
    assert preview["status"] == "preview" and preview["scope"] == "atoms"
    assert preview["atoms"][0] == {"atom_id": A, "description": f"{A} card", "who_id": "x:user:1",
                                   "when_ts": "2024-05-01", "source_url": f"https://example/{A}"}
    assert [a["atom_id"] for a in preview["atoms"]] == ids
    consent = preview["consent"]
    assert consent[:2] == ["This removes these 3 atoms.",
                           "Their text and search entries are deleted."]
    assert len(consent) == len(set(consent))
    assert sum("bookmarks" in line for line in consent) == 1      # two saved X posts, one warning
    assert any("Oracle refresh" in line for line in consent)       # B came from a footprint
    assert any("may contain these atoms" in line for line in consent)
    assert list(conn.iterdump()) == before


def test_many_atoms_go_in_one_confirm(conn):
    done = forget.atoms(conn, [A, B], confirm=True)
    assert done["status"] == "forgotten" and len(done["atoms"]) == 2
    assert _ids(conn) == set()
    for atom_id in (A, B):
        for table, where in forget._ATOM_REFERENCES.items():
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}",
                                {"atom_id": atom_id}).fetchone()[0] == 0


def test_many_atoms_roll_back_together(conn, monkeypatch):
    """One transaction: the second atom failing leaves the first in the store too."""
    before = list(conn.iterdump())
    real = forget.atom

    def flaky(c, atom_id, **kw):
        if atom_id == B:
            raise sqlite3.OperationalError("disk I/O error")
        return real(c, atom_id, **kw)
    monkeypatch.setattr(forget, "atom", flaky)
    with pytest.raises(sqlite3.OperationalError):
        forget.atoms(conn, [A, B], confirm=True)
    assert list(conn.iterdump()) == before


def test_an_unknown_id_refuses_the_whole_set(conn):
    before = list(conn.iterdump())
    for confirm in (False, True):
        out = forget.atoms(conn, [A, "x:404", B, "x:405"], confirm=confirm)
        assert out["status"] == "not_found" and out["unknown"] == ["x:404", "x:405"]
        assert "x:404, x:405" in out["message"] and "Nothing was removed" in out["message"]
    assert list(conn.iterdump()) == before


def test_tool_takes_a_list_of_atoms(conn, monkeypatch):
    tool = _forget_tool(monkeypatch)
    assert tool(atom_id=A, atom_ids=[B])["status"] == "error"
    assert tool(atom_ids=[A, B], oracle="Alice")["status"] == "error"
    assert tool(atom_ids=[A, B], source="https://a.test")["status"] == "error"
    assert tool(atom_ids=[A, B], delete_atoms=True)["status"] == "error"
    assert tool(atom_ids=[])["status"] == "error"
    assert tool(atom_ids=[A]) == tool(atom_id=A)                   # one id is atom_id
    assert tool(atom_ids=[A, "x:404"])["unknown"] == ["x:404"]
    assert tool(atom_ids=[A, B])["status"] == "preview"
    assert _ids(conn) == {A, B}
    assert tool(atom_ids=[A, B], confirm=True)["status"] == "forgotten"
    assert _ids(conn) == set()


def _alice_with_footprint(conn):
    """Alice: the saved post `A` (fixture), two posts her own X brought in, and one from her
    blog, a second member of her cluster."""
    schema.upsert_entity(conn, "x:user:1", name="Alice", profile={"handle": "alice"})
    schema.upsert_entity(conn, "blog:alice.test", name="Alice")
    conn.execute("UPDATE entities SET canonical_id='x:user:1' WHERE entity_id='blog:alice.test'")
    conn.commit()
    roster.upsert_oracle(conn, "x:user:1", name="Alice")
    pairs.seed_from_entities(conn)
    _add(conn, "xprofile:20", "x", "opinion", "x:user:1", [], "alice footprint one")
    _add(conn, "xprofile:21", "x", "opinion", "x:user:1", [], "alice footprint two")
    _add(conn, "blog:alice.test/p", "blog", "opinion", "blog:alice.test", [],
         "alice blog post")
    return ["blog:alice.test/p", "xprofile:20", "xprofile:21"]


def _ids(conn):
    return {r[0] for r in conn.execute("SELECT atom_id FROM atoms")}


def test_forgetting_oracle_with_delete_atoms_removes_only_what_her_sources_brought_in(
        conn):
    footprint = _alice_with_footprint(conn)
    before = list(conn.iterdump())
    preview = forget_oracle.forget(conn, "@alice", delete_atoms=True)
    assert preview["status"] == "preview"
    assert preview["atoms_deleted"] == 3 and preview["atoms_kept"] == 1
    assert "footprint_atom_ids" not in preview
    assert any("There is no undo" in line for line in preview["consent"])
    assert any("are deleted, with their text and search entries." in line
               for line in preview["consent"])
    assert not any("chunks" in line for line in preview["consent"])   # an atom has no chunks
    assert list(conn.iterdump()) == before

    done = forget_oracle.forget(conn, "@alice", confirm=True, delete_atoms=True)
    assert done["status"] == "forgotten" and done["atoms_deleted"] == 3
    assert "footprint_atom_ids" not in done and "atoms_not_deleted" not in done
    assert _ids(conn) == {A, B}                    # her saved post and someone else's stay
    for atom_id in footprint:
        assert conn.execute("SELECT COUNT(*) FROM bodies WHERE atom_id=?",
                            (atom_id,)).fetchone()[0] == 0
    # Their index entries go with the body rows: nothing of hers is still findable.
    assert conn.execute("SELECT COUNT(*) FROM bodies_fts WHERE bodies_fts MATCH 'alice'"
                        ).fetchone()[0] == 0
    assert roster.list_oracles(conn) == [] and registry.list_sources(conn) == []


def test_forgetting_an_oracle_deletes_the_repos_of_the_github_account_added_for_them(
        conn):
    """Her GitHub, added for her, is hers: its repos go with her. A repo she only linked,
    someone else's account, stays."""
    from pipeline.kb import resolve
    schema.upsert_entity(conn, "x:user:1", name="Alice", profile={"handle": "alice"})
    roster.upsert_oracle(conn, "x:user:1", name="Alice")
    conn.execute("INSERT INTO oracle_review_items (canonical_id, source_type, source_url, "
                 "source_key, kind, status) VALUES ('x:user:1', 'github', "
                 "'https://github.com/alice', 'github.com/alice', 'source', 'added')")
    conn.commit()
    _add(conn, "github:alice/tool", "github", "artifact", "github:alice", [],
         "alice tool repo")
    _add(conn, "github:python/cpython", "github", "artifact", "github:python",
         [], "cpython", entry_mode="author_referenced")
    resolve.resolve_entities(conn)

    done = forget_oracle.forget(conn, "@alice", confirm=True, delete_atoms=True)
    assert done["status"] == "forgotten" and done["atoms_deleted"] == 1
    assert _ids(conn) == {A, B, "github:python/cpython"}


def _ann_and_bob_share_a_blog(conn):
    """Alice added the team blog first, then Bob; each has a post of their own."""
    for eid, name in (("x:user:1", "Alice"), ("x:user:2", "Bob")):
        schema.upsert_entity(conn, eid, name=name, profile={"handle": name.lower()})
        roster.upsert_oracle(conn, eid, name=name)
        conn.execute("INSERT INTO oracle_review_items (canonical_id, source_type, source_url, "
                     "source_key, kind, status) VALUES (?, 'blog', 'https://team.dev', "
                     "'team.dev', 'source', 'added')", (eid,))
    conn.commit()
    schema.upsert_entity(conn, "blog:team.dev", identity_links=["https://team.dev"])
    _add(conn, "xprofile:30", "x", "opinion", "x:user:2", [], "bob footprint")
    _add(conn, "blog:team.dev/p", "blog", "opinion", "blog:team.dev", [], "team blog post")
    _add(conn, "xprofile:20", "x", "opinion", "x:user:1", [], "alice footprint")


def test_forgetting_the_first_adder_of_a_shared_blog_keeps_it_for_the_other(conn):
    """The blog is Alice's while she has it, but Bob added it too: forgetting Alice deletes her
    own posts only, and the blog's posts are Bob's afterwards."""
    from pipeline.kb import resolve
    _ann_and_bob_share_a_blog(conn)
    resolve.resolve_entities(conn)
    assert schema.current_canonical(conn, "blog:team.dev") == \
        schema.current_canonical(conn, "x:user:1")

    done = forget_oracle.forget(conn, "@alice", confirm=True, delete_atoms=True)
    assert done["status"] == "forgotten" and done["atoms_deleted"] == 1
    assert _ids(conn) == {A, B, "xprofile:30", "blog:team.dev/p"}
    assert schema.current_canonical(conn, "blog:team.dev") == \
        schema.current_canonical(conn, "x:user:2")


def test_forgetting_a_shared_source_of_one_oracle_leaves_the_others_add(conn):
    from pipeline.forget import source as forget_source
    from pipeline.kb import resolve
    _ann_and_bob_share_a_blog(conn)
    resolve.resolve_entities(conn)

    done = forget_source.forget(conn, "https://team.dev", oracle="x:user:1", confirm=True,
                                delete_atoms=True)
    assert done["status"] == "forgotten" and done["atoms_deleted"] == 0
    assert "blog:team.dev/p" in _ids(conn)
    assert {(r[0], r[1]) for r in conn.execute(
        "SELECT canonical_id, status FROM oracle_review_items")} == {
        ("x:user:1", "dismissed"), ("x:user:2", "added")}
    assert schema.current_canonical(conn, "blog:team.dev") == \
        schema.current_canonical(conn, "x:user:2")


def test_forgetting_oracle_without_delete_atoms_still_keeps_every_atom(conn):
    _alice_with_footprint(conn)
    preview = forget_oracle.forget(conn, "@alice")
    assert preview["atoms_kept"] == 4 and "atoms_deleted" not in preview
    assert forget_oracle.forget(conn, "@alice", confirm=True)["status"] == "forgotten"
    assert len(_ids(conn)) == 5


def test_one_atom_that_will_not_delete_leaves_the_rest_deleted(conn, monkeypatch):
    _alice_with_footprint(conn)
    real = forget.atom

    def flaky(c, atom_id, **kw):
        if atom_id == "xprofile:20":
            raise sqlite3.OperationalError("disk I/O error")
        return real(c, atom_id, **kw)
    monkeypatch.setattr(forget, "atom", flaky)
    done = forget_oracle.forget(conn, "@alice", confirm=True, delete_atoms=True)
    assert done["status"] == "forgotten"
    assert done["atoms_deleted"] == 2 and done["atoms_not_deleted"] == 1
    assert "disk I/O error" in done["errors"][0]
    assert _ids(conn) == {A, B, "xprofile:20"}


def test_tool_takes_delete_atoms_for_an_oracle_only(conn, monkeypatch):
    tool = _forget_tool(monkeypatch)
    _alice_with_footprint(conn)
    assert tool(atom_id=A, delete_atoms=True)["status"] == "error"
    assert tool(oracle="@alice", delete_atoms=True)["atoms_deleted"] == 3
    assert tool(oracle="@alice", delete_atoms=True, confirm=True)["atoms_deleted"] == 3
    assert _ids(conn) == {A, B}


def test_forgetting_oracle_clears_what_the_background_scrapers_kept_for_them(conn):
    """Left behind, a re-add would find discovery done and every found source pulled."""
    from pipeline.oracles.sources import scraper_tables as scrapers
    from pipeline.oracles import asks
    from pipeline.oracles import windows
    scrapers.init_schema(conn)
    schema.upsert_entity(conn, "x:user:1", name="Alice", profile={"handle": "alice"})
    roster.upsert_oracle(conn, "x:user:1", name="Alice")
    roster.upsert_oracle(conn, "x:user:2", name="Bob")
    for cid in ("x:user:1", "x:user:2"):
        asks.record_ask(conn, cid, web_from=windows.ALL_TIME)
        conn.execute("INSERT INTO found_sources (canonical_id, source_type, url, found_at, "
                     "pulled_at) VALUES (?, 'blog', 'https://a.test', 'then', 'then')", (cid,))
        conn.execute("INSERT INTO milestones_shown VALUES (?, 'archives', 'then')", (cid,))
        conn.execute("INSERT INTO archive_walks(key,canonical_id,generation,spec) VALUES(?,?,'then','{}')", (cid, cid))
        conn.execute("INSERT INTO archive_units(walk,generation,kind,key,data) VALUES(?,'then','page','one','{}')", (cid,))
        conn.execute("INSERT INTO web_recommendation_reads(canonical_id,url) VALUES(?,'https://a.test')", (cid,))
    conn.commit()
    assert forget_oracle.forget(conn, "Alice", confirm=True)["status"] == "forgotten"
    for table in ("oracle_asks", "found_sources", "milestones_shown", "archive_walks", "web_recommendation_reads"):
        assert [r[0] for r in conn.execute(f"SELECT canonical_id FROM {table}")] == ["x:user:2"]


def test_forget_removes_archive_children(conn):
    from pipeline.oracles.sources import scraper_tables
    scraper_tables.init_schema(conn)
    roster.upsert_oracle(conn, "x:user:1", name="Alice")
    conn.execute("INSERT INTO archive_walks(key,canonical_id,generation,spec) VALUES('walk','x:user:1','now','{}')")
    conn.execute("INSERT INTO archive_units(walk,generation,kind,key,data) VALUES('walk','now','page','one','{}')")
    conn.commit()
    assert forget_oracle.forget(conn, "x:user:1", confirm=True)["status"] == "forgotten"
    assert conn.execute("SELECT count(*) FROM archive_units").fetchone()[0] == 0


def test_the_oracle_preview_says_late_arrivals_are_not_kept(conn):
    """Removal commits at once; the consent said it "waits for a retry after it finishes"."""
    _alice_with_footprint(conn)
    consent = " ".join(forget_oracle.forget(conn, "@alice")["consent"])
    assert "waits for a retry" not in consent
    assert "is not kept" in consent


def test_a_call_names_exactly_one_target():
    """The tool's argument rules, checked on plain values."""
    def named(**kw):
        args = {"atom_id": None, "atom_ids": None, "oracle": None, "source": None, **kw}
        delete_atoms = args.pop("delete_atoms", False)
        return name_target(**args, delete_atoms=delete_atoms)

    assert "message" in named()
    assert "message" in named(atom_id=A, oracle="Alice")
    assert "message" in named(atom_id=A, delete_atoms=True)
    assert "message" in named(atom_ids=[])
    assert "message" in named(oracle="  ")
    assert named(atom_ids=[A]) == {"scope": "atom", "atom_id": A}
    assert named(atom_ids=[A, B]) == {"scope": "atoms", "atom_ids": [A, B]}
    assert named(oracle=" @alice ", delete_atoms=True) == {"scope": "oracle", "oracle": "@alice"}
    assert named(source=" https://a.dev ", oracle="Bob") == {
        "scope": "source", "source": "https://a.dev", "oracle": "Bob"}
