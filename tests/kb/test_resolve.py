"""Stage-3 entity resolution (attested-only). Pure over the `entities` table — no network,
no embedder. Proves the merge RULE (attest→self and self↔self merge; attest↔attest NEVER),
the squatter defense, canonical_id materialization + idempotency, dry-run, and the stats."""
from __future__ import annotations

import pytest

from pipeline.kb import resolve, schema


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _ent(conn, eid, links=None, name=None):
    schema.upsert_entity(conn, eid, name=name, identity_links=links)


def _canon(conn, eid):
    row = conn.execute("SELECT canonical_id FROM entities WHERE entity_id=?", (eid,)).fetchone()
    return row["canonical_id"] if row else None


# ── _url_sets: the per-platform self/attests split ─────────────────────────────────

def test_url_sets_x_is_attests_substack_is_self():
    s, a = resolve._url_sets("x:user:1", ["https://carol.substack.com"])
    assert s == frozenset() and a == frozenset({"carol.substack.com"})
    s, a = resolve._url_sets("substack:carol", ["https://carol.substack.com/"])
    assert s == frozenset({"carol.substack.com"}) and a == frozenset()


def test_url_sets_drops_empty_and_none_links():
    assert resolve._url_sets("x:user:1", None) == (frozenset(), frozenset())
    assert resolve._url_sets("x:user:1", ["", None]) == (frozenset(), frozenset())


def test_url_sets_blog_is_self():
    # A blog entity's stored link IS its home (self) — the resolve.py _SELF_PLATFORMS fix.
    s, a = resolve._url_sets("blog:simonwillison.net", ["https://simonwillison.net"])
    assert s == frozenset({"simonwillison.net"}) and a == frozenset()


# ── the core merge rule ────────────────────────────────────────────────────────────

def test_attested_x_website_to_substack_home_merges(conn):
    _ent(conn, "x:user:1", ["https://carol.substack.com"], name="Carol")   # X website → substack
    _ent(conn, "substack:carol", ["https://carol.substack.com"], name="Carol Writes")
    st = resolve.resolve_entities(conn)
    assert _canon(conn, "x:user:1") == _canon(conn, "substack:carol")
    assert st.cross_platform == 1 and st.components == 1 and st.merged_entities == 2


def test_attested_x_website_to_blog_home_merges(conn):
    # The blog analog of the Substack merge: an X website field attesting a blog home unifies the
    # `blog:{host}` footprint entity into the Oracle canonical (blog link counts as `self`).
    _ent(conn, "x:user:1", ["https://simonwillison.net"], name="Simon")
    _ent(conn, "blog:simonwillison.net", ["https://simonwillison.net"], name="Simon Willison")
    st = resolve.resolve_entities(conn)
    assert _canon(conn, "x:user:1") == _canon(conn, "blog:simonwillison.net")
    assert st.cross_platform == 1 and st.components == 1 and st.merged_entities == 2


def test_intra_substack_handle_vs_subdomain_merges(conn):
    # saved-post path (handle id) + subs path (subdomain id) for ONE pub → both store pub url
    _ent(conn, "substack:carol", ["https://carol.substack.com"])
    _ent(conn, "substack:carolnews", ["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    assert _canon(conn, "substack:carol") == _canon(conn, "substack:carolnews")


def test_two_people_sharing_a_third_link_do_NOT_merge(conn):
    # both X websites point at ycombinator.com — a shared ATTEST, not a self → stay separate.
    # This is the squatter defense: attest∩attest must never merge.
    _ent(conn, "x:user:1", ["https://www.ycombinator.com"])
    _ent(conn, "x:user:2", ["https://ycombinator.com"])
    st = resolve.resolve_entities(conn)
    assert _canon(conn, "x:user:1") != _canon(conn, "x:user:2")
    assert st.components == 2 and st.merged_entities == 0


def test_unlinked_entities_are_singletons(conn):
    _ent(conn, "x:user:1", None)
    _ent(conn, "x:user:2", [])
    st = resolve.resolve_entities(conn)
    assert _canon(conn, "x:user:1") == "x:user:1"
    assert _canon(conn, "x:user:2") == "x:user:2"
    assert st.components == 2 and st.merged_entities == 0


def test_transitive_merge_across_three_rows(conn):
    # X website → substack (attest→self), plus a 2nd substack id sharing the same pub self
    _ent(conn, "x:user:1", ["https://carol.substack.com"])
    _ent(conn, "substack:carol", ["https://carol.substack.com"])
    _ent(conn, "substack:carolnews", ["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    roots = {_canon(conn, e) for e in ("x:user:1", "substack:carol", "substack:carolnews")}
    assert len(roots) == 1


def test_canonical_id_is_min_representative(conn):
    _ent(conn, "x:user:9", ["https://carol.substack.com"])
    _ent(conn, "substack:carol", ["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    # min("substack:carol", "x:user:9") == "substack:carol"  ('s' < 'x')
    assert _canon(conn, "x:user:9") == "substack:carol"
    assert _canon(conn, "substack:carol") == "substack:carol"


def test_custom_domain_substack_merges(conn):
    # substack on a custom domain: self = bare host; X website points at the same host
    _ent(conn, "x:user:1", ["https://www.stratechery.com"])
    _ent(conn, "substack:stratechery", ["https://stratechery.com"])
    resolve.resolve_entities(conn)
    assert _canon(conn, "x:user:1") == _canon(conn, "substack:stratechery")


def test_distinct_people_stay_distinct(conn):
    _ent(conn, "x:user:1", ["https://carol.substack.com"])
    _ent(conn, "substack:carol", ["https://carol.substack.com"])
    _ent(conn, "x:user:2", ["https://dave.substack.com"])
    _ent(conn, "substack:dave", ["https://dave.substack.com"])
    st = resolve.resolve_entities(conn)
    assert _canon(conn, "x:user:1") == _canon(conn, "substack:carol")
    assert _canon(conn, "x:user:2") == _canon(conn, "substack:dave")
    assert _canon(conn, "x:user:1") != _canon(conn, "x:user:2")
    assert st.components == 2 and st.cross_platform == 2


# ── materialization semantics ──────────────────────────────────────────────────────

def test_resolution_is_idempotent(conn):
    _ent(conn, "x:user:1", ["https://carol.substack.com"])
    _ent(conn, "substack:carol", ["https://carol.substack.com"])
    a = resolve.resolve_entities(conn).as_dict()
    b = resolve.resolve_entities(conn).as_dict()
    assert a == b
    assert a["duplicate_rows_collapsed"] == 1


def test_dry_run_computes_stats_but_writes_nothing(conn):
    _ent(conn, "x:user:1", ["https://carol.substack.com"])
    _ent(conn, "substack:carol", ["https://carol.substack.com"])
    st = resolve.resolve_entities(conn, dry_run=True)
    assert st.cross_platform == 1                       # stats still computed
    assert _canon(conn, "x:user:1") is None             # but nothing persisted
    assert _canon(conn, "substack:carol") is None


def test_empty_entities_table_is_safe(conn):
    st = resolve.resolve_entities(conn)
    assert st.total_entities == 0 and st.components == 0


def test_backfills_entity_for_atom_author(conn):
    # an atom whose author has NO entity row → resolution backfills it, then resolves it
    conn.execute("INSERT INTO atoms (atom_id, source_type, who_id) VALUES ('x:5','x','x:user:5')")
    conn.commit()
    st = resolve.resolve_entities(conn)
    assert _canon(conn, "x:user:5") == "x:user:5"       # backfilled + resolved as a singleton
    assert st.total_entities == 1


def test_reindex_after_new_link_repoints(conn):
    # start unmerged (no link on the X side), then the X website appears on a re-pull → merge
    _ent(conn, "x:user:1", None)
    _ent(conn, "substack:carol", ["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    assert _canon(conn, "x:user:1") == "x:user:1"       # singleton for now
    _ent(conn, "x:user:1", ["https://carol.substack.com"])   # website discovered on re-pull
    resolve.resolve_entities(conn)
    assert _canon(conn, "x:user:1") == _canon(conn, "substack:carol")
