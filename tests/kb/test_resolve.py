"""Stage-3 entity resolution (attested-only). Pure over the `entities` table — no network,
no embedder. Proves the merge rule (attest→self and self↔self merge; attest↔attest never),
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
    s, a = resolve._url_sets("x:user:1", ["https://carol.substack.com"], "Carol_X")
    assert s == frozenset({"x.com/carol_x"}) and a == frozenset({"carol.substack.com"})
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
    # both X websites point at ycombinator.com — a shared attest, not a self → stay separate.
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
    assert _canon(conn, "x:user:1") == "x:user:1"       # singleton until its website is known
    _ent(conn, "x:user:1", ["https://carol.substack.com"])   # website discovered on re-pull
    resolve.resolve_entities(conn)
    assert _canon(conn, "x:user:1") == _canon(conn, "substack:carol")


# ── a source added for an Oracle is the user's word that it is theirs ───────────────

def _review_item(conn, cid, url, status="added", kind="source"):
    conn.execute("INSERT INTO oracle_review_items (canonical_id, source_type, source_url, "
                 "source_key, kind, status) VALUES (?, 'substack', ?, ?, ?, ?)",
                 (cid, url, url, kind, status))
    conn.commit()


def test_an_added_source_joins_the_oracle_it_was_added_to(conn):
    _ent(conn, "x:user:1", ["https://carol.dev"])                 # the bio names only the site
    _ent(conn, "substack:carol", ["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    assert _canon(conn, "substack:carol") == "substack:carol"

    _review_item(conn, "x:user:1", "https://carol.substack.com")
    resolve.resolve_entities(conn)
    assert _canon(conn, "substack:carol") == _canon(conn, "x:user:1")


def test_a_dismissed_or_pending_or_also_item_attests_nothing(conn):
    _ent(conn, "x:user:1", [])
    _ent(conn, "substack:a", ["https://a.substack.com"])
    _ent(conn, "substack:b", ["https://b.substack.com"])
    _ent(conn, "substack:c", ["https://c.substack.com"])
    _review_item(conn, "x:user:1", "https://a.substack.com", status="dismissed")
    _review_item(conn, "x:user:1", "https://b.substack.com", status="pending")
    _review_item(conn, "x:user:1", "https://c.substack.com", kind="also")
    resolve.resolve_entities(conn)
    assert {_canon(conn, e) for e in ("substack:a", "substack:b", "substack:c")} == {
        "substack:a", "substack:b", "substack:c"}


def test_an_x_profile_refresh_cannot_undo_an_added_source(conn):
    _ent(conn, "x:user:1", ["https://carol.dev"])
    _ent(conn, "substack:carol", ["https://carol.substack.com"])
    _review_item(conn, "x:user:1", "https://carol.substack.com")
    _ent(conn, "x:user:1", ["https://carol.dev/new"])             # the next X refresh rewrites it
    resolve.resolve_entities(conn)
    assert _canon(conn, "substack:carol") == _canon(conn, "x:user:1")


def test_an_x_account_added_to_an_oracle_joins_it_though_its_bio_names_another_home(conn):
    """A Substack-rooted Oracle whose profile declares their X, and an X bio that links only
    their site: the account is its profile, `x.com/{handle}`, which the added source names."""
    schema.upsert_entity(conn, "x:user:1", identity_links=["https://carol.dev"],
                         profile={"handle": "Carol_X"})
    _ent(conn, "substack:carol", ["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    assert _canon(conn, "x:user:1") == "x:user:1"

    _review_item(conn, "substack:carol", "https://twitter.com/carol_x")
    resolve.resolve_entities(conn)
    assert _canon(conn, "x:user:1") == _canon(conn, "substack:carol")


# ── a GitHub account added for an Oracle joins it, and only that way ─────────────────

KARPATHY = "x:user:33836629"


def _github(conn, owner, links=None):
    """A GitHub account as stores hold it: `github:{owner}`, with no link (its repos write it)
    or its own URL."""
    _ent(conn, f"github:{owner}", links, name=owner)


def test_a_github_account_added_for_an_oracle_joins_it_and_a_repo_they_linked_does_not(conn):
    """Live shape: github.com/karpathy added for the Karpathy Oracle. A repo of someone else's
    that his posts link (`author_referenced`) is not his, and nothing added it."""
    schema.upsert_entity(conn, KARPATHY, identity_links=["https://karpathy.ai"],
                         profile={"handle": "karpathy"})
    _github(conn, "karpathy", ["https://github.com/karpathy"])
    _github(conn, "python")
    conn.execute("INSERT INTO atoms (atom_id, source_type, who_id, entry_mode) "
                 "VALUES ('github:python/cpython', 'github', 'github:python', 'author_referenced')")
    _review_item(conn, KARPATHY, "https://github.com/karpathy")
    resolve.resolve_entities(conn)
    assert _canon(conn, "github:karpathy") == _canon(conn, KARPATHY)
    assert _canon(conn, "github:python") == "github:python"


def test_a_github_link_in_anyones_profile_merges_nothing(conn):
    """An X website naming a repo (`github.com/google/jax`) or their own GitHub, with no source
    added: GitHub's address is matched against added sources only."""
    schema.upsert_entity(conn, "x:user:1", identity_links=["https://github.com/google/jax"],
                         profile={"handle": "jaxdev"})
    schema.upsert_entity(conn, KARPATHY, identity_links=["https://github.com/karpathy"],
                         profile={"handle": "karpathy"})
    _github(conn, "google", ["https://github.com/google"])
    _github(conn, "karpathy", ["https://github.com/karpathy"])
    resolve.resolve_entities(conn)
    assert {_canon(conn, e) for e in ("x:user:1", KARPATHY, "github:google",
                                      "github:karpathy")} == {
        "x:user:1", KARPATHY, "github:google", "github:karpathy"}


def test_one_github_account_added_for_two_oracles_joins_the_first(conn):
    """`layr-labs` added for two Oracles joins the one that added it first, and only that one."""
    schema.upsert_entity(conn, "x:user:1", profile={"handle": "gajesh"})
    schema.upsert_entity(conn, "x:user:2", profile={"handle": "sreeram"})
    _github(conn, "layr-labs", ["https://github.com/layr-labs"])
    _review_item(conn, "x:user:1", "https://github.com/layr-labs")
    _review_item(conn, "x:user:2", "https://github.com/layr-labs")
    resolve.resolve_entities(conn)
    assert _canon(conn, "github:layr-labs") == _canon(conn, "x:user:1")
    assert _canon(conn, "x:user:2") == "x:user:2"


# ── a source added for several Oracles joins the first that added it ────────────────

def _x(conn, eid, handle, website=None):
    schema.upsert_entity(conn, eid, identity_links=[website] if website else None,
                         profile={"handle": handle})


def _added_at(conn, cid, url, when, status="added"):
    """An added row whose add is at `when` (`updated_at`, set when a row becomes added)."""
    conn.execute("INSERT INTO oracle_review_items (canonical_id, source_type, source_url, "
                 "source_key, kind, status, created_at, updated_at) "
                 "VALUES (?, 'blog', ?, ?, 'source', ?, ?, ?)", (cid, url, url, status, when, when))
    conn.commit()


def _team_blog_for_ann_then_bob(conn):
    _x(conn, "x:user:1", "ann", "https://ann.dev")
    _x(conn, "x:user:2", "bob", "https://bob.dev")
    _ent(conn, "blog:team.dev", ["https://team.dev"])
    _added_at(conn, "x:user:2", "https://team.dev", "2026-10-11 10:00:00")   # Bob's row first …
    conn.execute("UPDATE oracle_review_items SET updated_at='2026-10-11 12:00:00'")
    _added_at(conn, "x:user:1", "https://team.dev", "2026-10-11 11:00:00")   # … Ann's add first
    conn.commit()


def test_one_blog_added_for_two_x_oracles_joins_the_first_adder(conn):
    _team_blog_for_ann_then_bob(conn)
    resolve.resolve_entities(conn)
    assert _canon(conn, "blog:team.dev") == _canon(conn, "x:user:1")
    assert _canon(conn, "x:user:2") == "x:user:2"


def test_the_first_add_is_when_a_row_became_added_not_when_it_was_found(conn):
    """Ann's row was found first but stayed pending until the user added it, after Bob's
    discovery had added the blog: it stays Bob's, and a later add never takes it from him."""
    _x(conn, "x:user:1", "ann")
    _x(conn, "x:user:2", "bob")
    _ent(conn, "blog:team.dev", ["https://team.dev"])
    _added_at(conn, "x:user:1", "https://team.dev", "2026-10-11 09:00:00", status="pending")
    _added_at(conn, "x:user:2", "https://team.dev", "2026-10-11 10:00:00")
    conn.execute("UPDATE oracle_review_items SET status='added', updated_at='2026-10-11 11:00:00' "
                 "WHERE canonical_id='x:user:1'")
    conn.commit()
    resolve.resolve_entities(conn)
    assert _canon(conn, "blog:team.dev") == _canon(conn, "x:user:2")
    assert _canon(conn, "x:user:1") == "x:user:1"


def test_one_url_added_for_two_substack_oracles_merges_nothing(conn):
    """An added URL is not a Substack-rooted Oracle's own address: two of them adding one URL
    merged at once, before any entity for it existed."""
    _ent(conn, "substack:ann", ["https://ann.substack.com"])
    _ent(conn, "substack:bob", ["https://bob.substack.com"])
    _review_item(conn, "substack:ann", "https://team.dev")
    _review_item(conn, "substack:bob", "https://team.dev")
    resolve.resolve_entities(conn)
    assert _canon(conn, "substack:ann") != _canon(conn, "substack:bob")


def test_one_github_url_added_for_an_x_and_a_substack_oracle_joins_the_first(conn):
    _x(conn, "x:user:1", "gajesh")
    _ent(conn, "substack:bob", ["https://bob.substack.com"])
    _github(conn, "layr-labs", ["https://github.com/layr-labs"])
    _review_item(conn, "x:user:1", "https://github.com/layr-labs")
    _review_item(conn, "substack:bob", "https://github.com/layr-labs")
    resolve.resolve_entities(conn)
    assert _canon(conn, "github:layr-labs") == _canon(conn, "x:user:1")
    assert _canon(conn, "substack:bob") == "substack:bob"


def test_two_oracles_naming_two_accounts_of_one_third_person_stay_apart(conn):
    """Ann added a site and Bob an X account that are one other person: it goes to Ann, who
    named it first, and Bob is not pulled in through it."""
    _x(conn, "x:user:1", "ann")
    _x(conn, "x:user:2", "bob")
    _x(conn, "x:user:9", "cat", "https://cat.dev")
    _ent(conn, "blog:cat.dev", ["https://cat.dev"])
    _review_item(conn, "x:user:1", "https://cat.dev")
    _review_item(conn, "x:user:2", "https://x.com/cat")
    resolve.resolve_entities(conn)
    assert _canon(conn, "blog:cat.dev") == _canon(conn, "x:user:9") == _canon(conn, "x:user:1")
    assert _canon(conn, "x:user:2") == "x:user:2"


def test_a_source_already_in_one_oracles_own_cluster_stays_there(conn):
    """Bob's X website is the team blog and Bob added it too; Ann added it first. Joining Ann
    would make Ann and Bob one person, so it stays with Bob."""
    _x(conn, "x:user:1", "ann")
    _x(conn, "x:user:2", "bob", "https://team.dev")
    _ent(conn, "blog:team.dev", ["https://team.dev"])
    _review_item(conn, "x:user:1", "https://team.dev")
    _review_item(conn, "x:user:2", "https://team.dev")
    resolve.resolve_entities(conn)
    assert _canon(conn, "blog:team.dev") == _canon(conn, "x:user:2")
    assert _canon(conn, "x:user:1") == "x:user:1"


def test_a_second_adders_claim_never_pulls_the_first_adder_in(conn):
    """The loop trap: once the blog joined Ann, Bob's claim on it names Ann's cluster. It must
    wait, not join Ann into Bob."""
    _team_blog_for_ann_then_bob(conn)
    _ent(conn, "blog:bob.dev", ["https://bob.dev"])        # Bob's own site, in his cluster
    _added_at(conn, "x:user:2", "https://bob.dev", "2026-10-11 13:00:00")
    resolve.resolve_entities(conn)
    resolve.resolve_entities(conn)
    assert _canon(conn, "blog:team.dev") == _canon(conn, "x:user:1")
    assert _canon(conn, "x:user:1") != _canon(conn, "x:user:2")


def test_resolve_gives_one_answer_whatever_the_order(conn, tmp_path):
    """Twice in a row, and on a store whose entities and rows went in the other way round."""
    _team_blog_for_ann_then_bob(conn)
    first = resolve.resolve_entities(conn).merges
    assert resolve.resolve_entities(conn).merges == first

    other = schema.connect(tmp_path / "other.db")
    _ent(other, "blog:team.dev", ["https://team.dev"])
    _x(other, "x:user:2", "bob", "https://bob.dev")
    _x(other, "x:user:1", "ann", "https://ann.dev")
    _added_at(other, "x:user:1", "https://team.dev", "2026-10-11 11:00:00")
    _added_at(other, "x:user:2", "https://team.dev", "2026-10-11 12:00:00")
    assert resolve.resolve_entities(other).merges == first
    other.close()


def test_a_substack_oracle_still_gets_what_it_alone_added(conn):
    """Alex Imas's shape: his X (bio: his site) and his site, both added for his Substack."""
    _ent(conn, "substack:aleximas", ["https://aleximas.substack.com"])
    _x(conn, "x:user:176", "alexolegimas", "http://www.aleximas.com")
    _ent(conn, "blog:aleximas.com", ["https://www.aleximas.com"])
    _review_item(conn, "substack:aleximas", "https://twitter.com/alexolegimas")
    _review_item(conn, "substack:aleximas", "https://www.aleximas.com")
    resolve.resolve_entities(conn)
    assert {_canon(conn, e) for e in ("substack:aleximas", "x:user:176",
                                      "blog:aleximas.com")} == {"blog:aleximas.com"}


def test_dismissing_the_first_add_hands_the_source_to_the_next(conn):
    _team_blog_for_ann_then_bob(conn)
    conn.execute("UPDATE oracle_review_items SET status='dismissed' WHERE canonical_id='x:user:1'")
    conn.commit()
    resolve.resolve_entities(conn)
    assert _canon(conn, "blog:team.dev") == _canon(conn, "x:user:2")
    assert _canon(conn, "x:user:1") == "x:user:1"


def test_one_person_confirmed_twice_still_gets_what_both_added(conn):
    """The user confirmed Ann's Substack and Ann's X as two Oracles. Her Substack added her X,
    which makes them one person, so the site both added is one person's."""
    _ent(conn, "substack:ann", ["https://ann.substack.com"])
    _x(conn, "x:user:1", "ann")
    _ent(conn, "blog:ann.dev", ["https://ann.dev"])
    _review_item(conn, "substack:ann", "https://x.com/ann")
    _review_item(conn, "substack:ann", "https://ann.dev")
    _review_item(conn, "x:user:1", "https://ann.dev")
    resolve.resolve_entities(conn)
    assert {_canon(conn, e) for e in ("substack:ann", "x:user:1", "blog:ann.dev")} == {
        "blog:ann.dev"}


def test_the_first_adder_searches_and_refreshes_a_shared_source(conn):
    """The blog joins Ann: her scoped search finds its post and Bob's does not, and it gets a
    refresh pair, under Ann."""
    from pipeline.oracles import pairs, registry, roster
    from pipeline.retrieval.local import LocalKB
    from tests.kb.test_export import _add
    _team_blog_for_ann_then_bob(conn)
    for eid, name in (("x:user:1", "Ann"), ("x:user:2", "Bob")):
        roster.upsert_oracle(conn, eid, name=name)
    _add(conn, "blog:team.dev/p", "blog", "opinion", "blog:team.dev", [],
         "zebracorn team blog post")
    resolve.resolve_entities(conn)
    pairs.seed_from_entities(conn)

    kb = LocalKB(conn, "mine", None)

    def found(who):
        return [h["atom_id"] for h in kb.search("zebracorn", who=who)["hits"]]
    assert found("@ann") == ["blog:team.dev/p"] and found("@bob") == []
    assert {(r.canonical_id, r.source_type, r.source_key) for r in registry.list_sources(conn)
            if r.source_type == "blog" and "team.dev" in r.source_key} == {
        (_canon(conn, "x:user:1"), "blog", "https://team.dev")}

