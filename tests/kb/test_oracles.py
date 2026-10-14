"""Confirm — writing picks into the `oracles` table, the phantom-id guard, idempotency,
and resolve-at-confirm (a raw handle → a resolved oracle). The Substack-URL path is network-free;
the X-handle fetch is monkeypatched (no twitterapi call, no key)."""
from __future__ import annotations

import pytest

from pipeline.kb import resolve, schema


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _person(conn, eid, *, name=None, links=None):
    schema.upsert_entity(conn, eid, name=name, identity_links=links)


# ── confirm from ranked canonical_ids ──────────────────────────────────────────

def test_confirm_ranked_pick_writes_one_oracle(conn):
    _person(conn, "x:user:1", name="Carol")
    out = admit.confirm(conn, canonical_ids=["x:user:1"])
    assert out["confirmed"] == [{"canonical_id": "x:user:1", "name": "Carol", "source": "screen"}]
    assert out["total_oracles"] == 1 and roster.on_roster(conn, "x:user:1")


def test_confirm_phantom_id_is_guarded_not_written(conn):
    out = admit.confirm(conn, canonical_ids=["x:user:does-not-exist"])
    assert out["unknown"] == ["x:user:does-not-exist"]
    assert out["confirmed"] == [] and out["total_oracles"] == 0


def test_confirm_is_idempotent(conn):
    _person(conn, "x:user:1", name="Carol")
    admit.confirm(conn, canonical_ids=["x:user:1"])
    admit.confirm(conn, canonical_ids=["x:user:1"])
    assert len(roster.list_oracles(conn)) == 1


def test_confirm_uses_canonical_name_across_cluster(conn):
    # signal name sits on the substack row; the X row has the richer name → confirm reflects it
    _person(conn, "x:user:1", name="Carol Ada", links=["https://carol.substack.com"])
    _person(conn, "substack:carol", name="carol", links=["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    canon = schema.get_entity(conn, "x:user:1")["canonical_id"]
    out = admit.confirm(conn, canonical_ids=[canon])
    assert out["confirmed"][0]["name"] == "Carol Ada"


# ── resolve-at-confirm (the free-form floor) ────────────────────────────────────

def test_resolve_at_confirm_substack_url_needs_no_identity_fetch(conn, no_venue):
    """Rooting a site URL asks OpenAlex once whether the host is a research venue. A Substack URL
    still keys itself off the URL alone and needs no per-person identity fetch."""
    out = admit.confirm(conn, add_handles=["https://carol.substack.com"])
    assert out["unresolved"] == [] and len(out["confirmed"]) == 1
    c = out["confirmed"][0]
    assert c["source"] == "freeform" and c["canonical_id"].startswith("substack:")
    assert roster.on_roster(conn, c["canonical_id"])


def test_resolve_at_confirm_blog_url_mints_a_blog_entity(conn, no_venue):
    # A generic http… home is a blog, keyed `blog:{host}`, not `substack:{host}` (which would
    # send a personal site to the Substack cluster).
    out = admit.confirm(conn, add_handles=["https://simonwillison.net"])
    assert out["unresolved"] == [] and len(out["confirmed"]) == 1
    c = out["confirmed"][0]
    assert c["canonical_id"].startswith("blog:") and c["source"] == "freeform"
    assert roster.on_roster(conn, c["canonical_id"])


def test_resolve_at_confirm_substack_url_stays_substack(conn, no_venue):
    # A substack.com URL still keys `substack:`.
    out = admit.confirm(conn, add_handles=["https://carol.substack.com"])
    assert out["confirmed"][0]["canonical_id"].startswith("substack:")


def test_resolve_at_confirm_x_handle_mints_and_confirms(conn, monkeypatch):
    monkeypatch.setattr(identify, "fetch_x_identity", lambda h: {
        "user_id": "999", "display_name": "Pasted Person", "bio": "builder",
        "site": "https://pasted.com", "verified": True, "followers": 5000, "handle": "pasted"})
    out = admit.confirm(conn, add_handles=["@pasted"])
    assert len(out["confirmed"]) == 1
    c = out["confirmed"][0]
    assert c["canonical_id"] == "x:user:999" and c["source"] == "freeform"
    ent = schema.get_entity(conn, "x:user:999")
    assert ent is not None and "pasted.com" in (ent["identity_links"] or "")


def test_resolve_at_confirm_unresolved_handle_is_reported_not_crashed(conn):
    # no monkeypatch → the real fetch runs but fails (no key/network) → reported, nothing written
    from unittest import mock
    with mock.patch.object(identify, "fetch_x_identity", return_value=None):
        out = admit.confirm(conn, add_handles=["@ghost"])
    assert out["unresolved"] == ["@ghost"] and out["confirmed"] == []
    assert out["total_oracles"] == 0


def test_confirmed_oracles_carries_footprint_members(conn):
    _person(conn, "x:user:1", name="Carol", links=["https://carol.substack.com"])
    _person(conn, "substack:carol", name="Carol", links=["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    canon = schema.get_entity(conn, "x:user:1")["canonical_id"]
    admit.confirm(conn, canonical_ids=[canon])
    got = roster.confirmed_oracles(conn)
    assert len(got) == 1
    members = {m["entity_id"] for m in got[0]["members"]}
    assert members == {"x:user:1", "substack:carol"}          # both footprints reach the pull


# ── the backstop on a rendered diagnostic ──────────────────────────────────────
# ── one publication, one adapter ────────────────────────────────────────────────
#
# The defect these exist for: a Substack on a custom domain
# is two cluster members (`substack:{host}` + `blog:{host}`), resolve correctly merges them into
# one person, and both stayed pullable. One archive landed 180 times as `blog:`
# atoms and 180 times as `substack:` atoms — same `source_url` on every pair, 8,771 duplicate
# chunks, ~14MB of duplicate text in the embedding index, and no error anywhere. It was invisible
# until `oracle_refresh` began running, because until then only one of the two was ever reached.
from pipeline.oracles import roster
from pipeline.oracles.pairs import FOOTPRINT_ENTRY_MODE, _one_adapter_per_publication, pairs_for_oracle, seed_from_entities
from pipeline.oracles import registry
from pipeline.oracles import identify
from pipeline.oracles import admit


def _atoms(conn, who_id, source_type, n=1):
    """n FOOTPRINT atoms from one adapter — the evidence `_one_adapter_per_publication` decides
    on. `entry_mode` is load-bearing: a `user-saved` atom says the user bookmarked a post, not
    that the adapter can pull the publication, and counting one would hand a publication to an
    adapter that cannot pull it."""
    for i in range(n):
        schema.upsert_atom(conn, {"atom_id": f"{source_type}:{who_id}:{i}",
                                  "source_type": source_type, "what_kind": "opinion",
                                  "who_id": who_id, "when_ts": "2026-01-01",
                                  "description": "x",
                                  "entry_mode": FOOTPRINT_ENTRY_MODE})


def test_a_substack_on_a_custom_domain_is_pulled_once(conn):
    """Neither adapter has pulled yet, so substack takes the open case."""
    pairs = [("x", "dwarkesh_sp"), ("substack", "https://www.dwarkesh.com"),
             ("blog", "https://www.dwarkesh.com")]
    assert _one_adapter_per_publication(conn, pairs, ["blog:dwarkesh.com"]) == [
        ("x", "dwarkesh_sp"), ("substack", "https://www.dwarkesh.com")]


def test_the_www_that_made_two_spellings_look_like_two_publications(conn):
    """The live cluster carried `https://www.hyperdimensional.co` on its substack member and
    `https://hyperdimensional.co` on its blog member. One publication, two strings."""
    pairs = [("substack", "https://www.hyperdimensional.co"),
             ("blog", "https://hyperdimensional.co")]
    assert _one_adapter_per_publication(conn, pairs, ["blog:hyperdimensional.co"]) == [
        ("substack", "https://www.hyperdimensional.co")]


def test_the_adapter_that_already_works_keeps_the_publication(conn):
    """The case that caught the first rule out. A publication's essays can pull fine from the blog
    adapter while its Substack is refused by the single-author eligibility gate — the publication
    is "Hyperdimensional" and its author is not. Always preferring substack would retire the
    source that worked for one that returns `skipped` forever, leaving 116 essays in the store
    with nothing registered that could ever refresh them."""
    _atoms(conn, "blog:hyperdimensional.co", "blog", n=3)
    pairs = [("substack", "https://www.hyperdimensional.co"),
             ("blog", "https://hyperdimensional.co")]

    assert _one_adapter_per_publication(conn, pairs, ["blog:hyperdimensional.co"]) == [
        ("blog", "https://hyperdimensional.co")]


def test_substack_keeps_the_publication_once_it_has_actually_pulled(conn):
    """The winner keeps winning, so the choice cannot oscillate between passes."""
    _atoms(conn, "blog:dwarkesh.com", "blog", n=2)
    _atoms(conn, "substack:www.dwarkesh.com", "substack", n=2)
    pairs = [("substack", "https://www.dwarkesh.com"), ("blog", "https://www.dwarkesh.com")]

    assert _one_adapter_per_publication(
        conn, pairs, ["blog:dwarkesh.com", "substack:www.dwarkesh.com"]) == [
        ("substack", "https://www.dwarkesh.com")]


def test_a_plain_blog_keeps_its_blog_pair(conn):
    """The collapse is about a duplicate, never about preferring one adapter. Someone whose site
    is not a Substack has only the blog adapter, and dropping it would drop their whole corpus."""
    pairs = [("x", "gajesh"), ("blog", "https://gajesh.com")]
    assert _one_adapter_per_publication(conn, pairs, ["blog:gajesh.com"]) == pairs


def test_two_different_publications_both_survive(conn):
    pairs = [("substack", "https://a.substack.com"), ("blog", "https://b.example.com")]
    assert _one_adapter_per_publication(conn, pairs, ["x:user:1"]) == pairs


def test_pairs_for_oracle_collapses_a_real_merged_cluster(conn):
    from pipeline.kb import resolve
    _person(conn, "x:user:1", name="Dwarkesh", links=["https://www.dwarkesh.com"])
    _person(conn, "substack:www.dwarkesh.com", name="Pod", links=["https://www.dwarkesh.com"])
    _person(conn, "blog:dwarkesh.com", name="Dwarkesh", links=["https://www.dwarkesh.com"])
    resolve.resolve_entities(conn)
    head = schema.current_canonical(conn, "x:user:1")

    pairs, _who = pairs_for_oracle(conn, head)

    assert ("substack", "https://www.dwarkesh.com") in pairs
    assert not [p for p in pairs if p[0] == "blog"], f"blog pair survived: {pairs}"


# ── the stale cluster head ──────────────────────────────────────────────────────
def test_a_merge_that_moves_the_head_reads_as_the_new_head_everywhere(conn):
    """One stale row, three wrong answers — `trusted_atoms` 80 against a true 884, an Oracle's
    display name resolving to NULL (so 182 posts were attributed to "substack"), and a second
    `oracle_sources` registration under the dead id. The row keeps its confirm-time id and every
    read folds it (`schema.ROSTER_HEADS_SQL`, `roster`)."""
    from pipeline.oracles.sources import pair_rows
    from pipeline.kb import resolve
    _person(conn, "x:user:9", name="Dwarkesh", links=["https://www.dwarkesh.com"])
    resolve.resolve_entities(conn)
    roster.upsert_oracle(conn, "x:user:9", name="Dwarkesh", source="screen")

    # The blog member arrives later and sorts below the X id, moving the head.
    _person(conn, "blog:dwarkesh.com", name="Dwarkesh", links=["https://www.dwarkesh.com"])
    resolve.resolve_entities(conn)

    head = schema.current_canonical(conn, "x:user:9")
    assert head == "blog:dwarkesh.com"
    assert roster.heads(conn) == {head}
    assert [o["canonical_id"] for o in roster.confirmed_oracles(conn)] == [head]
    assert roster.on_roster(conn, head)
    seed_from_entities(conn, include_x=False)
    assert {r.canonical_id for r in pair_rows.walked(conn, ("blog", "substack"))} == {head}


def test_the_name_join_that_silently_returned_null_now_resolves(conn):
    """The concrete cost of the stale row: `_dispatch` passes `row.name` as `author_name`, and a
    NULL there is what made 182 substack atoms read "substack ·" instead of the author."""
    from pipeline.kb import resolve
    _person(conn, "x:user:9", name="Dwarkesh", links=["https://www.dwarkesh.com"])
    resolve.resolve_entities(conn)
    roster.upsert_oracle(conn, "x:user:9", name="Dwarkesh Patel", source="screen")
    _person(conn, "blog:dwarkesh.com", name="Dwarkesh", links=["https://www.dwarkesh.com"])
    resolve.resolve_entities(conn)

    seed_from_entities(conn)

    assert {r.name for r in registry.list_sources(conn)} == {"Dwarkesh Patel"}


def test_two_oracles_that_turn_out_to_be_one_person_keep_the_older_confirmation(conn):
    """`canonical_id` is the primary key, so a merge can collide. The oldest confirmation is the
    one the user would recognise; `INSERT OR REPLACE` would pick by insertion order instead."""
    from pipeline.kb import resolve
    _person(conn, "x:user:9", name="D", links=["https://www.dwarkesh.com"])
    _person(conn, "blog:dwarkesh.com", name="D", links=[])
    resolve.resolve_entities(conn)
    roster.upsert_oracle(conn, "blog:dwarkesh.com", name="Older", source="screen")
    conn.execute("UPDATE oracles SET confirmed_at='2020-01-01' WHERE canonical_id=?",
                 ("blog:dwarkesh.com",))
    roster.upsert_oracle(conn, "x:user:9", name="Newer", source="screen")
    conn.commit()

    # Link them: one cluster, head = blog:dwarkesh.com (sorts first).
    schema.upsert_entity(conn, "blog:dwarkesh.com", name="D",
                         identity_links=["https://www.dwarkesh.com"])
    resolve.resolve_entities(conn)

    rows = roster.confirmed_oracles(conn)
    assert len(rows) == 1 and rows[0]["canonical_id"] == "blog:dwarkesh.com"
    assert rows[0]["name"] == "Older" and rows[0]["confirmed_at"] == "2020-01-01"
    assert roster.heads(conn) == {"blog:dwarkesh.com"}
    seed_from_entities(conn, include_x=False)
    assert {r.name for r in registry.list_sources(conn)} == {"Older"}


def test_an_already_registered_blog_row_is_retired_when_substack_covers_it(conn):
    """Registration is not the loop's input. `refresh_all` reads `oracle_sources`, so
    collapsing the pair stops a new duplicate and does nothing about a row already written —
    which would keep re-walking the same publication forever."""
    from pipeline.kb import resolve
    _person(conn, "substack:www.dwarkesh.com", name="Pod", links=["https://www.dwarkesh.com"])
    _person(conn, "blog:dwarkesh.com", name="Dwarkesh", links=["https://www.dwarkesh.com"])
    resolve.resolve_entities(conn)
    head = schema.current_canonical(conn, "blog:dwarkesh.com")
    roster.upsert_oracle(conn, head, name="Dwarkesh Patel", source="screen")
    # The row a pre-collapse seed would have written.
    registry.upsert_source(conn, registry.SourceRow(head, "blog", "https://www.dwarkesh.com"))

    out = seed_from_entities(conn, include_x=False)

    kinds = {(r.source_type, r.source_key) for r in registry.list_sources(conn)}
    assert ("substack", "https://www.dwarkesh.com") in kinds
    assert not [k for k in kinds if k[0] == "blog"], f"blog row survived: {kinds}"
    assert out["retired"] == 1


def test_a_blog_only_oracles_row_is_never_retired(conn):
    """The retire is about a duplicate. A person whose site is not a Substack has one source, and
    retiring it would delete their whole corpus's registration plus its pull history."""
    from pipeline.kb import resolve
    _person(conn, "blog:gajesh.com", name="Gajesh", links=["https://gajesh.com"])
    resolve.resolve_entities(conn)
    roster.upsert_oracle(conn, "blog:gajesh.com", name="Gajesh", source="screen")

    out = seed_from_entities(conn, include_x=False)

    assert {(r.source_type, r.source_key) for r in registry.list_sources(conn)} == \
           {("blog", "https://gajesh.com")}
    assert out["retired"] == 0


def test_a_pair_stranded_under_a_pre_merge_id_is_adopted_with_its_history(conn):
    """The same pair, registered twice, pulled twice — `refresh_all` reads every row in the
    table with no join to `oracles`, so a row left under a dead id keeps costing a request every
    cycle against the API whose rate limit already defers this user's backlog."""
    from pipeline.kb import resolve
    _person(conn, "substack:www.dwarkesh.com", name="Pod", links=["https://www.dwarkesh.com"])
    _person(conn, "blog:dwarkesh.com", name="Dwarkesh", links=["https://www.dwarkesh.com"])
    resolve.resolve_entities(conn)
    head = schema.current_canonical(conn, "blog:dwarkesh.com")
    roster.upsert_oracle(conn, head, name="Dwarkesh Patel", source="screen")
    # The pre-merge row: it is the one that did the pull.
    registry.upsert_source(conn, registry.SourceRow("substack:www.dwarkesh.com", "substack",
                                          "https://www.dwarkesh.com"))
    registry.record_pull(conn, registry.SourceRow("substack:www.dwarkesh.com", "substack",
                                        "https://www.dwarkesh.com"), last_status="ingested")

    seed_from_entities(conn, include_x=False)

    rows = [r for r in registry.list_sources(conn) if r.source_type == "substack"]
    assert len(rows) == 1, f"the pair is still registered twice: {rows}"
    assert rows[0].canonical_id == head and rows[0].name == "Dwarkesh Patel"
    assert rows[0].last_pulled_at, "adoption dropped the pull history and will re-walk the archive"


def test_adoption_never_touches_a_different_persons_rows(conn):
    """Adoption is keyed on `current_canonical` resolving to this Oracle. B is a separate cluster,
    so B's row keeps B's id — a rule that matters because the alternative silently moves one
    person's pull history onto another's registry."""
    from pipeline.kb import resolve
    _person(conn, "x:user:1", name="A", links=[])
    _person(conn, "x:user:2", name="B", links=[])
    resolve.resolve_entities(conn)
    roster.upsert_oracle(conn, "x:user:1", name="A", source="screen")
    registry.upsert_source(conn, registry.SourceRow("x:user:2", "x", "b_handle"))

    seed_from_entities(conn, include_x=True)

    rows = registry.list_sources(conn)
    assert [(r.canonical_id, r.source_key) for r in rows] == [("x:user:2", "b_handle")]


def test_a_merge_that_moves_the_head_takes_the_oracles_pairs_with_it(conn):
    """A site discovery added became the head of a person's cluster, and their X pair stayed under
    the old id until that site's first pull finished, 6.5 minutes later: `progress` showed no X
    side for them and the timeline did not walk them. The merge moves the pairs itself."""
    from pipeline.kb import resolve
    from pipeline.oracles import milestones
    _person(conn, "x:user:1", name="Yoni", links=["https://carol.dev"])
    resolve.resolve_entities(conn)
    roster.upsert_oracle(conn, "x:user:1", name="Yoni", source="screen")
    registry.upsert_source(conn, registry.SourceRow("x:user:1", "x", "yoni"))
    registry.record_pull(conn, registry.SourceRow("x:user:1", "x", "yoni"),
                         last_status="ingested")

    _person(conn, "blog:carol.dev", name="Yoni", links=["https://carol.dev"])
    resolve.resolve_entities(conn)

    head = schema.current_canonical(conn, "x:user:1")
    assert head == "blog:carol.dev"
    (row,) = registry.list_sources(conn)
    assert row.canonical_id == head and row.last_pulled_at
    (pick,) = milestones.for_picks(conn)
    assert pick["newest_posts"] is True and pick["reach"] is not None


def test_a_saved_post_is_not_evidence_that_the_adapter_works(conn):
    """A cluster holding exactly one substack atom — put there by the saved-posts import, because
    the user had bookmarked one of the person's essays. Counting it as evidence would hand the
    publication to the adapter the eligibility gate refuses, leaving its 116 blog essays in the
    store with nothing registered that could refresh them. `entry_mode` is the difference
    between "the user read this" and "this source works"."""
    _atoms(conn, "blog:hyperdimensional.co", "blog", n=3)
    schema.upsert_atom(conn, {"atom_id": "substack:saved-1", "source_type": "substack",
                              "what_kind": "opinion", "who_id": "substack:deanwball",
                              "when_ts": "2026-06-26", "description": "a post the user saved",
                              "entry_mode": "user-saved"})
    pairs = [("substack", "https://www.hyperdimensional.co"),
             ("blog", "https://hyperdimensional.co")]

    kept = _one_adapter_per_publication(
        conn, pairs, ["blog:hyperdimensional.co", "substack:deanwball"])

    assert kept == [("blog", "https://hyperdimensional.co")]
