"""Stage-4 confirm — writing picks into the `oracles` table, the phantom-id guard, idempotency,
and resolve-at-confirm (a raw handle → a resolved oracle). The Substack-URL path is network-free;
the X-handle fetch is monkeypatched (no twitterapi call, no key)."""
from __future__ import annotations

import pytest

from pipeline.kb import oracles, resolve, schema


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
    out = oracles.confirm(conn, canonical_ids=["x:user:1"])
    assert out["confirmed"] == [{"canonical_id": "x:user:1", "name": "Carol", "source": "screen"}]
    assert out["total_oracles"] == 1 and schema.is_oracle(conn, "x:user:1")


def test_confirm_phantom_id_is_guarded_not_written(conn):
    out = oracles.confirm(conn, canonical_ids=["x:user:does-not-exist"])
    assert out["unknown"] == ["x:user:does-not-exist"]
    assert out["confirmed"] == [] and out["total_oracles"] == 0


def test_confirm_is_idempotent(conn):
    _person(conn, "x:user:1", name="Carol")
    oracles.confirm(conn, canonical_ids=["x:user:1"])
    oracles.confirm(conn, canonical_ids=["x:user:1"])
    assert len(schema.list_oracles(conn)) == 1


def test_confirm_uses_canonical_name_across_cluster(conn):
    # signal name sits on the substack row; the X row has the richer name → confirm reflects it
    _person(conn, "x:user:1", name="Carol Ada", links=["https://carol.substack.com"])
    _person(conn, "substack:carol", name="carol", links=["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    canon = schema.get_entity(conn, "x:user:1")["canonical_id"]
    out = oracles.confirm(conn, canonical_ids=[canon])
    assert out["confirmed"][0]["name"] == "Carol Ada"


# ── resolve-at-confirm (the free-form floor) ────────────────────────────────────

def test_resolve_at_confirm_substack_url_needs_no_identity_fetch(conn, no_venue, solo_site):
    """Renamed 2026-09-08: rooting a site URL is no longer network-free, because it asks OpenAlex
    once whether the host is a research venue. What still holds — and what this tested — is that
    a Substack URL keys itself off the URL alone and needs no per-person identity fetch."""
    out = oracles.confirm(conn, add_handles=["https://carol.substack.com"])
    assert out["unresolved"] == [] and len(out["confirmed"]) == 1
    c = out["confirmed"][0]
    assert c["source"] == "freeform" and c["canonical_id"].startswith("substack:")
    assert schema.is_oracle(conn, c["canonical_id"])


def test_resolve_at_confirm_blog_url_mints_a_blog_entity(conn, no_venue, solo_site):
    # A generic http… home is a BLOG, keyed `blog:{host}` — NOT the mis-minted `substack:{host}`
    # the old branch produced for every URL (which sent a personal site to the Substack cluster).
    out = oracles.confirm(conn, add_handles=["https://simonwillison.net"])
    assert out["unresolved"] == [] and len(out["confirmed"]) == 1
    c = out["confirmed"][0]
    assert c["canonical_id"].startswith("blog:") and c["source"] == "freeform"
    assert schema.is_oracle(conn, c["canonical_id"])


def test_resolve_at_confirm_substack_url_stays_substack(conn, no_venue, solo_site):
    # The split must not regress the Substack branch — a substack.com URL still keys `substack:`.
    out = oracles.confirm(conn, add_handles=["https://carol.substack.com"])
    assert out["confirmed"][0]["canonical_id"].startswith("substack:")


def test_resolve_at_confirm_x_handle_mints_and_confirms(conn, monkeypatch):
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: {
        "user_id": "999", "display_name": "Pasted Person", "bio": "builder",
        "site": "https://pasted.com", "verified": True, "followers": 5000, "handle": "pasted"})
    out = oracles.confirm(conn, add_handles=["@pasted"])
    assert len(out["confirmed"]) == 1
    c = out["confirmed"][0]
    assert c["canonical_id"] == "x:user:999" and c["source"] == "freeform"
    ent = schema.get_entity(conn, "x:user:999")
    assert ent is not None and "pasted.com" in (ent["identity_links"] or "")


def test_resolve_at_confirm_unresolved_handle_is_reported_not_crashed(conn):
    # no monkeypatch → the real fetch runs but fails (no key/network) → reported, nothing written
    from unittest import mock
    with mock.patch.object(oracles, "_fetch_x_identity", return_value=None):
        out = oracles.confirm(conn, add_handles=["@ghost"])
    assert out["unresolved"] == ["@ghost"] and out["confirmed"] == []
    assert out["total_oracles"] == 0


def test_confirmed_oracles_carries_footprint_members(conn):
    _person(conn, "x:user:1", name="Carol", links=["https://carol.substack.com"])
    _person(conn, "substack:carol", name="Carol", links=["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    canon = schema.get_entity(conn, "x:user:1")["canonical_id"]
    oracles.confirm(conn, canonical_ids=[canon])
    got = oracles.confirmed_oracles(conn)
    assert len(got) == 1
    members = {m["entity_id"] for m in got[0]["members"]}
    assert members == {"x:user:1", "substack:carol"}          # Stage-5 gets both footprints


# ── the backstop on a rendered diagnostic ──────────────────────────────────────
def test_an_ordinary_summary_renders_whole():
    from pipeline.kb.oracles import _detail

    assert _detail({"source": "x-footprint", "added": 55}) == \
           str({"source": "x-footprint", "added": 55})


def test_a_summary_carrying_a_cache_is_cut_and_says_so():
    """⚠️ THE BACKSTOP, not the fix — adapters not putting caches in summaries is the fix. This
    is what keeps the NEXT one from reaching a host: `str()` renders whatever an adapter returns,
    and one leak already cost `progress` its readability at 110KB."""
    from pipeline.kb.oracles import _DETAIL_MAX, _detail

    out = _detail({"payloads": {f"u{i}": "body" * 200 for i in range(40)}})

    assert len(out) < _DETAIL_MAX + 300
    assert "truncated" in out and "`stats`" in out, "a clipped summary must not read as a short one"


# ── one publication, one adapter ────────────────────────────────────────────────
#
# THE DEFECT THESE EXIST FOR, measured 2026-09-14 on a live store: a Substack on a custom domain
# is two cluster members (`substack:{host}` + `blog:{host}`), resolve correctly merges them into
# one person, and BOTH stayed pullable. Dwarkesh Patel's archive landed 180 times as `blog:`
# atoms and 180 times as `substack:` atoms — same `source_url` on every pair, 8,771 duplicate
# chunks, ~14MB of duplicate text in the embedding index, and no error anywhere. It was invisible
# until `oracle_refresh` began running, because until then only one of the two was ever reached.
from pipeline.kb import oracle_refresh_state as _st


def _atoms(conn, who_id, source_type, n=1):
    """n FOOTPRINT atoms from one adapter — the evidence `_one_adapter_per_publication` decides
    on. `entry_mode` is load-bearing: a `user-saved` atom says the user bookmarked a post, not
    that the adapter can pull the publication, and counting one handed Dean W. Ball's essays to
    the adapter that cannot."""
    for i in range(n):
        schema.upsert_atom(conn, {"atom_id": f"{source_type}:{who_id}:{i}",
                                  "source_type": source_type, "what_kind": "opinion",
                                  "who_id": who_id, "when_ts": "2026-01-01",
                                  "description": "x",
                                  "entry_mode": _st.FOOTPRINT_ENTRY_MODE})


def test_a_substack_on_a_custom_domain_is_pulled_once(conn):
    """Neither adapter has pulled yet, so substack takes the open case."""
    pairs = [("x", "dwarkesh_sp"), ("substack", "https://www.dwarkesh.com"),
             ("blog", "https://www.dwarkesh.com")]
    assert _st._one_adapter_per_publication(conn, pairs, ["blog:dwarkesh.com"]) == [
        ("x", "dwarkesh_sp"), ("substack", "https://www.dwarkesh.com")]


def test_the_www_that_made_two_spellings_look_like_two_publications(conn):
    """The live cluster carried `https://www.hyperdimensional.co` on its substack member and
    `https://hyperdimensional.co` on its blog member. One publication, two strings."""
    pairs = [("substack", "https://www.hyperdimensional.co"),
             ("blog", "https://hyperdimensional.co")]
    assert _st._one_adapter_per_publication(conn, pairs, ["blog:hyperdimensional.co"]) == [
        ("substack", "https://www.hyperdimensional.co")]


def test_the_adapter_that_already_works_keeps_the_publication(conn):
    """⚠️ THE 2026-09-15 CORRECTION, and the case that caught the first rule out. Dean W. Ball's
    essays pull fine from the BLOG adapter while his Substack is refused by the single-author
    eligibility gate — the publication is "Hyperdimensional" and its author is not. Always
    preferring substack retired the source that worked for one that returns `skipped` forever,
    leaving 116 essays in the store with nothing registered that could ever refresh them."""
    _atoms(conn, "blog:hyperdimensional.co", "blog", n=3)
    pairs = [("substack", "https://www.hyperdimensional.co"),
             ("blog", "https://hyperdimensional.co")]

    assert _st._one_adapter_per_publication(conn, pairs, ["blog:hyperdimensional.co"]) == [
        ("blog", "https://hyperdimensional.co")]


def test_substack_keeps_the_publication_once_it_has_actually_pulled(conn):
    """The winner keeps winning, so the choice cannot oscillate between passes."""
    _atoms(conn, "blog:dwarkesh.com", "blog", n=2)
    _atoms(conn, "substack:www.dwarkesh.com", "substack", n=2)
    pairs = [("substack", "https://www.dwarkesh.com"), ("blog", "https://www.dwarkesh.com")]

    assert _st._one_adapter_per_publication(
        conn, pairs, ["blog:dwarkesh.com", "substack:www.dwarkesh.com"]) == [
        ("substack", "https://www.dwarkesh.com")]


def test_a_plain_blog_keeps_its_blog_pair(conn):
    """The collapse is about a DUPLICATE, never about preferring one adapter. Someone whose site
    is not a Substack has only the blog adapter, and dropping it would drop their whole corpus."""
    pairs = [("x", "gajesh"), ("blog", "https://gajesh.com")]
    assert _st._one_adapter_per_publication(conn, pairs, ["blog:gajesh.com"]) == pairs


def test_two_different_publications_both_survive(conn):
    pairs = [("substack", "https://a.substack.com"), ("blog", "https://b.example.com")]
    assert _st._one_adapter_per_publication(conn, pairs, ["x:user:1"]) == pairs


def test_pairs_for_oracle_collapses_a_real_merged_cluster(conn):
    from pipeline.kb import resolve
    _person(conn, "x:user:1", name="Dwarkesh", links=["https://www.dwarkesh.com"])
    _person(conn, "substack:www.dwarkesh.com", name="Pod", links=["https://www.dwarkesh.com"])
    _person(conn, "blog:dwarkesh.com", name="Dwarkesh", links=["https://www.dwarkesh.com"])
    resolve.resolve_entities(conn)
    head = schema.current_canonical(conn, "x:user:1")

    pairs, _who = _st.pairs_for_oracle(conn, head)

    assert ("substack", "https://www.dwarkesh.com") in pairs
    assert not [p for p in pairs if p[0] == "blog"], f"blog pair survived: {pairs}"


# ── the stale cluster head ──────────────────────────────────────────────────────
def test_a_merge_that_moves_the_head_repoints_the_oracle_row(conn):
    """⚠️ ONE STALE ROW, THREE WRONG ANSWERS — `trusted_atoms` 80 against a true 884, an Oracle's
    display name resolving to NULL (so 182 posts were attributed to "substack"), and a second
    `oracle_sources` registration under the dead id. `current_canonical` absorbs this on read and
    `confirmed_oracles` remembers to call it; every other join did not."""
    from pipeline.kb import resolve
    _person(conn, "x:user:9", name="Dwarkesh", links=["https://www.dwarkesh.com"])
    resolve.resolve_entities(conn)
    schema.upsert_oracle(conn, "x:user:9", name="Dwarkesh", source="screen")

    # The blog member arrives later and sorts below the X id, moving the head.
    _person(conn, "blog:dwarkesh.com", name="Dwarkesh", links=["https://www.dwarkesh.com"])
    resolve.resolve_entities(conn)

    head = schema.current_canonical(conn, "x:user:9")
    assert head == "blog:dwarkesh.com"
    assert [o["canonical_id"] for o in schema.list_oracles(conn)] == [head]
    assert schema.is_oracle(conn, head) and not schema.is_oracle(conn, "x:user:9")


def test_the_name_join_that_silently_returned_null_now_resolves(conn):
    """The concrete cost of the stale row: `_dispatch` passes `row.name` as `author_name`, and a
    NULL there is what made 182 substack atoms read "substack ·" instead of the author."""
    from pipeline.kb import resolve
    _person(conn, "x:user:9", name="Dwarkesh", links=["https://www.dwarkesh.com"])
    resolve.resolve_entities(conn)
    schema.upsert_oracle(conn, "x:user:9", name="Dwarkesh Patel", source="screen")
    _person(conn, "blog:dwarkesh.com", name="Dwarkesh", links=["https://www.dwarkesh.com"])
    resolve.resolve_entities(conn)

    _st.seed_from_entities(conn)

    assert {r.name for r in _st.list_sources(conn)} == {"Dwarkesh Patel"}


def test_two_oracles_that_turn_out_to_be_one_person_keep_the_older_confirmation(conn):
    """`canonical_id` is the primary key, so a merge can collide. The oldest confirmation is the
    one the user would recognise; `INSERT OR REPLACE` would pick by insertion order instead."""
    from pipeline.kb import resolve
    _person(conn, "x:user:9", name="D", links=["https://www.dwarkesh.com"])
    _person(conn, "blog:dwarkesh.com", name="D", links=[])
    resolve.resolve_entities(conn)
    schema.upsert_oracle(conn, "blog:dwarkesh.com", name="Older", source="screen")
    conn.execute("UPDATE oracles SET confirmed_at='2020-01-01' WHERE canonical_id=?",
                 ("blog:dwarkesh.com",))
    schema.upsert_oracle(conn, "x:user:9", name="Newer", source="screen")
    conn.commit()

    # Link them: now one cluster, head = blog:dwarkesh.com (sorts first).
    schema.upsert_entity(conn, "blog:dwarkesh.com", name="D",
                         identity_links=["https://www.dwarkesh.com"])
    resolve.resolve_entities(conn)

    rows = schema.list_oracles(conn)
    assert len(rows) == 1 and rows[0]["canonical_id"] == "blog:dwarkesh.com"
    assert rows[0]["name"] == "Older" and rows[0]["confirmed_at"] == "2020-01-01"


def test_an_already_registered_blog_row_is_retired_when_substack_covers_it(conn):
    """⚠️ REGISTRATION IS NOT THE LOOP'S INPUT. `refresh_all` reads `oracle_sources`, so
    collapsing the pair stops a NEW duplicate and does nothing about the row an earlier version
    already wrote — which would keep re-walking the same publication forever."""
    from pipeline.kb import resolve
    _person(conn, "substack:www.dwarkesh.com", name="Pod", links=["https://www.dwarkesh.com"])
    _person(conn, "blog:dwarkesh.com", name="Dwarkesh", links=["https://www.dwarkesh.com"])
    resolve.resolve_entities(conn)
    head = schema.current_canonical(conn, "blog:dwarkesh.com")
    schema.upsert_oracle(conn, head, name="Dwarkesh Patel", source="screen")
    # The row a pre-collapse seed would have written.
    _st.upsert_source(conn, _st.SourceRow(head, "blog", "https://www.dwarkesh.com"))

    out = _st.seed_from_entities(conn, include_x=False)

    kinds = {(r.source_type, r.source_key) for r in _st.list_sources(conn)}
    assert ("substack", "https://www.dwarkesh.com") in kinds
    assert not [k for k in kinds if k[0] == "blog"], f"blog row survived: {kinds}"
    assert out["retired"] == 1


def test_a_blog_only_oracles_row_is_never_retired(conn):
    """The retire is about a DUPLICATE. A person whose site is not a Substack has one source, and
    retiring it would delete their whole corpus's registration plus its pull history."""
    from pipeline.kb import resolve
    _person(conn, "blog:gajesh.com", name="Gajesh", links=["https://gajesh.com"])
    resolve.resolve_entities(conn)
    schema.upsert_oracle(conn, "blog:gajesh.com", name="Gajesh", source="screen")

    out = _st.seed_from_entities(conn, include_x=False)

    assert {(r.source_type, r.source_key) for r in _st.list_sources(conn)} == \
           {("blog", "https://gajesh.com")}
    assert out["retired"] == 0


def test_a_pair_stranded_under_a_pre_merge_id_is_adopted_with_its_history(conn):
    """⚠️ THE SAME PAIR, REGISTERED TWICE, PULLED TWICE — `refresh_all` reads every row in the
    table with no join to `oracles`, so a row left under a dead id keeps costing a request every
    cycle against the API whose rate limit already defers this user's backlog."""
    from pipeline.kb import resolve
    _person(conn, "substack:www.dwarkesh.com", name="Pod", links=["https://www.dwarkesh.com"])
    _person(conn, "blog:dwarkesh.com", name="Dwarkesh", links=["https://www.dwarkesh.com"])
    resolve.resolve_entities(conn)
    head = schema.current_canonical(conn, "blog:dwarkesh.com")
    schema.upsert_oracle(conn, head, name="Dwarkesh Patel", source="screen")
    # The pre-merge row: it is the one that DID the pull.
    _st.upsert_source(conn, _st.SourceRow("substack:www.dwarkesh.com", "substack",
                                          "https://www.dwarkesh.com"))
    _st.record_pull(conn, _st.SourceRow("substack:www.dwarkesh.com", "substack",
                                        "https://www.dwarkesh.com"), last_status="ingested")

    _st.seed_from_entities(conn, include_x=False)

    rows = [r for r in _st.list_sources(conn) if r.source_type == "substack"]
    assert len(rows) == 1, f"the pair is still registered twice: {rows}"
    assert rows[0].canonical_id == head and rows[0].name == "Dwarkesh Patel"
    assert rows[0].last_pulled_at, "adoption dropped the pull history and will re-walk the archive"


def test_adoption_never_touches_a_different_persons_rows(conn):
    """Adoption is keyed on `current_canonical` resolving to THIS Oracle. B is a separate cluster,
    so B's row keeps B's id — a rule that matters because the alternative silently moves one
    person's pull history onto another's registry."""
    from pipeline.kb import resolve
    _person(conn, "x:user:1", name="A", links=[])
    _person(conn, "x:user:2", name="B", links=[])
    resolve.resolve_entities(conn)
    schema.upsert_oracle(conn, "x:user:1", name="A", source="screen")
    _st.upsert_source(conn, _st.SourceRow("x:user:2", "x", "b_handle"))

    _st.seed_from_entities(conn, include_x=True)

    rows = _st.list_sources(conn)
    assert [(r.canonical_id, r.source_key) for r in rows] == [("x:user:2", "b_handle")]


def test_a_saved_post_is_not_evidence_that_the_adapter_works(conn):
    """⚠️ THE 2026-09-15 REGRESSION, in one test. Dean W. Ball's cluster held exactly ONE substack
    atom — put there by the SAVED-POSTS import, because the user had bookmarked one of his essays.
    Counting it as evidence handed his publication to the adapter the eligibility gate refuses,
    and his 116 blog essays were left in the store with nothing registered that could refresh
    them. `entry_mode` is the difference between "the user read this" and "this source works"."""
    _atoms(conn, "blog:hyperdimensional.co", "blog", n=3)
    schema.upsert_atom(conn, {"atom_id": "substack:saved-1", "source_type": "substack",
                              "what_kind": "opinion", "who_id": "substack:deanwball",
                              "when_ts": "2026-06-26", "description": "a post the user saved",
                              "entry_mode": "user-saved"})
    pairs = [("substack", "https://www.hyperdimensional.co"),
             ("blog", "https://hyperdimensional.co")]

    kept = _st._one_adapter_per_publication(
        conn, pairs, ["blog:hyperdimensional.co", "substack:deanwball"])

    assert kept == [("blog", "https://hyperdimensional.co")]
