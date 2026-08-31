"""retrieve.resolve_who — a handle/URL/id → the entity cluster(s) it names.

The read-side lookup, and deliberately NOT the one in `oracles._resolve_handle`: that one pays
twitterapi.io and mints entities (correct for onboarding), this one is pure local SQL over what
the store already has (the only thing a query is allowed to cost).

The load-bearing test here is `test_a_single_id_would_have_missed_half_their_work`: an atom
carries its author's PER-PLATFORM id, so one person is several ids, and a resolver that returned
just one would report a fraction of someone's output as all of it.
"""
from __future__ import annotations

import pytest

from pipeline.kb import resolve, schema
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.raw_store import write_snapshot
from pipeline.kb.retrieve import candidate_atom_ids, resolve_who, search_atoms

X_ID = "x:user:33836629"
GH_ID = "github:Karpathy"          # the API's casing — NOT what a user types
BLOG_ID = "blog:karpathy.github.io"


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _person(conn):
    """One person, three platform rows, cross-linked by a shared home URL so Stage-3 resolution
    merges them into a single cluster — the ordinary shape after an Oracle footprint expand."""
    home = "https://karpathy.github.io"
    schema.upsert_entity(conn, X_ID, name="Andrej Karpathy", identity_links=[home], profile={"handle": "karpathy"})
    schema.upsert_entity(conn, GH_ID, name="karpathy", identity_links=[home])
    schema.upsert_entity(conn, BLOG_ID, name="Andrej Karpathy", identity_links=[home])
    resolve.resolve_entities(conn)


def _atom(conn, emb, atom_id, source_type, who_id, text):
    raw_ref, raw_hash = write_snapshot(source_type, atom_id, text)
    store_atom(conn, emb, atom=dict(
        atom_id=atom_id, source_type=source_type, what_kind="opinion", who_id=who_id,
        when_ts="2024-05-01", when_precision="day", about_entities=[],
        source_url=f"https://example/{atom_id}", raw_ref=raw_ref, raw_hash=raw_hash,
        description=f"{atom_id} card", payload={}, entry_mode="user-saved",
    ), snapshot_text=text)


# ── the three storage shapes one handle can live in ────────────────────────────

def test_x_handle_resolves_through_the_profile_not_the_id(conn):
    # X's id is numeric, so the handle is NOT reconstructable from it — it lives in `profile`.
    _person(conn)
    got = resolve_who(conn, "@karpathy")
    assert len(got) == 1
    assert got[0]["name"] == "Andrej Karpathy"
    assert got[0]["handle"] == "karpathy"


def test_the_at_sign_is_optional(conn):
    _person(conn)
    assert resolve_who(conn, "karpathy") == resolve_who(conn, "@karpathy")


def test_github_handle_matches_the_apis_casing_not_the_users(conn):
    # GitHub logins are case-insensitive but the atom keys on the API's canonical casing, so a
    # user typing the lowercase form must still land on `github:Karpathy`.
    schema.upsert_entity(conn, GH_ID, name="karpathy")
    assert [c["who_ids"] for c in resolve_who(conn, "karpathy")] == [[GH_ID]]


def test_a_blog_url_resolves_through_the_same_id_the_ingest_side_minted(conn):
    _person(conn)
    got = resolve_who(conn, "https://karpathy.github.io")
    assert len(got) == 1 and BLOG_ID in got[0]["who_ids"]


def test_an_id_passes_straight_through(conn):
    # A caller that already HAS an id shouldn't need a different entry point.
    _person(conn)
    assert resolve_who(conn, X_ID)[0]["canonical_id"] \
        == resolve_who(conn, "karpathy")[0]["canonical_id"]


# ── the cluster, and why one id is not enough ─────────────────────────────────

def test_resolve_returns_every_platform_id_for_one_person(conn):
    _person(conn)
    got = resolve_who(conn, "karpathy")
    assert len(got) == 1, "three platform rows for one person must collapse to ONE cluster"
    assert got[0]["who_ids"] == sorted([X_ID, GH_ID, BLOG_ID])


def test_a_single_id_would_have_missed_half_their_work(conn, fake_embedder):
    """The reason `who_ids` is a list. An atom carries its author's PER-PLATFORM id, so
    filtering on any ONE of them silently returns a fraction of what that person wrote —
    and presents it as everything."""
    _person(conn)
    _atom(conn, fake_embedder, "x:1", "x", X_ID, "thoughts on an autonomous agent")
    _atom(conn, fake_embedder, "github:Karpathy/nanogpt", "github", GH_ID,
          "a minimal agent framework")

    one_id = candidate_atom_ids(conn, None, None, None, X_ID)
    whole_person = candidate_atom_ids(conn, None, None, None,
                                      resolve_who(conn, "karpathy")[0]["who_ids"])
    assert one_id == {"x:1"}                                        # half their output…
    assert whole_person == {"x:1", "github:Karpathy/nanogpt"}       # …vs all of it


def test_the_cluster_is_only_as_good_as_stage3_resolution(conn):
    """The known ceiling on this whole feature, pinned so it is a documented boundary rather
    than a surprise. Stage-3 unions on `self ∩ attests` — someone has to BE a URL for the
    entities that LINK to it to merge. Two platform rows that merely both point at the same
    home stay SEPARATE, because two different people can each link to one company page.

    The consequence for the caller: `resolve_who` reports two clusters for one person, and a
    caller that takes only the first gets half their work. It UNDER-returns; it never returns
    the wrong person's atoms. Fixing it means attesting the link (Stage-3's job), not loosening
    the match here — a looser rule would merge strangers who share an employer."""
    home = "https://acme.example.com"
    schema.upsert_entity(conn, "github:root", name="Root", identity_links=[home])
    schema.upsert_entity(conn, X_ID, name="Root", identity_links=[home], profile={"handle": "root"})
    resolve.resolve_entities(conn)
    assert len(resolve_who(conn, "root")) == 2       # one PERSON, two clusters — not yet merged

    # Add the entity that IS that home, and the same two rows now merge into one cluster.
    schema.upsert_entity(conn, "blog:acme.example.com", name="Root", identity_links=[home])
    resolve.resolve_entities(conn)
    assert len(resolve_who(conn, "root")) == 1


def test_two_different_people_stay_two_clusters(conn):
    # The same handle string on two platforms with NO attested link between them. Reporting one
    # merged person would be a false identity claim, so they stay separate and the caller
    # decides — the resolver never guesses a merge that Stage-3 didn't attest.
    schema.upsert_entity(conn, "github:karpathy", name="A")
    schema.upsert_entity(conn, "substack:karpathy", name="B")
    got = resolve_who(conn, "karpathy")
    assert len(got) == 2
    assert {c["who_ids"][0] for c in got} == {"github:karpathy", "substack:karpathy"}


# ── fail-safe ─────────────────────────────────────────────────────────────────

def test_an_unknown_handle_resolves_to_nobody(conn):
    _person(conn)
    assert resolve_who(conn, "@someone-else-entirely") == []


def test_empty_input_resolves_to_nobody(conn):
    assert resolve_who(conn, "") == []
    assert resolve_who(conn, None) == []


def test_resolve_is_read_only(conn):
    """It must never mint an entity for someone the store has never seen. Unlike the onboarding
    resolver, this runs on every query — inventing a row per unresolved handle would let a
    search write to the store."""
    before = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    resolve_who(conn, "@nobody")
    resolve_who(conn, "https://nobody.example.com")
    assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == before


def test_a_bare_handle_does_not_seed_a_bogus_blog_id(conn):
    # `blog_entity_id` will happily turn "karpathy" into `blog:karpathy`; the URL arm only runs
    # when the input actually looks like one, or every handle would match a phantom blog.
    schema.upsert_entity(conn, "blog:karpathy", name="Not them")
    assert resolve_who(conn, "karpathy") == []


# ── the empty-author-set trap ─────────────────────────────────────────────────

def test_an_empty_id_list_matches_nothing_never_everything(conn, fake_embedder):
    """A failed resolution yields an empty author list. If the pre-filter read that as "no
    filter", the caller would get the WHOLE store back and present a stranger's atoms as the
    person they asked for. The empty list must mean nobody."""
    _person(conn)
    _atom(conn, fake_embedder, "x:1", "x", X_ID, "thoughts on an autonomous agent")
    assert candidate_atom_ids(conn, None, None, None, []) == set()
    assert search_atoms(conn, "agent", fake_embedder, who_id=[]).hits == []


def test_an_empty_string_still_means_no_filter(conn, fake_embedder):
    # The other empty case, and it reads the opposite way on purpose: a host filling an unused
    # optional slot with "" means "no filter", not "nobody".
    _person(conn)
    _atom(conn, fake_embedder, "x:1", "x", X_ID, "thoughts on an autonomous agent")
    assert candidate_atom_ids(conn, None, None, None, "") is None
