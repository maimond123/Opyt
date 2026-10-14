"""Stage 6 — academic identity URLs as Oracle roots, and the ORCID merge rule.

The rootable set is what OPYT can pull, which is OpenAlex and nothing else: an ORCID (one lookup
away from an author id) and an OpenAlex author URL (the id itself). Every other registry is
refused by name.

Offline: the OpenAlex lookups are monkeypatched at the adapter.
"""
from __future__ import annotations

import pytest

from pipeline.platforms.papers import openalex
from pipeline.kb import resolve, schema
from pipeline.oracles import reference
from pipeline.oracles import identify


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


_ORCID = "https://orcid.org/0000-0002-4027-364X"
_ARNOLD = {"id": "https://openalex.org/A5043841592", "display_name": "Frances H. Arnold",
           "works_count": 928}


# ── admission ────────────────────────────────────────────────────────────────────

def test_an_academic_identity_url_no_longer_mints_a_blog_entity(conn, monkeypatch):
    """orcid.org must not be admitted as a blog: it would become `blog:orcid.org/0000-…`, and
    `oracle_refresh` would point the blog adapter at ORCID and scrape a researcher's identity
    record as if it were their archive."""
    monkeypatch.setattr(openalex.OpenAlexClient, "author_by_orcid", lambda self, o: _ARNOLD)

    cid = identify.resolve_handle(conn, _ORCID)

    assert cid == "openalex:A5043841592"
    assert not conn.execute("SELECT 1 FROM entities WHERE entity_id LIKE 'blog:%'").fetchone()


def test_an_orcid_resolves_to_the_id_that_has_a_works_feed(conn, monkeypatch):
    """`orcid:{id}` would be an Oracle whose every refresh found nothing — only an OpenAlex author
    id has a works feed. The ORCID is still stored as the identity link, which is what merges this
    entity with a `scholar:` one for the same person."""
    monkeypatch.setattr(openalex.OpenAlexClient, "author_by_orcid", lambda self, o: _ARNOLD)

    identify.resolve_handle(conn, _ORCID)

    row = schema.get_entity(conn, "openalex:A5043841592")
    assert row["name"] == "Frances H. Arnold"
    assert _ORCID in (row["identity_links"] or "")


def test_a_failed_orcid_lookup_refuses_rather_than_falling_back_to_a_blog(conn, monkeypatch):
    """Falling back is the exact behaviour this change exists to stop."""
    def _boom(self, o):
        raise RuntimeError("openalex is down")
    monkeypatch.setattr(openalex.OpenAlexClient, "author_by_orcid", _boom)

    assert identify.resolve_handle(conn, _ORCID) is None
    assert not conn.execute("SELECT 1 FROM entities").fetchone()


def test_a_semantic_scholar_author_page_is_refused_and_writes_nothing(conn):
    """A `scholar:{id}` entity would have no pullable pair, so the URL would confirm an Oracle
    with zero sources.

    OpenAlex is what OPYT pulls papers through, and it subsumes Semantic Scholar as a works index
    (Karpathy 24/24 keyed papers present; Arnold 398/408, 9 of the 10 misses being Protein Data
    Bank depositions). So the root has to be an id OpenAlex is keyed by."""
    assert identify.resolve_handle(
        conn, "https://www.semanticscholar.org/author/F-Arnold/2081297") is None
    assert not conn.execute("SELECT 1 FROM entities").fetchone()


def test_an_unpullable_registry_is_refused_without_a_network_call(conn, monkeypatch):
    """`confirm(add_handles=[…])` reaches `_resolve_handle` without passing `_unsupported_root`,
    so the refusal has to live here too. An unpullable prefix must not fall through to the ORCID
    lookup: a refusal that depends on a network call failing is not one."""
    monkeypatch.setattr(openalex.OpenAlexClient, "author_by_orcid",
                        lambda self, o: pytest.fail(f"no lookup should be made for {o!r}"))

    assert identify.resolve_handle(
        conn, "https://www.semanticscholar.org/author/F-Arnold/2081297") is None


# ── the OpenAlex author URL ──────────────────────────────────────────────────────

def test_an_openalex_author_url_roots_on_the_id_it_carries(conn, monkeypatch):
    """The one academic URL that needs no lookup to be pullable — it is the id the pull takes.

    It is also the shape the refusal copy names, which is what makes refusing Semantic Scholar
    cheap for a researcher who has no ORCID."""
    monkeypatch.setattr(openalex.OpenAlexClient, "author",
                        lambda self, aid: {"id": f"https://openalex.org/{aid}",
                                           "display_name": "Andrej Karpathy", "works_count": 25})

    cid = identify.resolve_handle(conn, "https://openalex.org/A5009290031")

    assert cid == "openalex:A5009290031"
    assert schema.get_entity(conn, cid)["name"] == "Andrej Karpathy"
    assert not conn.execute("SELECT 1 FROM entities WHERE entity_id LIKE 'blog:%'").fetchone()


def test_an_openalex_author_url_is_not_name_searched(conn, monkeypatch):
    """An `A…` id resolves through `author`, an `S…` through `source`, and neither pays the venue
    name search. Sending `openalex` to `sources_by_name` is what produced `blog:openalex.org`."""
    monkeypatch.setattr(openalex.OpenAlexClient, "author", lambda self, aid: None)
    monkeypatch.setattr(openalex.OpenAlexClient, "sources_by_name",
                        lambda self, n: pytest.fail("a pasted id must not be name-searched"))

    assert identify.resolve_handle(conn, "https://openalex.org/A5009290031") \
        == "openalex:A5009290031"


# ── every pullable root really is pullable ────────────────────────────────────────────

# One sample URL per prefix `_SCHOLAR_ROOTS` marks pullable. The test below fails when a prefix is
# marked True without one, so the claim is always checked against the thing that does the pulling.
_PULLABLE_SAMPLES = {"orcid.org/": _ORCID}


def test_every_root_marked_pullable_actually_yields_a_pullable_pair(conn, monkeypatch):
    """`_SCHOLAR_ROOTS`'s value means "OPYT can turn this URL into an id it can pull work by", and
    `oracle_refresh_state.pair_from_member` is what decides that. The two must agree: a True entry
    whose entity yields no pair is an Oracle confirmed with zero sources."""
    from pipeline.oracles import pairs

    monkeypatch.setattr(openalex.OpenAlexClient, "author_by_orcid", lambda self, o: _ARNOLD)
    pullable = [p for p, ok in reference.SCHOLAR_ROOTS.items() if ok]

    assert set(pullable) == set(_PULLABLE_SAMPLES), "a pullable prefix needs a sample URL here"
    for prefix in pullable:
        cid = identify.resolve_handle(conn, _PULLABLE_SAMPLES[prefix])
        assert cid, prefix
        assert pairs.pair_from_member(schema.get_entity(conn, cid)) is not None, prefix


@pytest.mark.parametrize("url", [
    "https://www.semanticscholar.org/author/F-Arnold/2081297",
    "https://scholar.google.com/citations?user=abc",
    "https://dblp.org/pid/12/3456.html",
    "https://www.researchgate.net/profile/Frances-Arnold",
    "https://arxiv.org/a/arnold_f_1",
])
def test_a_registry_with_no_works_feed_is_refused_with_the_shape_that_works(url):
    """Refused, not admitted-and-broken. These publish no free author→works feed, so an entity
    minted from one would be an Oracle whose every refresh had nothing to pull. The message names
    what to pass instead, the same way the X branch does."""
    reason = reference.unsupported_root(url)

    assert reason and "no author→works feed" in reason
    assert "orcid.org" in reason and "openalex.org/A" in reason


def test_a_personal_site_is_still_a_blog_root():
    assert reference.unsupported_root("https://simonwillison.net") is None


# ── the merge ────────────────────────────────────────────────────────────────────

def test_two_registries_sharing_an_orcid_become_one_person(conn):
    """A separate `openalex:` prefix costs nothing. Both carry the ORCID, both
    count it as self, so the existing union-find merges them with no new code."""
    schema.upsert_entity(conn, "openalex:A5043841592", name="Frances H. Arnold",
                         identity_links=[_ORCID])
    schema.upsert_entity(conn, "scholar:2081297", name="F. Arnold", identity_links=[_ORCID])
    conn.commit()

    resolve.resolve_entities(conn)

    heads = {r[0] for r in conn.execute(
        "SELECT COALESCE(canonical_id, entity_id) FROM entities")}
    assert len(heads) == 1


def test_two_researchers_sharing_an_institution_do_not_merge(conn):
    """Only an ORCID counts as self. Two people in one department share a lab page;
    nobody can claim another person's ORCID."""
    lab = "https://www.caltech.edu/people/arnold-group"
    schema.upsert_entity(conn, "openalex:A1", name="One", identity_links=[lab])
    schema.upsert_entity(conn, "openalex:A2", name="Two", identity_links=[lab])
    conn.commit()

    resolve.resolve_entities(conn)

    heads = {r[0] for r in conn.execute(
        "SELECT COALESCE(canonical_id, entity_id) FROM entities")}
    assert heads == {"openalex:A1", "openalex:A2"}


def test_only_the_orcid_counts_as_self_even_when_other_links_are_stored(conn):
    """Asserted at `_url_sets` rather than only through a merge, because this is what stops the
    rule's safety depending on every distant writer being disciplined about `identity_links`."""
    selfs, attests = resolve._url_sets(
        "openalex:A1", [_ORCID, "https://www.caltech.edu/people/arnold-group"])

    assert selfs == frozenset({"orcid.org/0000-0002-4027-364X"})
    assert "orcid.org/0000-0002-4027-364X" not in attests
    assert attests                                   # the lab page is an outbound attestation
