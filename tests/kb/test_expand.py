"""Stage-5 footprint helpers (expand.py) — rooting + routing, fully offline.

`expand` owns no loop: it roots an Oracle for discovery (`_root_profile`)
and holds the lookback presets. Routing one discovered source is `first_pull.onboard_footprint`'s,
tested in `test_onboard_footprint.py`. It has no driver of its own (the
loop is `oracles._ingest_oracle`), so these tests exercise the helpers directly.
"""
from __future__ import annotations

import pytest

from pipeline.kb import expand, resolve, schema
from pipeline.oracles import windows
from pipeline.timeparse import utc_now
from pipeline.oracles import roster
from pipeline.oracles import admit


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _oracle_person(conn, eid="x:user:1", *, name="Carol", handle="carol"):
    """Upsert an X person (handle in profile), resolve so canonical_id=self, confirm as Oracle."""
    schema.upsert_entity(conn, eid, name=name, profile={"handle": handle} if handle else None)
    resolve.resolve_entities(conn)          # sets canonical_id (Phase-2 does this in the real flow)
    admit.confirm(conn, canonical_ids=[eid])
    return [o for o in roster.confirmed_oracles(conn) if o["canonical_id"] == eid][0]


# ── rooting: which profile discovery starts from ──────────────────────────────

def test_x_member_roots_on_its_handle(conn):
    o = _oracle_person(conn)
    assert expand._root_profile(conn, o) == {"seed": "carol", "seed_type": "x"}


def _oracle_substack(conn, eid="substack:carol", *, name="Carol"):
    """Upsert a Substack-ONLY person (no x:user member), resolve, confirm as Oracle."""
    schema.upsert_entity(conn, eid, name=name, identity_links=["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    admit.confirm(conn, canonical_ids=[eid])
    return [o for o in roster.confirmed_oracles(conn) if o["canonical_id"] == eid][0]


def test_substack_only_oracle_roots_on_its_substack(conn):
    """De-X-rooting: a Substack-ONLY Oracle roots on that handle, with no X anywhere."""
    o = _oracle_substack(conn)
    assert expand._root_profile(conn, o) == {"seed": "carol", "seed_type": "substack"}


def test_blog_only_oracle_roots_on_its_blog(conn):
    schema.upsert_entity(conn, "blog:carol.dev", name="Carol",
                         identity_links=["https://carol.dev"])
    resolve.resolve_entities(conn)
    admit.confirm(conn, canonical_ids=["blog:carol.dev"])
    o = [x for x in roster.confirmed_oracles(conn) if x["canonical_id"] == "blog:carol.dev"][0]
    assert expand._root_profile(conn, o) == {"seed": "https://carol.dev", "seed_type": "blog"}


def test_no_rootable_profile_is_none(conn):
    """An X person with no handle and no Substack/blog member → nothing to root discovery on.
    None, not a crash: the caller reports it."""
    schema.upsert_entity(conn, "x:user:2", name="NoHandle")  # no profile.handle
    resolve.resolve_entities(conn)
    admit.confirm(conn, canonical_ids=["x:user:2"])
    o = [x for x in roster.confirmed_oracles(conn) if x["canonical_id"] == "x:user:2"][0]
    assert expand._root_profile(conn, o) is None


def test_oracle_read_survives_canonical_shift(conn):
    """Regression: a footprint merge AFTER confirm shifts the cluster head (blog: sorts below
    x:user:), so the oracle's STORED canonical_id goes stale. confirmed_oracles + _x_handle must
    still recover the full cluster + handle by re-anchoring to the current head."""
    schema.upsert_entity(conn, "x:user:1", name="Carol", profile={"handle": "carol"})
    resolve.resolve_entities(conn)                              # head = x:user:1
    admit.confirm(conn, canonical_ids=["x:user:1"])
    # simulate the post-footprint-resolve state: blog:carol merged in and became the new head
    schema.upsert_entity(conn, "blog:carol", name="carol")
    conn.execute("UPDATE entities SET canonical_id='blog:carol' "
                 "WHERE entity_id IN ('x:user:1', 'blog:carol')")
    conn.commit()
    o = roster.confirmed_oracles(conn)[0]
    assert o["canonical_id"] == "blog:carol"                    # reports the CURRENT head
    assert {m["entity_id"] for m in o["members"]} == {"x:user:1", "blog:carol"}
    assert expand._x_handle(conn, "x:user:1") == "carol"        # stale stored id still resolves


# ── the profile pages discovery reads ──────────────────────

def test_member_profiles_are_the_root_then_every_other_profile_of_theirs(conn):
    """The root as `_root_profile` picks it, then each other profile page of theirs: another X,
    each Substack read by the publication it is (its id may be an author handle, which names no
    publication), each GitHub account, each ORCID. A blog's home is not a profile page, an
    OpenAlex author with no ORCID has none, and one publication is read once."""
    schema.upsert_entity(conn, "x:user:1", name="Carol", profile={"handle": "carol"})
    schema.upsert_entity(conn, "x:user:2", name="Carol Labs", profile={"handle": "carollabs"})
    conn.execute("UPDATE entities SET canonical_id='x:user:1'")
    conn.commit()
    o = {"canonical_id": "x:user:1", "members": [
        {"entity_id": "x:user:1", "identity_links": None},
        {"entity_id": "x:user:2", "identity_links": None},
        {"entity_id": "blog:carol.dev", "identity_links": '["https://carol.dev"]'},
        {"entity_id": "substack:carolwrites", "identity_links": '["https://www.carol.news"]'},
        {"entity_id": "substack:www.carol.news", "identity_links": '["https://www.carol.news"]'},
        {"entity_id": "github:Carol", "identity_links": None},
        {"entity_id": "openalex:A1",
         "identity_links": '["https://orcid.org/0000-0001-2345-6789"]'},
        {"entity_id": "openalex:A2", "identity_links": None}]}
    assert expand.member_profiles(conn, o) == [
        {"seed": "carol", "seed_type": "x"},
        {"seed": "carollabs", "seed_type": "x"},
        {"seed": "https://www.carol.news", "seed_type": "substack"},
        {"seed": "Carol", "seed_type": "github"},
        {"seed": "0000-0001-2345-6789", "seed_type": "orcid"}]


def test_a_substack_root_is_not_read_again_under_its_authors_handle(conn):
    """The Art of the Realizable: `substack:realizable` roots it, and `substack:mraginsky` is
    its author, the same publication."""
    o = {"canonical_id": "substack:realizable", "members": [
        {"entity_id": "substack:realizable",
         "identity_links": '["https://realizable.substack.com"]'},
        {"entity_id": "substack:mraginsky",
         "identity_links": '["https://realizable.substack.com"]'}]}
    assert expand.member_profiles(conn, o) == [{"seed": "realizable", "seed_type": "substack"}]


def test_no_root_reads_no_profile(conn):
    """A GitHub-only Oracle has no root, so discovery reads nothing for it, as before."""
    o = {"canonical_id": "github:carol", "members": [{"entity_id": "github:carol"}]}
    assert expand.member_profiles(conn, o) == []


# ── Lookback selectors (onboarding: X window + Substack/blog window) ───────────

def test_lookback_presets_match_spec():
    # X: 6mo/1yr/2yr (a hard 2yr ceiling). Substack/blog: 1yr/2yr/5yr/all ('all' = no bound).
    assert windows.X_LOOKBACK_PRESETS == {"6mo": 183, "1yr": 365, "2yr": 730}
    assert windows.WEB_LOOKBACK_PRESETS == {"1yr": 365, "2yr": 730, "5yr": 1825, "all": None}
    assert windows.since("web", "all", utc_now()) is None                   # 'all' → no lower bound
    assert windows.since("web", "1yr", utc_now()) is not None


# ── identity_links helpers ────────────────────────────────────────────────────

def test_first_url_reads_json_string_or_list():
    assert expand._first_url('["https://a.dev", "x"]') == "https://a.dev"
    assert expand._first_url(["https://b.dev"]) == "https://b.dev"
    assert expand._first_url(None) is None
    assert expand._first_url("not-a-url") is None


def test_blog_home_reconstructed_from_id_when_no_links():
    o = {"members": [{"entity_id": "blog:simonwillison.net", "identity_links": None}]}
    assert expand._blog_home(o) == "https://simonwillison.net"
