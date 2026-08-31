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

def test_resolve_at_confirm_substack_url_is_network_free(conn):
    out = oracles.confirm(conn, add_handles=["https://carol.substack.com"])
    assert out["unresolved"] == [] and len(out["confirmed"]) == 1
    c = out["confirmed"][0]
    assert c["source"] == "freeform" and c["canonical_id"].startswith("substack:")
    assert schema.is_oracle(conn, c["canonical_id"])


def test_resolve_at_confirm_blog_url_mints_a_blog_entity(conn):
    # A generic http… home is a BLOG, keyed `blog:{host}` — NOT the mis-minted `substack:{host}`
    # the old branch produced for every URL (which sent a personal site to the Substack cluster).
    out = oracles.confirm(conn, add_handles=["https://simonwillison.net"])
    assert out["unresolved"] == [] and len(out["confirmed"]) == 1
    c = out["confirmed"][0]
    assert c["canonical_id"].startswith("blog:") and c["source"] == "freeform"
    assert schema.is_oracle(conn, c["canonical_id"])


def test_resolve_at_confirm_substack_url_stays_substack(conn):
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
