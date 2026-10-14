"""The Oracle-refresh freshness registry: flat TTLs, the atoms-derived cursor, and seeding.

Every assertion runs against the atom rail — `atoms`/`entities`/`oracles`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pipeline.kb import schema
from pipeline.oracles import windows
from pipeline.oracles import roster
from pipeline.oracles import registry
from pipeline.oracles import pairs

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)


def _row(stype: str, *, last=None, cursor=None, key="k"):
    return registry.SourceRow(canonical_id="x:user:1", source_type=stype, source_key=key,
                        last_pulled_at=last, cursor_ts=cursor)


def _iso(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def _seed_atom(conn, atom_id, *, source_type, who_id, when_ts):
    schema.upsert_atom(conn, {"atom_id": atom_id, "source_type": source_type,
                              "who_id": who_id, "when_ts": when_ts, "description": "d"})


def _member(conn, entity_id, *, profile=None, identity_links=None):
    """The exact row shape `schema.entities_for_canonical` supplies to source registration."""
    schema.upsert_entity(conn, entity_id, profile=profile, identity_links=identity_links)
    return conn.execute(
        "SELECT entity_id, identity_links, profile FROM entities WHERE entity_id=?", (entity_id,)
    ).fetchone()


# ── flat TTLs ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("stype,base", [("x", 72.0), ("substack", 168.0),
                                        ("blog", 336.0), ("github", 336.0)])
def test_ttl_boundary_per_type(stype, base):
    """Stale iff elapsed >= the pair's effective TTL — the whole staleness rule, one line.
    Effective = the type's flat base × this pair's stable jitter, so the boundary is asserted
    against `pair_ttl_hours` rather than a hardcoded number that jitter would make a lie."""
    ttl = windows.pair_ttl_hours(_row(stype))
    assert windows.ttl_hours(stype) == base
    assert abs(ttl - base) <= base * windows.TTL_JITTER
    assert not windows.is_stale(_row(stype, last=_iso(ttl - 1)), NOW)
    assert windows.is_stale(_row(stype, last=_iso(ttl)), NOW)
    assert windows.is_stale(_row(stype, last=_iso(ttl + 1)), NOW)


def test_never_pulled_is_infinitely_stale_and_sorts_first():
    never = _row("x")
    assert windows.is_stale(never, NOW)
    assert windows.staleness_hours(never, NOW) == float("inf")
    lagged = _row("x", last=_iso(100))
    assert windows.staleness_hours(lagged, NOW) == pytest.approx(100 - windows.pair_ttl_hours(lagged))


# ── TTL jitter ──────────────────────────────────────────────────────────────────
def test_jitter_is_stable_for_a_given_pair():
    """The load-bearing property. A `random()` here would make `is_stale` nondeterministic —
    the exact thing the repeat-run harness verifies. Same key must mean same factor, always."""
    args = ("x:user:1", "x", "willccbb")
    assert windows.jitter_factor(*args) == windows.jitter_factor(*args) == windows.jitter_factor(*args)
    row = _row("x", last=_iso(70))
    assert [windows.is_stale(row, NOW) for _ in range(5)] == [windows.is_stale(row, NOW)] * 5


def test_jitter_stays_inside_the_band():
    factors = [windows.jitter_factor("x:user:1", "blog", f"https://s{i:03d}.com") for i in range(500)]
    assert min(factors) >= 1 - windows.TTL_JITTER
    assert max(factors) <= 1 + windows.TTL_JITTER


def test_jitter_de_synchronizes_a_batch_seeded_at_one_instant():
    """Every pair an `add_oracle` registers inherits the SAME `oracles.ingest_to`, so without
    jitter the whole batch falls due in the same second — and re-clusters every cycle, because a
    batch refreshed together gets stamped together. This asserts the batch FANS OUT instead."""
    rows = [registry.SourceRow(canonical_id="x:user:1", source_type="blog",
                         source_key=f"https://s{i:03d}.com", last_pulled_at=_iso(320))
            for i in range(200)]
    due = [windows.is_stale(r, NOW) for r in rows]
    assert 0 < sum(due) < len(rows), "the batch should split, not fire (or stall) as one block"
    spread = max(windows.pair_ttl_hours(r) for r in rows) - min(windows.pair_ttl_hours(r) for r in rows)
    assert spread > 40                                  # ~67h of fan-out on a 336h base


def test_jitter_differs_across_source_types_for_one_oracle():
    """The key includes source_type, so an Oracle's own pairs do not share a factor either."""
    f = [windows.jitter_factor("x:user:1", t, "k") for t in ("x", "substack", "blog", "github")]
    assert len(set(f)) == 4


def test_unparseable_stamp_is_stale_not_a_crash():
    """Fail-safe: a corrupt cache row re-pulls rather than raising."""
    assert windows.is_stale(_row("x", last="not-a-date"), NOW)


def test_unknown_source_type_falls_back_to_a_week():
    assert windows.ttl_hours("youtube") == windows.DEFAULT_TTL_HOURS


# ── the cursor, over the cluster not one actor ──────────────────────────────────
def test_latest_atom_ts_maxes_over_every_cluster_member(kb_home):
    conn = schema.connect()
    registry.init_state_schema(conn)
    _seed_atom(conn, "x:1", source_type="x", who_id="x:user:1", when_ts="2026-07-01")
    _seed_atom(conn, "x:2", source_type="x", who_id="x:user:1", when_ts="2026-07-20")
    _seed_atom(conn, "b:1", source_type="blog", who_id="blog:a.com", when_ts="2026-08-01")
    try:
        assert registry.latest_atom_ts(conn, "x", ["x:user:1", "blog:a.com"]) == "2026-07-20"
        assert registry.latest_atom_ts(conn, "blog", ["x:user:1", "blog:a.com"]) == "2026-08-01"
        assert registry.latest_atom_ts(conn, "github", ["x:user:1"]) is None
        assert registry.latest_atom_ts(conn, "x", []) is None          # no members → no cursor, no SQL
    finally:
        conn.close()


# ── entity → pair mapping ───────────────────────────────────────────────────────
def test_pair_from_member_per_platform(kb_home):
    conn = schema.connect()
    try:
        assert pairs.pair_from_member(_member(conn, "x:user:9", profile={"handle": "willccbb"})) \
            == ("x", "willccbb")
        assert pairs.pair_from_member(_member(conn, "substack:foo",
                                            identity_links=["https://foo.substack.com"])) \
            == ("substack", "https://foo.substack.com")
        assert pairs.pair_from_member(_member(conn, "blog:willcb.com")) == ("blog", "https://willcb.com")
        assert pairs.pair_from_member(_member(conn, "github:willccbb")) == ("github", "willccbb")
    finally:
        conn.close()


def test_pair_from_member_refuses_unpullable_members(kb_home):
    # An X entity with no stored handle: the adapter pulls `from:handle`, so an id is not enough.
    conn = schema.connect()
    try:
        assert pairs.pair_from_member(_member(conn, "x:user:9")) is None
        assert pairs.pair_from_member(_member(conn, "org:acme.com")) is None
        assert pairs.pair_from_member(_member(conn, "blog:unknown")) is None
        # `github:{owner}/{name}` is an atom id, never a feed.
        assert pairs.pair_from_member(_member(conn, "github:willccbb/vllm")) is None
    finally:
        conn.close()


def test_substack_handle_id_reconstructs_a_publication_url(kb_home):
    """`substack_entity_id` keys on the author handle when it has one and on the host otherwise —
    a dot is the only thing telling the two apart."""
    conn = schema.connect()
    try:
        assert pairs.pair_from_member(_member(conn, "substack:bob")) == ("substack",
                                                                         "https://bob.substack.com")
        assert pairs.pair_from_member(_member(conn, "substack:news.example.com")) \
            == ("substack", "https://news.example.com")
    finally:
        conn.close()


def test_github_owner_recovered_from_identity_links():
    """A GitHub node that never folded into the canonical cluster."""
    assert pairs.github_owners_from_links('["https://github.com/willccbb"]') == ["willccbb"]
    assert pairs.github_owners_from_links('["https://github.com/o/repo"]') == []
    assert pairs.github_owners_from_links(None) == []


# ── seeding ─────────────────────────────────────────────────────────────────────
def _make_oracle(conn, *, with_github_link=False):
    links = ["https://willcb.com"] + (["https://github.com/willccbb"] if with_github_link else [])
    schema.upsert_entity(conn, "x:user:1", name="Will", identity_links=links, profile={"handle": "willccbb"})
    schema.upsert_entity(conn, "blog:willcb.com", name="Will", identity_links=["https://willcb.com"])
    schema.set_canonical_ids(conn, {"x:user:1": "x:user:1", "blog:willcb.com": "x:user:1"})
    roster.upsert_oracle(conn, "x:user:1", name="Will")


def test_seed_registers_one_row_per_source_and_is_idempotent(kb_home):
    conn = schema.connect()
    registry.init_state_schema(conn)
    try:
        _make_oracle(conn, with_github_link=True)
        _seed_atom(conn, "x:1", source_type="x", who_id="x:user:1", when_ts="2026-07-20")

        first = pairs.seed_from_entities(conn)
        rows = registry.list_sources(conn)
        assert first["pairs"] == 3
        assert {(r.source_type, r.source_key) for r in rows} == {
            ("x", "willccbb"), ("blog", "https://willcb.com"), ("github", "willccbb")}
        assert all(r.status == "trusted" for r in rows)
        # cursor comes from the corpus; seeding stamps no pull, so `last_pulled_at` stays NULL
        assert next(r for r in rows if r.source_type == "x").cursor_ts == "2026-07-20"
        assert next(r for r in rows if r.source_type == "x").last_pulled_at is None

        pairs.seed_from_entities(conn)
        assert len(registry.list_sources(conn)) == 3        # re-running duplicates nothing
    finally:
        conn.close()


def test_seeding_claims_no_coverage(kb_home):
    """Seeding registers rows; it must never claim a pull happened.

    Copying `oracles.ingest_to` into `last_pulled_at` would be wrong: that column is written
    unconditionally at the end of an onboarding ingest, including on a run whose X pull raised.
    A person with zero X atoms would get a row claiming a fresh X pull, `upsert_source`'s
    COALESCE would freeze it, and no rail would ever pick them up again."""
    conn = schema.connect()
    registry.init_state_schema(conn)
    try:
        _make_oracle(conn)
        pairs.seed_from_entities(conn)
        row = next(r for r in registry.list_sources(conn) if r.source_type == "x")
        assert row.last_pulled_at is None
        assert row.covered_from is None
        assert windows.is_stale(row, NOW)                       # infinitely stale → sorts first
        assert windows.staleness_hours(row, NOW) == float("inf")
    finally:
        conn.close()


def test_reseed_never_rewinds_a_pair_the_loop_already_refreshed(kb_home):
    """The registry is re-seeded after every ingest, so this is the property that keeps that safe:
    a stored `last_pulled_at` must survive a later seed."""
    conn = schema.connect()
    registry.init_state_schema(conn)
    try:
        _make_oracle(conn)
        pairs.seed_from_entities(conn)
        row = next(r for r in registry.list_sources(conn) if r.source_type == "x")
        registry.record_pull(conn, row, last_status="ingested", cursor_ts="2026-08-08", stamp=True)
        fresh_stamp = next(r for r in registry.list_sources(conn)
                           if r.source_type == "x").last_pulled_at

        pairs.seed_from_entities(conn)
        assert next(r for r in registry.list_sources(conn)
                    if r.source_type == "x").last_pulled_at == fresh_stamp
    finally:
        conn.close()


# ── covered_from: the backward frontier ────────────────────────────────────────
def test_covered_from_only_widens(kb_home):
    """`covered_from` answers "how far back do we go", so a narrower later pull must not shrink
    it — the atoms from the wider pull are still in the store."""
    conn = schema.connect()
    registry.init_state_schema(conn)
    try:
        _make_oracle(conn)
        pairs.seed_from_entities(conn)
        row = next(r for r in registry.list_sources(conn) if r.source_type == "x")

        registry.record_pull(conn, row, last_status="ingested", covered_from="2026-03-01", stamp=True)
        assert _x(conn).covered_from == "2026-03-01"
        registry.record_pull(conn, row, last_status="ingested", covered_from="2026-06-01", stamp=True)
        assert _x(conn).covered_from == "2026-03-01"       # narrower → unchanged
        registry.record_pull(conn, row, last_status="ingested", covered_from="2025-01-01", stamp=True)
        assert _x(conn).covered_from == "2025-01-01"       # wider → adopted
    finally:
        conn.close()


def test_covered_from_none_leaves_the_frontier_alone(kb_home):
    """None means "this pull had no lower bound to report", not "unbounded" — so a stored NULL
    means exactly one thing, no lower bound recorded."""
    conn = schema.connect()
    registry.init_state_schema(conn)
    try:
        _make_oracle(conn)
        pairs.seed_from_entities(conn)
        row = next(r for r in registry.list_sources(conn) if r.source_type == "x")
        registry.record_pull(conn, row, last_status="ingested", covered_from="2026-03-01", stamp=True)
        registry.record_pull(conn, row, last_status="ingested", covered_from=None, stamp=True)
        assert _x(conn).covered_from == "2026-03-01"
    finally:
        conn.close()


def _x(conn):
    return next(r for r in registry.list_sources(conn) if r.source_type == "x")


def test_record_pull_stamp_false_leaves_the_pair_stale(kb_home):
    """A blocked host is not an author who went quiet — it must not buy a full TTL of silence."""
    conn = schema.connect()
    registry.init_state_schema(conn)
    try:
        _make_oracle(conn)
        pairs.seed_from_entities(conn)
        row = next(r for r in registry.list_sources(conn) if r.source_type == "x")
        registry.record_pull(conn, row, last_status="blocked", stamp=False)
        after = next(r for r in registry.list_sources(conn) if r.source_type == "x")
        assert after.last_status == "blocked"
        assert after.last_pulled_at is None
        assert windows.is_stale(after, NOW)
    finally:
        conn.close()


def test_requested_from_only_widens_and_none_leaves_it(kb_home):
    """The durable record that someone asked for a window. Same widen-only
    contract as `covered_from` — a later shallower request must not erase a 1-year ask, and None
    means "this pull carried no user window", never "clear it"."""
    conn = schema.connect()
    registry.init_state_schema(conn)
    try:
        _make_oracle(conn)
        pairs.seed_from_entities(conn)
        row = _x(conn)

        registry.record_request(conn, row, "2026-03-01")
        assert _x(conn).requested_from == "2026-03-01"
        registry.record_request(conn, row, "2026-06-01")
        assert _x(conn).requested_from == "2026-03-01"     # narrower → unchanged
        registry.record_request(conn, row, "2025-01-01")
        assert _x(conn).requested_from == "2025-01-01"     # wider → adopted
        registry.record_request(conn, row, None)
        assert _x(conn).requested_from == "2025-01-01"

        # And the outcome writer never touches it — one owner for the column.
        registry.record_pull(conn, row, last_status="ingested", stamp=True)
        assert _x(conn).requested_from == "2025-01-01"
    finally:
        conn.close()


@pytest.mark.parametrize("orphan_pulled_later", [True, False])
def test_adopting_an_orphan_keeps_its_topic_filter_and_the_older_coverage(kb_home,
                                                                          orphan_pulled_later):
    """When the head's row already exists, adopting an orphan must carry more than the pull
    history: the OpenAlex pair's `topic_filter` (else the next walk runs unfiltered) and the older
    `covered_from` (it must never narrow)."""
    conn = schema.connect()
    try:
        registry.init_state_schema(conn)
        schema.upsert_entity(conn, "x:user:1", name="A")
        schema.upsert_entity(conn, "x:user:2", name="A")
        early, late = "2026-01-01T00:00:00+00:00", "2026-09-01T00:00:00+00:00"
        registry.upsert_source(conn, registry.SourceRow(
            "x:user:1", "openalex", "A123", topic_filter="T1", covered_from="2020-01-01",
            last_pulled_at=late if orphan_pulled_later else early))
        registry.upsert_source(conn, registry.SourceRow(
            "x:user:2", "openalex", "A123", covered_from="2024-01-01",
            last_pulled_at=early if orphan_pulled_later else late))
        schema.set_canonical_ids(conn, {"x:user:1": "x:user:2", "x:user:2": "x:user:2"})

        assert pairs.adopt_orphaned_sources(conn) == 1

        (row,) = registry.list_sources(conn)
        assert (row.canonical_id, row.topic_filter, row.covered_from) == (
            "x:user:2", "T1", "2020-01-01")
        assert row.last_pulled_at == late
    finally:
        conn.close()
