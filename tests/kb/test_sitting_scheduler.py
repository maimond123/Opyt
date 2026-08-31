"""The scheduler — which region gets read next, proven offline against planted geometry.

The claims here are the ones that would be invisible in production if they broke:

  • THE PRIORITY ORDER IS D16's. A region somebody pointed at outranks a fracture's leftovers. Get
    this wrong and six sub-regions off one fracture starve the region a person actually asked for,
    while every counter reports healthy progress.
  • ONE PAID READ PER RUN. This is what replaced the deleted daily cap, so it is the whole runaway
    guard. A second read in one pass is the failure it exists to stop.
  • NEW MASS IS COUNTED ON `first_seen`. Keyed on `ingested_at` instead, one re-scrape pass makes
    the entire corpus look new and every region claims a re-read at once, at full price, having
    gained nothing. The two columns are equal until the first re-observation, which is exactly why
    this needs a test rather than an eyeball.
  • THE BREAKER STOPS A FAILING LOOP, and `ever_ran` catches the loop that never started — the
    failure a breaker is structurally blind to, and this repo's actual one.
  • A CONTINUATION REUSES THE STORED ANCHOR. Re-resolving the phrase would grow a DIFFERENT region
    wearing the same name and inherit the chain's redundancy baseline anyway.
  • FRACTURE-OR-CONTINUE IS DECIDED BY SEPARABILITY. `ZOOM_K_MIN = 2` forces a split, so the
    single-conversation case is the one that can silently go wrong: both arms are tested.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from pipeline.timeparse import utc_iso

import numpy as np
import pytest

from pipeline import llm_client
from pipeline.kb import frontier_queries as fq
from pipeline.kb import ingest_common
from pipeline.kb import schema
from pipeline.kb import sitting_builder as sb
from pipeline.kb import sitting_reader as sr
from pipeline.kb import sitting_store as sst
from pipeline.kb import sitting_scheduler as ss

from .conftest import last_run

# Wide for the same reason `test_sitting_zoom` is wide: at 8 dimensions the noise term has nowhere
# to live and unrelated members land at cosine ~0.4 by chance, which trips the near-duplicate
# ceiling and turns a planted 12-atom cluster into one admitted atom.
DIM = 24
_NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
_USAGE = {"model": "test/model", "in_tokens": 900, "out_tokens": 300, "cost_usd": 0.0021}


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    from pipeline.kb.embed import ensure_kb_meta
    ensure_kb_meta(c, "fake", DIM, "local", "", storage_dtype="float32")
    yield c
    c.close()


# ── geometry ────────────────────────────────────────────────────────────────────
def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _axis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _atom(conn, atom_id, vec, *, when="2026-08-01", who="x:user:1", chars=800,
          first_seen: str | None = None, ingested_at: str | None = None,
          entry_mode: str = "user-saved"):
    conn.execute("INSERT INTO atoms (atom_id, source_type, who_id, when_ts, entry_mode) "
                 "VALUES (?,?,?,?,?)", (atom_id, "x", who, when, entry_mode))
    if first_seen is not None:
        conn.execute("UPDATE atoms SET first_seen = ? WHERE atom_id = ?", (first_seen, atom_id))
    if ingested_at is not None:
        conn.execute("UPDATE atoms SET ingested_at = ? WHERE atom_id = ?", (ingested_at, atom_id))
    text = f"{atom_id} body " + ("word " * max(1, chars // 5))
    conn.execute("INSERT INTO chunks (atom_id, seq, char_start, char_end, text, vector) "
                 "VALUES (?,0,0,?,?,?)", (atom_id, len(text), text, _unit(vec).tobytes()))
    conn.commit()


# Same weights as the zoom fixture, and the noise term is load-bearing for the same reason: points
# an epsilon apart are near-duplicates at redundancy ~1.0, so the builder SKIPS them and a planted
# cluster arrives as a single admitted atom.
SHARED, CLUSTER, NOISE = 0.55, 0.669, 0.5


def _planted(conn, axes, per=12, *, rng_seed=0, **kw):
    """One cluster per axis, `per` atoms each — a region whose true part count is KNOWN."""
    rng = np.random.default_rng(rng_seed)
    made = []
    for ax in axes:
        for i in range(per):
            g = rng.normal(size=DIM).astype(np.float32)
            g[0] = g[ax] = 0.0
            g /= np.linalg.norm(g) + 1e-9
            v = np.zeros(DIM, dtype=np.float32)
            v[0], v[ax] = SHARED, CLUSTER
            aid = f"a:c{ax}-{i}"
            _atom(conn, aid, _unit(v + NOISE * g), **kw)
            made.append(aid)
    return made


def _region(conn, label="mlx", *, floor=0.68, when=_NOW, **kw):
    """A two-atom region seeded from an atom — the cheap shape for the queue tests."""
    _atom(conn, f"a:{label}-seed", _axis(0), who="x:alice", **kw)
    _atom(conn, f"a:{label}-near", _unit(0.85 * _axis(0) + 0.53 * _axis(1)), who="x:bob", **kw)
    return sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=[f"a:{label}-seed"], label=label),
                            floor=floor, now=when)


# ── a stubbed transport ─────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, text):
        self.text, self.model = text, "fake-model"
        self.input_tokens, self.output_tokens, self.cost_usd = 100, 20, 0.01
        self.raw = {}


def _reply(role, *, system, user, **kw):
    """A stubbed read that CITES THE SITTING IT WAS GIVEN.

    Reading the atom ids back out of the rendered prompt rather than hard-coding one is not
    decoration: `reader_core.validate` drops a query whose ids resolve to nothing, and a run with no
    surviving query is recorded as FAILED. A fixed citation therefore turns every scheduler test
    that reads a planted region into a failure test by accident.
    """
    # Anchored on the per-atom HEADING, not on any parenthesis: the document header also carries
    # `(centroid)` and `(vector seed)`, and a loose match picks those up as citations.
    ids = re.findall(r"^### .*\(([^()]+)\)\s*$", user, re.M)
    return _Resp(json.dumps({"consensus": "the arc moved", "queries": [
        {"text": "gated DeltaNet attention", "target_sources": ["arxiv"],
         "rationale": "because", "atom_ids": ids[:2]}]}))


@pytest.fixture()
def ready(monkeypatch):
    """A usable backend whose call is stubbed. Reads succeed and spend nothing."""
    monkeypatch.setattr(llm_client, "preflight", lambda role: None)
    monkeypatch.setattr("pipeline.ingestion.utils.load_yaml_config", lambda: {})
    monkeypatch.delenv("OPYT_FRONTIER_BACKEND", raising=False)
    monkeypatch.setattr(llm_client, "call", _reply)


@pytest.fixture()
def broken(monkeypatch):
    """A backend whose every call raises — the shape the breaker exists for."""
    monkeypatch.setattr(llm_client, "preflight", lambda role: None)
    monkeypatch.setattr("pipeline.ingestion.utils.load_yaml_config", lambda: {})
    monkeypatch.delenv("OPYT_FRONTIER_BACKEND", raising=False)

    def _boom(role, **kw):
        raise RuntimeError("provider exploded")
    monkeypatch.setattr(llm_client, "call", _boom)


def _reads(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM sittings WHERE read_at IS NOT NULL").fetchone()[0]


def _asked(conn, sitting_id):
    """Evidence that a queries read was attempted here and did not land — what promotes a built
    region into the `pointed` RETRY lane. A bare build queues nothing (Option C)."""
    fq.record_run(conn, generator="sitting:mlx", sitting_id=sitting_id,
                  status="failed", reason="provider exploded")


# ── the queue ───────────────────────────────────────────────────────────────────
def test_an_empty_store_claims_nothing_and_spends_nothing(conn):
    res = ss.run_sitting_scheduler(conn, now=_NOW)
    assert res["status"] == "skipped" and res["reason"] == "nothing claimable"
    assert _reads(conn) == 0


def test_a_previewed_region_nobody_read_is_never_claimed(conn, ready):
    """Option C, the money leak pinned. Consumption subscribes; construction does not — a bare
    `preview` (a build with no failed-read evidence) must sit forever, never entering the queue and
    never getting an unattended paid read."""
    _region(conn, "mlx", when=_NOW)
    assert ss.claims(conn) == []
    res = ss.run_sitting_scheduler(conn, now=_NOW)
    assert res["status"] == "skipped" and res["reason"] == "nothing claimable"
    assert _reads(conn) == 0


def test_a_read_that_failed_is_retried(conn, ready):
    """The other half of Option C: a region whose `queries` read was attempted and did not land is
    NOT a bare preview — it is claimed as `pointed` and read on the next pass."""
    rec = _region(conn, "mlx", when=_NOW)
    _asked(conn, rec["sitting_id"])
    queue = ss.claims(conn)
    assert [c["channel"] for c in queue] == ["pointed"]
    assert queue[0]["sitting_id"] == rec["sitting_id"]
    res = ss.run_sitting_scheduler(conn, now=_NOW)
    assert res["status"] == "ok" and res["claim"]["channel"] == "pointed"
    assert _reads(conn) == 1


def test_a_host_lens_receipt_alone_is_never_claimed(conn):
    """Option C Part 2, the leak Part 1 alone left open. A briefing (or any host-side lens) writes
    a RECEIPT row via `sitting_reader.record_lens_run` — `lens` set, no consensus, no queries —
    and that receipt alone must not make the region look like a failed `queries` attempt.
    Without the `lens` filter, asking for a free briefing would make the scheduler spend an API
    call generating standing queries nobody asked for — the exact bug Option C exists to close, one
    layer down. See docs/plans/2026-08-16-lens-reads-subscribe-a-region.md, Part 2."""
    rec = _region(conn, "mlx", when=_NOW)
    sr.record_lens_run(conn, rec["sitting_id"], "briefing", usage=_USAGE, ref=_NOW)
    assert ss.claims(conn) == []
    res = ss.run_sitting_scheduler(conn, now=_NOW)
    assert res["status"] == "skipped" and res["reason"] == "nothing claimable"
    assert _reads(conn) == 0


def test_a_failed_queries_attempt_is_still_retried_alongside_a_lens_receipt(conn, ready):
    """The receipt and the failure-evidence row are not mutually exclusive in practice — a region
    can be both previewed, lens-read, AND have had a failed `queries` attempt. The lens receipt
    must not mask the retry evidence that is genuinely there."""
    rec = _region(conn, "mlx", when=_NOW)
    sr.record_lens_run(conn, rec["sitting_id"], "trajectory", usage=_USAGE, ref=_NOW)
    _asked(conn, rec["sitting_id"])
    queue = ss.claims(conn)
    assert [c["channel"] for c in queue] == ["pointed"]
    res = ss.run_sitting_scheduler(conn, now=_NOW)
    assert res["status"] == "ok" and res["claim"]["channel"] == "pointed"


def _fake_parent(conn, sitting_id, *, read_at=None, built_at):
    """A minimal `sittings` row with `region_key IS NULL` — invisible to `_new_mass_claims` (which
    requires `region_key IS NOT NULL`). Building it through `_region` + `_read_now` instead would
    make it a real, geometrically-anchored region and pull unrelated fixture atoms into its
    new-mass count as an unwanted side effect."""
    conn.execute(
        "INSERT INTO sittings (sitting_id, built_at, seed_kind, floor, ceiling, "
        "  budget_tokens, stop, read_at) VALUES (?,?,?,?,?,?,?,?)",
        (sitting_id, utc_iso(built_at), "query", 0.68, 0.9, 4000, "saturation",
         utc_iso(read_at) if read_at else None))
    conn.commit()


def test_a_fracture_off_an_unread_parent_is_never_claimed(conn):
    """The new parent clause. Nobody consumed the parent, so its sub-region is not a leftover from
    a read region — it is a hand-run `zoom` nobody asked for, and Option C must not claim it."""
    _fake_parent(conn, "unread-parent", built_at=_NOW - timedelta(days=25))
    leftover = _region(conn, "sub", when=_NOW - timedelta(days=5))
    conn.execute("UPDATE sittings SET parent_sitting_id = ? WHERE sitting_id = ?",
                 ("unread-parent", leftover["sitting_id"]))
    conn.commit()
    assert ss.claims(conn) == []


def test_a_pointed_region_outranks_a_fracture_leftover(conn):
    """D16. Sub-regions ARE unread regions, so a coverage-first ranking would let six of them off
    one fracture starve the region a person actually asked for."""
    _fake_parent(conn, "a-parent", built_at=_NOW - timedelta(days=26), read_at=_NOW - timedelta(days=25))
    leftover = _region(conn, "sub", when=_NOW - timedelta(days=5))
    conn.execute("UPDATE sittings SET parent_sitting_id = ? WHERE sitting_id = ?",
                 ("a-parent", leftover["sitting_id"]))
    conn.commit()
    pointed = _region(conn, "mlx", when=_NOW)
    _asked(conn, pointed["sitting_id"])

    queue = ss.claims(conn)
    assert [c["channel"] for c in queue] == ["pointed", "sub_region"]
    # The leftover is FIVE DAYS OLDER and still loses. Age only orders within a tier.
    assert queue[0]["sitting_id"] == pointed["sitting_id"]


def test_only_one_claim_is_taken_per_run(conn, ready):
    """The bound that replaced the deleted daily cap. Two pointed regions, one pass, one read."""
    a = _region(conn, "mlx", when=_NOW - timedelta(days=1))
    b = _region(conn, "kiss", when=_NOW)
    _asked(conn, a["sitting_id"])
    _asked(conn, b["sitting_id"])
    res = ss.run_sitting_scheduler(conn, now=_NOW)
    assert res["status"] == "ok" and res["queued"] == 2
    assert _reads(conn) == 1


def test_the_oldest_pointed_region_goes_first(conn):
    """Within a tier, age decides: a region that has been waiting is the one rotting."""
    old = _region(conn, "mlx", when=_NOW - timedelta(days=3))
    new = _region(conn, "kiss", when=_NOW)
    _asked(conn, old["sitting_id"])
    _asked(conn, new["sitting_id"])
    assert ss.claims(conn)[0]["sitting_id"] == old["sitting_id"]


# ── new mass ────────────────────────────────────────────────────────────────────
def _read_now(conn, sitting_id, at=_NOW):
    sst.mark_read(conn, sitting_id, status="ok", at=at)


# Every arrival stamp is a full DAY after `_NOW`, and that gap is load-bearing rather than tidy:
# new mass is "arrived since the last read", the fixtures read at `_NOW`, and an atom stamped the
# same hour would sit on the wrong side of a strict `>` for reasons that have nothing to do with
# what the test is checking.
_LATER = "2026-08-17 09:00:00"
_LATER_STILL = "2026-08-17 10:00:00"
_RAN = _NOW + timedelta(days=1, hours=2)


def _arrive(conn, n, *, first_seen, ingested_at=None, near=0.9, tag="a", **kw):
    """`n` atoms that clear the floor against axis 0, arriving at a stated time."""
    for i in range(n):
        _atom(conn, f"a:new-{tag}-{i}",
              _unit(near * _axis(0) + np.sqrt(1 - near ** 2) * _axis(2 + i % 4)),
              first_seen=first_seen, ingested_at=ingested_at, **kw)


def test_new_mass_is_counted_on_first_seen_not_ingested_at(conn):
    """THE TRAP THIS COLUMN EXISTS FOR. `upsert_atom` refreshes `ingested_at` on every
    re-observation, so keyed on it a re-scrape of old saves reads as a corpus full of new material
    and every region claims a re-read at once, at full price, having gained nothing."""
    rec = _region(conn, "mlx", first_seen="2026-01-01 00:00:00")
    _read_now(conn, rec["sitting_id"])
    # Old arrivals, re-observed AFTER the read: `ingested_at` says new, `first_seen` says January.
    _arrive(conn, 8, first_seen="2026-01-02 00:00:00", ingested_at=_LATER)
    assert ss.claims(conn) == []

    # The same eight atoms, this time genuinely new.
    conn.execute("UPDATE atoms SET first_seen = ? WHERE atom_id LIKE 'a:new-%'", (_LATER,))
    conn.commit()
    queue = ss.claims(conn)
    assert [c["channel"] for c in queue] == ["new_mass"] and queue[0]["new_mass"] == 8


def test_frontier_arrivals_never_open_the_wallet(conn):
    """RULED 2026-08-24: new mass counts HUMAN-lane arrivals only.

    Frontier atoms enter sittings and get read — they just cannot TRIGGER a paid read. Counting
    them closes the loop (a read mints queries, the queries pull atoms, those atoms trigger the
    next paid read), which makes the spend cadence machine-determined instead of tracking David's
    own engagement rate. Enforced twice over — `_arrivals` and `_relevance` both filter on
    `HUMAN_ATTESTED` — and this pins the OUTCOME, so either filter alone regressing is visible.
    """
    rec = _region(conn, "mlx", first_seen="2026-01-01 00:00:00")
    _read_now(conn, rec["sitting_id"])
    _arrive(conn, 8, first_seen=_LATER, entry_mode="frontier")
    assert ss.claims(conn) == []

    # The same eight atoms, human-attested: the identical mass now claims a re-read.
    conn.execute("UPDATE atoms SET entry_mode = 'user-saved' WHERE atom_id LIKE 'a:new-%'")
    conn.commit()
    queue = ss.claims(conn)
    assert [c["channel"] for c in queue] == ["new_mass"] and queue[0]["new_mass"] == 8


def test_a_promoted_atom_arrives_when_it_was_promoted_not_when_it_was_crawled(conn):
    """PROMOTION OPENS THE WALLET (RULED 2026-08-25) — and it is `promoted_at` that makes it so.

    A frontier atom's `first_seen` is the date the crawler found it, typically months before the
    deposit that promoted it. Keyed on `first_seen` alone, eight brand-new engagements arrive
    already stale, clear no `>` against the last read, and buy nothing at all — the ruling would be
    inert while looking implemented. `_arrivals` reads `COALESCE(promoted_at, first_seen)`.
    """
    rec = _region(conn, "mlx", first_seen="2026-01-01 00:00:00")
    _read_now(conn, rec["sitting_id"])
    # Crawled in January, long before the read: invisible to the trigger while it stays machine-lane.
    _arrive(conn, 8, first_seen="2026-01-02 00:00:00", entry_mode="frontier")
    assert ss.claims(conn) == []

    for aid in [r[0] for r in conn.execute("SELECT atom_id FROM atoms WHERE atom_id LIKE 'a:new-%'")]:
        ingest_common.promote_atom(conn, aid, "user-saved")
    queue = ss.claims(conn)
    assert [c["channel"] for c in queue] == ["new_mass"] and queue[0]["new_mass"] == 8
    # `first_seen` is untouched by promotion: the arrival date is still the arrival date.
    assert conn.execute("SELECT first_seen FROM atoms WHERE atom_id = 'a:new-a-0'").fetchone()[0] \
        == "2026-01-02 00:00:00"


def test_new_mass_below_the_threshold_makes_no_claim(conn):
    """`max(3, 0.20 x size)` — two new atoms is under the floor of three, so nothing is owed."""
    rec = _region(conn, "mlx", first_seen="2026-01-01 00:00:00")
    _read_now(conn, rec["sitting_id"])
    _arrive(conn, 2, first_seen=_LATER)
    assert ss.claims(conn) == []


def test_a_big_region_needs_proportionally_more_new_material(conn):
    """`max`, not `min`: reading cost tracks region SIZE, so an expensive region has to justify
    itself with more than the three atoms that would trigger a small one."""
    _planted(conn, (1,), per=25, first_seen="2026-01-01 00:00:00")
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, vector=_axis(0), label="big"),
                           floor=0.50, now=_NOW)
    _read_now(conn, rec["sitting_id"])
    size = rec["region_atoms"] + len(rec["seed_atom_ids"])
    assert size == 25                       # big enough that the fraction, not the floor, binds

    _arrive(conn, 4, first_seen=_LATER, near=0.95, tag="few")
    assert ss.claims(conn) == []            # over the floor of 3, under 20% of 25
    _arrive(conn, 2, first_seen=_LATER_STILL, near=0.95, tag="more")
    assert [c["channel"] for c in ss.claims(conn)] == ["new_mass"]


def test_a_region_with_a_build_already_waiting_is_not_also_claimed_for_new_mass(conn):
    """Otherwise one region is claimed twice and the same material is read twice for the money."""
    rec = _region(conn, "mlx", first_seen="2026-01-01 00:00:00")
    _read_now(conn, rec["sitting_id"])
    _arrive(conn, 8, first_seen=_LATER)
    retry = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:mlx-seed"], label="mlx"),
                             floor=0.68, now=_NOW + timedelta(minutes=1))
    _asked(conn, retry["sitting_id"])
    assert [c["channel"] for c in ss.claims(conn)] == ["pointed"]


def test_a_new_mass_rebuild_stays_on_the_notebook_chain(conn, ready):
    """REVERSED 2026-08-24. A new-mass rebuild used to start a FRESH lineage so the re-read could
    see the whole conversation again in order. Under chained parts that link is the notebook: what
    earlier parts established travels forward as claims every later read must answer to, and a
    NULL `continues` orphans the whole history at exactly the moment new material arrived to
    compare it against. It also stops re-paying to read text already distilled.
    """
    rec = _region(conn, "mlx", first_seen="2026-01-01 00:00:00")
    _read_now(conn, rec["sitting_id"])
    _arrive(conn, 8, first_seen=_LATER)
    res = ss.run_sitting_scheduler(conn, now=_RAN)
    assert res["status"] == "ok" and res["claim"]["channel"] == "new_mass"

    fresh = sst.get_sitting(conn, res["regrew"])
    assert fresh["continues"] == rec["sitting_id"]
    assert sb.chain_atom_ids(conn, fresh["sitting_id"])          # the walk finds the ancestor
    assert fresh["region_key"] == sst.get_sitting(conn, rec["sitting_id"])["region_key"]
    # Same region, so the two reads share a query set instead of starting a parallel one.
    assert {r["generator"] for r in conn.execute(
        "SELECT DISTINCT generator FROM frontier_queries")} == {"sitting:mlx"}


# ── remainder: fracture or continue ─────────────────────────────────────────────
def _over_budget(conn, axes, *, budget, per=12):
    """A region the budget cut short, already read — the shape a remainder claim keys on."""
    _planted(conn, axes, per=per, first_seen="2026-01-01 00:00:00")
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, vector=_axis(0), label="parent"),
                           floor=0.50, budget_tokens=budget, now=_NOW)
    assert rec["stop"] == "budget", rec["stop"]
    _read_now(conn, rec["sitting_id"])
    return rec


def test_a_separable_region_fractures_and_one_piece_is_read_now(conn, ready):
    """D15 — size only GATES; separability decides. Two planted conversations wearing one label."""
    parent = _over_budget(conn, (1, 2), budget=3_000)
    res = ss.run_sitting_scheduler(conn, now=_RAN)
    assert res["claim"]["channel"] == "remainder" and res["action"] == "fracture"
    assert res["persisted"] > 1
    kids = conn.execute("SELECT sitting_id, read_at FROM sittings WHERE parent_sitting_id = ?",
                        (parent["sitting_id"],)).fetchall()
    assert len(kids) == res["persisted"]
    # ONE PAID READ, not one per sub-region. The siblings become `sub_region` claims.
    assert sum(1 for k in kids if k["read_at"]) == 1
    assert [c["channel"] for c in ss.claims(conn)] == ["sub_region"] * (len(kids) - 1)


def test_one_conversation_continues_instead_of_fracturing(conn, ready):
    """⚠️ THE ARM THAT CAN SILENTLY GO WRONG. `ZOOM_K_MIN = 2` forces k-means to place two
    centroids even in a region that is one conversation, so `_merge_overlaps` is the only thing
    between a manufactured seam and two persisted sub-sittings."""
    parent = _over_budget(conn, (1,), budget=1_200)
    res = ss.run_sitting_scheduler(conn, now=_RAN)
    assert res["claim"]["channel"] == "remainder" and res["action"] == "continue"
    assert res["persisted"] == 0
    part2 = conn.execute("SELECT * FROM sittings WHERE continues = ?",
                         (parent["sitting_id"],)).fetchone()
    assert part2 is not None and part2["read_at"] is not None


def test_a_continuation_reuses_the_stored_anchor_and_never_re_resolves_the_phrase(conn, ready,
                                                                                  monkeypatch):
    """Re-embedding the phrase would select a different top-k — the corpus moved — so part N+1 would
    grow a DIFFERENT region wearing the same name. It is also the only reason a continuation costs
    nothing: a stored anchor is a blob read where a phrase is a metered embed call."""
    def _never(*a, **kw):
        raise AssertionError("the continuation re-resolved its seed phrase")
    monkeypatch.setattr(sb, "_vector_seed_atoms", _never)

    parent = _over_budget(conn, (1,), budget=1_200)
    res = ss.run_sitting_scheduler(conn, now=_RAN)
    assert res["action"] == "continue"
    part2 = conn.execute("SELECT * FROM sittings WHERE continues = ?",
                         (parent["sitting_id"],)).fetchone()
    # SAME REGION: the dials and the seed identity carry through, so the read refreshes this
    # region's queries rather than opening a parallel set.
    assert part2["region_key"] == sst.get_sitting(conn, parent["sitting_id"])["region_key"]
    # And the chain is inherited: part 2 must not re-admit what part 1 already read.
    covered = set(sb.chain_atom_ids(conn, parent["sitting_id"]))
    part2_atoms = {r["atom_id"] for r in conn.execute(
        "SELECT atom_id FROM sitting_atoms WHERE sitting_id = ? AND is_seed = 0",
        (part2["sitting_id"],))}
    assert not (part2_atoms & covered)


def test_a_claimed_remainder_moves_down_the_chain_rather_than_repeating(conn, ready):
    """The two NOT EXISTS clauses. Without them the HEAD of a chain is claimed on every pass and the
    rail pays forever for a region it has already handled.

    A chain terminates when a part SATURATES, which is the only honest stopping condition — "this
    region is now fully read" is a fact about the corpus, not a count. Part 2 here takes the whole
    remainder, so it saturates and the claim retires. `stop` being EXACT is what makes that work:
    reporting 'budget' merely because the last atom crossed the line would claim a part 3 forever
    against an empty pool.
    """
    parent = _over_budget(conn, (1,), budget=1_200)
    ss.run_sitting_scheduler(conn, now=_RAN)
    part2 = conn.execute("SELECT sitting_id, stop FROM sittings WHERE continues = ?",
                         (parent["sitting_id"],)).fetchone()
    again = [c["sitting_id"] for c in ss.claims(conn) if c["channel"] == "remainder"]
    assert parent["sitting_id"] not in again, "the head was claimed twice"
    assert part2["stop"] == "saturation" and again == []


def test_a_part_that_leaves_material_behind_still_owes_the_next_one(conn, ready):
    """The other arm, and the one the exactness is FOR: part 2 that runs out of budget with atoms
    still on the table keeps the region's claim alive, so the chain continues."""
    parent = _over_budget(conn, (1,), budget=700)
    ss.run_sitting_scheduler(conn, now=_RAN)
    part2 = conn.execute("SELECT sitting_id, stop FROM sittings WHERE continues = ?",
                         (parent["sitting_id"],)).fetchone()
    assert part2["stop"] == "budget"
    assert [c["sitting_id"] for c in ss.claims(conn)
            if c["channel"] == "remainder"] == [part2["sitting_id"]]


# ── the guards ──────────────────────────────────────────────────────────────────
def test_three_consecutive_failures_open_the_breaker(conn, broken):
    """D10's first half. Each pass leaves the region unread, so the claim survives and the loop
    would otherwise retry a dead provider on every session open, forever."""
    for i in range(4):
        r = _region(conn, f"r{i}", when=_NOW - timedelta(days=4 - i))
        _asked(conn, r["sitting_id"])
    for i in range(3):
        assert ss.run_sitting_scheduler(conn, now=_NOW)["status"] == "failed"
    res = ss.run_sitting_scheduler(conn, now=_NOW)
    assert res["status"] == "skipped" and "breaker OPEN" in res["reason"]
    assert _reads(conn) == 0


def test_a_success_resets_the_failure_count(conn, monkeypatch, ready):
    """Two failures then a success must not leave the loop one bad night from tripping."""
    mlx = _region(conn, "mlx", when=_NOW - timedelta(days=2))
    _region(conn, "kiss", when=_NOW - timedelta(days=1))
    _asked(conn, mlx["sitting_id"])
    monkeypatch.setattr(llm_client, "call",
                        lambda role, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    for _ in range(2):
        ss.run_sitting_scheduler(conn, now=_NOW)
    monkeypatch.setattr(llm_client, "call", _reply)
    assert ss.run_sitting_scheduler(conn, now=_NOW)["status"] == "ok"
    assert ss.health(conn)["breaker_open"] is False


def test_health_names_a_loop_that_has_never_run(conn):
    """D10's SECOND half, and the one a breaker cannot give you: zero runs produce zero failures,
    which reads exactly like health. Four of this repo's six spawners failed this way."""
    assert ss.health(conn)["needs_attention"] is False      # nothing waiting: stay silent
    rec = _region(conn, "mlx", when=_NOW)
    _asked(conn, rec["sitting_id"])
    h = ss.health(conn)
    assert h["ever_ran"] is False and h["claims_waiting"] == 1
    assert h["needs_attention"] is True and "never run" in h["note"]


def test_health_goes_quiet_once_the_loop_has_run(conn, ready):
    _region(conn, "mlx", when=_NOW)
    ss.run_sitting_scheduler(conn, now=_NOW)
    h = ss.health(conn)
    assert h["ever_ran"] is True and h["needs_attention"] is False


def test_a_plan_takes_no_action_and_records_no_run(conn, ready):
    """`--plan` is named apart from the reader's `--dry-run`, which DOES make the paid call. A plan
    that recorded a run would also make `ever_ran` report a live loop that has never disposed."""
    rec = _region(conn, "mlx", when=_NOW)
    _asked(conn, rec["sitting_id"])
    res = ss.run_sitting_scheduler(conn, plan_only=True, now=_NOW)
    assert res["status"] == "plan" and len(res["claims"]) == 1
    assert _reads(conn) == 0
    assert conn.execute("SELECT COUNT(*) FROM frontier_reader_runs "
                        "WHERE generator = ?", (ss.GENERATOR,)).fetchone()[0] == 0


def test_the_schedulers_run_row_is_not_mistaken_for_a_regions_read(conn, ready):
    """One paid read leaves TWO rows in `frontier_reader_runs` — the region's, and the scheduler's
    claim. They must stay separately addressable: `last_run(generator=...)` is how a job asks
    "when did I last run", and the answer decides whether it skips. Merge them and the scheduler's
    bookkeeping row would answer a region's question about its own reads."""
    rec = _region(conn, "mlx", when=_NOW)
    _asked(conn, rec["sitting_id"])
    ss.run_sitting_scheduler(conn, now=_NOW)
    assert last_run(conn, generator=ss.GENERATOR)["generator"] == ss.GENERATOR
    assert last_run(conn, generator="sitting:mlx")["generator"] == "sitting:mlx"
    assert last_run(conn, generator="sitting:never-built") is None


def test_a_read_that_skips_does_not_count_against_the_breaker(conn, ready):
    """A stale claim is not a broken loop. Counting it as a failure would trip the breaker on
    bookkeeping and stop a rail that is working."""
    rec = _region(conn, "mlx", when=_NOW)
    _asked(conn, rec["sitting_id"])
    # Make the claim stale the way only a race could: read it out from under the queue.
    queue = ss.claims(conn)
    sr.read_sitting(conn, rec["sitting_id"], now=_NOW)
    assert queue[0]["sitting_id"] == rec["sitting_id"]
    assert ss.run_sitting_scheduler(conn, now=_NOW)["status"] == "skipped"
    assert ss.health(conn)["breaker_open"] is False
