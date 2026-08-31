"""The lenses (Job L), proven offline with a stubbed transport.

⚠️ A SITTING-SCOPED LENS SPENDS NOW (amended 2026-08-24). It is map-reduce: one model call per
uncached part of the region's chain, then the host reduces the per-part outputs in-session. What
these lock:

  • ONE CALL PER UNCACHED (PART, LENS), and ZERO on re-invocation. Closed parts are frozen, so the
    cache needs no invalidation rule and steady state is one call for the open tail. A cache that
    silently missed would make every lens on every region cost the whole chain, every time.
  • THE LOOP IS CODE. A host handed a cursor protocol and told to walk the chain itself answers
    early from partial material — rejected on exactly that ground.
  • `claim` ENTERS AT THE JOIN, NEVER THE MAP. A claim in the map instruction would force the claim
    into the cache key and re-map every part per question, forever.
  • A PART THAT CANNOT BE MAPPED IS NAMED, NOT HIDDEN. A reconcile over a region with a hole in it
    must say which stretch is absent or the join reads as complete.
  • `sprouts` STILL CALLS NOTHING. It has no chain, no part, and nothing frozen to cache against.
  • A RECEIPT NEVER STAMPS `read_at`. That column is the `queries` lens's alone — see
    docs/plans/2026-08-16-lens-reads-subscribe-a-region.md, Part 2.
  • AN UNKNOWN LENS OR A MISSING `sitting_id` RAISES `LensError`, A CALLER MISTAKE, not a runtime
    failure — the tool layer turns it into `status: "error"`, never `"failed"`.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline import llm_client
from pipeline.kb import frontier_queries as fq
from pipeline.kb import schema
from pipeline.kb import sitting_builder as sb
from pipeline.kb import sitting_store as sst
from pipeline.kb import sitting_lenses as sl

DIM = 8


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    from pipeline.kb.embed import ensure_kb_meta
    ensure_kb_meta(c, "fake", DIM, "local", "", storage_dtype="float32")
    yield c
    c.close()


def _unit(*w) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[:len(w)] = w
    return v / (np.linalg.norm(v) + 1e-9)


ANCHOR = _unit(1)


def _atom(conn, atom_id, vec, *, who="x:user:1", when="2026-08-01"):
    conn.execute("INSERT INTO atoms (atom_id, source_type, who_id, when_ts, entry_mode) "
                 "VALUES (?,?,?,?,'user-saved')", (atom_id, "x", who, when))
    text = f"{atom_id} body " + ("word " * 200)
    conn.execute("INSERT INTO chunks (atom_id, seq, char_start, char_end, text, vector) "
                 "VALUES (?,0,0,?,?,?)", (atom_id, len(text), text, vec.tobytes()))
    conn.commit()


@pytest.fixture()
def sitting(conn):
    _atom(conn, "a:seed", ANCHOR, who="x:alice")
    _atom(conn, "a:near", _unit(0.85, 0.53), who="x:bob")
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"], label="mlx"),
                           floor=0.68)
    return rec["sitting_id"]


class _Resp:
    def __init__(self, text):
        self.text, self.model = text, "fake-model"
        self.input_tokens, self.output_tokens, self.cost_usd = 100, 20, 0.01
        self.raw = {}


@pytest.fixture(autouse=True)
def transport(monkeypatch):
    """A usable backend whose call is stubbed. Returns the recorded (system, user) pairs — the
    counter every cache assertion here reads."""
    monkeypatch.setattr(llm_client, "preflight", lambda role: None)
    monkeypatch.setattr("pipeline.ingestion.utils.load_yaml_config", lambda: {})
    monkeypatch.delenv("OPYT_FRONTIER_BACKEND", raising=False)
    calls: list[dict] = []

    def _call(role, *, system, user, **kw):
        calls.append({"system": system, "user": user})
        return _Resp(f"map output {len(calls)}")
    monkeypatch.setattr(llm_client, "call", _call)
    return calls


# ── the four sitting-scoped lenses ───────────────────────────────────────────────
@pytest.mark.parametrize("lens", sl.SITTING_LENSES)
def test_a_sitting_scoped_lens_returns_the_join_rule_and_the_mapped_parts(conn, sitting, lens,
                                                                          transport):
    """The `document` is the per-part MAP outputs, not the region's raw text. That compaction is
    the point: the claims notebook is the paid-read chain's memory and was judged too lossy to be
    a lens's memory too."""
    res = sl.read_lens(conn, lens, sitting_id=sitting)
    assert res["status"] == "ok" and res["lens"] == lens and res["sitting_id"] == sitting
    assert res["instruction"] and "map output 1" in res["document"]
    assert "## Context (chronological)" not in res["document"]
    assert len(transport) == 1 and [p["part"] for p in res["parts"]] == [1]


@pytest.mark.parametrize("lens", sl.SITTING_LENSES)
def test_a_sitting_scoped_lens_writes_a_receipt_but_never_reads_the_sitting(conn, sitting, lens):
    """The whole point of the receipt: `sitting_scheduler` must be able to see it, and `read_at`
    must stay untouched so the `queries` lens can still run this region later."""
    sl.read_lens(conn, lens, sitting_id=sitting)
    row = conn.execute("SELECT * FROM frontier_reader_runs WHERE sitting_id = ? AND lens = ?",
                       (sitting, lens)).fetchone()
    assert row is not None and row["status"] == "ok"
    assert row["consensus"] is None and row["emitted"] is None
    # The run row carries what the map spent. Reporting $0 for a call that spent would make the
    # lens rail invisible in every spend report there is.
    assert row["cost_usd"] == 0.01 and row["model"] == "fake-model"
    assert sst.get_sitting(conn, sitting)["read_at"] is None


def test_disconfirmation_names_the_default_target_when_no_claim_is_given(conn, sitting):
    res = sl.read_lens(conn, "disconfirmation", sitting_id=sitting)
    assert "most strongly" in res["instruction"]


def test_disconfirmation_carries_the_callers_claim_verbatim(conn, sitting):
    res = sl.read_lens(conn, "disconfirmation", sitting_id=sitting,
                       claim="agentic payments need no human in the loop")
    assert "agentic payments need no human in the loop" in res["instruction"]


def test_gaps_forces_the_nearest_miss_framing_and_never_a_bare_absence(conn, sitting):
    """D7/D21 rules this MANDATORY — the one rule in this module that must never be softened."""
    res = sl.read_lens(conn, "gaps", sitting_id=sitting, claim="who bears the loss")
    assert "who bears the loss" in res["instruction"]
    assert "CLOSEST" in res["instruction"]
    assert "not an acceptable answer" in res["instruction"]


def test_a_sitting_scoped_lens_with_no_sitting_id_needs_a_query_or_atom_ids(conn):
    with pytest.raises(sl.LensError):
        sl.read_lens(conn, "briefing")


def test_an_unknown_lens_raises(conn, sitting):
    with pytest.raises(sl.LensError):
        sl.read_lens(conn, "not-a-real-lens", sitting_id=sitting)


def test_an_unknown_sitting_id_raises_key_error(conn):
    with pytest.raises(KeyError):
        sl.read_lens(conn, "briefing", sitting_id="does-not-exist")


def test_a_lens_call_on_an_already_read_region_still_works(conn, sitting):
    """Unlike `read`, a lens has no "already read" refusal — the `queries` read is a different
    consumer of the same region and neither gates the other."""
    sst.mark_read(conn, sitting)
    assert sl.read_lens(conn, "briefing", sitting_id=sitting)["status"] == "ok"


# ── map-reduce over the part chain (RULED 2026-08-24) ───────────────────────────
@pytest.fixture()
def chain(conn):
    """A three-part chain of one region, oldest stretch first — each part holding real material,
    because a part that admits only its seeds now (correctly) reports `saturation` and would end
    the chain at one."""
    import datetime as _dt
    _atom(conn, "a:seed", ANCHOR, who="x:alice", when="2026-01-01")
    for i in range(3):
        # Distinct AXES, not near-parallel points: three atoms an epsilon apart are near-duplicates
        # and the ceiling drops two of them, leaving parts 2 and 3 with nothing to read.
        v = [0.90] + [0.0] * 7
        v[2 + i] = 0.43
        _atom(conn, f"a:{i}", _unit(*v), when=f"2026-0{i + 2}-01")
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"], label="mlx")
    # Sized from the real atom cost rather than guessed: the seed is re-admitted into every part,
    # so the budget has to clear the seed and exactly one more atom.
    probe = sb.build_sitting(conn, seed, floor=0.68, persist=False)
    per = {a["atom_id"]: a["tokens"] for a in probe["admissions"]}
    budget = per["a:seed"] + 1
    ids, prev = [], None
    for i in range(3):
        rec = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=budget, continues=prev,
                               now=_dt.datetime(2026, 8, 1, 12, 0, i, tzinfo=_dt.timezone.utc))
        assert len([a for a in rec["admissions"] if not a["is_seed"]]) == 1
        ids.append(rec["sitting_id"])
        prev = rec["sitting_id"]
    return ids


def test_every_part_of_the_chain_is_mapped_once(conn, chain, transport):
    """THE LOOP IS CODE, not model discretion — a host walking a cursor protocol itself answers
    early from partial material, and that is exactly what this replaces."""
    res = sl.read_lens(conn, "trajectory", sitting_id=chain[-1])
    assert len(transport) == 3
    assert [p["part"] for p in res["parts"]] == [1, 2, 3]
    assert [p["sitting_id"] for p in res["parts"]] == chain
    for n in ("map output 1", "map output 2", "map output 3"):
        assert n in res["document"]


def test_a_second_call_costs_nothing(conn, chain, transport):
    """A closed part is FROZEN, so it is mapped once per lens EVER — which is why this cache needs
    no invalidation rule at all, and why steady state on any region is one call for the open tail.
    A silently-missing cache would make every lens on every region re-pay the whole chain."""
    first = sl.read_lens(conn, "trajectory", sitting_id=chain[-1])
    n = len(transport)
    again = sl.read_lens(conn, "trajectory", sitting_id=chain[-1])
    assert len(transport) == n
    assert again["document"] == first["document"]
    assert all(p["cached"] for p in again["parts"])
    assert again["spent_usd"] == 0.0 and first["spent_usd"] > 0.0


def test_each_lens_maps_the_chain_separately(conn, chain, transport):
    """The cache key is (sitting_id, lens): a briefing of a part answers a different question than
    its trajectory, and sharing the row would serve one lens's output as the other's."""
    sl.read_lens(conn, "trajectory", sitting_id=chain[-1])
    sl.read_lens(conn, "briefing", sitting_id=chain[-1])
    assert len(transport) == 6
    lenses = {r[0] for r in conn.execute("SELECT DISTINCT lens FROM sitting_lens_outputs")}
    assert lenses == {"trajectory", "briefing"}


def test_the_parts_are_labelled_with_the_stretch_they_cover(conn, chain, transport):
    """The host reduces across time, so a block with no dates on it cannot be joined end to end."""
    res = sl.read_lens(conn, "trajectory", sitting_id=chain[-1])
    assert "Part 1 of 3 — covering 2026-02-01" in res["document"]
    # Seeds are re-admitted into every part, so a span counting them would report the seed's own
    # date for all three and the host could not join anything end to end.
    assert [p["covering"] for p in res["parts"]] == ["2026-02-01–2026-02-01",
                                                    "2026-03-01–2026-03-01",
                                                    "2026-04-01–2026-04-01"]


def test_the_claim_never_reaches_the_map(conn, chain, transport):
    """A claim in the map instruction forces the claim into the cache key and re-maps every part
    per question, forever. The target enters at the JOIN, which the host performs with it in hand."""
    sl.read_lens(conn, "gaps", sitting_id=chain[-1], claim="who bears the loss")
    assert all("who bears the loss" not in c["system"] for c in transport)

    n = len(transport)
    res = sl.read_lens(conn, "gaps", sitting_id=chain[-1], claim="an entirely different question")
    assert len(transport) == n, "a new question re-mapped the chain"
    assert "an entirely different question" in res["instruction"]


def test_a_part_that_cannot_be_mapped_is_named(conn, chain, monkeypatch, transport):
    """FAIL-SAFE, but never silent: a lens is prose for a person, so a region missing one stretch
    is a degraded answer while refusing the whole lens over one bad call is a worse one. The join
    must say which stretch is absent or it reads as complete."""
    def _boom(role, *, system, user, **kw):
        raise RuntimeError("transport down")
    monkeypatch.setattr(llm_client, "call", _boom)

    res = sl.read_lens(conn, "briefing", sitting_id=chain[-1])
    assert res["status"] == "ok" and res["parts"] == []
    assert res["missing_parts"] == [1, 2, 3] and "missing that stretch" in res["note"]
    assert conn.execute("SELECT COUNT(*) FROM sitting_lens_outputs").fetchone()[0] == 0


def test_an_oversized_part_is_capped_before_the_call(conn, sitting, monkeypatch, transport):
    """`render_sitting` is used directly here, not through `render_prompt`, which is where the
    input cap normally lives — so it has to be re-applied or a long part is sent uncapped."""
    monkeypatch.setattr(sl.sr, "MAX_INPUT_CHARS", 200)
    sl.read_lens(conn, "briefing", sitting_id=sitting)
    assert len(transport[0]["user"]) < 300
    assert "TRUNCATED" in transport[0]["user"]


def test_the_reconciled_output_is_never_persisted(conn, chain, transport):
    """RULED 2026-08-25. The reduce's input is the LIVE region, so storing its output buys the
    invalidation problem forever plus a second record of state-over-time that can disagree with the
    claims notebook. Cache what is frozen; recompute what is live."""
    sl.read_lens(conn, "trajectory", sitting_id=chain[-1])
    rows = conn.execute("SELECT sitting_id FROM sitting_lens_outputs").fetchall()
    assert sorted(r[0] for r in rows) == sorted(chain), "only per-PART rows may exist"


# ── sprouts — no sitting_id, no receipt ──────────────────────────────────────────
def test_sprouts_needs_no_sitting_id(conn):
    _atom(conn, "a:orphan", ANCHOR)
    res = sl.read_lens(conn, "sprouts")
    assert res["status"] == "ok" and res["atoms"] == 1
    assert "a:orphan" in res["document"]


def test_sprouts_writes_no_receipt_and_calls_nothing(conn, transport):
    """It has no `sitting_id`, so there is nothing for `sitting_scheduler` to key a receipt on, and
    no chain and no frozen part for a map call to cache against. It is what this whole module used
    to be, and it stays that way."""
    _atom(conn, "a:orphan", ANCHOR)
    sl.read_lens(conn, "sprouts")
    assert conn.execute("SELECT COUNT(*) FROM frontier_reader_runs").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM sitting_lens_outputs").fetchone()[0] == 0
    assert transport == []


def test_sprouts_ignores_a_stray_sitting_id(conn, sitting):
    """`sprouts` is not about one region — a `sitting_id` passed alongside it must be a no-op, not
    an error, so a caller that always passes one from a prior preview does not need a special case."""
    res = sl.read_lens(conn, "sprouts", sitting_id=sitting)
    assert res["status"] == "ok" and "sitting_id" not in res
