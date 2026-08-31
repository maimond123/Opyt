"""Reading a sitting, proven offline with a stubbed transport.

The output contract (parse, validate, provenance repair, upsert identity) is proven once in
`test_reader_core.py` against the same `reader_core`. What is NEW here, and what these lock:

  • A FAILED READ LEAVES THE SITTING UNREAD. The read stamp is what moves atoms out of never-read
    mass; stamping on failure would report coverage the corpus never got.
  • DORMANCY IS SCOPED PER REGION. Under one shared generator label, reading one region would age
    out every other region's queries — dormant in three sittings.
  • RUN ROWS ARE SCOPED. `frontier_reader_runs` is written by every region and by the scheduler, so
    an unscoped read lets one job's run prove to another that it had already run.
  • THE ONE REMAINING CAP BOUNDS A SWEEP, not a use: an already-read sitting is never re-read. The
    daily read cap that used to sit beside it was DELETED, and its absence is pinned — a per-use
    permission gate reads like a safety feature and gets re-added otherwise.
  • `prompt_only` SPENDS NOTHING. It is the mode that shows the exact bytes a read would send.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from pipeline import llm_client
from pipeline.kb import frontier_queries as fq
from pipeline.kb import schema
from pipeline.kb import sitting_builder as sb
from pipeline.kb import sitting_store as sst
from pipeline.kb import sitting_reader as sr

from .conftest import last_run

DIM = 8
_NOW = datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)


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


def _at_cos(c: float, axis: int = 1) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[0], v[axis] = c, float(np.sqrt(max(0.0, 1.0 - c * c)))
    return v / (np.linalg.norm(v) + 1e-9)


def _atom(conn, atom_id, vec, *, who="x:user:1", when="2026-08-01", entry_mode="user-saved"):
    conn.execute("INSERT INTO atoms (atom_id, source_type, who_id, when_ts, entry_mode) "
                 "VALUES (?,?,?,?,?)", (atom_id, "x", who, when, entry_mode))
    text = f"{atom_id} body " + ("word " * 200)
    conn.execute("INSERT INTO chunks (atom_id, seq, char_start, char_end, text, vector) "
                 "VALUES (?,0,0,?,?,?)", (atom_id, len(text), text, vec.tobytes()))
    conn.commit()


@pytest.fixture()
def sitting(conn):
    """A two-atom sitting, built and unread."""
    _atom(conn, "a:seed", ANCHOR, who="x:alice")
    _atom(conn, "a:near", _at_cos(0.85), who="x:bob")
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"], label="mlx"),
                           floor=0.68)
    return rec["sitting_id"]


def _body(queries, verdicts=None) -> str:
    import json
    out = {"consensus": "the arc moved", "queries": queries}
    if verdicts is not None:
        out["verdicts"] = verdicts
    return json.dumps(out)


def _v(text, verdict="keep", *, atom_ids=("a:seed",)):
    return {"text": text, "verdict": verdict, "reason": "because",
            "atom_ids": list(atom_ids)}


def _q(text, *, atom_ids=("a:seed",)):
    return {"text": text, "target_sources": ["arxiv"], "rationale": "because",
            "atom_ids": list(atom_ids)}


class _Resp:
    def __init__(self, text):
        self.text, self.model = text, "fake-model"
        self.input_tokens, self.output_tokens, self.cost_usd = 100, 20, 0.01
        self.raw = {}


@pytest.fixture()
def ready(monkeypatch):
    """A usable API backend whose call is stubbed."""
    monkeypatch.setattr(llm_client, "preflight", lambda role: None)
    monkeypatch.setattr("pipeline.ingestion.utils.load_yaml_config", lambda: {})
    monkeypatch.delenv("OPYT_FRONTIER_BACKEND", raising=False)


def _answer(monkeypatch, body: str, seen: dict | None = None):
    def _call(role, *, system, user, **kw):
        if seen is not None:
            seen.update(role=role, system=system, user=user)
        return _Resp(body)
    monkeypatch.setattr(llm_client, "call", _call)


# ── the happy path ──────────────────────────────────────────────────────────────
def test_a_successful_read_emits_queries_and_stamps_the_sitting(conn, sitting, ready, monkeypatch):
    _answer(monkeypatch, _body([_q("gated DeltaNet attention"), _q("muon optimizer")]))
    res = sr.read_sitting(conn, sitting, now=_NOW)
    assert res["status"] == "ok" and res["emitted"] == 2 and res["new"] == 2
    assert sst.get_sitting(conn, sitting)["read_at"] is not None
    assert sst.coverage(conn)["read"] == 2          # both atoms now count as covered


def test_the_machine_lane_quota_is_enforced_inside_a_real_read(conn, ready, monkeypatch):
    """The WIRING, not the arithmetic (that is test_frontier_query_ownership.py). Post-parse and
    pre-upsert is the only point holding resolved citations and the generator — and it must be in
    code, never in the prompt: a limit the model is merely asked to respect is not a limit.

    The drop is never silent: it lands in `notes`, which the run row's `reason` carries."""
    _atom(conn, "a:seed", ANCHOR, who="x:alice")
    _atom(conn, "a:found", _at_cos(0.85), who="x:bob", entry_mode="frontier")
    sid = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"], label="mlx"),
                           floor=0.68)["sitting_id"]
    _answer(monkeypatch, _body([_q(f"m{i}", atom_ids=["a:found"]) for i in range(5)]
                               + [_q("human thread", atom_ids=["a:seed"])]))

    res = sr.read_sitting(conn, sid, now=_NOW)
    assert res["status"] == "ok" and res["emitted"] == 4     # 3 machine + the human one
    lanes = {r["normalized"]: r["lane"] for r in conn.execute(
        "SELECT normalized, lane FROM frontier_queries")}
    assert lanes == {"m0": "machine", "m1": "machine", "m2": "machine",
                     "human thread": "human"}
    reason = conn.execute("SELECT reason FROM frontier_reader_runs").fetchone()["reason"]
    assert "m3" in reason and "m4" in reason


def test_the_generator_is_scoped_to_the_region(conn, sitting, ready, monkeypatch):
    _answer(monkeypatch, _body([_q("gated DeltaNet attention")]))
    sr.read_sitting(conn, sitting, now=_NOW)
    row = conn.execute("SELECT generator FROM frontier_queries").fetchone()
    assert row["generator"] == "sitting:mlx"


def test_reading_one_region_does_not_age_out_another(conn, ready, monkeypatch):
    """THE BUG A SHARED GENERATOR LABEL WOULD CAUSE: three sittings and every other region's
    queries go dormant, because the sweep ages whatever this run did not re-emit."""
    fq.upsert_queries(conn, [_q("kisspeptin analog")], generator="sitting:kiss1r")
    _atom(conn, "a:seed", ANCHOR)
    _atom(conn, "a:near", _at_cos(0.85))
    sid = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"], label="mlx"),
                           floor=0.68)["sitting_id"]
    _answer(monkeypatch, _body([_q("mlx kernel")]))
    for _ in range(3):
        sr.read_sitting(conn, sid, force=True, now=_NOW)
    other = conn.execute("SELECT status, miss_count FROM frontier_queries "
                         "WHERE generator='sitting:kiss1r'").fetchone()
    assert other["status"] == "active" and other["miss_count"] == 0


# ── failure leaves no trace ─────────────────────────────────────────────────────
def test_a_failed_call_leaves_the_sitting_unread_and_writes_no_queries(conn, sitting, ready,
                                                                       monkeypatch):
    def _boom(role, *, system, user, **kw):
        raise RuntimeError("provider exploded")
    monkeypatch.setattr(llm_client, "call", _boom)
    res = sr.read_sitting(conn, sitting, now=_NOW)
    assert res["status"] == "failed"
    assert sst.get_sitting(conn, sitting)["read_at"] is None
    assert conn.execute("SELECT COUNT(*) FROM frontier_queries").fetchone()[0] == 0
    assert sst.coverage(conn)["read"] == 0
    row = conn.execute("SELECT status, sitting_id FROM frontier_reader_runs").fetchone()
    assert row["status"] == "failed" and row["sitting_id"] == sitting


def test_an_unparseable_body_writes_nothing(conn, sitting, ready, monkeypatch):
    _answer(monkeypatch, "I'm afraid I can't do that")
    res = sr.read_sitting(conn, sitting, now=_NOW)
    assert res["status"] == "failed"
    assert sst.get_sitting(conn, sitting)["read_at"] is None


def test_a_preflight_failure_writes_no_queries_and_leaves_it_unread(conn, sitting, monkeypatch):
    """No call was attempted, so nothing was spent and the region still owes a read."""
    monkeypatch.setattr("pipeline.ingestion.utils.load_yaml_config", lambda: {})
    monkeypatch.setattr(llm_client, "preflight", lambda role: "no API key")
    res = sr.read_sitting(conn, sitting, now=_NOW)
    assert res["status"] == "failed"
    assert sst.get_sitting(conn, sitting)["read_at"] is None


def test_out_of_credits_is_named_rather_than_reported_as_a_broken_call(conn, sitting, ready,
                                                                       monkeypatch):
    """MONEY-ABSENT is the one spend condition that still gates, and it fails LOUD because the
    remedy is a human action. Rejected before inference, so nothing was spent."""
    class _402(RuntimeError):
        status = 402
    monkeypatch.setattr(llm_client, "call",
                        lambda role, **kw: (_ for _ in ()).throw(_402("no credit")))
    res = sr.read_sitting(conn, sitting, now=_NOW)
    assert "OUT OF CREDITS" in res["reason"]
    assert sst.get_sitting(conn, sitting)["read_at"] is None


# ── the one cap that remains ────────────────────────────────────────────────────
def test_a_read_sitting_is_never_re_read(conn, sitting, ready, monkeypatch):
    """Same atoms over the same corpus state — a second read is the same input for the same money."""
    calls = []
    monkeypatch.setattr(llm_client, "call",
                        lambda role, **kw: calls.append(1) or _Resp(_body([_q("x")])))
    sr.read_sitting(conn, sitting, now=_NOW)
    res = sr.read_sitting(conn, sitting, now=_NOW)
    assert res["status"] == "skipped" and "already read" in res["reason"]
    assert len(calls) == 1


def test_no_per_use_cap_stands_between_the_user_and_a_read(conn, sitting, ready, monkeypatch):
    """⚠️ PINS A DELETION. A daily read cap used to sit here, and it was a per-use permission gate:
    consent belongs at the deposit, never at each use, and refusing because the user asked eight
    times today is exactly what that rule forbids. It also guarded a cost that no longer exists —
    seven cents a day at the role's current model — and `--force` deliberately did not override it,
    so its only escape was an env var that then got set permanently.

    A test asserting the ABSENCE of a limit is unusual, and it is here because the limit reads like
    a safety feature: without this, the next person to worry about runaway spend re-adds it."""
    assert not hasattr(sr, "DAILY_READS")
    assert not hasattr(sr, "_paid_reads_since")
    _answer(monkeypatch, _body([_q("x")]))
    for _ in range(12):                      # far past the old cap of 8
        assert sr.read_sitting(conn, sitting, force=True, now=_NOW)["status"] == "ok"


def test_an_empty_sitting_is_skipped_without_calling(conn, ready, monkeypatch):
    calls = []
    monkeypatch.setattr(llm_client, "call", lambda role, **kw: calls.append(1))
    _atom(conn, "a:seed", ANCHOR)
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    conn.execute("UPDATE chunks SET vector = NULL")
    conn.commit()
    sid = sb.build_sitting(conn, seed, floor=0.68)["sitting_id"]
    assert sr.read_sitting(conn, sid, now=_NOW)["status"] == "skipped"
    assert calls == []


# ── runs are scoped, so one job's row never answers another's question ──────────
#
# ⚠️ TWO TESTS HERE ASSERTED THE SPLIT AGAINST `bookmark_reader` AND WENT WITH IT (2026-08-16, D13).
# The claim did not go with it: `frontier_reader_runs` is still written by several jobs — one row
# per REGION under `sitting:<slug>`, plus `sitting-scheduler`'s own claim rows — and an unscoped
# read still lets one job's run prove to another that it had already run. The surviving proof is
# `test_sitting_scheduler.py::test_the_schedulers_run_row_is_not_mistaken_for_a_regions_read`,
# which exercises the two labels that both exist today.
def test_one_regions_run_row_is_not_visible_to_another_region(conn, sitting, ready, monkeypatch):
    """The same failure the deleted cross-rail tests guarded, at the granularity that remains."""
    _answer(monkeypatch, _body([_q("x")]))
    sr.read_sitting(conn, sitting, now=_NOW)
    assert last_run(conn, status="ok", generator="sitting:mlx") is not None
    assert last_run(conn, status="ok", generator="sitting:kiss1r") is None


# ── the prompt ──────────────────────────────────────────────────────────────────
def test_prompt_only_spends_nothing_and_shows_the_exact_prompt(conn, sitting, ready, monkeypatch):
    """⚠️ NAMED APART FROM THE TOOL'S `preview` ACTION, deliberately. This one means "show me the
    exact bytes that would be sent"; that one means "build the region and report its scope". Both
    are free, both are called before a read, and one word for two questions is how a caller ends up
    spending on the question they did not ask."""
    calls = []
    monkeypatch.setattr(llm_client, "call", lambda role, **kw: calls.append(1))
    res = sr.read_sitting(conn, sitting, prompt_only=True, now=_NOW)
    assert res["status"] == "prompt-only" and calls == []
    assert "a:seed" in res["prompt"] and res["est_tokens"] > 0
    assert conn.execute("SELECT COUNT(*) FROM frontier_reader_runs").fetchone()[0] == 0
    assert sst.get_sitting(conn, sitting)["read_at"] is None


def test_the_reader_is_shown_this_regions_standing_queries_only(conn, sitting, ready, monkeypatch):
    fq.upsert_queries(conn, [_q("mlx kernel autotune")], generator="sitting:mlx")
    fq.upsert_queries(conn, [_q("kisspeptin analog")], generator="sitting:kiss1r")
    seen = {}
    _answer(monkeypatch, _body([_q("x")]), seen)
    sr.read_sitting(conn, sitting, now=_NOW)
    assert "- mlx kernel autotune" in seen["user"]
    assert "kisspeptin" not in seen["user"]
    assert "=== SITTING ===" in seen["user"]


def test_the_first_read_of_a_region_shows_no_empty_heading(conn, sitting, ready, monkeypatch):
    seen = {}
    _answer(monkeypatch, _body([_q("x")]), seen)
    sr.read_sitting(conn, sitting, now=_NOW)
    assert "CURRENT STANDING" not in seen["user"]
    assert seen["user"].startswith("# Sitting:")


def test_the_prompt_carries_the_author_concentration(conn, ready, monkeypatch):
    """A single-author region has to be visible to the reader, or it queries that author's own
    repos — which David already follows."""
    _atom(conn, "a:seed", ANCHOR, who="x:taelin")
    _atom(conn, "a:2", _at_cos(0.85), who="x:taelin")
    sid = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)["sitting_id"]
    seen = {}
    _answer(monkeypatch, _body([_q("x")]), seen)
    sr.read_sitting(conn, sid, now=_NOW)
    assert "x:taelin wrote 100%" in seen["user"]
    assert "WHEN ONE AUTHOR DOMINATES" in seen["system"]


def test_an_oversized_sitting_is_truncated_at_the_END(conn, sitting, ready, monkeypatch):
    """The opposite of the bookmark window's rule, and deliberately so: this render is
    chronological, so cutting the head would delete the beginning of the story."""
    monkeypatch.setattr(sr, "MAX_INPUT_CHARS", 400)
    out = sr.render_prompt(conn, sitting)
    assert out.startswith("# Sitting:") and out.endswith("input budget]")


# ── provenance ──────────────────────────────────────────────────────────────────
def test_citations_are_checked_against_the_sitting_not_the_corpus(conn, sitting, ready,
                                                                  monkeypatch):
    """A query citing an atom that was not in this sitting has no provenance here, so it is dropped
    rather than stored with a citation to material the model never saw."""
    _atom(conn, "a:elsewhere", _unit(0, 1))
    _answer(monkeypatch, _body([_q("real one", atom_ids=["a:seed"]),
                                _q("phantom", atom_ids=["a:elsewhere"])]))
    res = sr.read_sitting(conn, sitting, now=_NOW)
    texts = {r["text"] for r in conn.execute("SELECT text FROM frontier_queries")}
    assert texts == {"real one"} and res["emitted"] == 1


def _part(conn, *, continues, tag="two", when="2026-01-01"):
    """One more part of the same region, built and unread. Seeds are re-admitted into every part,
    so `a:seed` is NOT an ancestor-only atom — `a:near` is."""
    _atom(conn, f"a:{tag}", _at_cos(0.85, axis=2), when=when)
    return sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"], label="mlx"),
                            floor=0.68, continues=continues,
                            now=_NOW + timedelta(seconds=1))["sitting_id"]


def test_a_query_citing_only_an_ancestor_atom_is_dropped(conn, sitting, ready, monkeypatch):
    """THE NARROW GATE, RULED 2026-08-25. The claims lens widens its citation gate to the whole
    chain; this one deliberately does not. Part 2's preamble names ancestor atoms, so the model CAN
    cite them — and a query that cites nothing else is claiming provenance in text this run never
    read. It is dropped, and a query citing a shown atom in the same response is unaffected."""
    _answer(monkeypatch, _body([_q("part one thread", atom_ids=["a:seed"])]))
    sr.read_sitting(conn, sitting, now=_NOW)

    p2 = _part(conn, continues=sitting)
    assert "a:near" not in {a["atom_id"] for a in sst.get_sitting(conn, p2)["admissions"]}
    _answer(monkeypatch, _body([_q("born of the notebook alone", atom_ids=["a:near"]),
                                _q("born of this part", atom_ids=["a:two"])]))
    res = sr.read_sitting(conn, p2, now=_NOW)

    assert res["status"] == "ok" and res["emitted"] == 1
    texts = {r["text"] for r in conn.execute("SELECT text FROM frontier_queries")}
    assert "born of this part" in texts and "born of the notebook alone" not in texts


def test_the_dropped_query_is_recorded_on_the_run_row(conn, sitting, ready, monkeypatch):
    """Where the accepted cost shows up. The read's own result carries no `notes`, so this row is
    the only place a discarded thread is legible after the fact."""
    p2 = _part(conn, continues=sitting)
    _answer(monkeypatch, _body([_q("notebook only", atom_ids=["a:near"]),
                                _q("kept", atom_ids=["a:two"])]))
    sr.read_sitting(conn, p2, now=_NOW)
    row = conn.execute("SELECT reason FROM frontier_reader_runs WHERE sitting_id = ?",
                       (p2,)).fetchone()
    assert "notebook only" in (row["reason"] or "")


def test_dry_run_writes_nothing_but_reports_the_queries(conn, sitting, ready, monkeypatch):
    _answer(monkeypatch, _body([_q("gated DeltaNet attention")]))
    res = sr.read_sitting(conn, sitting, dry_run=True, now=_NOW)
    assert res["status"] == "dry-run" and len(res["queries"]) == 1
    assert conn.execute("SELECT COUNT(*) FROM frontier_queries").fetchone()[0] == 0
    assert sst.get_sitting(conn, sitting)["read_at"] is None


def test_unread_sittings_lists_the_queue_biggest_first(conn, sitting, ready, monkeypatch):
    queue = sr.unread_sittings(conn)
    assert [q["sitting_id"] for q in queue] == [sitting]
    _answer(monkeypatch, _body([_q("x")]))
    sr.read_sitting(conn, sitting, now=_NOW)
    assert sr.unread_sittings(conn) == []


def test_no_standing_reads_the_region_as_if_for_the_first_time(conn, sitting, ready, monkeypatch):
    """A measurement mode. "Did THIS sitting generate these queries?" cannot be answered while the
    model is shown the answer — measured on the real corpus, a region that shrank to 873 tokens
    still re-emitted 14 queries from its earlier, far richer read."""
    fq.upsert_queries(conn, [_q("mlx kernel autotune")], generator="sitting:mlx")
    seen = {}
    _answer(monkeypatch, _body([_q("x")]), seen)
    sr.read_sitting(conn, sitting, standing=False, now=_NOW)
    assert "mlx kernel autotune" not in seen["user"]
    assert "CURRENT STANDING" not in seen["user"]


# ── verdicts: the decay signal this rail took over (D11) ────────────────────────
def test_a_drop_actually_slows_the_query_down(conn, sitting, ready, monkeypatch):
    """THE REGRESSION TEST FOR THE `votable` FLIP, and the one failure the rest of the suite cannot
    see. `apply_verdicts` writes the claim's `miss_count` whatever the flag says, so a `dropped`
    counter of 1 and a run row that looks healthy are produced EITHER WAY. What separates the two
    worlds is whether `_sync_speed` — which takes the MIN over VOTABLE claims only — then carries
    that drop onto the query row stage 2 actually reads. Left non-votable, this rail would record
    every decision and change nothing, which is worse than staying silent because it looks like
    success. So assert the query's OWN speed, never the counter.

    ⚠️ THE QUERY IS CREATED BY A READ, NOT BY A DIRECT `upsert_queries` CALL, AND THAT IS THE WHOLE
    TEST. `upsert_queries` defaults to `votable=True`, so a hand-seeded query registers the
    generator as votable and the drop then lands correctly no matter what the reader passes. This
    test was written that way first and it PASSED against the bug it exists to catch. Only the real
    two-read sequence exercises the flag the READER declares: read 1 registers the generator while
    emitting, read 2 is shown its own query and drops it. Both assertions below were confirmed to
    fail independently with `votable=False`.
    """
    _answer(monkeypatch, _body([_q("mlx kernel autotune")]))
    sr.read_sitting(conn, sitting, now=_NOW)
    assert conn.execute("SELECT votable FROM frontier_generators WHERE generator='sitting:mlx'"
                        ).fetchone()[0] == 1, "the reader registered itself as unable to vote"

    _answer(monkeypatch, _body([], [_v("mlx kernel autotune", "drop")]))
    res = sr.read_sitting(conn, sitting, force=True, now=_NOW)
    assert res["status"] == "ok" and res["dropped"] == 1
    speed = conn.execute("SELECT miss_count FROM frontier_queries WHERE text=?",
                         ("mlx kernel autotune",)).fetchone()[0]
    assert speed == 1, "the drop was recorded but never reached the query's speed"


def test_a_keep_restores_full_speed(conn, sitting, ready, monkeypatch):
    """A drop is reversible — the whole reason the prompt can tell the model to judge honestly
    rather than defensively."""
    _answer(monkeypatch, _body([_q("mlx kernel autotune")]))
    sr.read_sitting(conn, sitting, now=_NOW)
    _answer(monkeypatch, _body([], [_v("mlx kernel autotune", "drop")]))
    sr.read_sitting(conn, sitting, force=True, now=_NOW)
    _answer(monkeypatch, _body([], [_v("mlx kernel autotune", "keep")]))
    sr.read_sitting(conn, sitting, force=True, now=_NOW)
    assert conn.execute("SELECT miss_count FROM frontier_queries WHERE text=?",
                        ("mlx kernel autotune",)).fetchone()[0] == 0


def test_a_read_that_only_renders_verdicts_is_a_good_read(conn, sitting, ready, monkeypatch):
    """The steady state this rail is heading for: a settled region whose every thread is already
    covered opens no new query. The old queries-only rejection would have failed exactly that."""
    fq.upsert_queries(conn, [_q("mlx kernel autotune")], generator="sitting:mlx")
    _answer(monkeypatch, _body([], [_v("mlx kernel autotune", "keep")]))
    res = sr.read_sitting(conn, sitting, now=_NOW)
    assert res["status"] == "ok" and res["emitted"] == 0 and res["kept"] == 1
    assert sst.get_sitting(conn, sitting)["read_at"] is not None


def test_unverdicted_is_shown_minus_decided(conn, sitting, ready, monkeypatch):
    """Watched rather than merely recorded: a rising `unverdicted` means the survival signal is
    degrading while `kept` and `dropped` sit still and look healthy."""
    fq.upsert_queries(conn, [_q("one"), _q("two"), _q("three")], generator="sitting:mlx")
    _answer(monkeypatch, _body([], [_v("one", "keep"), _v("two", "drop")]))
    res = sr.read_sitting(conn, sitting, now=_NOW)
    assert (res["kept"], res["dropped"], res["unverdicted"]) == (1, 1, 1)
    row = conn.execute("SELECT kept, dropped, unverdicted FROM frontier_reader_runs "
                       "WHERE status='ok'").fetchone()
    assert (row["kept"], row["dropped"], row["unverdicted"]) == (1, 1, 1)


def test_a_verdict_on_a_query_never_shown_moves_nothing(conn, sitting, ready, monkeypatch):
    """`shown=running` is a provenance check, not a convenience. A verdict on a query this region
    was never shown is a hallucination, and applying it would move a counter on evidence that does
    not exist."""
    fq.upsert_queries(conn, [_q("kisspeptin analog")], generator="sitting:kiss1r")
    _answer(monkeypatch, _body([_q("x")], [_v("kisspeptin analog", "drop")]))
    res = sr.read_sitting(conn, sitting, now=_NOW)
    assert res["dropped"] == 0
    assert conn.execute("SELECT miss_count FROM frontier_queries WHERE text=?",
                        ("kisspeptin analog",)).fetchone()[0] == 0


def test_keeping_and_re_emitting_one_query_counts_it_once(conn, sitting, ready, monkeypatch):
    """The likeliest way a run inflates a counter, and part of why `shown` is passed: without it
    `validate` cannot see the collision, and the verdict and the upsert each bump `emit_count`."""
    fq.upsert_queries(conn, [_q("mlx kernel autotune")], generator="sitting:mlx")
    before = conn.execute("SELECT emit_count FROM frontier_queries WHERE text=?",
                          ("mlx kernel autotune",)).fetchone()[0]
    _answer(monkeypatch, _body([_q("mlx kernel autotune")], [_v("mlx kernel autotune", "keep")]))
    sr.read_sitting(conn, sitting, now=_NOW)
    after = conn.execute("SELECT emit_count FROM frontier_queries WHERE text=?",
                         ("mlx kernel autotune",)).fetchone()[0]
    assert after == before + 1


def test_the_prompt_asks_for_a_verdict_on_every_running_query(conn, sitting, ready, monkeypatch):
    """Retiring `bookmark_reader` (D13, done 2026-08-16) removed the only other verdict producer,
    so if this prompt
    ever stops asking, decay does not become conservative — it disappears."""
    seen = {}
    _answer(monkeypatch, _body([_q("x")]), seen)
    sr.read_sitting(conn, sitting, now=_NOW)
    assert "STEP 2 — VERDICTS" in seen["system"]
    assert "Render a verdict on EVERY ONE of them" in seen["system"]


# ── record_lens_run — the Option C Part 2 receipt (Job L) ──────────────────────────────────────
_USAGE = {"model": "test/model", "in_tokens": 900, "out_tokens": 300, "cost_usd": 0.0021}


def test_a_lens_receipt_carries_its_cost_and_stamps_no_read_at(conn, sitting):
    """The whole contract: a run row with `lens` set and the map call's cost on it, `read_at` left
    untouched. Stamping `read_at` here would mean the `queries` lens could never run this region
    again — the opposite of the intent (docs/plans/2026-08-16-lens-reads-subscribe-a-region.md,
    Part 2). `consensus` stays NULL: that column is what separates a receipt from a queries read."""
    sr.record_lens_run(conn, sitting, "briefing", usage=_USAGE, ref=_NOW)
    row = conn.execute("SELECT * FROM frontier_reader_runs WHERE sitting_id = ?",
                       (sitting,)).fetchone()
    assert row["lens"] == "briefing" and row["status"] == "ok"
    assert row["consensus"] is None and row["cost_usd"] == _USAGE["cost_usd"]
    assert sst.get_sitting(conn, sitting)["read_at"] is None


def test_a_lens_receipt_is_scoped_to_the_regions_generator(conn, sitting):
    """Same `sitting:<slug>` generator the `queries` lens would use — a receipt is still a fact
    about this region, and `last_run(generator=...)` should be able to find it."""
    sr.record_lens_run(conn, sitting, "trajectory", usage=_USAGE, ref=_NOW)
    row = last_run(conn, generator="sitting:mlx")
    assert row is not None and row["lens"] == "trajectory"


def test_a_lens_receipt_on_an_unknown_sitting_does_not_raise(conn):
    """Never a caller's problem — a bookkeeping row that cannot be written must not turn a lens
    read into an error the person sees."""
    sr.record_lens_run(conn, "no-such-sitting", "briefing", usage=_USAGE, ref=_NOW)  # no raise
