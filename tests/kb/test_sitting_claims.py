"""Extracting claims from a sitting (Job N), proven offline with a stubbed transport.

Mirrors `test_sitting_reader.py`'s conventions for the shared machinery (already proven once in
`test_reader_core.py`). What is NEW here, and what these lock:

  • CLAIMS AND QUERIES READ INDEPENDENTLY. Each lens's "never re-read once read" guard lives in
    `sitting_reads`, keyed per lens — reading a region for `claims` must not block, or be blocked
    by, its `queries` read.
  • `falsified_by` IS REQUIRED. A claim with nothing to contradict is dropped rather than stored —
    it is not the falsifiable claim the prompt asked for.
  • A ZERO-CLAIM RESPONSE FAILS THE READ. `minimax-m3` returned a well-formed, empty
    `{"claims": []}` after reading real input and still billed — `parsed` and a clean
    `finish_reason` both looked healthy, so this has to be an explicit rejection.
  • `middle_share` LANDS ON THE RUN ROW, same D22 signal `queries` writes.
  • THE NOTEBOOK IS THE REGION'S MEMORY (RULED 2026-08-24). Part N reads holding parts 1..N-1's
    claims, the citation gate is widened to the whole chain so an ancestor claim can be REFUTED and
    not merely added to, and the claims receipt is a DEBT the next chain walk collects — never a
    gate on closing a part.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from pipeline import llm_client
from pipeline.kb import frontier_queries as fq
from pipeline.kb import schema
from pipeline.kb import sitting_builder as sb
from pipeline.kb import sitting_claims as scl
from pipeline.kb import sitting_store as sst
from pipeline.kb import sitting_reader as sr

DIM = 8
_NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)


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


def _atom(conn, atom_id, vec, *, who="x:user:1", when="2026-08-01"):
    conn.execute("INSERT INTO atoms (atom_id, source_type, who_id, when_ts, entry_mode) "
                 "VALUES (?,?,?,?,'user-saved')", (atom_id, "x", who, when))
    text = f"{atom_id} body " + ("word " * 200)
    conn.execute("INSERT INTO chunks (atom_id, seq, char_start, char_end, text, vector) "
                 "VALUES (?,0,0,?,?,?)", (atom_id, len(text), text, vec.tobytes()))
    conn.commit()


@pytest.fixture()
def sitting(conn):
    """A two-atom sitting, built and unread (by either lens)."""
    _atom(conn, "a:seed", ANCHOR, who="x:alice")
    _atom(conn, "a:near", _at_cos(0.85), who="x:bob")
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"], label="mlx"),
                           floor=0.68)
    return rec["sitting_id"]


def _c(claim, *, falsified_by="a verified snapshot would show otherwise", atom_ids=("a:seed",)):
    return {"claim": claim, "falsified_by": falsified_by, "atom_ids": list(atom_ids)}


def _body(claims) -> str:
    return json.dumps({"claims": claims})


class _Resp:
    def __init__(self, text):
        self.text, self.model = text, "fake-model"
        self.input_tokens, self.output_tokens, self.cost_usd = 100, 20, 0.01
        self.raw = {}


@pytest.fixture()
def ready(monkeypatch):
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
def test_a_successful_read_writes_claims_and_stamps_the_lens(conn, sitting, ready, monkeypatch):
    _answer(monkeypatch, _body([_c("x402 processed $1.6M/month"), _c("ERC-8004 hit 97,713 agents")]))
    res = scl.read_claims(conn, sitting, now=_NOW)
    assert res["status"] == "ok" and len(res["claims"]) == 2
    assert sst.lens_read_state(conn, sitting, "claims")["read_status"] == "ok"
    assert len(sst.get_claims(conn, sitting)) == 2


def test_a_claims_read_does_not_stamp_queries_read_at(conn, sitting, ready, monkeypatch):
    """The independence Option C's whole design depends on — a `claims` read must not look like a
    `queries` read to the scheduler, which only ever consults `sittings.read_at`."""
    _answer(monkeypatch, _body([_c("x")]))
    scl.read_claims(conn, sitting, now=_NOW)
    assert sst.get_sitting(conn, sitting)["read_at"] is None


def test_a_queries_read_does_not_block_a_claims_read_or_vice_versa(conn, sitting, ready,
                                                                    monkeypatch):
    _q = {"text": "gated DeltaNet attention", "target_sources": ["arxiv"],
          "rationale": "because", "atom_ids": ["a:seed"]}
    _answer(monkeypatch, json.dumps({"consensus": "moved", "queries": [_q]}))
    sr.read_sitting(conn, sitting, now=_NOW)
    assert sst.get_sitting(conn, sitting)["read_at"] is not None

    _answer(monkeypatch, _body([_c("x")]))
    res = scl.read_claims(conn, sitting, now=_NOW)
    assert res["status"] == "ok"


def test_claims_are_never_re_read_once_read(conn, sitting, ready, monkeypatch):
    calls = []
    monkeypatch.setattr(llm_client, "call",
                        lambda role, **kw: calls.append(1) or _Resp(_body([_c("x")])))
    scl.read_claims(conn, sitting, now=_NOW)
    res = scl.read_claims(conn, sitting, now=_NOW)
    assert res["status"] == "skipped" and "already read" in res["reason"]
    assert len(calls) == 1


# ── falsified_by is required ────────────────────────────────────────────────────
def test_a_claim_with_no_falsified_by_is_dropped(conn, sitting, ready, monkeypatch):
    good = _c("ERC-8004 hit 97,713 agents")
    bad = {"claim": "adoption is growing", "falsified_by": "", "atom_ids": ["a:seed"]}
    _answer(monkeypatch, _body([good, bad]))
    res = scl.read_claims(conn, sitting, now=_NOW)
    assert res["status"] == "ok" and len(res["claims"]) == 1
    assert res["claims"][0]["claim"] == "ERC-8004 hit 97,713 agents"


# ── the empty response that looked healthy on every other signal ───────────────
def test_an_empty_claims_list_fails_the_read(conn, sitting, ready, monkeypatch):
    """⚠️ PINS THE minimax-m3 FINDING. A well-formed, empty {"claims": []} passes `parsed` and a
    clean `finish_reason`; only an explicit reject on zero claims catches it."""
    _answer(monkeypatch, _body([]))
    res = scl.read_claims(conn, sitting, now=_NOW)
    assert res["status"] == "failed"
    assert sst.lens_read_state(conn, sitting, "claims") is None
    assert sst.get_claims(conn, sitting) == []


# ── failure leaves no trace ─────────────────────────────────────────────────────
def test_a_failed_call_leaves_the_lens_unread_and_writes_no_claims(conn, sitting, ready,
                                                                    monkeypatch):
    def _boom(role, *, system, user, **kw):
        raise RuntimeError("provider exploded")
    monkeypatch.setattr(llm_client, "call", _boom)
    res = scl.read_claims(conn, sitting, now=_NOW)
    assert res["status"] == "failed"
    assert sst.lens_read_state(conn, sitting, "claims") is None
    assert sst.get_claims(conn, sitting) == []
    row = conn.execute("SELECT status, lens, sitting_id FROM frontier_reader_runs").fetchone()
    assert row["status"] == "failed" and row["lens"] == "claims" and row["sitting_id"] == sitting


def test_out_of_credits_is_named_rather_than_reported_as_a_broken_call(conn, sitting, ready,
                                                                       monkeypatch):
    class _402(RuntimeError):
        status = 402
    monkeypatch.setattr(llm_client, "call",
                        lambda role, **kw: (_ for _ in ()).throw(_402("no credit")))
    res = scl.read_claims(conn, sitting, now=_NOW)
    assert "OUT OF CREDITS" in res["reason"]
    assert sst.lens_read_state(conn, sitting, "claims") is None


def test_an_empty_sitting_is_skipped_without_calling(conn, ready, monkeypatch):
    calls = []
    monkeypatch.setattr(llm_client, "call", lambda role, **kw: calls.append(1))
    _atom(conn, "a:seed", ANCHOR)
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    conn.execute("UPDATE chunks SET vector = NULL")
    conn.commit()
    sid = sb.build_sitting(conn, seed, floor=0.68)["sitting_id"]
    assert scl.read_claims(conn, sid, now=_NOW)["status"] == "skipped"
    assert calls == []


# ── provenance ──────────────────────────────────────────────────────────────────
def test_citations_are_checked_against_the_sitting_not_the_corpus(conn, sitting, ready,
                                                                   monkeypatch):
    _atom(conn, "a:elsewhere", _unit(0, 1))
    _answer(monkeypatch, _body([_c("real one", atom_ids=["a:seed"]),
                                _c("phantom", atom_ids=["a:elsewhere"])]))
    res = scl.read_claims(conn, sitting, now=_NOW)
    assert res["status"] == "ok" and len(res["claims"]) == 1
    assert res["claims"][0]["claim"] == "real one"


# ── D22, made standing ───────────────────────────────────────────────────────────
def test_middle_share_lands_on_the_run_row(conn, sitting, ready, monkeypatch):
    _answer(monkeypatch, _body([_c("x", atom_ids=["a:seed", "a:near"])]))
    scl.read_claims(conn, sitting, now=_NOW)
    row = conn.execute("SELECT middle_share FROM frontier_reader_runs WHERE lens='claims'"
                       ).fetchone()
    assert row["middle_share"] is not None


# ── the prompt ──────────────────────────────────────────────────────────────────
def test_prompt_only_spends_nothing_and_shows_the_exact_prompt(conn, sitting, ready, monkeypatch):
    calls = []
    monkeypatch.setattr(llm_client, "call", lambda role, **kw: calls.append(1))
    res = scl.read_claims(conn, sitting, prompt_only=True, now=_NOW)
    assert res["status"] == "prompt-only" and calls == []
    assert "a:seed" in res["prompt"] and res["est_tokens"] > 0
    assert conn.execute("SELECT COUNT(*) FROM frontier_reader_runs").fetchone()[0] == 0
    assert sst.lens_read_state(conn, sitting, "claims") is None


def test_the_prompt_carries_no_standing_list(conn, sitting, ready, monkeypatch):
    """`claims` has nothing analogous to `queries`' survivors — the prompt is the sitting alone."""
    seen = {}
    _answer(monkeypatch, _body([_c("x")]), seen)
    scl.read_claims(conn, sitting, now=_NOW)
    assert "CURRENT STANDING" not in seen["user"]
    assert seen["user"].startswith("# Sitting:")


def test_the_settled_prompt_bans_forecasting(conn, sitting, ready, monkeypatch):
    """⚠️ PINS THE BUG A REAL RUN COST MONEY TO FIND: the first draft asked for claims falsifiable
    by a FUTURE observation, and that word alone drove the model into forecasting instead of
    describing the material."""
    seen = {}
    _answer(monkeypatch, _body([_c("x")]), seen)
    scl.read_claims(conn, sitting, now=_NOW)
    assert "Do not forecast" in seen["system"]
    assert "FUTURE" not in seen["system"]


def test_dry_run_writes_nothing_but_reports_the_claims(conn, sitting, ready, monkeypatch):
    _answer(monkeypatch, _body([_c("x")]))
    res = scl.read_claims(conn, sitting, dry_run=True, now=_NOW)
    assert res["status"] == "dry-run" and len(res["claims"]) == 1
    assert sst.get_claims(conn, sitting) == []
    assert sst.lens_read_state(conn, sitting, "claims") is None


# ── the notebook (RULED 2026-08-24) ─────────────────────────────────────────────
def _part(conn, *, continues=None, tag="p", when="2026-01-01"):
    """One more part of the same region, built and unread."""
    _atom(conn, f"a:{tag}", _at_cos(0.85, axis=2), when=when)
    return sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"], label="mlx"),
                            floor=0.68, continues=continues,
                            now=_NOW.replace(second=len(tag) % 50))["sitting_id"]


def test_part_two_reads_holding_part_ones_claims(conn, sitting, ready, monkeypatch):
    """THE CROSS-PART MEMORY. Old parts appear only as distilled claims — never as re-read text —
    which is what makes an unbounded region readable at a bounded price, and what lets part 2
    recognise that the audit it is reading GUTS the number part 1 established.
    """
    _answer(monkeypatch, _body([_c("x402 processed $24M/month, per Bloomberg")]))
    scl.read_claims(conn, sitting, now=_NOW)
    sst.mark_read(conn, sitting)

    p2 = _part(conn, continues=sitting, tag="two")
    seen: dict = {}
    _answer(monkeypatch, _body([_c("the audit puts it at $1.6M", atom_ids=["a:two"])]), seen)
    scl.read_claims(conn, p2, now=_NOW)

    prompt = seen["user"]
    assert "ESTABLISHED SO FAR" in prompt
    assert "$24M/month, per Bloomberg" in prompt, "part 1's claim is not in part 2's memory"
    assert "CONFIRM, REVISE or REFUTE" in prompt
    assert "a:seed body" not in prompt.split("=== THIS PART ===")[0], \
        "part 1's TEXT was re-rendered — the notebook exists so it is not"


def test_the_first_part_gets_no_preamble(conn, sitting, ready, monkeypatch):
    """Nothing established yet, so nothing to answer to. An empty notebook heading would only
    invite the model to fill it."""
    seen: dict = {}
    _answer(monkeypatch, _body([_c("x")]), seen)
    scl.read_claims(conn, sitting, now=_NOW)
    assert "ESTABLISHED SO FAR" not in seen["user"]


def test_a_claim_citing_only_an_ancestor_atom_survives_the_gate(conn, sitting, ready, monkeypatch):
    """THE MECHANICAL CHECK THE RULING LEFT OPEN. The preamble asks part 2 to refute part 1's
    claims, and the honest way to refute one is to cite the ancestor atom it rested on. Gated on
    the SHOWN set alone, `validate_claims` drops exactly those citations — so the notebook could be
    added to but never corrected, which is the one thing it exists for."""
    _answer(monkeypatch, _body([_c("part one said $24M")]))
    scl.read_claims(conn, sitting, now=_NOW)
    sst.mark_read(conn, sitting)

    p2 = _part(conn, continues=sitting, tag="two")
    # `a:near`, NOT `a:seed`: seeds are re-admitted into every part, so a seed citation would
    # resolve against part 2's own shown set and prove nothing. `a:near` is part 1's alone.
    assert "a:near" not in {a["atom_id"] for a in sst.get_sitting(conn, p2)["admissions"]}
    _answer(monkeypatch, _body([_c("that number was wash trading; refutes C1",
                                   atom_ids=["a:near"])]))
    res = scl.read_claims(conn, p2, now=_NOW)
    assert res["status"] == "ok" and res["claims"][0]["atom_ids"] == ["a:near"]


def test_a_claims_failure_costs_a_notebook_entry_not_the_part(conn, sitting, ready, monkeypatch):
    """THE ORDERING. The queries read closes the part and stamps `read_at`; claims runs after. Run
    the other way round, a bad claims call blocks a part that was ready to close."""
    _answer(monkeypatch, json.dumps({"consensus": "moved", "queries": [
        {"text": "gated DeltaNet attention", "target_sources": ["arxiv"],
         "rationale": "because", "atom_ids": ["a:seed"]}]}))
    sr.read_sitting(conn, sitting, now=_NOW)
    _answer(monkeypatch, _body([]))                      # zero claims → an explicit failure
    assert scl.read_claims(conn, sitting, now=_NOW)["status"] == "failed"

    assert sst.get_sitting(conn, sitting)["read_at"] is not None, "the part did not close"
    assert sst.lens_read_state(conn, sitting, "claims") is None
    assert sst.get_claims(conn, sitting) == []


def test_the_next_chain_walk_collects_the_unpaid_debt(conn, sitting, ready, monkeypatch):
    """And the debt does not vanish. An ancestor read but never distilled is collected just before
    part N+1 is rendered — the moment the hole in its own memory would start to matter."""
    sst.mark_read(conn, sitting)                         # read, never distilled
    p2 = _part(conn, continues=sitting, tag="two")
    _answer(monkeypatch, _body([_c("recovered from part one")]))

    paid = scl.collect_notebook_debt(conn, p2, now=_NOW)
    assert [x["sitting_id"] for x in paid] == [sitting]
    assert len(sst.get_claims(conn, sitting)) == 1
    assert sr.chain_claims(conn, p2)[0]["part"] == 1


def test_an_unread_ancestor_owes_nothing(conn, sitting, ready, monkeypatch):
    """Never read means never part of the story — collecting there would pay to distil material
    nobody has looked at, which is the read rail's job and its budget, not the notebook's."""
    p2 = _part(conn, continues=sitting, tag="two")
    monkeypatch.setattr(llm_client, "call",
                        lambda *a, **kw: pytest.fail("paid to distil an unread ancestor"))
    assert scl.collect_notebook_debt(conn, p2, now=_NOW) == []
