"""The shared machinery behind the reading rail, proven offline (stubbed transports).

⚠️ THIS FILE WAS `test_bookmark_reader.py` UNTIL 2026-08-16, AND THE RENAME IS THE POINT. That
module is deleted (D13), but roughly half of its tests never tested it: they tested `reader_core`'s
output contract, `frontier_queries`' verdict and retirement semantics, and the headless `claude -p`
transport — all of which SURVIVE, and all of which the sitting rail now depends on alone. Deleting
the file with the module would have removed the only proof of any of that, silently, in a commit
whose diff read as a straight retirement. What went with the module is only what was specific to
its 90-day window: `window_atoms`, `fit_to_budget`, `trigger_check`, and the single-flight cap.

The prose is kept VERBATIM wherever the claim did not change. Several of these comments record a
failure that cost real time to find, and re-deriving them from a test name is not possible.

What this file locks:

  • IDENTITY SURVIVES A RE-RUN. A confirmed query lands on the same row, so stage 2's per-query
    watermark is not silently reset every night.
  • THE SYSTEM NEVER REMOVES A QUERY. Survival is an explicit verdict, a query nobody verdicts is
    untouched, and a drop only slows it down. Retirement is a human act, and an AST sweep proves
    exactly one module in the pipeline can write it.
  • VALIDATION IS HARD, AND EVERY DROP IS VISIBLE. JSON mode buys valid JSON, never the RIGHT JSON.
  • PROVENANCE IS CHECKED, NOT TRUSTED. A cited atom_id is repaired against the atoms actually
    shown, or the query dies rather than joining to nothing later.
  • THE PORTABLE TRANSPORT IS THE DEFAULT, and the opt-in CLI path stays clean — no tools, no MCP,
    a neutral cwd, and the window on stdin.

Where a claim needs a whole run to prove, the run is a SITTING read — the one reading rail left.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from pipeline import llm_client
from pipeline.kb import frontier_queries as fq
from pipeline.kb import reader_core as core
from pipeline.kb import schema
from pipeline.kb import sitting_builder as sb
from pipeline.kb import sitting_reader as sr

from .conftest import last_run

_NOW = datetime(2026, 8, 9, 12, 0, 0, tzinfo=timezone.utc)

# A stand-in for whichever rail owns a query. It used to be `bookmark_reader.GENERATOR`; the
# semantics under test are the STORE's, and they are identical for any label, so a literal keeps
# these tests from re-acquiring a dependency on one rail's module constant.
GEN = "sitting:test"


# ── fixtures / helpers ──────────────────────────────────────────────────────────
@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _q(text: str, *, sources=("arxiv",), atom_ids=("x:1",), rationale="because"):
    return {"text": text, "target_sources": list(sources),
            "atom_ids": list(atom_ids), "rationale": rationale}


def _v(text, verdict="keep", *, atom_ids=("x:1",), reason="still live"):
    return {"text": text, "verdict": verdict, "reason": reason, "atom_ids": list(atom_ids)}


def _body(queries, consensus="the consensus", verdicts=None):
    obj = {"consensus": consensus, "queries": queries}
    if verdicts is not None:
        obj["verdicts"] = verdicts
    return json.dumps(obj)


def _fake_call(text: str, *, cost: float = 0.42):
    def call(role, *, system, user, **kw):
        return type("R", (), {"text": text, "model": "fake/model", "input_tokens": 100,
                              "output_tokens": 50, "cost_usd": cost, "raw": {}})()
    return call


@pytest.fixture()
def ready(monkeypatch):
    monkeypatch.setattr(llm_client, "preflight", lambda role: None)


# ── a real sitting, for the claims that need a whole run ────────────────────────
DIM = 8


def _unit(*w) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[:len(w)] = w
    return v / (np.linalg.norm(v) + 1e-9)


@pytest.fixture()
def sitting(conn):
    """The smallest region that can be read: two atoms, built and unread.

    Deliberately a SITTING and not a hand-written row. These tests prove what happens end to end on
    the surviving rail, so the fixture has to be something `sitting_reader` will actually read.
    """
    from pipeline.kb.embed import ensure_kb_meta
    ensure_kb_meta(conn, "fake", DIM, "local", "", storage_dtype="float32")
    for aid, vec, who in (("a:seed", _unit(1), "x:alice"), ("a:near", _unit(0.85, 0.53), "x:bob")):
        conn.execute("INSERT INTO atoms (atom_id, source_type, who_id, when_ts, entry_mode) "
                     "VALUES (?,?,?,?,'user-saved')", (aid, "x", who, "2026-08-01"))
        text = f"{aid} body " + ("word " * 200)
        conn.execute("INSERT INTO chunks (atom_id, seq, char_start, char_end, text, vector) "
                     "VALUES (?,0,0,?,?,?)", (aid, len(text), text, vec.tobytes()))
    conn.commit()
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"], label="mlx"),
                           floor=0.68)
    return rec["sitting_id"]


def _row(conn, normalized):
    return conn.execute("SELECT * FROM frontier_queries WHERE normalized=?",
                        (normalized,)).fetchone()


# ── upsert identity: stage 2's watermark depends on it ──────────────────────────
def test_reemission_lands_on_the_same_row(conn):
    first = fq.upsert_queries(conn, [_q("Muon  Optimizer Convergence")],
                              generator=GEN, now="2026-01-01T00:00:00+00:00")
    second = fq.upsert_queries(conn, [_q("muon optimizer convergence", rationale="newer")],
                               generator=GEN, now="2026-02-01T00:00:00+00:00")
    assert first["new"] == 1 and second["new"] == 0 and second["refreshed"] == 1
    assert first["query_ids"] == second["query_ids"]           # case/whitespace is not identity
    rows = list(conn.execute("SELECT * FROM frontier_queries"))
    assert len(rows) == 1                                       # never a twin
    assert rows[0]["emit_count"] == 2 and rows[0]["miss_count"] == 0
    assert rows[0]["created_at"] == "2026-01-01T00:00:00+00:00"       # origin preserved
    assert rows[0]["last_emitted_at"] == "2026-02-01T00:00:00+00:00"  # recency refreshed
    assert rows[0]["rationale"] == "newer"                      # provenance is the LATEST read's


def test_intra_run_duplicates_do_not_inflate_emit_count(conn):
    """Two emitted queries differing only by case are ONE sighting, not a sighting plus a
    confirmation."""
    res = fq.upsert_queries(conn, [_q("Same Query"), _q("same query")], generator=GEN)
    assert (res["new"], res["refreshed"]) == (1, 0)
    # And the text lists the watchlist diff reads carry the same collapse — naming it twice would
    # tell the user two questions were added when one was.
    assert res["new_texts"] == ["same query"] and res["refreshed_texts"] == []
    assert conn.execute("SELECT emit_count FROM frontier_queries").fetchone()[0] == 1


# ── verdicts: decay, never delete ───────────────────────────────────────────────
def test_a_keep_confirms_and_a_drop_slows(conn):
    fq.upsert_queries(conn, [_q("stays"), _q("fades")], generator=GEN,
                      now="2026-01-01T00:00:00+00:00")
    res = fq.apply_verdicts(conn, [_v("stays"), _v("fades", "drop")],
                            generator=GEN, now="2026-02-01T00:00:00+00:00")
    assert res == {"kept": 1, "dropped": 1, "unmatched": 0}
    stays, fades = _row(conn, "stays"), _row(conn, "fades")
    assert stays["miss_count"] == 0 and stays["emit_count"] == 2
    assert stays["last_emitted_at"] == "2026-02-01T00:00:00+00:00"
    assert fades["miss_count"] == 1
    assert fades["last_emitted_at"] == "2026-01-01T00:00:00+00:00"   # a drop confirms nothing


def test_a_query_with_no_verdict_is_untouched(conn):
    """THE ENTIRE POINT. v1 read silence as a miss, and at the query cap silence just meant
    "something else was added this run" — three live queries were retired on that arithmetic."""
    fq.upsert_queries(conn, [_q("unmentioned")], generator=GEN,
                      now="2026-01-01T00:00:00+00:00")
    before = tuple(_row(conn, "unmentioned"))
    fq.apply_verdicts(conn, [], generator=GEN, now="2026-06-01T00:00:00+00:00")
    assert tuple(_row(conn, "unmentioned")) == before             # byte-identical, not aged


def test_drops_accumulate_and_a_later_keep_restores_full_speed(conn):
    """Demotion has to be REVERSIBLE, or it is deletion on a delay. A thread that goes quiet for
    months and then produces something new must come straight back to daily."""
    fq.upsert_queries(conn, [_q("quiet")], generator=GEN)
    for expected in (1, 2, 3):
        fq.apply_verdicts(conn, [_v("quiet", "drop")], generator=GEN)
        assert _row(conn, "quiet")["miss_count"] == expected
    fq.apply_verdicts(conn, [_v("quiet")], generator=GEN)
    assert _row(conn, "quiet")["miss_count"] == 0
    assert _row(conn, "quiet")["status"] == "active"              # and never left the query set


def test_verdicts_are_scoped_to_one_generator(conn):
    """One region's read must not move another region's counters — otherwise whichever rail ran
    last owns every other rail's query set."""
    fq.upsert_queries(conn, [_q("mine")], generator="sitting:kiss1r")
    res = fq.apply_verdicts(conn, [_v("mine", "drop")], generator="sitting:mlx")
    assert res["unmatched"] == 1 and res["dropped"] == 0
    assert _row(conn, "mine")["miss_count"] == 0


def test_an_uncited_keep_is_honoured_but_leaves_provenance_alone(conn):
    """FAIL-SAFE. A model that forgets its atom_ids must not cost the query its life. The stale
    provenance is left visible rather than blanked, because a citation that silently disappears is
    worse than one that is visibly old."""
    fq.upsert_queries(conn, [_q("cited", atom_ids=["x:original"])], generator=GEN)
    fq.apply_verdicts(conn, [_v("cited", atom_ids=[])], generator=GEN)
    row = _row(conn, "cited")
    assert row["miss_count"] == 0 and row["emit_count"] == 2      # honoured
    assert json.loads(row["source_atom_ids"]) == ["x:original"]   # untouched, not blanked


def test_a_cited_keep_refreshes_provenance(conn):
    fq.upsert_queries(conn, [_q("cited", atom_ids=["x:old"])], generator=GEN)
    fq.apply_verdicts(conn, [_v("cited", atom_ids=["x:new"])], generator=GEN)
    assert json.loads(_row(conn, "cited")["source_atom_ids"]) == ["x:new"]


def test_a_verdict_on_an_unknown_query_moves_nothing(conn):
    fq.upsert_queries(conn, [_q("real")], generator=GEN)
    res = fq.apply_verdicts(conn, [_v("invented", "drop")], generator=GEN)
    assert res["unmatched"] == 1
    assert conn.execute("SELECT COUNT(*) FROM frontier_queries").fetchone()[0] == 1


def test_case_drift_in_an_echoed_verdict_still_lands_on_the_row(conn):
    """Identity is the NORMALIZED string, so a model that re-capitalizes its echo has still
    rendered a verdict on that query — not on a new one."""
    fq.upsert_queries(conn, [_q("gated DeltaNet attention")], generator=GEN)
    assert fq.apply_verdicts(conn, [_v("Gated  deltanet Attention", "drop")],
                             generator=GEN)["dropped"] == 1
    assert _row(conn, "gated deltanet attention")["miss_count"] == 1


# ── retirement: the only door out, and only David opens it ──────────────────────
def test_retire_is_the_only_thing_that_stops_a_query_executing(conn):
    fq.upsert_queries(conn, [_q("done with this")], generator=GEN)
    assert fq.retire_query(conn, "Done With This") is True        # matched on the normalized form
    assert _row(conn, "done with this")["status"] == "retired"
    assert fq.active_queries(conn) == []                          # gone from execution...
    assert _row(conn, "done with this") is not None               # ...but never deleted
    assert fq.unretire_query(conn, "done with this") is True
    assert len(fq.active_queries(conn)) == 1


def test_nothing_automatic_writes_retired(conn):
    """PINNED INVARIANT. Every automatic path is exercised against a query, and none of them may
    reach 'retired' — the decay tiers exist precisely so no machine path ever needs to."""
    fq.upsert_queries(conn, [_q("survivor")], generator=GEN)
    for _ in range(50):                                           # far past any old threshold
        fq.apply_verdicts(conn, [_v("survivor", "drop")], generator=GEN)
    fq.upsert_queries(conn, [_q("survivor")], generator=GEN)      # collision path too
    assert _row(conn, "survivor")["status"] == "active"
    assert _row(conn, "survivor")["miss_count"] == 0              # ...zeroed by the collision
    assert len(fq.active_queries(conn)) == 1


def test_only_one_module_in_the_pipeline_can_write_retired():
    """The behavioural test above proves the paths it exercises; this proves there are no OTHERS.
    A future rail could add its own automatic retirement and every existing test would stay green,
    which is exactly how v1's arithmetic retirement sat unnoticed for twelve days.

    Read via the AST, not grep, so the prose ABOUT retirement — which is everywhere in these
    modules — does not trip it. Whitespace is stripped from each literal so a differently-spaced
    `SET status = 'retired'` is caught too.
    """
    import ast

    root = Path(__file__).resolve().parents[2] / "pipeline"
    writers = set()
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "setstatus='retired'" in "".join(node.value.split()).lower():
                    writers.add(str(path.relative_to(root.parent)))
    assert writers == {"pipeline/kb/frontier_queries.py"}


def test_re_emitting_a_retired_querys_text_does_not_resurrect_it(conn):
    """A retired query is hidden from the reader, so re-inventing its wording is an ordinary thing
    for the model to do. It must not overturn David's decision by accident."""
    fq.upsert_queries(conn, [_q("retired thread")], generator=GEN)
    fq.retire_query(conn, "retired thread")
    fq.upsert_queries(conn, [_q("retired thread")], generator=GEN)
    assert _row(conn, "retired thread")["status"] == "retired"
    assert fq.active_queries(conn) == []


# ── validation: the provider guarantees valid JSON, not the right JSON ──────────
def test_more_than_max_queries_is_clamped_and_noted():
    obj = {"consensus": "c", "queries": [_q(f"query number {i}") for i in range(40)]}
    _, queries, _, notes = core.validate(obj)
    assert len(queries) == core.MAX_QUERIES
    assert any("clamped" in n for n in notes)


def test_going_over_the_new_query_budget_is_noted_not_enforced():
    """An extra good thread costs one standing query; dropping it costs the thread. But a model
    ignoring the budget is exactly what makes a query set balloon, so it has to be visible."""
    obj = {"consensus": "c", "queries": [_q(f"query number {i}") for i in range(8)]}
    _, queries, _, notes = core.validate(obj)
    assert len(queries) == 8                                     # kept
    assert any(f"budget {core.MAX_NEW_QUERIES}" in n for n in notes)


def test_malformed_items_are_dropped_and_noted():
    obj = {"consensus": "c", "queries": [
        _q("good one"),
        {"text": "", "target_sources": ["arxiv"], "atom_ids": ["x:1"]},      # empty text
        {"text": "no sources", "target_sources": ["twitter"], "atom_ids": ["x:1"]},  # unroutable
        {"text": "no provenance", "target_sources": ["arxiv"], "atom_ids": []},      # uncited
        "not an object",
    ]}
    _, queries, _, notes = core.validate(obj)
    assert [q["text"] for q in queries] == ["good one"]
    assert len(notes) >= 4                                       # every drop is visible


def test_unknown_target_sources_are_stripped_but_the_query_survives():
    _, queries, _, _ = core.validate({"consensus": "c", "queries": [
        _q("mixed", sources=["arxiv", "myspace"])]})
    assert queries[0]["target_sources"] == ["arxiv"]


# ── verdict parsing ─────────────────────────────────────────────────────────────
def test_verdicts_are_parsed_and_matched_to_the_shown_queries():
    obj = {"consensus": "c", "queries": [],
           "verdicts": [_v("alpha thread"), _v("beta thread", "drop")]}
    _, _, verdicts, notes = core.validate(obj, known_atom_ids={"x:1"},
                                          shown=["alpha thread", "beta thread"])
    assert [(v["text"], v["verdict"]) for v in verdicts] == [
        ("alpha thread", "keep"), ("beta thread", "drop")]
    assert notes == []                                            # a complete run says nothing


def test_an_absent_verdicts_section_is_silence_not_an_error():
    """The shape of the first-ever read: nothing is running, so there is nothing to judge."""
    _, queries, verdicts, notes = core.validate({"consensus": "c", "queries": [_q("first")]},
                                                shown=[])
    assert verdicts == [] and notes == [] and len(queries) == 1


def test_a_verdict_on_a_query_that_was_never_shown_is_dropped():
    """A hallucinated verdict would move a counter on evidence that does not exist."""
    _, _, verdicts, notes = core.validate(
        {"consensus": "c", "queries": [], "verdicts": [_v("never shown", "drop")]},
        shown=["real one"])
    assert verdicts == []
    assert any("not a query that was shown" in n for n in notes)


def test_shown_queries_with_no_verdict_are_counted():
    """Not an error — they are untouched by design. But a reader that stops verdicting has lost
    the whole survival signal while every counter sits still and looks healthy."""
    _, _, verdicts, notes = core.validate(
        {"consensus": "c", "queries": [], "verdicts": [_v("judged")]},
        known_atom_ids={"x:1"}, shown=["judged", "ignored a", "ignored b"])
    assert len(verdicts) == 1
    assert any("2 of 3 shown queries got no verdict" in n for n in notes)


def test_an_unreadable_verdict_becomes_no_verdict_rather_than_a_drop():
    """FAIL-SAFE: a garbled call must never cost a query its speed."""
    _, _, verdicts, notes = core.validate(
        {"consensus": "c", "queries": [],
         "verdicts": [{"text": "a thread", "verdict": "maybe?"}]},
        shown=["a thread"])
    assert verdicts == []
    assert any("treated as no verdict" in n for n in notes)


def test_a_query_both_kept_and_re_emitted_is_counted_once():
    """The old prompt trained exactly this for two months, so it is the likeliest way a run
    inflates a counter: the verdict and the upsert would each bump `emit_count`."""
    _, queries, verdicts, notes = core.validate(
        {"consensus": "c", "queries": [_q("gated DeltaNet attention"), _q("genuinely new")],
         "verdicts": [_v("gated DeltaNet attention")]},
        known_atom_ids={"x:1"}, shown=["gated DeltaNet attention"])
    assert [q["text"] for q in queries] == ["genuinely new"]      # the duplicate is gone
    assert len(verdicts) == 1                                     # the decision survives
    assert any("already verdicted" in n for n in notes)


def test_verdict_citations_get_the_same_bare_id_repair_as_queries():
    _, _, verdicts, notes = core.validate(
        {"consensus": "c", "queries": [], "verdicts": [_v("a thread", atom_ids=["2083"])]},
        known_atom_ids={"x:2083"}, shown=["a thread"])
    assert verdicts[0]["atom_ids"] == ["x:2083"]
    assert any("repaired 1" in n for n in notes)


def test_a_bare_atom_id_is_repaired_against_what_was_shown():
    """The live model strips the source prefix and cites `2083...` for `[x:2083...]`. Provenance
    is machine-checkable against the exact material it was shown, so repair it rather than trusting
    it — a citation that joins to nothing fails silently later instead of loudly now."""
    _, queries, _, notes = core.validate(
        {"consensus": "c", "queries": [_q("gated DeltaNet", atom_ids=["2083", "x:9999"])]},
        known_atom_ids={"x:2083", "x:9999"})
    assert queries[0]["atom_ids"] == ["x:2083", "x:9999"]      # repaired + exact both survive
    assert any("repaired 1" in n for n in notes)


def test_an_ambiguous_or_unknown_citation_is_dropped_not_guessed():
    _, queries, _, notes = core.validate(
        {"consensus": "c", "queries": [
            _q("ambiguous", atom_ids=["777"]),                 # matches two sources
            _q("invented", atom_ids=["x:does-not-exist"]),
            _q("fine", atom_ids=["x:1"])]},
        known_atom_ids={"x:777", "substack:777", "x:1"})
    assert [q["text"] for q in queries] == ["fine"]
    assert sum("no resolvable atom_ids" in n for n in notes) == 2


class _Proc:
    def __init__(self, stdout="", stderr=""):
        self.stdout, self.stderr = stdout, stderr


def test_cli_failure_reports_the_reason_not_the_zeroed_usage_block():
    """A failed `claude -p` prints its envelope to stdout with the message in `result`, AFTER a
    `usage` block of zeros. Truncating the raw envelope reported only the zeros — measured on the
    first live sitting read, 2026-08-24, whose entire recorded reason said nothing happened."""
    env = {"is_error": True, "stop_reason": "stop_sequence", "subtype": "error_during_execution",
           "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                     "padding": "x" * 600},
           "result": "Claude AI usage limit reached"}
    msg = core._cli_failure(_Proc(stdout=json.dumps(env)))
    assert "Claude AI usage limit reached" in msg
    assert "error_during_execution" in msg and "stop_sequence" in msg


def test_cli_failure_falls_back_to_raw_output_when_there_is_no_envelope():
    assert "ENOENT" in core._cli_failure(_Proc(stderr="dyld: ENOENT"))
    assert "segfault" in core._cli_failure(_Proc(stdout="segfault"))
    assert core._cli_failure(_Proc()) == ""          # nothing to say, and it says nothing


@pytest.mark.parametrize("body", ["not json at all", "", "[1,2,3]", '{"queries": '])
def test_unparseable_bodies_yield_none(body):
    assert core.parse_response(body) is None


def test_a_fenced_body_still_parses():
    assert core.parse_response('```json\n{"consensus":"c","queries":[]}\n```')["consensus"] == "c"


# ── failure paths: a failure writes NOTHING ─────────────────────────────────────
def test_a_bad_response_leaves_an_EXISTING_query_set_untouched(conn, sitting, ready, monkeypatch):
    """Not "no rows were written" — BYTE-IDENTICAL. The rows that were already there are the ones a
    half-applied read would corrupt, and a store that starts empty cannot show that.

    Re-driven through the sitting reader when `bookmark_reader` was retired; the claim is the
    store's, so the rail that exercises it does not matter.
    """
    fq.upsert_queries(conn, [_q("pre-existing")], generator="sitting:mlx")
    before = [tuple(r) for r in conn.execute(
        "SELECT * FROM frontier_queries ORDER BY query_id")]
    monkeypatch.setattr(llm_client, "call", _fake_call("total garbage"))
    res = sr.read_sitting(conn, sitting, now=_NOW)
    assert res["status"] == "failed" and "unparseable" in res["reason"]
    assert [tuple(r) for r in conn.execute(
        "SELECT * FROM frontier_queries ORDER BY query_id")] == before
    run = last_run(conn)
    # The call WAS attempted, so the recorded figure is the real charge when the response carried
    # one, and 0.0 only when it did not — never NULL, which means "never reached the provider".
    assert run["status"] == "failed" and run["cost_usd"] is not None


def test_truncation_is_reported_as_truncation_not_as_bad_json(conn, sitting, ready, monkeypatch):
    """Both look like unparseable JSON downstream, but one is fixed by raising `max_tokens` and the
    other by changing the prompt. The model's reasoning is billed against the same ceiling as its
    answer, so this is a live failure mode on a long read."""
    def call(role, *, system, user, **kw):
        return type("R", (), {
            "text": '{"consensus": "cut off mid',
            "model": "m", "input_tokens": 1, "output_tokens": 32000, "cost_usd": 0.9,
            "raw": {"choices": [{"finish_reason": "length"}]}})()
    monkeypatch.setattr(llm_client, "call", call)
    res = sr.read_sitting(conn, sitting, now=_NOW)
    assert res["status"] == "failed" and "truncated at max_tokens" in res["reason"]


# ── transport B: the headless `claude -p` path ──────────────────────────────────
def _cli_envelope(result_text: str, *, ok: bool = True):
    return json.dumps({
        "is_error": not ok, "subtype": "success" if ok else "error_during_execution",
        "result": result_text, "total_cost_usd": 0.21,
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 216000,
                  "cache_read_input_tokens": 0, "output_tokens": 900},
        "modelUsage": {"claude-sonnet-5": {}}})


@pytest.fixture()
def cli_backend(monkeypatch):
    monkeypatch.setenv("OPYT_FRONTIER_BACKEND", core.BACKEND_CLI)
    monkeypatch.setattr("shutil.which", lambda n: "/usr/local/bin/claude" if n == "claude" else None)


def _spy_run(captured, envelope):
    def run(cmd, **kw):
        captured["cmd"], captured["kw"] = cmd, kw
        return type("P", (), {"returncode": 0, "stdout": envelope, "stderr": ""})()
    return run


def test_the_shipped_default_backend_is_the_portable_one(monkeypatch):
    """CLAUDE.md requires the core to run on ANY MCP client. A CLI default would silently give
    Cursor/Windsurf/Desktop users no Frontier at all."""
    monkeypatch.delenv("OPYT_FRONTIER_BACKEND", raising=False)
    monkeypatch.setattr("pipeline.ingestion.utils.load_yaml_config", lambda: {})
    assert core.resolve_backend() == core.BACKEND_API


def test_an_unreadable_config_still_picks_the_portable_backend(monkeypatch):
    monkeypatch.delenv("OPYT_FRONTIER_BACKEND", raising=False)
    def boom():
        raise OSError("no config")
    monkeypatch.setattr("pipeline.ingestion.utils.load_yaml_config", boom)
    assert core.resolve_backend() == core.BACKEND_API


def test_the_cli_model_ships_as_sonnet_and_is_settable_from_settings(monkeypatch):
    """A settings key, not env-only. The detached spawn inherits the MCP server's environment, not
    David's shell, so an env-only knob is inert exactly where the job runs unattended."""
    monkeypatch.delenv("OPYT_FRONTIER_CLI_MODEL", raising=False)
    monkeypatch.setattr("pipeline.ingestion.utils.load_yaml_config", lambda: {})
    assert core.resolve_cli_model() == "sonnet"                    # mirrors the API path
    monkeypatch.setattr("pipeline.ingestion.utils.load_yaml_config",
                        lambda: {"frontier": {"cli_model": "opus"}})
    assert core.resolve_cli_model() == "opus"
    monkeypatch.setenv("OPYT_FRONTIER_CLI_MODEL", "haiku")
    assert core.resolve_cli_model() == "haiku"                     # env still wins for a one-off


def test_the_resolved_model_reaches_the_command(monkeypatch, cli_backend):
    cap = {}
    monkeypatch.setenv("OPYT_FRONTIER_CLI_MODEL", "opus")
    monkeypatch.setattr(subprocess, "run",
                        _spy_run(cap, _cli_envelope('{"consensus":"c","queries":[]}')))
    core.call_claude_cli("s", "u")
    assert cap["cmd"][cap["cmd"].index("--model") + 1] == "opus"


def test_the_cli_runs_in_a_neutral_dir_with_no_tools_and_stdin_input(monkeypatch, cli_backend):
    """The cleanliness contract, asserted flag by flag. The cwd is the one that actually bit:
    running from the repo injected 36,917 tokens of CLAUDE.md and project context versus 8,205
    from a neutral dir — ~28K tokens of OPYT's own opinions leaking into a job whose whole point
    is reading the material without a prior."""
    cap = {}
    monkeypatch.setattr(subprocess, "run",
                        _spy_run(cap, _cli_envelope('{"consensus":"c","queries":[]}')))
    core.call_claude_cli("SYSTEM TEXT", "WINDOW TEXT")
    cmd, kw = cap["cmd"], cap["kw"]
    assert "-p" in cmd and "--system-prompt" in cmd                  # replaces, never appends
    assert cmd[cmd.index("--system-prompt") + 1] == "SYSTEM TEXT"
    assert cmd[cmd.index("--allowed-tools") + 1] == ""               # a completion, not an agent
    assert "--strict-mcp-config" in cmd and cmd[cmd.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert kw["input"] == "WINDOW TEXT"                              # ~1MB — never argv
    assert kw["cwd"] and "frontier-reader-" in kw["cwd"]             # empty temp dir, not the repo
    assert kw["timeout"] == core.CLI_TIMEOUT_S


def test_the_cli_counts_cached_prompt_tokens_as_input(monkeypatch, cli_backend):
    """The material is sent once and cached, so `input_tokens` alone reports ~2 tokens for a
    200K-token read — the run row would claim it barely read anything."""
    monkeypatch.setattr(subprocess, "run",
                        _spy_run({}, _cli_envelope('{"consensus":"c","queries":[]}')))
    r = core.call_claude_cli("s", "u")
    assert r.input_tokens == 216002 and r.output_tokens == 900


def test_the_run_names_the_model_that_did_the_work(monkeypatch, cli_backend):
    """Claude Code bills auxiliary steps to a small model in the same `modelUsage` map, so taking
    the first key logged `claude-haiku-4-5` for a run driven by Sonnet. On a role justified
    entirely by "not the cheap model", naming the wrong one is worse than naming none."""
    env = json.loads(_cli_envelope('{"consensus":"c","queries":[]}'))
    env["modelUsage"] = {
        "claude-haiku-4-5": {"inputTokens": 10, "outputTokens": 20},          # auxiliary, first key
        "claude-sonnet-5": {"cacheCreationInputTokens": 216000, "outputTokens": 900}}
    monkeypatch.setattr(subprocess, "run", _spy_run({}, json.dumps(env)))
    assert core.call_claude_cli("s", "u").model == "claude-cli:claude-sonnet-5"


@pytest.mark.parametrize("stdout, frag", [
    ("not json", "non-JSON"),
    (json.dumps({"is_error": True, "subtype": "error", "result": "boom"}), "reported failure"),
])
def test_a_broken_cli_response_raises_rather_than_returning_junk(monkeypatch, cli_backend,
                                                                 stdout, frag):
    monkeypatch.setattr(subprocess, "run", _spy_run({}, stdout))
    with pytest.raises(RuntimeError, match=frag):
        core.call_claude_cli("s", "u")


def test_a_full_read_on_the_cli_backend_writes_queries(conn, sitting, monkeypatch, cli_backend):
    monkeypatch.setattr(subprocess, "run", _spy_run(
        {}, _cli_envelope(_body([_q("muon optimizer", atom_ids=["a:seed"])]))))
    res = sr.read_sitting(conn, sitting, now=_NOW)
    assert res["status"] == "ok" and res["emitted"] == 1
    assert last_run(conn, status="ok")["model"].startswith("claude-cli:")


def test_a_missing_cli_degrades_open_without_spending(conn, sitting, monkeypatch):
    """A preflight failure never reached a provider, so `cost_usd` stays NULL — the distinction
    that separates "we could not call" from "we called and it failed"."""
    monkeypatch.setenv("OPYT_FRONTIER_BACKEND", core.BACKEND_CLI)
    monkeypatch.setattr("shutil.which", lambda n: None)
    res = sr.read_sitting(conn, sitting, now=_NOW)
    assert res["status"] == "failed" and "not on PATH" in res["reason"]
    assert last_run(conn)["cost_usd"] is None


def test_an_unknown_backend_fails_closed_rather_than_guessing(conn, sitting, monkeypatch):
    monkeypatch.setenv("OPYT_FRONTIER_BACKEND", "gpt-by-carrier-pigeon")
    res = sr.read_sitting(conn, sitting, now=_NOW)
    assert res["status"] == "failed" and "unknown backend" in res["reason"]
