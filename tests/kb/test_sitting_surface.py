"""The scope report and the `sitting` tool, proven offline.

What these lock, and every one is a failure that would be invisible in production:

  • A PREVIEW SPENDS NO MODEL CALL. It is the free look at a region whose size the caller cannot
    predict — a phrase resolves to 4 atoms or 236 and only the corpus knows which.
  • IT IS A SCOPE CHECK, NEVER A SPEND GATE. `read` must be reachable without it, or the free look
    becomes the per-use permission gate the consent rule forbids.
  • THE WARNINGS WARN AND NEVER REFUSE. Whether a 3-day region is the right input depends on the
    question, and only the caller knows the question.
  • A SCOPE REPORT READS NO ATOM'S TEXT. Stitching a 236-atom region to count its authors would put
    the whole region into RAM for a free call.
  • THE TOOL IS TRANSPORT-AGNOSTIC. It must never assume which reader backend is configured.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.kb import schema
from pipeline.kb import sitting_builder as sb
from pipeline.kb import sitting_render as sre
from pipeline.kb import sitting_store as sst
from pipeline.kb import sitting_surface as ss

# Wider than the other sitting suites' 8, and the reason is the near-duplicate ceiling. Two vectors
# built on the SAME axis at similar cosines are nearly parallel, so they are skipped as reposts and
# a "12-atom region" quietly admits four. One axis per atom keeps mutual redundancy at c**2 = 0.64,
# comfortably under the 0.95 ceiling, so these regions are the size they say they are.
DIM = 32


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    from pipeline.kb.embed import ensure_kb_meta
    ensure_kb_meta(c, "fake", DIM, "local", "", storage_dtype="float32")
    yield c
    c.close()


def _at_cos(c: float, axis: int = 1) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[0], v[axis] = c, float(np.sqrt(max(0.0, 1.0 - c * c)))
    return v / (np.linalg.norm(v) + 1e-9)


ANCHOR = _at_cos(1.0)


def _atom(conn, atom_id, vec, *, who="x:user:1", when="2026-08-01", chars=400):
    conn.execute("INSERT INTO atoms (atom_id, source_type, who_id, when_ts, entry_mode) "
                 "VALUES (?,?,?,?,'user-saved')", (atom_id, "x", who, when))
    text = f"{atom_id} body " + ("word " * max(1, chars // 5))
    conn.execute("INSERT INTO chunks (atom_id, seq, char_start, char_end, text, vector) "
                 "VALUES (?,0,0,?,?,?)", (atom_id, len(text), text, vec.tobytes()))
    conn.commit()


def _region(conn, n=12, *, who="x:alice", start_day=1, span_days=200):
    """A standalone-tier region spread over `span_days`, all by one author unless told otherwise."""
    _atom(conn, "a:seed", ANCHOR, who=who, when="2026-01-01")
    for i in range(n):
        day = start_day + (i * span_days // max(1, n))
        month, dom = 1 + day // 28, 1 + day % 28
        _atom(conn, f"a:{i}", _at_cos(0.80, axis=1 + i), who=who,
              when=f"2026-{min(month, 12):02d}-{dom:02d}")
    return sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"], label="mlx"),
                            floor=0.68)


# ── the report ──────────────────────────────────────────────────────────────────
def test_scope_reports_size_span_and_authors_without_reading_any_text(conn):
    """⚠️ THE `with_text=False` CONTRACT. `_atom_bodies` calls itself "the ONLY place text enters
    RAM"; a free scope check over a 236-atom region must not be the thing that breaks that."""
    rec = _region(conn, n=12, who="x:alice")
    seen = []
    real = sre._atom_bodies

    def _spy(c, ids, **kw):
        seen.append(kw.get("with_text", True))
        return real(c, ids, **kw)

    import pipeline.kb.sitting_render as _sre
    _sre._atom_bodies, ss.sre._atom_bodies = _spy, _spy
    try:
        rep = ss.scope(conn, rec)
    finally:
        _sre._atom_bodies, ss.sre._atom_bodies = real, real

    assert seen == [False]                       # asked for metadata only
    assert rep["atoms"] == rec["atoms"] and rep["tokens"] > 0
    assert rep["first"] and rep["last"] and rep["days"] > 0
    assert rep["authors"] == 1 and rep["top_author"] == "x:alice"
    assert rep["sitting_id"] == rec["sitting_id"]


def test_the_sample_names_atoms_in_the_order_a_read_would_see_them(conn):
    """Chronological, matching the render — a sample in admission order would show the caller a
    different region than the one that gets read."""
    rec = _region(conn, n=6)
    rep = ss.scope(conn, rec)
    whens = [s["when"] for s in rep["sample"]]
    assert whens == sorted(whens)
    assert len(rep["sample"]) <= ss.SAMPLE_ATOMS


def test_undated_atoms_are_counted_apart_from_the_span(conn):
    """⚠️ THEY MUST NOT BE FOLDED IN. Undated atoms sort LAST in the render and have no position in
    time, so treating a missing date as any particular date either stretches the span or crushes
    it — and both readings look like data."""
    _atom(conn, "a:seed", ANCHOR, when="2026-01-01")
    _atom(conn, "a:dated", _at_cos(0.85), when="2026-06-01")
    _atom(conn, "a:undated", _at_cos(0.84, axis=2), when="")
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    rep = ss.scope(conn, rec)
    assert rep["undated"] == 1
    assert rep["first"] == "2026-01-01" and rep["last"] == "2026-06-01"


# ── the warnings ────────────────────────────────────────────────────────────────
def test_a_healthy_region_warns_about_nothing(conn):
    """The negative control. A warning list that is never empty is a banner, and a banner printed
    on every call trains the reader to skip the one call where it matters."""
    rec = _region(conn, n=12, who="x:alice")
    conn.executemany("UPDATE atoms SET who_id = ? WHERE atom_id = ?",
                     [(f"x:w{i}", f"a:{i}") for i in range(12)])
    conn.commit()
    assert ss.scope(conn, rec)["warnings"] == []


def test_a_sprout_region_is_flagged_as_too_thin_for_standing_queries(conn):
    rec = _region(conn, n=3, who="x:a")                   # 4 admitted = sprout tier
    conn.executemany("UPDATE atoms SET who_id = ? WHERE atom_id = ?",
                     [(f"x:w{i}", f"a:{i}") for i in range(3)])
    conn.commit()
    warns = " ".join(ss.scope(conn, rec)["warnings"])
    assert "sprout" in warns and str(sb.TIER_STANDALONE_MIN) in warns


def test_a_narrow_span_is_flagged_as_having_no_arc(conn):
    """Reading in publication order exists to show how a conversation MOVED. A window this narrow
    holds a snapshot, and the caller should know that before paying to look for movement in it."""
    rec = _region(conn, n=12, span_days=2)
    warns = " ".join(ss.scope(conn, rec)["warnings"])
    assert "snapshot" in warns and "arc" in warns


def test_a_single_author_region_is_flagged(conn):
    """MEASURED, not guessed: a 76-atom sitting that was ~85% one person produced queries pointing
    back at that person's own repos — things the user already had."""
    rec = _region(conn, n=12, who="x:taelin")
    warns = " ".join(ss.scope(conn, rec)["warnings"])
    assert "x:taelin" in warns and "100%" in warns


def test_a_budget_stopped_region_says_it_is_part_one_of_more(conn):
    _atom(conn, "a:seed", ANCHOR, when="2026-01-01", chars=4000)
    for i in range(8):
        _atom(conn, f"a:{i}", _at_cos(0.80, axis=1 + i), who=f"x:w{i}",
              when=f"2026-0{1 + i % 8}-01", chars=4000)
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68,
                           budget_tokens=1200)
    assert rec["stop"] == "budget"
    assert "part 1" in " ".join(ss.scope(conn, rec)["warnings"])


def test_an_unknown_lens_says_silence_is_not_approval(conn):
    """A lens with no rules yet must not come back looking healthy — "no warnings defined" and "no
    problems found" are different facts and they render identically if this is missed."""
    rec = _region(conn, n=12)
    warns = ss.lens_warnings(ss.scope(conn, rec), lens="not-a-real-lens")
    assert warns and "not approval" in warns[0]


def test_each_sitting_scoped_lens_gets_its_own_tier_wording(conn):
    """Job L. `queries`' "generate standing queries" reason is nonsense for a lens that emits no
    queries — each lens must name ITS OWN reason a thin region is a poor fit."""
    rec = _region(conn, n=3, who="x:a")                    # 4 admitted = sprout tier
    scope = ss.scope(conn, rec, lens="briefing")
    for lens in ("briefing", "trajectory", "disconfirmation", "gaps"):
        warns = " ".join(ss.lens_warnings(scope, lens=lens))
        assert "sprout" in warns and str(sb.TIER_STANDALONE_MIN) in warns
        assert "standing queries" not in warns


def test_only_trajectory_and_queries_warn_on_a_narrow_span(conn):
    """D6's named example. `trajectory` needs an arc to trace the same way `queries` does; a
    briefing or a disconfirmation pass is not the same failure over a short window."""
    rec = _region(conn, n=12, span_days=2)
    scope = ss.scope(conn, rec, lens="briefing")
    for lens in ("queries", "trajectory"):
        assert "arc" in " ".join(ss.lens_warnings(scope, lens=lens))
    for lens in ("briefing", "disconfirmation", "gaps"):
        assert "arc" not in " ".join(ss.lens_warnings(scope, lens=lens))


def test_only_queries_warns_on_a_single_author_region(conn):
    """MEASURED for `queries` alone — a single-author region generating self-referential queries.
    No analogous measured failure exists for the host-side lenses, so they stay silent on it."""
    rec = _region(conn, n=12, who="x:taelin")
    scope = ss.scope(conn, rec, lens="briefing")
    assert "x:taelin" in " ".join(ss.lens_warnings(scope, lens="queries"))
    for lens in ("briefing", "trajectory", "disconfirmation", "gaps"):
        assert "x:taelin" not in " ".join(ss.lens_warnings(scope, lens=lens))


def test_a_budget_stop_warns_every_sitting_scoped_lens(conn):
    """Universal, unlike tier/span/author: every lens reads the same partial document when the
    budget cut a build short, so every lens needs to know it is reading part 1."""
    _atom(conn, "a:seed", ANCHOR, when="2026-01-01", chars=4000)
    for i in range(8):
        _atom(conn, f"a:{i}", _at_cos(0.80, axis=1 + i), who=f"x:w{i}",
              when=f"2026-0{1 + i % 8}-01", chars=4000)
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68,
                           budget_tokens=1200)
    assert rec["stop"] == "budget"
    for lens in ("queries", "briefing", "trajectory", "disconfirmation", "gaps"):
        assert "part 1" in " ".join(ss.lens_warnings(ss.scope(conn, rec, lens=lens), lens=lens))


# ── the tool ────────────────────────────────────────────────────────────────────
class _Mcp:
    """The two lines of FastMCP this module actually uses."""

    def __init__(self):
        self.tools = {}

    def tool(self, *a, **kw):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture()
def tool(conn):
    from mcp_server.sitting_tools import register_sitting_tools
    m = _Mcp()
    register_sitting_tools(m)
    return m.tools["sitting"]


@pytest.fixture()
def mapped(monkeypatch):
    """A stubbed transport for the lens MAP call. Returns the recorded calls."""
    from pipeline import llm_client

    class _Resp:
        text, model = "mapped", "fake-model"
        input_tokens, output_tokens, cost_usd = 100, 20, 0.01
        raw: dict = {}

    calls: list = []
    monkeypatch.setattr(llm_client, "preflight", lambda role: None)
    monkeypatch.setattr("pipeline.ingestion.utils.load_yaml_config", lambda: {})
    monkeypatch.delenv("OPYT_FRONTIER_BACKEND", raising=False)
    monkeypatch.setattr(llm_client, "call",
                        lambda role, **kw: calls.append(kw) or _Resp())
    return calls


def test_preview_builds_a_region_and_calls_no_model(conn, tool, monkeypatch):
    from pipeline import llm_client
    calls = []
    monkeypatch.setattr(llm_client, "call", lambda role, **kw: calls.append(1))
    _region(conn, n=12)
    res = tool(action="preview", atom_ids=["a:seed"])
    assert res["status"] == "preview" and calls == []
    assert res["atoms"] > 0 and res["sitting_id"]
    # PERSISTED: building is free and a row is what makes the region addressable for a later read.
    assert sst.get_sitting(conn, res["sitting_id"]) is not None


def test_read_is_reachable_without_a_preview_first(conn, tool, monkeypatch):
    """⚠️ PINS THAT PREVIEW IS NOT A GATE. Consent lives at the deposit, never at each use, so a
    free look that a read REQUIRED would be exactly the per-use permission step that rule forbids.
    This is the assertion that stops preview quietly becoming mandatory."""
    from pipeline import llm_client
    import json
    _region(conn, n=12)

    class _R:
        text = json.dumps({"consensus": "it moved",
                           "queries": [{"text": "gated deltanet", "target_sources": ["arxiv"],
                                        "rationale": "why", "atom_ids": ["a:seed"]}]})
        model, input_tokens, output_tokens, cost_usd = "m", 1, 1, 0.0
        raw = {}

    monkeypatch.setattr("pipeline.ingestion.utils.load_yaml_config",
                        lambda: {"frontier": {"backend": "api"}})
    monkeypatch.setattr(llm_client, "preflight", lambda role: None)
    monkeypatch.setattr(llm_client, "call", lambda role, **kw: _R())
    res = tool(action="read", atom_ids=["a:seed"])
    assert res["status"] == "ok" and res["emitted"] == 1
    assert res["scope"]["atoms"] > 0            # the scope comes back anyway, unasked


def test_an_unresolvable_phrase_is_an_error_not_an_empty_region(conn, tool):
    """"Nothing matched" and "this corner of the corpus is empty" are different facts. An empty
    region would render them identically and the caller would believe the second."""
    res = tool(action="preview", atom_ids=["a:nonexistent"])
    assert res["status"] == "error"


def test_a_bad_action_is_refused_before_anything_opens(conn, tool):
    assert tool(action="zoom")["status"] == "error"
    assert "preview" in tool(action="zoom")["reason"]


def test_render_returns_the_document_without_re_reading(conn, tool, monkeypatch):
    from pipeline import llm_client
    calls = []
    monkeypatch.setattr(llm_client, "call", lambda role, **kw: calls.append(1))
    rec = _region(conn, n=12)
    res = tool(action="render", sitting_id=rec["sitting_id"])
    assert res["status"] == "ok" and "## Context (chronological)" in res["document"]
    assert calls == []


def test_render_of_an_unknown_id_is_an_error_not_a_traceback(conn, tool):
    assert tool(action="render", sitting_id="deadbeef")["status"] == "error"


def test_the_tool_never_names_a_transport_or_calls_a_read_free(conn):
    """⚠️ CLIENT-AGNOSTIC. The shipped default is the metered API; a machine may opt into the
    subscription transport. A docstring saying "free" is false on one of those, and it is the
    sentence a host reads when deciding whether to spend the user's money without asking."""
    from mcp_server import sitting_tools
    m = _Mcp()
    sitting_tools.register_sitting_tools(m)
    doc = m.tools["sitting"].__doc__.lower()
    assert "claude-cli" not in doc and "subscription" not in doc
    assert "spends" in doc


# ── action="lens" (Job L) ─────────────────────────────────────────────────────
def test_a_lens_call_maps_the_region_and_hands_back_the_join(conn, tool, mapped):
    """AMENDED 2026-08-24: a lens used to call no model at all. It now MAPS each part of the chain
    and hands the host the per-part outputs plus the join rule — never the region's raw text, which
    is the compaction the whole change exists for."""
    rec = _region(conn, n=12)
    res = tool(action="lens", lens="briefing", sitting_id=rec["sitting_id"])
    assert res["status"] == "ok" and len(mapped) == 1
    assert res["instruction"] and "mapped" in res["document"]
    assert "## Context (chronological)" not in res["document"]


def test_a_second_lens_call_on_the_same_region_costs_nothing(conn, tool, mapped):
    """A part is frozen once built, so its map output never goes stale and there is no
    invalidation rule to get wrong."""
    rec = _region(conn, n=12)
    tool(action="lens", lens="briefing", sitting_id=rec["sitting_id"])
    tool(action="lens", lens="briefing", sitting_id=rec["sitting_id"])
    assert len(mapped) == 1


def test_a_lens_call_builds_and_reads_in_one_step_from_a_fresh_query(conn, tool, mapped):
    """Same convenience `read` has with a fresh `query` — no separate preview required first."""
    _region(conn, n=12)
    res = tool(action="lens", lens="trajectory", atom_ids=["a:seed"])
    assert res["status"] == "ok" and res["sitting_id"]


def test_sprouts_needs_no_sitting_id_query_or_atom_ids(conn, tool):
    _atom(conn, "a:orphan", ANCHOR)
    res = tool(action="lens", lens="sprouts")
    assert res["status"] == "ok" and res["atoms"] == 1


def test_a_lens_call_with_no_lens_name_is_an_error(conn, tool):
    _region(conn, n=12)
    res = tool(action="lens")
    assert res["status"] == "error" and "lens name" in res["reason"]


def test_a_lens_call_with_an_unknown_lens_is_an_error(conn, tool):
    rec = _region(conn, n=12)
    res = tool(action="lens", lens="not-a-real-lens", sitting_id=rec["sitting_id"])
    assert res["status"] == "error"


def test_a_sitting_scoped_lens_with_nothing_to_build_from_is_an_error(conn, tool):
    res = tool(action="lens", lens="briefing")
    assert res["status"] == "error" and "sitting_id" in res["reason"]


def test_a_lens_call_on_an_unknown_sitting_id_is_an_error_not_a_traceback(conn, tool):
    res = tool(action="lens", lens="briefing", sitting_id="deadbeef")
    assert res["status"] == "error"


def test_disconfirmation_and_gaps_accept_a_claim(conn, tool, mapped):
    rec = _region(conn, n=12)
    res = tool(action="lens", lens="gaps", sitting_id=rec["sitting_id"], claim="who bears the loss")
    assert res["status"] == "ok" and "who bears the loss" in res["instruction"]


def test_preview_computes_warnings_for_the_lens_the_caller_intends_to_use(conn, tool):
    """D6: `preview` warns on a degenerate lens/region pair, not only for `queries`."""
    _region(conn, n=12, span_days=2)
    res = tool(action="preview", atom_ids=["a:seed"], lens="trajectory")
    assert any("arc" in w for w in res["warnings"])
