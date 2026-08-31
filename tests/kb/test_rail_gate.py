"""The rail spend boundary's routability gate — `rail_runtime.models_unroutable`.

This is the wiring that makes `model_routing.preflight` a mechanism instead of a report: every
rail calls it after `load_rail_env()`, before any lock or spend. The contract under test is the
asymmetry, because it is the easiest thing to get backwards:

  dead (ok=False)  → blocks, with a reason naming the models
  fragile/unknown  → logs, NEVER blocks
  preflight raises → proceeds — an outage in the outage-detector is not an outage
"""
from __future__ import annotations

import unittest.mock as um

from pipeline import model_routing
from pipeline.kb import rail_runtime


def _rep(**over):
    rep = {"ok": True, "dead": [], "fragile": [], "unknown": [], "checked": {}, "deny": []}
    rep.update(over)
    return rep


def test_ok_report_proceeds_silently():
    with um.patch.object(model_routing, "preflight", return_value=_rep()):
        assert rail_runtime.models_unroutable("test-rail") is None


def test_dead_blocks_and_names_the_model():
    rep = _rep(ok=False, dead=[("qwen/qwen3-embedding-8b", "atom-KB embeddings")])
    with um.patch.object(model_routing, "preflight", return_value=rep):
        reason = rail_runtime.models_unroutable("test-rail")
    assert reason is not None
    assert "qwen/qwen3-embedding-8b" in reason and "deny-list" in reason


def test_fragile_and_unknown_never_block(monkeypatch):
    logged = []
    monkeypatch.setattr("pipeline.ingestion.utils.log", lambda m: logged.append(m))
    rep = _rep(fragile=[("a/b", "emb", ["nebius"])], unknown=[("c/d", "role:x")])
    with um.patch.object(model_routing, "preflight", return_value=rep):
        assert rail_runtime.models_unroutable("test-rail") is None
    assert any("FRAGILE" in m for m in logged)     # warned out loud, in the rail's log


def test_a_raising_preflight_proceeds():
    with um.patch.object(model_routing, "preflight", side_effect=RuntimeError("boom")):
        assert rail_runtime.models_unroutable("test-rail") is None
