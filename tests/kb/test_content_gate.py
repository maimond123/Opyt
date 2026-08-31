"""content_gate — the EXTRACTIVE keep/drop gate, proven fully offline (stubbed llm_client).

These lock the MECHANISM + the fail-safe contract: paragraph split, batch budget, VERBATIM
reassembly of kept units, whole-page reject when every unit drops, conservative default-keep
for any index the model omits, and — load-bearing — DEGRADE-TO-KEEP-ALL on preflight failure,
call error, or unparseable output (a gate outage may only ADD junk, never DELETE writing).

Judgment QUALITY (does it drop the right things?) is NOT proven anywhere. It was measured once
against a hand-built gold-set in a scratch directory that was never committed, so there is no
standing check on it — treat that as an open gap, not as coverage living elsewhere.
"""
from __future__ import annotations

import json
import re

import pytest

from pipeline import llm_client
from pipeline.kb import content_gate

# This module drives the REAL content gate (against a faked llm_client), so it opts out of
# tests/kb/conftest.py's autouse keep-all stub.
pytestmark = pytest.mark.real_gate


FM = "---\nsource: blog\n---\n\n"


def _fake_call(verdict_map: dict[int, str]):
    """An llm_client.call stub: for each batch, answer every index it was SHOWN (parsed from the
    `[i]` labels in the prompt) with verdict_map[i], defaulting to 'keep'. Mirrors json_object mode."""
    def call(role, *, system, user, **kw):
        shown = [int(m) for m in re.findall(r"\[(\d+)\]", user)]
        obj = {str(i): verdict_map.get(i, "keep") for i in shown}
        return type("R", (), {"text": json.dumps(obj)})()
    return call


def _partial_call(returned: dict[int, str]):
    """A stub that returns ONLY the given indices (omitting the rest) — to prove a missing index
    defaults to KEEP (conservative)."""
    def call(role, *, system, user, **kw):
        return type("R", (), {"text": json.dumps({str(k): v for k, v in returned.items()})})()
    return call


@pytest.fixture()
def ready(monkeypatch):
    """Role is available (preflight passes) — the normal path."""
    monkeypatch.setattr(llm_client, "preflight", lambda role: None)


# ── mechanism ──────────────────────────────────────────────────────────────────────

def test_split_units_collapses_blank_runs():
    # Multiple blank lines (incl. whitespace-only) collapse; empty segments drop; unit text is
    # kept VERBATIM apart from the delimiter newlines (indentation inside a unit is preserved).
    assert content_gate._split_units("A\n\nB\n \n\n\nC") == ["A", "B", "C"]
    assert content_gate._split_units("  - indented item\n\nnext") == ["  - indented item", "next"]


def test_batch_respects_unit_and_char_budget():
    units = [f"u{i}" for i in range(90)]                     # tiny units → count budget bites first
    batches = content_gate._batch(units)
    assert [len(b) for b in batches] == [40, 40, 10]
    big = ["x" * 20_000, "y" * 20_000]                       # each over the char budget → its own batch
    assert content_gate._batch(big) == [[0], [1]]


def test_mixed_keep_drop_reassembles_verbatim(ready, monkeypatch):
    monkeypatch.setattr(llm_client, "call", _fake_call({1: "drop", 3: "drop"}))
    md = FM + "Alpha para.\n\nBravo para.\n\nCharlie para.\n\nDelta para."
    v = content_gate.classify_page(md)
    assert v.keep == [True, False, True, False]
    # kept text = frontmatter + ONLY the kept units, exact author words, order preserved
    assert v.kept_text == FM + "Alpha para.\n\nCharlie para."
    assert "Bravo" not in v.kept_text and "Delta" not in v.kept_text


def test_all_drop_rejects_whole_page(ready, monkeypatch):
    monkeypatch.setattr(llm_client, "call", _fake_call({0: "drop", 1: "drop"}))
    v = content_gate.classify_page(FM + "Nav home about.\n\nSubscribe now.")
    assert v.kept_text is None                               # wrong-source → no atom
    assert content_gate.gate(FM + "Nav home about.\n\nSubscribe now.") is None


def test_missing_index_defaults_keep(ready, monkeypatch):
    # Model returns a verdict for index 0 only; index 1 is OMITTED → must stay KEEP (conservative).
    monkeypatch.setattr(llm_client, "call", _partial_call({0: "drop"}))
    v = content_gate.classify_page(FM + "Menu junk.\n\nReal insight worth keeping.")
    assert v.keep == [False, True]
    assert v.kept_text == FM + "Real insight worth keeping."


# ── fail-safe: degrade to KEEP-ALL, never silent-drop ───────────────────────────────

def test_degrade_keep_all_on_preflight_fail(monkeypatch):
    monkeypatch.setattr(llm_client, "preflight", lambda role: "OPENROUTER_API_KEY not set")
    md = FM + "Real content A.\n\nReal content B."
    v = content_gate.classify_page(md)
    assert v.degraded is True and v.n_calls == 0
    assert v.kept_text == md                                 # returned UNCHANGED, byte-for-byte
    assert content_gate.gate(md) == md


def test_degrade_keep_all_on_call_error(ready, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("provider 500")
    monkeypatch.setattr(llm_client, "call", boom)
    v = content_gate.classify_page(FM + "Keep me one.\n\nKeep me two.")
    assert v.degraded is True                                # per-batch degrade → all units survive
    assert v.keep == [True, True] and v.kept_text is not None


def test_unparseable_response_keeps_batch(ready, monkeypatch):
    monkeypatch.setattr(llm_client, "call",
                        lambda *a, **k: type("R", (), {"text": "sorry I cannot comply"})())
    v = content_gate.classify_page(FM + "Alpha.\n\nBravo.")
    assert v.degraded is True and v.keep == [True, True]     # no usable verdicts → keep, never drop


def test_regex_fallback_salvages_loose_json(ready, monkeypatch):
    # Not strict JSON, but the keep/drop pairs are recoverable → the batch is NOT wasted.
    monkeypatch.setattr(llm_client, "call",
                        lambda *a, **k: type("R", (), {"text": 'here: 0: "drop", 1: "keep" ok'})())
    v = content_gate.classify_page(FM + "Drop this nav.\n\nKeep this prose.")
    assert v.keep == [False, True]


def test_empty_body_is_noop(monkeypatch):
    monkeypatch.setattr(llm_client, "preflight", lambda role: None)
    v = content_gate.classify_page(FM)                       # frontmatter only, no body units
    assert v.units == [] and v.n_calls == 0 and v.kept_text == FM
