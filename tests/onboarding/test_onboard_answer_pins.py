"""
tests/onboarding/test_onboard_answer_pins.py

Every answer `onboard` gives, recorded word for word, with the side effects each call has.

The copy `onboard` returns IS its behavior — a host model renders it
to the user — so a change to one sentence, one key or the order of the keys is a
behavior change. Each scenario below fakes only boundaries OUTSIDE the unit (the session probes, the worker report, the queue, the browsers, the store's counts), never a name
inside `onboard_tools`, `onboard_state` or `pipeline/onboarding/`, so the same scenarios run unchanged across refactors.

The expected answers live in `onboard_answers.json`, recorded from the code, current bugs included
(`scenario start_bogus_refused`). Re-record only for an intended behavior change:
    OPYT_RECORD_ONBOARD_PINS=1 python -m pytest tests/onboarding/test_onboard_answer_pins.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_server import onboard_tools
from opyt_core import github_device
from pipeline.curation import clock
from pipeline.ingestion.utils import SyncAuthError
from pipeline.kb import schema
from pipeline.platforms import connections
from pipeline.platforms.browser import cookies, login
from pipeline.worker import job_queue, liveness
from pipeline.oracles import roster
from pipeline.oracles import ranking
from pipeline.topics import store as topics_store

GOLDEN = Path(__file__).with_name("onboard_answers.json")

_RUNNING = {"ok": True, "status": "running", "cause": None, "step": None, "detail": {}}
_DOWN = {"ok": False, "status": "disabled", "cause": "The helper is switched off.",
         "step": "Turn it back on in Login Items.", "detail": {"plist": "x.plist"}}
_UNSUPPORTED = {"ok": False, "status": "unsupported", "cause": "No helper here.", "step": None,
                "detail": {}}


def _world(monkeypatch, home: Path, *, x=False, sub=False, sub_ready=True, oracles=0, watch=False, read=(), platform="darwin",
           worker=_RUNNING, cached=None, login_start=None,
           profile=False, candidates=0, atoms=0, queued=True, skipped=False,
           github_app=True) -> list:
    """Set every boundary `onboard` reads from, and return the list each side effect appends to."""
    effects: list = []
    monkeypatch.setenv("OPYT_HOME", str(home))
    # The optional keys, which a `start` word's step reads: none, whoever runs the suite.
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    if not github_app:
        monkeypatch.setattr(github_device, "CLIENT_ID", "")
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(connections, "_connected_x", lambda: x)
    monkeypatch.setattr(connections, "_connected_substack", lambda: sub)
    monkeypatch.setattr(connections, "managed_substack_session_ready", lambda **_kw: sub_ready)
    monkeypatch.setattr(liveness, "repair_or_report",
                        lambda **_kw: effects.append("liveness.repair_or_report") or worker)
    monkeypatch.setattr(liveness, "repair_or_report_cached",
                        lambda **_kw: cached if cached is not None else worker)
    monkeypatch.setattr(job_queue, "request_now",
                        lambda rail, **_kw: effects.append(f"request_now:{rail}") or queued)

    def _start(url):
        effects.append(f"login.start:{url}")
        if isinstance(login_start, Exception):
            raise login_start
        return SimpleNamespace(label="Opyt login — Chrome")
    monkeypatch.setattr(login, "start", _start)
    monkeypatch.setattr(cookies, "opyt_session_backends",
                        lambda: [SimpleNamespace(key="chrome")] if profile else [])
    monkeypatch.setattr(ranking, "rank_candidates", lambda _conn, **_kw: [object()] * candidates)

    def _count_atoms(_conn, _source_type=None):
        if isinstance(atoms, Exception):
            raise atoms
        return atoms
    monkeypatch.setattr(schema, "count_atoms", _count_atoms)

    conn = schema.connect()
    try:
        for i in range(oracles):
            roster.upsert_oracle(conn, f"person:{i}", f"Person {i}")
        if watch:
            topics_store.new_topic(conn, name="Protein design", words=["protein design"],
                                   people=[], now="2026-10-11T00:00:00Z")
        for collector in read:
            clock.record_run(conn, collector, status="ok")
        conn.commit()
    finally:
        conn.close()
    if skipped:
        (home / "onboard_sources_skipped").touch()
    return effects


# name -> (world knobs, the calls made in order as (source, start))
SCENARIOS: dict[str, tuple[dict, list[tuple]]] = {
    # ── a `start` word no step takes ─────────────────────────────────────────────────────────
    # `openrouter` and `trial` are retired words.
    "start_openrouter_refused": ({"x": True, "read": ("x_lists",)}, [(None, "openrouter")]),
    "start_trial_refused": ({"x": True, "read": ("x_lists",)}, [(None, "trial")]),
    "start_bogus_refused": ({"x": True, "read": ("x_lists",), "candidates": 2},
                            [(None, "bogus")]),
    # ── off macOS ─────────────────────────────────────────────────────────────────────────────
    "linux_unsupported": ({"platform": "linux", "cached": _UNSUPPORTED}, [(None, None)]),
    "linux_with_a_running_worker": ({"platform": "linux", "skipped": True}, [(None, None)]),
    # ── sources ───────────────────────────────────────────────────────────────────────────────
    "sources_menu": ({}, [(None, None)]),
    "sources_menu_with_a_profile": ({"profile": True}, [(None, None)]),
    "sources_menu_worker_down": ({"worker": _DOWN}, [(None, None)]),
    "source_bogus": ({}, [("twitter", None)]),
    "source_research": ({}, [("research", None)]),
    "source_blog": ({}, [("blog", None)]),
    "source_skip": ({}, [("skip", None), (None, None)]),
    "connect_x_local": ({}, [("x", None)]),
    "connect_x_local_refused": ({"login_start": SyncAuthError("No Chrome on this machine.")},
                                [("x", None)]),
    "connect_substack_local": ({}, [("substack", None)]),
    "connect_while_done": ({"x": True, "read": ("x_lists",)}, [("substack", None)]),
    # ── the curation walk ─────────────────────────────────────────────────────────────────────
    "curation_x_connected": ({"x": True}, [(None, None)]),
    "curation_both_connected_worker_down": ({"x": True, "sub": True, "worker": _DOWN},
                                            [(None, None)]),
    "curation_substack_not_ready": ({"sub": True, "sub_ready": False}, [(None, None)]),
    "curation_queue_refused": ({"x": True, "queued": False}, [(None, None)]),
    # ── done ──────────────────────────────────────────────────────────────────────────────────
    "done_people_to_choose": ({"x": True, "read": ("x_lists",), "candidates": 7},
                              [(None, None)]),
    "done_found_nobody": ({"x": True, "sub": True, "read": ("x_lists", "substack_follows")},
                          [(None, None)]),
    "done_an_oracle_named": ({"oracles": 1}, [(None, None)]),
    "done_skipped_with_atoms": ({"skipped": True, "atoms": 3}, [(None, None)]),
    "done_skipped_atoms_unreadable": ({"skipped": True, "atoms": RuntimeError("locked")},
                                      [(None, None)]),
    "done_blind_store_with_a_watch": ({"watch": True}, [(None, None)]),
    "done_blind_store": ({"skipped": True}, [(None, None)]),
    # A build without GitHub's OAuth app (`github_device.CLIENT_ID` empty).
    "start_github_before_its_app_exists": ({"github_app": False}, [(None, "github")]),
    "done_blind_store_x_connected_elsewhere": ({"skipped": True, "sub": True,
                                                "read": ("substack_follows",)}, [(None, None)]),
}


def _run(name: str, monkeypatch, home: Path) -> dict:
    knobs, calls = SCENARIOS[name]
    effects = _world(monkeypatch, home, **knobs)
    m = SimpleNamespace(tools={})
    m.tool = lambda **kw: (lambda fn: m.tools.setdefault(fn.__name__, fn))
    onboard_tools.register_onboard_tools(m)
    answers = [m.tools["onboard"](source=source, start=start) for source, start in calls]
    markers = sorted(p.name for p in home.iterdir()
                     if p.name == "onboard_sources_skipped")
    return {"answers": answers, "effects": effects, "markers": markers}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_onboard_answers_are_unchanged(name, monkeypatch, tmp_path):
    got = json.loads(json.dumps(_run(name, monkeypatch, tmp_path)))
    if os.environ.get("OPYT_RECORD_ONBOARD_PINS"):
        recorded = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        recorded[name] = got
        GOLDEN.write_text(json.dumps(dict(sorted(recorded.items())), indent=1,
                                     ensure_ascii=False) + "\n")
        pytest.skip("recorded")
    expected = json.loads(GOLDEN.read_text())[name]
    # Key order is part of the answer (the handoff's order), so compare the
    # serialized form, not just the dicts.
    assert json.dumps(got, ensure_ascii=False) == json.dumps(expected, ensure_ascii=False)
