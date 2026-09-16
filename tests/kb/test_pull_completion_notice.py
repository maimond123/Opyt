"""The rider — how a user who walked away ever learns their pull landed.

⚠️ OPYT CANNOT PUSH, AND THAT IS THE PREMISE. An MCP server speaks when it is called and never
otherwise. `push_catchup` publishes the SERVED copy of the knowledge base and notifies nobody;
`needs_attention` is a flag that needs a tool call to carry it. That is the whole inventory,
checked 2026-09-14. So a pull that finished while the conversation was closed rides back on
whatever the user does next — including `search`, which has nothing to do with Oracles.
"""
from __future__ import annotations

import pytest

from mcp_server import atoms_tools, oracle_tools
from pipeline.kb import pull_runs


PICKS = [{"canonical_id": "c:a", "name": "Andrej"},
         {"canonical_id": "c:d", "name": "Dean"},
         {"canonical_id": "c:s", "name": "Soren"}]


@pytest.fixture()
def conn(kb_home):
    c = pull_runs.connect()
    yield c
    c.close()


def _finished(conn, *, reached=3):
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    for pick in PICKS[:reached]:
        pull_runs.finish_oracle(conn, run_id, pick["canonical_id"], {"atoms_added": 10})
    pull_runs.close_run(conn, run_id)
    return run_id


def test_a_running_pull_is_not_announced(conn):
    """There is nothing to tell them yet, and a notice would be a completion claim."""
    pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    assert pull_runs.completion_notice(conn) is None


def test_a_finished_pull_is_announced_with_counts(conn):
    _finished(conn)
    notice = pull_runs.completion_notice(conn)
    assert notice["status"] == "complete"
    assert notice["writers"] == 3 and notice["atoms"] == 30


def test_it_fires_exactly_once(conn):
    """Un-stamped it would ride on every call forever and become furniture — the same argument
    `screen` makes for omitting `omitted: 0`."""
    _finished(conn)
    assert pull_runs.completion_notice(conn) is not None
    assert pull_runs.completion_notice(conn) is None


def test_a_stopped_pull_names_who_is_still_owed_and_is_never_called_a_failure(conn):
    """The likely cause is that the user quit the app. What landed is durable, so the honest
    offer is to finish it."""
    _finished(conn, reached=1)
    notice = pull_runs.completion_notice(conn)

    assert notice["status"] == "stopped"
    assert notice["unreached"] == ["Dean", "Soren"]
    assert "picks up where this stopped" in notice["message"]
    assert "do NOT call it a failure" in notice["message"]


def test_a_completion_is_told_once_briefly_not_made_the_subject(conn):
    """They asked for something else. A pull landing is a clause, not a turn."""
    _finished(conn)
    assert "Do not make it the subject of the turn" in pull_runs.completion_notice(conn)["message"]


def test_it_carries_the_run_id_rather_than_a_presentation(conn):
    """Counts and names here; the full report is one `progress` call away. Building prose in the
    row store would put a second opinion about an ingest outcome where the rows live."""
    run_id = _finished(conn)
    notice = pull_runs.completion_notice(conn)
    assert notice["run_id"] == run_id
    assert "presentation" not in notice


# ── the four surfaces that carry it ────────────────────────────────────────────

def test_search_carries_it(conn, kb_home, monkeypatch):
    """A notice about Oracles on a tool that has nothing to do with them — because it is the
    thing a returning user is most likely to reach for."""
    _finished(conn)
    out: dict = {"hits": [], "notices": []}
    atoms_tools._attach_pull_notice(out)

    codes = [n["code"] for n in out["notices"]]
    assert "pull_finished" in codes


def test_the_screen_surface_carries_it(conn, kb_home):
    _finished(conn)
    out = oracle_tools._screen(conn, floor=15, limit=40)
    assert out["pull_finished"]["writers"] == 3


def test_progress_spends_the_notice_so_it_is_not_told_twice(conn, monkeypatch):
    """A host just handed the whole report will tell the user. Letting the rider announce the
    same completion again on their next `search` would have OPYT report one event twice."""
    monkeypatch.setattr(oracle_tools, "PROGRESS_WAIT_SECONDS", 0.0)
    _finished(conn)

    assert oracle_tools._progress(conn)["status"] == "complete"
    assert pull_runs.completion_notice(conn) is None


def test_an_unreadable_store_contributes_nothing_and_breaks_nothing(monkeypatch):
    """A courtesy on somebody else's answer must never be able to break it."""
    from pipeline.kb import pull_runs as pr
    monkeypatch.setattr(pr, "completion_notice", lambda *a, **k: 1 / 0)
    out: dict = {"hits": []}
    atoms_tools._attach_pull_notice(out)
    assert out == {"hits": []}
