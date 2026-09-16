"""`oracle(action='progress')` — where a running pull has got to.

⚠️ THE CONSTANT THIS FILE EXERCISES IS NOT A BUDGET ON WORK. `PROGRESS_WAIT_SECONDS` bounds how
long one QUESTION waits before answering; no pull can see it and none is cut when it expires.
`2026-09-14-sixty-second-wall.md` §A put a clock on the work instead and was reverted — a clock
is useless against a unit of work that cannot be interrupted, and an archive should not be.
"""
from __future__ import annotations

import time

import pytest

from mcp_server import oracle_tools
from pipeline.kb import pull_runs


PICKS = [{"canonical_id": "c:a", "name": "Andrej"},
         {"canonical_id": "c:d", "name": "Dean"},
         {"canonical_id": "c:s", "name": "Soren"}]


@pytest.fixture()
def conn(kb_home):
    c = pull_runs.connect()
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    """These are about WHAT is reported, not how long it is willing to wait for it. The one test
    that is about the wait sets its own."""
    monkeypatch.setattr(oracle_tools, "PROGRESS_WAIT_SECONDS", 0.0)


def test_no_pull_at_all_is_an_answer_not_an_error(conn):
    """A host handed an error reasons its way into a retry — measured 2026-09-14."""
    out = oracle_tools._progress(conn)
    assert out["status"] == "none"
    assert "error" not in out


def test_every_writer_is_named_including_the_untouched_ones(conn):
    """⚠️ An Oracle missing from a report reads as an Oracle who FAILED. `waiting` is what says
    "not yet" instead of letting silence say "nothing found"."""
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    pull_runs.start_oracle(conn, run_id, "c:a")
    pull_runs.finish_oracle(conn, run_id, "c:a", {"atoms_added": 24})
    pull_runs.start_oracle(conn, run_id, "c:d")

    out = oracle_tools._progress(conn)

    assert [d["name"] for d in out["done"]] == ["Andrej"]
    assert [d["name"] for d in out["in_flight"]] == ["Dean"]
    assert [d["name"] for d in out["waiting"]] == ["Soren"]
    assert out["done"][0]["atoms_added"] == 24


def test_in_flight_is_a_list_because_breadth_holds_everyone_at_once(conn):
    """Breadth runs over EVERYONE before anyone is deepened. Reporting a single name would be
    false for the first phase of every pull."""
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    for pick in PICKS:
        pull_runs.start_oracle(conn, run_id, pick["canonical_id"])

    assert len(oracle_tools._progress(conn)["in_flight"]) == 3


def test_a_running_pull_reports_running_and_carries_no_report_yet(conn):
    pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    out = oracle_tools._progress(conn)
    assert out["status"] == "running"
    assert "report" not in out


def test_a_finished_pull_carries_the_whole_report(conn):
    """The same report the call that started the pull would have returned if it had waited —
    read off the record, because there is no in-memory copy left to offer."""
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS[:1], lookback={"x": "since 2024"})
    pull_runs.finish_oracle(conn, run_id, "c:a",
                            {"oracle_id": "c:a", "name": "Andrej", "atoms_added": 24,
                             "results": []})
    pull_runs.close_run(conn, run_id)

    out = oracle_tools._progress(conn)

    assert out["status"] == "complete"
    assert out["report"]["ingested_oracles"] == 1
    assert out["report"]["lookback"] == {"x": "since 2024"}
    assert "presentation" in out["report"]


def test_a_dead_pull_reads_as_stopped_with_who_it_reached(conn):
    """The wedged-job shape — started, never finished, nothing alive — had to be cleared out of
    the live store by hand on 2026-09-14. It must never read as "still going"."""
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    pull_runs.finish_oracle(conn, run_id, "c:a", {"atoms_added": 24})
    conn.execute("UPDATE pull_runs SET heartbeat_at = 0.0 WHERE run_id = ?", (run_id,))
    conn.commit()

    out = oracle_tools._progress(conn)

    assert out["status"] == "stopped"
    assert len(out["done"]) == 1 and len(out["waiting"]) == 2
    assert out["report"]["ingested_oracles"] == 1


def test_a_named_run_is_answerable_after_a_newer_one_started(conn):
    """`run_id` addresses a specific pull, so a conversation that started one can still ask about
    it rather than about whatever ran most recently."""
    first = pull_runs.open_run(conn, kind="ingest", picks=PICKS[:1])
    pull_runs.close_run(conn, first)
    time.sleep(0.01)
    pull_runs.open_run(conn, kind="ingest", picks=PICKS[1:])

    assert oracle_tools._progress(conn, run_id=first)["run_id"] == first
    assert oracle_tools._progress(conn)["run_id"] != first


def test_it_returns_early_the_moment_a_writer_lands(conn, monkeypatch):
    """The narration property: the call does not sit out its whole wait once it has news."""
    monkeypatch.setattr(oracle_tools, "PROGRESS_WAIT_SECONDS", 30.0)
    monkeypatch.setattr(oracle_tools, "_PROGRESS_POLL_SECONDS", 0.01)
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)

    import threading
    threading.Timer(0.15, lambda: pull_runs.finish_oracle(
        pull_runs.connect(), run_id, "c:a", {"atoms_added": 7})).start()

    began = time.time()
    out = oracle_tools._progress(conn)
    elapsed = time.time() - began

    assert elapsed < 5.0                       # nowhere near the 30s it was willing to wait
    assert [d["name"] for d in out["done"]] == ["Andrej"]


def test_it_gives_up_waiting_and_still_answers(conn, monkeypatch):
    """A quiet pull — one writer deep inside a blog archive — still gets an answer, because the
    caller has to be able to say something. Nothing is cut; the pull never learns this happened."""
    monkeypatch.setattr(oracle_tools, "PROGRESS_WAIT_SECONDS", 0.25)
    monkeypatch.setattr(oracle_tools, "_PROGRESS_POLL_SECONDS", 0.01)
    pull_runs.open_run(conn, kind="ingest", picks=PICKS)

    began = time.time()
    out = oracle_tools._progress(conn)

    assert 0.2 < time.time() - began < 5.0
    assert out["status"] == "running" and out["done"] == []


def test_the_wait_is_well_inside_the_measured_client_wall():
    """61.2 / 61.3 / 62.3s measured. The headroom covers a slow first read on a cold store, not a
    longer wait — this constant must not creep upward."""
    assert oracle_tools.PROGRESS_WAIT_SECONDS <= 45.0


def test_an_unknown_action_names_progress(kb_home):
    """A host that cannot discover the action cannot loop on it."""
    class _FakeMCP:
        def __init__(self):
            self.tools = {}

        def tool(self, *a, **kw):
            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn
            return deco

    mcp = _FakeMCP()
    oracle_tools.register_oracle_tools(mcp)
    assert "progress" in mcp.tools["oracle"](action="nonsense")["error"]


# ── what the host is told ──────────────────────────────────────────────────────
# §B's asymmetry: a truncated call destroys its return value, but the tool DESCRIPTION was
# delivered at tool-list time and is still in context when the call comes back as an error. So
# routing policy lives here and nowhere else — `server.py`: "`instructions` is optional on
# InitializeResult and a client MAY drop it".

def _oracle_doc():
    class _FakeMCP:
        def __init__(self):
            self.tools = {}

        def tool(self, *a, **kw):
            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn
            return deco

    mcp = _FakeMCP()
    oracle_tools.register_oracle_tools(mcp)
    return mcp.tools["oracle"].__doc__


def test_the_host_is_told_to_loop_until_the_pull_is_not_running():
    """Without this the model learns a pull finished only if it happens to ask again. §C's lesson
    is that instruction is not enforcement — but an instruction that was never written cannot
    even be followed."""
    doc = _oracle_doc()
    assert "LOOP ON `progress`" in doc
    assert "only way you" in doc


def test_every_status_the_host_can_receive_is_explained():
    doc = _oracle_doc()
    for status in ("running", "complete", "stopped"):
        assert f'"{status}"' in doc


def test_a_stopped_pull_is_never_to_be_called_a_failure():
    """The app was quit or the machine slept. `done` is real and durable; the rest has not
    happened. That is not a crash and must not be narrated as one."""
    doc = _oracle_doc()
    assert "Do NOT call it a failure" in doc
    assert "picks up exactly where this stopped" in doc


def test_a_quiet_wait_is_explicitly_not_a_timeout():
    """⚠️ THE FAILURE THIS WHOLE PLAN EXISTS FOR, one level up. A `progress` call that answers
    with no news is the system working — a writer deep inside a blog archive produces nothing for
    a minute at a time. Left unsaid, "returned with nothing" becomes "timed out" becomes a
    diagnosis, which is precisely the 2026-09-14 chain."""
    doc = _oracle_doc()
    assert 'NEVER say a pull "timed out"' in doc
    assert "neither sees that clock nor is cut by it" in doc


def test_the_error_rule_now_points_at_progress_instead_of_freshness():
    """`oracle_freshness` reports per-source staleness; it cannot say "3 of 7 done, Dean is
    next". Sending a host there after a lost report gave it the wrong question to ask."""
    doc = _oracle_doc()
    assert "Do not start another ingest" in doc
    assert "`oracle(action='progress')` — that is exactly what it is for" in doc
