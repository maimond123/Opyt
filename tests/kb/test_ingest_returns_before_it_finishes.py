"""`oracle(action='ingest')` starts a pull and returns — it does not wait for one.

⚠️ THE MEASUREMENT THIS EXISTS FOR (2026-09-14, session 5e49b025). Nine ingest calls, six cut off
at the client's 60-second wall. The work was fine — atoms landed on every one of them. What died
was the RETURN VALUE, where `presentation`, `partial_note` and the whole §G vocabulary live, so
the model received one fact — `Error: Request timed out` — and used it as a universal explanation
for three unrelated things, none of which was a timeout. Four retries fired off the back of it,
each spending the X meter the background pass was draining.

A call that returns in under a second is never mute. That is the whole ruling
(`docs/plans/2026-09-14-no-call-waits-for-a-pull.md`), and these are its tests.

NOTE ON THE SEAM. `tests/conftest.py` makes `_spawn` run inline for every test in the suite, so a
real pull thread can never outlive a test and hit the network. The tests HERE are about the
announcement, so they replace that seam with a spawn that starts nothing.
"""
from __future__ import annotations

import pytest

from mcp_server import oracle_tools
from pipeline.kb import oracles, pull_runs, schema


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


@pytest.fixture()
def unstarted(monkeypatch):
    """A pull that is announced and never runs — which is what the caller sees in production,
    since the thread has done nothing measurable by the time the call returns."""
    started: list = []
    monkeypatch.setattr(oracle_tools, "_spawn", lambda target: started.append(target))
    return started


def _oracle(conn, cid, name):
    schema.upsert_entity(conn, cid, name=name, profile={"handle": name.lower()})
    schema.upsert_oracle(conn, cid, name=name)
    return {"canonical_id": cid, "name": name}


def _ingest(conn, picks, monkeypatch, **kw):
    from pipeline.ingestion import x_graphql
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: True)
    monkeypatch.setattr(oracles, "confirmed_oracles", lambda c: picks)
    from pipeline.kb import embed
    monkeypatch.setattr(embed, "get_kb_embedder", lambda: object())
    return oracle_tools._ingest(conn, canonical_ids=[p["canonical_id"] for p in picks],
                                force=False, x_lookback=kw.get("x_lookback", "1yr"),
                                web_lookback=None, scholar_lookback=None, scholar_topics=None)


def test_the_call_returns_without_the_pull_having_run(conn, unstarted, monkeypatch):
    """THE RULING, as one assertion. The pull is started; the call does not wait for it."""
    picks = [_oracle(conn, "x:user:1", "A"), _oracle(conn, "x:user:2", "B")]

    out = _ingest(conn, picks, monkeypatch)

    assert out["status"] == "started"
    assert len(unstarted) == 1                  # spawned, never joined
    assert pull_runs.get_run(conn, out["run_id"]).finished_at is None


def test_the_announcement_names_everyone_who_was_picked(conn, unstarted, monkeypatch):
    """An Oracle absent from what the host is handed reads as an Oracle who was not included."""
    picks = [_oracle(conn, "x:user:1", "A"), _oracle(conn, "x:user:2", "B")]

    out = _ingest(conn, picks, monkeypatch)

    assert {p["canonical_id"] for p in out["picks"]} == {"x:user:1", "x:user:2"}
    assert "A" in out["message"] and "B" in out["message"]


def test_the_announcement_names_the_next_call_to_make(conn, unstarted, monkeypatch):
    """⚠️ TWO CHANNELS FOR ONE INSTRUCTION. §C's lesson is that a tool description is instruction
    and not enforcement — a model read "do not call it again" and reasoned its way into a smaller
    retry anyway. The description tells the host to loop; this tells it, in the value it is
    holding, what to call next."""
    out = _ingest(conn, [_oracle(conn, "x:user:1", "A")], monkeypatch)

    assert out["next_call"] == "oracle(action='progress')"
    assert "until `status` is no longer 'running'" in out["host_note"]
    assert "did NOT wait" in out["host_note"]


def test_the_announcement_promises_no_completion_time(conn, unstarted, monkeypatch):
    """We do not know one. A writer with a large archive takes as long as their archive, and
    "about ten minutes" is a broken promise the first time somebody tracks a prolific one."""
    out = _ingest(conn, [_oracle(conn, "x:user:1", "A")], monkeypatch)

    said = out["message"].lower()
    for guess in ("minutes,", "10 min", "about ten", "should take", "in around", "eta"):
        assert guess not in said
    assert "a few minutes" in said              # a shape, not a number


def test_the_announcement_says_nothing_about_what_will_be_found(conn, unstarted, monkeypatch):
    """Explaining an absence nobody has measured yet is the second-order damage this plan was
    written to stop.

    Scoped to `message` — the sentence meant for a PERSON. `host_note` is addressed to the model
    and is allowed to use these words precisely because its job is to forbid them; the two must
    not be checked as one string, or the instruction would have to avoid naming what it bans."""
    out = _ingest(conn, [_oracle(conn, "x:user:1", "A")], monkeypatch)

    said = out["message"].lower()
    for word in ("timed out", "timeout", "rate limit", "failed", "error", "may be missing"):
        assert word not in said


def test_the_host_is_told_not_to_report_a_failure(conn, unstarted, monkeypatch):
    """The other half. Nothing HAS failed — the call returned exactly as designed — and a host
    with no instruction fills that silence with "timed out", measured three times on
    2026-09-14."""
    out = _ingest(conn, [_oracle(conn, "x:user:1", "A")], monkeypatch)

    assert "not tell the user anything has failed" in out["host_note"]
    assert "Do NOT start another ingest" in out["host_note"]


def test_the_windows_are_stated_up_front(conn, unstarted, monkeypatch):
    """The cost-consent surface has to survive the call no longer reporting an outcome — what was
    asked for is knowable the moment it is asked."""
    out = _ingest(conn, [_oracle(conn, "x:user:1", "A")], monkeypatch)

    assert out["lookback"]["x"].startswith("since ")


def test_a_run_that_finished_before_the_call_returned_reports_instead(conn, monkeypatch):
    """Not a special case — the same question asked of the same record, where the answer happens
    to be complete. It is also what keeps every existing ingest assertion meaningful under the
    inline test seam."""
    calls: list[str] = []

    def _fake(conn_, embedder, oracle, **kw):
        calls.append(oracle["canonical_id"])
        return {"oracle_id": oracle["canonical_id"], "atoms_added": 3, "results": []}

    monkeypatch.setattr(oracles, "_ingest_oracle", _fake)

    out = _ingest(conn, [_oracle(conn, "x:user:1", "A")], monkeypatch)

    assert out.get("status") != "started"
    assert out["ingested_oracles"] == 1 and "presentation" in out
    assert calls


def test_a_pull_that_dies_leaves_the_run_open_rather_than_marking_it_done(conn, monkeypatch):
    """⚠️ A `finally: close_run()` WOULD MARK UNFINISHED WORK DONE — the one thing the fail-safe
    invariant forbids. A crash must read as "4 of 7 are in", which is exactly true, and a fresh
    ingest then picks up where it stopped."""
    def _explode(conn_, embedder, oracle, **kw):
        raise RuntimeError("x.com hung up")

    monkeypatch.setattr(oracles, "_ingest_oracle", _explode)

    out = _ingest(conn, [_oracle(conn, "x:user:1", "A")], monkeypatch)

    run = pull_runs.get_run(conn, out["run_id"])
    assert run.finished_at is None
    assert pull_runs.run_status(run, alive=False) == "stopped"


def test_the_thread_opens_its_own_connection(conn, monkeypatch):
    """SQLite connections are thread-bound and the caller's belongs to a request that has already
    returned. Handing it down would work in-process and corrupt the moment it is really threaded."""
    seen: list = []

    def _fake(conn_, embedder, oracle, **kw):
        seen.append(conn_)
        return {"oracle_id": oracle["canonical_id"], "atoms_added": 1, "results": []}

    monkeypatch.setattr(oracles, "_ingest_oracle", _fake)

    _ingest(conn, [_oracle(conn, "x:user:1", "A")], monkeypatch)

    assert seen and seen[0] is not conn


def test_a_real_thread_is_a_daemon_and_is_not_joined(monkeypatch):
    """The seam under test, not the seam the suite installs. A non-daemon thread would hold the
    MCP server open at shutdown for as long as an archive takes."""
    import threading
    monkeypatch.undo()
    done = threading.Event()
    t = oracle_tools._spawn(done.set)
    assert isinstance(t, threading.Thread) and t.daemon
    assert done.wait(2.0)
