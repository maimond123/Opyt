"""Every rail in the registry has a producer, and the chained ones fire only on real work.

The defect this file exists for: a rail with no producer. A rail that was only ever fired by a
session-open hook would become unreachable once that hook goes — complete, tested, and never
launched. Nothing errors, and a rail that never runs looks exactly like a rail that runs and
finds nothing.

`test_every_registry_rail_has_a_producer` is the one that cannot rot. Add a rail to `RAILS`
without wiring an activation event and it fails, naming the rail.

The chained producers are conditional on purpose. A rail that re-queues its successor on every
pass is an hourly timer wearing a successor's name, and it would put the whole chain on a
"fires whether or not there is work" footing.
"""
from __future__ import annotations

from pipeline.platforms import connections
from pipeline.worker.home import LOCAL_HOME_ID
from pipeline.worker import home as worker_home

import time
from pathlib import Path

import pytest

from pipeline.worker import job_queue
from pipeline.worker.job_queue import RailJobStore
from pipeline.worker.registry import RAILS
from pipeline.oracles import admit


class _MCP:
    """The two-line stand-in for FastMCP that every tool-registration test in this repo uses."""

    def __init__(self):
        self.tools = {}

    def tool(self, **kw):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture()
def queue(tmp_path, monkeypatch):
    """A real control database plus a recorder of what each call site queued."""
    db = tmp_path / "rail_jobs.db"
    monkeypatch.setenv(worker_home.WORKER_DB_ENV, str(db))
    monkeypatch.delenv(worker_home.WORKER_HOME_ID_ENV, raising=False)
    RailJobStore(db)
    queued: list[str] = []
    real = job_queue.request_now

    def recording(rail, **kw):
        queued.append(rail)
        return real(rail, **kw)

    monkeypatch.setattr(job_queue, "request_now", recording)
    return queued


# ── the coverage invariant ──────────────────────────────────────────────────────
#
# One entry per rail: the event that gives it work, and where that event is recorded. Kept as
# data rather than prose so the assertion below can be exhaustive over `RAILS`.
PRODUCERS = {
    "scrapers":          "an ask being recorded (pipeline/oracles/admit.record, "
                         "called by `oracle(action='confirm')`, `oracle(action='ingest')` and "
                         "`add_oracle(confirm=True)`); a check on an Oracle's newest posts "
                         "(pipeline/oracles/checks.request_check); AND onboard seeing a "
                         "connected platform whose collectors are due or whose saved-list walk "
                         "never began (onboarding.kickoff._queue_curation_walk, _wake_scrapers). "
                         "Inside the process, a collector that lands a unit wakes scouting "
                         "(LoopProcess wakes idle consumers). AND standing queries added "
                         "(frontier.chain.queries_added): a topic's watches, written by "
                         "`watch` create or edit (topics.store.add_watches). AND the host keeping "
                         "finds (frontier.chain.finds_kept): their admission is the queue inside "
                         "the same process. AND a durable share grant "
                         "(sharing.share: push_gate.owe, then the wake)",
    "line":              "this install sharing: every publishing pass enrolls the home "
                         "(sharing.publishing.Publications.record, job_queue.enroll), which "
                         "creates the row once and never moves it, so an install that never "
                         "shared has no row. AND the Google connection finishing "
                         "(sharing.link.run), which wakes it at once. Stopped by the worker's "
                         "restart, it asks for itself (mcp_server.line.run)",
}


def test_every_registry_rail_has_a_producer():
    """A rail the worker can launch but nothing can activate never runs. The worker seeds no
    recurring jobs by design (ruling F2), so the registry and this map must stay in step."""
    assert set(RAILS) == set(PRODUCERS), (
        "a rail in RAILS with no activation event is unreachable; one in PRODUCERS with no "
        "registry entry is a row no worker dispatches"
    )


# ── a watch wakes the searches ──────────────────────────────────────────────────

def _watch_tool():
    """The `watch` tool, registered on the stand-in."""
    from mcp_server import topic_tools

    m = _MCP()
    topic_tools.register_topic_tools(m)
    return m.tools["watch"]


def _watch(watch) -> dict:
    return watch(action="create", name="Agent payments",
                 watches=[{"name": "x402", "sources": ["github"]}])


def test_a_topics_watch_schedules_the_rail_that_re_runs_it(queue, kb_home):
    """A watch must not stop at its first pull. Creating one writes a standing query and runs one
    scoped pull in-process; without a scheduled row the query is active and `rail_jobs.db` is
    empty, so nothing would ever run it again — a watch frozen at its first pull reads exactly
    like one finding nothing new.

    The row wakes the scrapers process, whose searches source owes the new pairs their first pull
    and, through their cadence, every repeat after it.
    """
    out = _watch(_watch_tool())

    assert [w["name"] for w in out["watches"]] == ["x402"] and queue == ["scrapers"]
    job = RailJobStore().get(LOCAL_HOME_ID, "scrapers")
    assert job is not None, (
        "the event that starts a pull must be the event that schedules its repeat; without a row "
        "this watch is pulled once and never again"
    )
    assert job.due_at <= time.time(), "the row is the first pull — it must be due now"


def test_the_repeat_never_postpones_work_another_producer_made_due(queue, kb_home):
    """`activate` keeps `MIN(due_at)`, and that is what makes the door safe to call on a store
    where another producer has already queued the scrapers for real, waiting work."""
    job_queue.request_now("scrapers")

    _watch(_watch_tool())

    assert RailJobStore().get(LOCAL_HOME_ID, "scrapers").due_at <= time.time()


# ── the class of chaining bugs ──────────────────────────────────────────────────
def test_no_rail_schedules_its_successor_from_inside_a_main():
    """The whole class, not single instances. A `request_now` inside a `main()` body is a
    side-effect only the CLI child performs, and any in-process caller of the rail skips it, so
    the successor is silently never queued. This asserts over the whole tree for that reason.

    Put the chaining in the `run_*` body instead: it is the one door every caller goes through.
    """
    import ast
    import subprocess

    root = Path(__file__).resolve().parents[2]
    tracked = subprocess.run(["git", "ls-files", "*.py"], cwd=root,
                             capture_output=True, text=True, check=True).stdout.split()
    offenders = []
    for rel in tracked:
        if rel.startswith("tests/") or not (root / rel).exists():
            continue
        tree = ast.parse((root / rel).read_text(), filename=rel)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == "main"):
                continue
            for inner in ast.walk(node):
                called = inner.func if isinstance(inner, ast.Call) else None
                name = (called.attr if isinstance(called, ast.Attribute)
                        else called.id if isinstance(called, ast.Name) else None)
                if name == "request_now":
                    offenders.append(f"{rel}:{inner.lineno}")
    assert offenders == [], (
        f"a rail queues its successor from inside main(): {offenders}. Only the `--once` child "
        f"reaches that code; every in-process caller of the rail's `run_*` body loses the "
        f"successor silently. Move the chaining into `run_*`."
    )


# ── the hosted context ──────────────────────────────────────────────────────────
#
# The question these tests answer offline: does a hosted child queue the same set of rails as a
# local one? The producers live in `onboard_tools` / `curation_catchup` / the watch tool, none of
# which branch on hosted vs local for whether to queue. Asserted here rather than assumed.
@pytest.fixture()
def hosted(tmp_path, monkeypatch):
    """A hosted home's environment: an already-validated subject as the home id, and the
    operator's ONE shared control database inherited."""
    subject = "104729183746152938471"
    shared = tmp_path / "var" / "rail_jobs.db"
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "homes" / subject))
    monkeypatch.setenv(worker_home.WORKER_HOME_ID_ENV, subject)
    monkeypatch.setenv(worker_home.WORKER_DB_ENV, str(shared))
    return subject, shared


def test_a_hosted_child_says_something_will_act_on_what_it_queues(hosted):
    """`queue_is_shared` keys on the home id, not on whether this platform can install a
    LaunchAgent. Reading the second would tell every remote user that nothing would act on the
    answer they had just given, on a box where a systemd unit claims exactly those rows."""
    assert worker_home.queue_is_shared() is True


def test_a_hosted_watch_schedules_its_repeat_the_same_way(hosted):
    """A hosted watch: same producer, same shared database, same subject."""
    subject, shared = hosted

    _watch(_watch_tool())

    job = RailJobStore(shared).get(subject, "scrapers")
    assert job is not None and job.due_at <= time.time()


# ── the first-run sequence, end to end ──────────────────────────────────────────
#
# The failure these exist for: Oracles confirmed, the worker installed and running, and
# `rail_jobs.db` holding no `oracle_refresh` row. Nothing errors, yet the host tells the user
# "it fills in on its own" and that is false.
#
# `test_every_registry_rail_has_a_producer` cannot catch it, because `oracle_refresh` has a
# producer on paper. A producer whose precondition can never be true passes that check:
# onboarding ends before the roster is picked (`onboard` answers `next_tool='oracle'`), so a
# producer that waits for confirmed Oracles never fires there. Only a test that walks the real
# order can catch that, which is what these two do.
def test_confirming_an_oracle_queues_the_scrapers(queue, kb_home, monkeypatch):
    """Onboarding ends with the roster still empty, so `confirm` is the producer that fires."""
    from pipeline.kb import schema

    monkeypatch.setattr(connections, "_connected_x", lambda: True)
    conn = schema.connect(kb_home / "opyt.db")
    schema.upsert_entity(conn, "x:user:1", name="Carol", profile={"handle": "carol"})

    out = admit.confirm_picks(conn, canonical_ids=["x:user:1"], add_handles=None)
    conn.close()

    assert out["confirmed"] and queue == ["scrapers"]
    job = RailJobStore().get(LOCAL_HOME_ID, "scrapers")
    assert job is not None and job.due_at <= time.time(), (
        "the roster exists and the user asked — a promise of background work with no row "
        "behind it is the 2026-09-14 defect verbatim"
    )


def test_a_confirmed_oracle_is_kept_current_with_nothing_asked(queue, kb_home, monkeypatch):
    """OPYT never asks: every confirmed Oracle is pulled and kept current, and to stop, the
    user removes the Oracle. So `confirm` wakes the scrapers, and once the first
    pull ages past its refresh time, new posts are owed with no answer on record."""
    from datetime import timedelta

    from pipeline.kb import schema
    from pipeline.oracles.sources.timeline.sources import NewPosts
    from pipeline.timeparse import utc_now

    monkeypatch.setattr(connections, "_connected_x", lambda: True)
    conn = schema.connect(kb_home / "opyt.db")
    schema.upsert_entity(conn, "x:user:1", name="Carol", profile={"handle": "carol"})

    out = admit.confirm_picks(conn, canonical_ids=["x:user:1"], add_handles=None)
    assert out["confirmed"] and queue == ["scrapers"]

    stale = (utc_now() - timedelta(days=30)).isoformat()
    conn.execute("UPDATE oracle_sources SET last_pulled_at=?", (stale,))
    conn.commit()
    assert NewPosts().next_unit(conn, utc_now()) is not None
    conn.close()
