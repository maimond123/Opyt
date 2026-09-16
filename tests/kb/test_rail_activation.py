"""Every rail in the registry has a producer, and the chained ones fire only on real work.

THE DEFECT THIS FILE EXISTS FOR, found 2026-09-07 while deleting the session-open spawners:
four of the eight rails had no producer at all. They ran only because `mcp_server/server.py`
fired all eight on session open, so deleting that block would have left `frontier_execute`,
`frontier_admit`, `candidate_probe`, and the recurring half of `curation_catchup` unreachable —
complete, tested, and never launched. That is the same shape as the hot-feed dead drop the
`retired-hot-feed-vault-drop` guard records: nothing errors, and a rail that never runs looks
exactly like a rail that runs and finds nothing.

`test_every_registry_rail_has_a_producer` is the one that cannot rot. Add a rail to `RAILS`
without wiring an activation event and it fails, naming the rail.

The chained producers are conditional on purpose. A rail that re-queues its successor on every
pass is an hourly timer wearing a successor's name, and it would put the whole chain back on the
"fires whether or not there is work" footing this migration removed.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from pipeline.kb import rail_jobs
from pipeline.kb.rail_jobs import LOCAL_HOME_ID, RailJobStore
from pipeline.kb.rail_worker import RAILS


class _MCP:
    """The two-line stand-in for FastMCP that every tool-registration test in this repo uses."""

    def __init__(self):
        self.tools = {}

    def tool(self):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture()
def queue(tmp_path, monkeypatch):
    """A real control database plus a recorder of what each call site queued."""
    db = tmp_path / "rail_jobs.db"
    monkeypatch.setenv(rail_jobs.WORKER_DB_ENV, str(db))
    monkeypatch.delenv(rail_jobs.WORKER_HOME_ID_ENV, raising=False)
    RailJobStore(db)
    queued: list[str] = []
    real = rail_jobs.request_now

    def recording(rail, **kw):
        queued.append(rail)
        return real(rail, **kw)

    monkeypatch.setattr(rail_jobs, "request_now", recording)
    return queued


# ── the coverage invariant ──────────────────────────────────────────────────────
#
# One entry per rail: the event that gives it work, and where that event is recorded. Kept as
# data rather than prose so the assertion below can be exhaustive over `RAILS`.
PRODUCERS = {
    "bookmark_catchup":  "onboard backlog consent (mcp_server/onboard_tools._queue_backlog), OR "
                         "a consented platform connected later, queued beside the Arm B import "
                         "that same call",
    "substack_saved_catchup": "onboard backlog consent (onboard_tools._queue_backlog), the same "
                              "answer as bookmark_catchup and a marker of its own — OR, and for "
                              "Substack this is the USUAL door, the later `source='substack'` "
                              "connect, since consent is answered while only X is live",
    "oracle_refresh":    "an Oracle being minted, with refresh consent already recorded "
                         "(pipeline/kb/oracles.activate_refresh, called from both `confirm` and "
                         "the `add_oracle` engine); ALSO onboard consent against a roster that "
                         "already exists (onboard_tools._queue_refresh), the re-consent door",
    "curation_catchup":  "onboard's curation phase, which grants curation consent",
    "push_catchup":      "a durable share grant (mcp_server/share_tools)",
    "sitting_scheduler": "a sitting claim written by read_lens (mcp_server/sitting_tools._dispose)",
    "frontier_execute":  "a sitting read that added standing queries "
                         "(sitting_scheduler.run_sitting_scheduler) — AND, for the hand-added "
                         "watch that goes through no read at all, the add itself "
                         "(sitting_tools._schedule_the_repeat)",
    "frontier_admit":    "a stage-2 pass that staged candidates "
                         "(frontier_execute.run_frontier_execute — the rail child AND the "
                         "watchlist add's first pull)",
    "candidate_probe":   "a curation pass in which a collector ran "
                         "(curation_catchup.run_curation_catchup — the rail child AND "
                         "`onboard`'s in-process Arm A walk)",
}


def test_every_registry_rail_has_a_producer():
    """A rail the worker can launch but nothing can activate never runs. The worker seeds no
    recurring jobs by design (ruling F2), so the registry and this map must stay in step."""
    assert set(RAILS) == set(PRODUCERS), (
        "a rail in RAILS with no activation event is unreachable; one in PRODUCERS with no "
        "registry entry is a row no worker dispatches"
    )


# ── the chained producers ───────────────────────────────────────────────────────
#
# ⚠️ EVERY ONE OF THESE CALLS THE RAIL'S PUBLIC `run_*`, NEVER ITS `main()`, and that is the
# contract as much as the queue assertions are. Until 2026-09-16 all three chains lived inside a
# `main()` body — reachable only from `python -m pipeline.kb.<rail> --once`, the worker's child —
# while the MCP side called the `run_*` body directly and silently lost the successor. Stubbing
# the inner `_run` tests the door BOTH callers share; stubbing `run_*` and calling `main()`, which
# is what these tests used to do, is precisely the shape that kept a green suite over a broken
# fresh install.
def _pass(module, result, monkeypatch) -> None:
    """Stub the rail's pass so only its chaining is under test."""
    monkeypatch.setattr(module, "_run", lambda *a, **kw: result)


def test_staging_candidates_makes_stage_three_due(queue, monkeypatch):
    from pipeline.kb import frontier_execute as fe
    _pass(fe, {"status": "ok", "candidates_new": 3}, monkeypatch)
    fe.run_frontier_execute(registry={})
    assert queue == ["frontier_admit"]
    assert RailJobStore().get(LOCAL_HOME_ID, "frontier_admit") is not None


def test_a_stage_two_pass_that_staged_nothing_queues_nothing(queue, monkeypatch):
    """The expensive half of Frontier is stage 3. An unconditional re-queue would run it hourly
    against an empty `new` queue forever."""
    from pipeline.kb import frontier_execute as fe
    _pass(fe, {"status": "ok", "candidates_new": 0}, monkeypatch)
    fe.run_frontier_execute(registry={})
    assert queue == []


def test_a_dry_run_stages_nothing_so_it_queues_nothing(queue, monkeypatch):
    from pipeline.kb import frontier_execute as fe
    _pass(fe, {"status": "dry-run", "candidates_new": 9}, monkeypatch)
    fe.run_frontier_execute(registry={}, dry_run=True)
    assert queue == []


def test_the_add_time_first_pull_makes_admit_due_like_the_rail_child(queue, monkeypatch):
    """§2b of the 2026-09-16 handoff. `sitting_tools._start_first_pull` calls
    `run_frontier_execute(query_ids=...)` on a thread when a watch is added — the scoped first
    pull. With the chain in `main()` those candidates landed as `new` with nothing queued to
    admit them, and recovered only once the whole chain happened to cycle."""
    from pipeline.kb import frontier_execute as fe
    _pass(fe, {"status": "ok", "candidates_new": 2}, monkeypatch)
    fe.run_frontier_execute(registry={}, query_ids={"q1"})
    assert queue == ["frontier_admit"]


def test_a_collector_that_pulled_makes_the_probe_due(queue, monkeypatch):
    from pipeline.kb import curation_catchup as cc
    _pass(cc, {"status": "ok", "ran": {"x_lists": {}}}, monkeypatch)
    cc.run_curation_catchup()
    assert queue == ["candidate_probe"]


def test_a_curation_pass_inside_every_floor_queues_nothing(queue, monkeypatch):
    from pipeline.kb import curation_catchup as cc
    _pass(cc, {"status": "ok", "ran": {}, "skipped_within_floor": ["x_lists"]}, monkeypatch)
    cc.run_curation_catchup()
    assert queue == []


def test_a_lease_lost_after_a_collector_ran_still_makes_the_probe_due(queue, monkeypatch):
    """A reclaimed lease stops the NEXT collector; it does not un-mint the candidates the ones
    that already walked wrote. `ran` is on that return for exactly this reason."""
    from pipeline.kb import curation_catchup as cc
    _pass(cc, {"status": "lease_lost", "ran": {"x_lists": {}}}, monkeypatch)
    cc.run_curation_catchup()
    assert queue == ["candidate_probe"]


def test_the_in_process_curation_walk_makes_the_probe_due_like_the_rail_child(queue, monkeypatch):
    """§2 of the 2026-09-16 handoff, and the one that was MEASURED failing. `onboard`'s Arm A
    calls `run_curation_catchup(force=True, platforms=...)` directly, so a chain living in
    `main()` never fired for it: a clean install had 1,018 screen candidates, no
    `candidate_probe` row and no `candidate_probe.log` — permanently, not slowly."""
    from mcp_server import onboard_tools
    from pipeline.kb import curation_catchup as cc
    _pass(cc, {"status": "ok", "ran": {"x_lists": {}}}, monkeypatch)
    onboard_tools._run_curation({"x"})
    assert queue == ["candidate_probe"]


def test_a_read_that_added_queries_makes_stage_two_due(queue, monkeypatch):
    from pipeline.kb import sitting_scheduler as ss
    _pass(ss, {"status": "ok", "read": {"new": 2, "refreshed": 5}}, monkeypatch)
    ss.run_sitting_scheduler()
    assert queue == ["frontier_execute"]


def test_a_read_that_only_refreshed_queries_queues_nothing(queue, monkeypatch):
    """A refreshed query is already standing and already scheduled. Re-queueing on a refresh
    would make every read restart stage 2, which is how a cadence becomes a busy loop."""
    from pipeline.kb import sitting_scheduler as ss
    _pass(ss, {"status": "ok", "read": {"new": 0, "refreshed": 7}}, monkeypatch)
    ss.run_sitting_scheduler()
    assert queue == []


def test_a_scheduler_pass_with_no_claim_queues_nothing(queue, monkeypatch):
    from pipeline.kb import sitting_scheduler as ss
    _pass(ss, {"status": "skipped", "reason": "nothing claimable", "claims": []}, monkeypatch)
    ss.run_sitting_scheduler()
    assert queue == []


def test_plan_only_spends_nothing_and_queues_nothing(queue, monkeypatch):
    from pipeline.kb import sitting_scheduler as ss
    _pass(ss, {"status": "plan", "claims": [], "read": {"new": 4}}, monkeypatch)
    ss.run_sitting_scheduler(plan_only=True)
    assert queue == []


def test_a_hand_added_watch_schedules_the_rail_that_re_runs_it(queue, kb_home, monkeypatch):
    """⚠️ FOUND BY THE 2026-09-16 RAIL AUDIT — the third instance of the class, and the one the
    handoff had not named. `sitting(action='watchlist', add=[...])` writes a standing query and
    runs ONE scoped pull in-process. Measured on a clean home before the fix: the query was
    active and `rail_jobs.db` was empty, so nothing would ever run it again — a watchlist frozen
    at its first pull reads exactly like one finding nothing new.

    DUE ONE CADENCE OUT, not now: the first pull is walking these same (query, source) pairs on a
    thread and `frontier_execute` has no single-flight lock, so a due-now child would duplicate
    its requests. What the user needs from this call is the ROW, not an immediate second pass.
    """
    from mcp_server import sitting_tools

    monkeypatch.setattr(sitting_tools, "_spawn", lambda target: None)   # no live first pull
    m = _MCP()
    sitting_tools.register_sitting_tools(m)

    out = m.tools["sitting"](action="watchlist", add=["agentic payments"])

    assert out["added"] == ["agentic payments"]
    job = RailJobStore().get(LOCAL_HOME_ID, "frontier_execute")
    assert job is not None, (
        "the event that starts a pull must be the event that schedules its repeat; without a row "
        "this watch is pulled once and never again"
    )
    assert job.due_at > time.time(), "due now would race the in-process first pull"


def test_the_repeat_never_postpones_work_another_producer_made_due(queue, kb_home, monkeypatch):
    """`activate` keeps `MIN(due_at)`, and that is what makes the delayed door safe to call on a
    store where a sitting read has already queued stage 2 for real, waiting work."""
    from mcp_server import sitting_tools
    from pipeline.kb import rail_jobs

    rail_jobs.request_now("frontier_execute")
    monkeypatch.setattr(sitting_tools, "_spawn", lambda target: None)
    m = _MCP()
    sitting_tools.register_sitting_tools(m)

    m.tools["sitting"](action="watchlist", add=["agentic payments"])

    assert RailJobStore().get(LOCAL_HOME_ID, "frontier_execute").due_at <= time.time()


# ── the class the chains used to belong to ──────────────────────────────────────
def test_no_rail_schedules_its_successor_from_inside_a_main():
    """⚠️ THE CLASS, not the two instances. A `request_now` inside a `main()` body is a
    side-effect only the CLI child performs, and every rail in this chain grew an in-process
    caller that skipped it — `onboard_tools._run_curation` for `curation_catchup`,
    `sitting_tools._start_first_pull` for `frontier_execute`. The third,
    `sitting_scheduler.main`, had no such caller on 2026-09-16 and was one refactor from joining
    them, which is why this asserts over the whole tree rather than over the two that broke.

    Put the chaining in the `run_*` body instead: it is the one door every caller goes through.
    """
    import ast
    import subprocess

    root = Path(__file__).resolve().parents[2]
    tracked = subprocess.run(["git", "ls-files", "*.py"], cwd=root,
                             capture_output=True, text=True, check=True).stdout.split()
    offenders = []
    for rel in tracked:
        if rel.startswith("tests/"):
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
# §3 OF THE 2026-09-16 HANDOFF, which is the one row of its table with no evidence behind it: no
# hosted run was ever exercised. What it asks is not "does the gateway work" but the narrower
# question these tests can answer offline — does a hosted child queue the SAME SET of rails as a
# local one? Every producer touched by §1 and §2 lives in `onboard_tools` / `curation_catchup` /
# `sitting_tools`, none of which branches on hosted vs local for WHETHER to queue, so a hosted
# user was missing exactly the same two rails and is fixed by exactly the same change. Asserted
# here rather than reasoned, because "confirm, do not assume" is the whole instruction.
@pytest.fixture()
def hosted(tmp_path, monkeypatch):
    """A hosted MCP child's environment, built the way `gateway/children._child_env` builds it:
    the gateway's already-validated subject as the home id, and the operator's ONE shared control
    database inherited (deliberately absent from `_GATEWAY_ONLY`, per that tuple's own comment)."""
    subject = "104729183746152938471"
    shared = tmp_path / "var" / "rail_jobs.db"
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "homes" / subject))
    monkeypatch.setenv(rail_jobs.WORKER_HOME_ID_ENV, subject)
    monkeypatch.setenv(rail_jobs.WORKER_DB_ENV, str(shared))
    return subject, shared


def test_a_hosted_child_says_something_will_act_on_what_it_queues(hosted):
    """The half that was already fixed, pinned from this side too: `queue_is_shared` keys on the
    home id, NOT on whether this platform can install a LaunchAgent. Reading the second told
    every remote user that nothing would act on the consent they had just given, on the box where
    a systemd unit was claiming exactly those rows."""
    assert rail_jobs.queue_is_shared() is True


def test_the_curation_chain_queues_under_the_gateways_subject_not_the_local_home(hosted,
                                                                                monkeypatch):
    """The §2 chain, hosted. It must land in the operator's shared database keyed by this user's
    subject — a row in the child's own home is one the single worker never opens, which is the
    silent total failure `worker_db_path()` raises to prevent."""
    subject, shared = hosted
    from mcp_server import onboard_tools
    from pipeline.kb import curation_catchup as cc

    _pass(cc, {"status": "ok", "ran": {"x_lists": {}}}, monkeypatch)
    onboard_tools._run_curation({"x"})

    assert [(j.home_id, j.rail) for j in RailJobStore(shared).list_jobs()] == [
        (subject, "candidate_probe")]
    assert not (Path(os.environ["OPYT_HOME"]) / "rail_jobs.db").exists()


def test_a_hosted_watch_schedules_its_repeat_the_same_way(hosted, monkeypatch):
    """The audit's third instance, hosted. Same producer, same shared database, same subject."""
    subject, shared = hosted
    from mcp_server import sitting_tools

    monkeypatch.setattr(sitting_tools, "_spawn", lambda target: None)
    m = _MCP()
    sitting_tools.register_sitting_tools(m)

    m.tools["sitting"](action="watchlist", add=["agentic payments"])

    job = RailJobStore(shared).get(subject, "frontier_execute")
    assert job is not None and job.due_at > time.time()


# ── ordering: the write is durable before the successor can be claimed ──────────
def test_the_successor_cannot_be_claimed_before_the_pass_that_queued_it_exits(queue,
                                                                              monkeypatch):
    """The worker refuses any rail for a home that already has an active child, so a queued
    successor waits for its producer's process. Without that, stage 3 could read the candidate
    table while stage 2 is still writing it."""
    store = RailJobStore()
    store.activate(LOCAL_HOME_ID, "frontier_execute", due_at=0)
    claimed = store.claim_next(home_id=LOCAL_HOME_ID)
    assert claimed.rail == "frontier_execute"

    from pipeline.kb import frontier_execute as fe
    _pass(fe, {"status": "ok", "candidates_new": 4}, monkeypatch)
    fe.run_frontier_execute(registry={})

    # The successor row exists and is due, but the producer's claim is still open.
    assert store.get(LOCAL_HOME_ID, "frontier_admit").due_at <= time.time()
    assert store.claim_next(home_id=LOCAL_HOME_ID) is None

    store.finish(claimed, 0, cadence=3600.0)
    assert store.claim_next(home_id=LOCAL_HOME_ID).rail == "frontier_admit"


# ── the first-run sequence, end to end ──────────────────────────────────────────
#
# THE DEFECT THESE EXIST FOR, found 2026-09-14 by reading a real Claude Desktop session back:
# five Oracles confirmed, `oracle_refresh_consent` on disk, the launchd worker installed and
# running — and `rail_jobs.db` held two rows, neither of them `oracle_refresh`. Nothing errored.
# The host was told to say "it fills in on its own" nine times across that session, and it was
# false every time.
#
# `test_every_registry_rail_has_a_producer` passed throughout, because `oracle_refresh` DID have
# a producer on paper. What it did not have was a producer whose precondition could ever be true:
# `_apply_consent` queues only when `_confirmed_oracles() > 0`, and onboarding asks for consent
# BEFORE the roster is picked (`onboard` answers `next_tool='oracle'`). The map above cannot
# catch that — it checks that a producer is named, not that the named producer can fire. Only a
# test that walks the real order can, which is what these two do.
def test_confirming_an_oracle_queues_the_refresh_the_user_consented_to(queue, kb_home):
    """Consent first against an EMPTY roster, exactly as onboarding does it, then confirm."""
    from pipeline.kb import oracle_refresh, oracles, schema

    oracle_refresh.grant_consent()                    # the onboard answer, roster still empty
    conn = schema.connect(kb_home / "opyt.db")
    schema.upsert_entity(conn, "x:user:1", name="Carol")

    out = oracles.confirm(conn, canonical_ids=["x:user:1"])
    conn.close()

    assert out["confirmed"] and out["refresh_queued"] is True
    assert queue == ["oracle_refresh"]
    job = RailJobStore().get(LOCAL_HOME_ID, "oracle_refresh")
    assert job is not None and job.due_at <= time.time(), (
        "the roster exists and the user said yes — a promise of background refresh with no row "
        "behind it is the 2026-09-14 defect verbatim"
    )


def test_a_declined_roster_still_mints_the_oracle_and_queues_nothing(queue, kb_home):
    """Consent is READ at the mint, never granted by it. Someone who answered `backlog` reads
    their Oracles on demand; minting one must not enrol them in a recurring spend they refused."""
    from pipeline.kb import oracles, schema

    conn = schema.connect(kb_home / "opyt.db")        # no consent marker written
    schema.upsert_entity(conn, "x:user:1", name="Carol")

    out = oracles.confirm(conn, canonical_ids=["x:user:1"])
    conn.close()

    assert out["confirmed"], "the mint itself is unconditional — only the schedule is gated"
    assert out["refresh_queued"] is False and queue == []
