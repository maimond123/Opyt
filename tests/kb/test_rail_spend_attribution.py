"""Every paid rail labels its own spend, and every paid rail has a ceiling that reads that label.

WHY THIS FILE IS CROSS-CUTTING rather than split across the five per-rail test files. The defect it
pins was invisible in any single rail: `ORACLE_REFRESH_DAILY_USD` and `BOOKMARK_CATCHUP_DAILY_USD`
were both $1.00 and both read `spend_today()` — the TOTAL for the UTC day — so two constants that
looked like two budgets were one shared budget, and each rail's own tests passed. Worse,
`frontier_execute` and `hopper` spent into that same total while checking nothing, so an UNGATED
rail could exhaust the pool and a GATED rail was what stopped. The property only exists BETWEEN the
rails, so it is tested between them.

The label is asserted, never the pull: each test stubs the first thing the rail's entry point calls,
reads `llm_client.current_rail()` from inside, and bails. No network, no store, no money.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from pipeline import llm_client
from pipeline.kb import (bookmark_catchup, curation_catchup, expand, frontier_execute, hopper,
                         oracle_refresh, oracles, probe_catchup, rail_budgets, schema)


@pytest.fixture
def isolated_stats(tmp_path):
    """Redirect api_stats.json to a tmp dir. Matches the local fixture in test_llm_client.py /
    test_llm_cost_accounting.py — `_override_stats_file_for_tests` rebuilds `_STATS` from the
    `_fresh_stats()` factory, so a copy of this fixture cannot drift from the production shape."""
    llm_client._override_stats_file_for_tests(tmp_path / "api_stats.json")
    yield tmp_path / "api_stats.json"
    llm_client._override_stats_file_for_tests(None)


class _Bail(Exception):
    """Stop the rail the instant the label has been read, so no test does the rail's real work."""


def _label_seen_by(monkeypatch, module, attr: str) -> dict:
    """Replace `module.attr` with a stub that records the active rail, then bails."""
    seen: dict = {}

    def _stub(*args, **kwargs):
        seen["rail"] = llm_client.current_rail()
        raise _Bail

    monkeypatch.setattr(module, attr, _stub)
    return seen


# ── the four detached rails ──────────────────────────────────────────────────────
#
# Each is spawned as its own process, so the label is set once at the entry point and is true for
# every thread in that child. The scope wraps `run_*` and NOT `main()`: `main()` only wraps `run_*`
# for the `--once` child, so labelling it would attribute the child's spend and miss every
# in-process call the MCP side makes directly.


def test_bookmark_catchup_labels_its_spend(monkeypatch):
    seen = _label_seen_by(monkeypatch, bookmark_catchup, "load_rail_env")
    with pytest.raises(_Bail):
        bookmark_catchup.run_bookmark_catchup()
    assert seen["rail"] == "bookmark_catchup"
    assert llm_client.current_rail() == "unattributed", "the label must not outlive the rail"


def test_oracle_refresh_labels_its_spend(monkeypatch):
    seen = _label_seen_by(monkeypatch, oracle_refresh, "load_rail_env")
    with pytest.raises(_Bail):
        oracle_refresh.run_oracle_refresh()
    assert seen["rail"] == "oracle_refresh"
    assert llm_client.current_rail() == "unattributed"


def test_curation_catchup_labels_its_spend(monkeypatch):
    """Labelled even though its four collectors are FREE. A rail that spends nothing today may
    spend something tomorrow, and an unlabelled rail is invisible rather than zero. (It still gets
    no ceiling — see `test_curation_catchup_has_no_ceiling`.)"""
    seen = _label_seen_by(monkeypatch, curation_catchup, "load_rail_env")
    with pytest.raises(_Bail):
        curation_catchup.run_curation_catchup()
    assert seen["rail"] == "curation_catchup"
    assert llm_client.current_rail() == "unattributed"


def test_candidate_probe_labels_its_spend(monkeypatch):
    """⚠️ THE RAIL THIS FILE WAS ALMOST WRITTEN WITHOUT. `probe_catchup` landed on main the same
    day as these meters and arrived UNLABELLED — its embed spend recorded as `unattributed`, which
    is the "fifth registry" warning in `rail_budgets` firing within hours of being written.

    Unlike `curation_catchup`, this one is NOT free: it embeds every probed candidate, measured at
    $0.0026/day at its ceiling. So the label is not a someday-it-might-spend precaution here — it
    is attribution for money that is being spent today."""
    seen = _label_seen_by(monkeypatch, probe_catchup, "load_rail_env")
    with pytest.raises(_Bail):
        probe_catchup.run_candidate_probe()
    assert seen["rail"] == "candidate_probe"
    assert llm_client.current_rail() == "unattributed", "the label must not outlive the rail"


def test_the_candidate_probe_label_matches_the_files_a_human_will_grep(kb_home):
    """The label is a key someone reads off a dollar figure and then searches for. Every other
    rail's label matches its module name because their artifacts do too; this rail's module is
    `probe_catchup` while its lock, log, stamp and spawner are all `candidate_probe`. Pinning the
    label to the ARTIFACTS rather than the module is deliberate — it is what makes the trail from
    a spend figure to `~/.opyt/candidate_probe.log` actually connect."""
    assert probe_catchup.RAIL == "candidate_probe"
    monkey = probe_catchup.spawn_candidate_probe
    assert callable(monkey)
    from opyt_core.paths import opyt_path
    assert opyt_path(f"{probe_catchup.RAIL}.log").name == "candidate_probe.log"
    assert opyt_path(f"{probe_catchup.RAIL}_last_spawn").name == "candidate_probe_last_spawn"


def test_frontier_execute_labels_its_spend(monkeypatch):
    seen: dict = {}

    def _stub(conn, **kwargs):
        seen["rail"] = llm_client.current_rail()
        return {"status": "skipped"}

    monkeypatch.setattr(frontier_execute, "_run", _stub)
    # A caller-supplied conn keeps `own` False, so nothing opens a DB.
    assert frontier_execute.run_frontier_execute(conn=object())["status"] == "skipped"
    assert seen["rail"] == "frontier_execute"
    assert llm_client.current_rail() == "unattributed"


# ── hopper, the one IN-PROCESS rail ──────────────────────────────────────────────
#
# The other four are detached children, so a process-wide label is true for their whole life.
# `hopper` runs inside the long-lived MCP server, so it needs an explicit scope that ENDS — and it
# can be entered while another rail's scope is already open, which is what the save/restore in
# `llm_client.rail` is for.
#
# The scope goes on `pipeline.kb.hopper.save`, NOT on the `hopper` function in
# mcp_server/hopper_tools.py. That one is a closure inside `register_hopper_tools(mcp)` and cannot
# be imported or called without a mock MCP server, so a label there would be untestable; `save` is
# the importable entry every caller (the tool included) actually goes through, and it covers the
# paid `preview` path too.


def test_hopper_labels_its_spend(isolated_stats, monkeypatch):
    seen: dict = {}

    def _stub_preview(conn, reference, **kwargs):
        seen["rail"] = llm_client.current_rail()
        return {"routable": False, "reference": reference}

    monkeypatch.setattr(hopper, "preview", _stub_preview)
    assert hopper.save(None, None, "not-a-url")["status"] == "unroutable"
    assert seen["rail"] == "hopper"
    assert llm_client.current_rail() == "unattributed"


def test_spend_outside_hopper_is_not_attributed_to_it(isolated_stats):
    llm_client.record_external_cost("openrouter-embed", 0.02)
    assert llm_client.spend_today_for_rail("hopper") == 0.0
    assert llm_client.spend_today_for_rail("unattributed") == 0.02


def test_hopper_nested_in_another_rail_restores_the_outer_label(isolated_stats, monkeypatch):
    """⚠️ THE IN-PROCESS HAZARD. `hopper` is the one rail that can start while another scope is
    open, so an inner scope that reset to unattributed on exit would silently un-attribute the
    REST of the outer rail's run — spend that is already committed to the outer rail's ceiling."""
    seen: dict = {}

    def _stub_preview(conn, reference, **kwargs):
        seen["rail"] = llm_client.current_rail()
        return {"routable": False, "reference": reference}

    monkeypatch.setattr(hopper, "preview", _stub_preview)
    with llm_client.rail("frontier_execute"):
        hopper.save(None, None, "not-a-url")
        llm_client.record_external_cost("openrouter-embed", 0.03)

    assert seen["rail"] == "hopper"
    assert llm_client.spend_today_for_rail("frontier_execute") == 0.03
    assert llm_client.spend_today_for_rail("unattributed") == 0.0


# ── each ceiling governs ITS OWN rail ────────────────────────────────────────────


def test_one_rails_spend_no_longer_exhausts_the_others_ceiling(isolated_stats):
    """THE DEFECT, stated as a test. Both constants read $1.00 and both gated on the TOTAL, so the
    two rails shared one pool and either could starve the other — and an UNGATED rail spending into
    that same pool (here `frontier_execute`) could starve BOTH while checking nothing itself."""
    # Comfortably PAST both $1.00 ceilings, so the old total-reading gates would both refuse. A
    # figure under the ceiling would pass against the buggy code too and prove nothing.
    with llm_client.rail("frontier_execute"):
        llm_client.record_external_cost("openrouter-embed", 1.50)
    assert llm_client.spend_today() == 1.50, "the total still sees it — this is attribution, not hiding"
    assert bookmark_catchup._daily_budget_exhausted() is False
    assert oracle_refresh._daily_budget_exhausted() is False


def test_bookmark_catchup_ceiling_trips_on_its_own_spend(isolated_stats):
    with llm_client.rail(bookmark_catchup.RAIL):
        llm_client.record_external_cost("openrouter-embed", bookmark_catchup.BOOKMARK_CATCHUP_DAILY_USD)
    assert bookmark_catchup._daily_budget_exhausted() is True
    assert oracle_refresh._daily_budget_exhausted() is False, "must not starve its neighbour"


def test_oracle_refresh_ceiling_trips_on_its_own_spend(isolated_stats):
    with llm_client.rail(oracle_refresh.RAIL):
        llm_client.record_external_cost("openrouter-embed", oracle_refresh.ORACLE_REFRESH_DAILY_USD)
    assert oracle_refresh._daily_budget_exhausted() is True
    assert bookmark_catchup._daily_budget_exhausted() is False, "must not starve its neighbour"


def test_the_gate_reads_the_label_the_rail_actually_sets(isolated_stats, monkeypatch):
    """⚠️ THE DRIFT THIS PREVENTS. A rail's name now appears twice — once in the decorator that
    labels its spend, once in the gate that reads that label. Two literals that must match is
    exactly what drifts, and a drifted pair fails SILENTLY: the rail spends, the meter fills under
    one name, and the ceiling reads an empty meter under the other. Both are bound to one module
    constant; this asserts the binding end to end, through the real entry point."""
    def _spend(*args, **kwargs):
        llm_client.record_external_cost("openrouter-embed", bookmark_catchup.BOOKMARK_CATCHUP_DAILY_USD)
        raise _Bail

    monkeypatch.setattr(bookmark_catchup, "load_rail_env", _spend)
    with pytest.raises(_Bail):
        bookmark_catchup.run_bookmark_catchup()
    assert bookmark_catchup._daily_budget_exhausted() is True


def test_an_unreadable_meter_blocks_NEITHER_rail(isolated_stats, monkeypatch):
    """Fail-safe, unchanged by the re-attribution: a gate that cannot read its meter must let the
    run through, not refuse it. A ceiling is a seatbelt, and a seatbelt that jams shut is a worse
    failure than one that does not fasten.

    BOTH rails, deliberately, even though test_bookmark_catchup.py already pins its own. "The other
    rail has the same shape, so it must behave the same" is precisely the assumption that let two
    identical $1.00 gates share one pool unnoticed."""
    monkeypatch.setattr(llm_client, "spend_today_for_rail",
                        lambda _: (_ for _ in ()).throw(RuntimeError("meter unreadable")))
    assert bookmark_catchup._daily_budget_exhausted() is False
    assert oracle_refresh._daily_budget_exhausted() is False


# ── every PAID rail has a ceiling ────────────────────────────────────────────────
#
# ⚠️ THE HALF THAT IS NOT OPTIONAL. Per-rail attribution WITHOUT this fixes the accounting and
# leaves the actual hole open: `frontier_execute` and `hopper` spend real money into the same
# total and, before this, checked nothing at all. So an UNGATED rail could exhaust the pool while
# the GATED rails were the ones that stopped.


def test_every_paid_rail_has_a_ceiling_and_a_gate():
    """The completeness property, asserted as one list. A new paid rail added without a ceiling
    reopens exactly the hole this work closed, and nothing else would notice."""
    paid = [(bookmark_catchup, "BOOKMARK_CATCHUP_DAILY_USD"),
            (oracle_refresh, "ORACLE_REFRESH_DAILY_USD"),
            (frontier_execute, "FRONTIER_EXECUTE_DAILY_USD"),
            (hopper, "HOPPER_DAILY_USD")]
    for module, const in paid:
        assert getattr(module, const) > 0, f"{module.__name__} has no ceiling amount"
        assert callable(module._daily_budget_exhausted), f"{module.__name__} has no gate"
        assert module.RAIL, f"{module.__name__} has no rail label to gate on"


def test_curation_catchup_gets_NO_ceiling():
    """⚠️ DELIBERATE, and the one exception. Its four collectors are free, so a gate there would be
    a seatbelt on a parked car — and it would make `budget_paused` mean two different things: 'this
    rail spent its money' and 'this rail that spends no money is off'. It is still LABELLED, so it
    reads as zero rather than as invisible."""
    assert not hasattr(curation_catchup, "CURATION_CATCHUP_DAILY_USD")
    assert not hasattr(curation_catchup, "_daily_budget_exhausted")
    assert curation_catchup.RAIL == "curation_catchup"


def test_oracle_ingest_labels_its_spend(monkeypatch):
    """⚠️ THE SECOND RAIL TO ARRIVE UNLABELLED, days after the first. `probe_catchup` landed
    unattributed on 2026-08-16; `oracles._ingest_oracle` has been spending unattributed for far
    longer, and reviving the cold-start anchor was about to add a paid probe to it.

    `_ingest_oracle` is IN-PROCESS, like `hopper` — it runs inside the long-lived MCP server via
    `add_oracle` and `oracle(action="ingest")` — so it needs a scope that ENDS, not a
    process-wide label.
    """
    # Probes `_root_profile`, the first thing `_ingest_oracle` calls. It used to probe
    # `trust.seed_roots`, which ran earlier still — until that layer was deleted (2026-08-23) for
    # having no reader. Any early, unconditional call inside the rail scope works here; what is
    # under test is the LABEL, not which function carries it.
    seen = _label_seen_by(monkeypatch, expand, "_root_profile")
    with pytest.raises(_Bail):
        oracles._ingest_oracle(None, None, {"canonical_id": "x:user:1", "members": []})
    assert seen["rail"] == "oracle_ingest"
    assert llm_client.current_rail() == "unattributed", "the label must not outlive the rail"


def test_oracle_ingest_captures_the_WHOLE_footprint_not_just_the_anchor(monkeypatch):
    """⚠️ READ THIS BEFORE SIZING A CEILING FOR THIS RAIL. The label was added for the ~$0.0051
    cold-start anchor, but the scope wraps the whole engine — so it also captures every embedding
    `onboard_footprint` writes and every twitterapi request `sync_x_footprint` makes. Those are
    the large numbers; the anchor is a rounding error beside them.

    That is CORRECT (total attribution is the point), and it means this rail's daily figure is not
    comparable to the anchor's per-call cost. Anyone reading `oracle_ingest` off a dollar report
    is looking at a footprint ingest, not at a search probe.
    """
    spent: dict = {}

    def _spend_inside_the_rail(conn, oracle):
        llm_client.record_external_cost("openrouter-embed", 0.25)      # stands in for the X pull
        spent["rail"] = llm_client.current_rail()
        raise _Bail

    monkeypatch.setattr(expand, "_root_profile", _spend_inside_the_rail)
    with pytest.raises(_Bail):
        oracles._ingest_oracle(None, None, {"canonical_id": "x:user:1", "members": []})
    assert spent["rail"] == "oracle_ingest"


def test_oracle_ingest_gets_NO_DOLLAR_ceiling():
    """⚠️ THE THIRD EXCEPTION, and a THIRD distinct reason — do not collapse these three.
    `curation_catchup` is exempt because it is free. `candidate_probe` is exempt because it is
    bounded in candidates. This one is exempt because it is bounded by CONSENT: every call is a
    user typing `add_oracle` or `oracle(action="ingest")` for one named person, and the number of
    Oracles is the bound. There is no schedule and no loop that can run it away.

    Pinned, not left implicit, because this rail spends MORE per invocation than two of the four
    that do have ceilings (see the test above). Someone will read that figure and reach for a
    budget; the honest answer is that a ceiling here would refuse work a user explicitly asked
    for, which is the failure mode `hopper`'s runaway guard is shaped to avoid rather than cause.

    ⚠️ If this rail ever gains a NON-user-initiated caller — a scheduler, a catch-up, a retry
    loop — this reasoning expires and it needs a guard. Nothing here enforces that; this comment
    is the warning.
    """
    assert not hasattr(oracles, "ORACLE_INGEST_DAILY_USD")
    assert not hasattr(oracles, "_daily_budget_exhausted")
    assert oracles.RAIL == "oracle_ingest"


def test_oracle_ingest_is_never_reported_as_paused(isolated_stats, kb_home):
    """No ceiling → never paused, however much is attributed to it."""
    with llm_client.rail(oracles.RAIL):
        llm_client.record_external_cost("openrouter-embed", 99.0)
    assert rail_budgets.paused_today() == []


def test_candidate_probe_gets_NO_DOLLAR_ceiling():
    """⚠️ THE SECOND EXCEPTION, and it is a DIFFERENT exception from `curation_catchup`'s. That
    rail is exempt because it is free. This one is exempt because it is already bounded IN ANOTHER
    UNIT: `PROBE_DAILY_CANDIDATES = 60` meters candidates, because candidates — X requests against
    one shared cookie session — are what is actually scarce here, and the money ($0.0026/day at
    that ceiling) is a rounding error beside it.

    So a dollar ceiling here would not add safety, it would add a SECOND bound that can disagree
    with the first: a rail refusing at $1.00 while its candidate budget says 40 to go, or the
    reverse. Pinned so nobody "completes the set" by adding one — the honest reading is that this
    rail is gated, just not in dollars."""
    assert probe_catchup.PROBE_DAILY_CANDIDATES > 0, "it must still be bounded in SOME unit"
    assert not hasattr(probe_catchup, "PROBE_CATCHUP_DAILY_USD")
    assert not hasattr(probe_catchup, "_daily_budget_exhausted")
    assert probe_catchup.RAIL == "candidate_probe"


def test_frontier_execute_pauses_on_its_own_spend(isolated_stats, kb_home):
    conn = schema.connect()
    try:
        with llm_client.rail(frontier_execute.RAIL):
            llm_client.record_external_cost("openrouter-embed",
                                            frontier_execute.FRONTIER_EXECUTE_DAILY_USD)
        out = frontier_execute.run_frontier_execute(conn=conn)
        assert out["status"] == "budget_paused"

        # ⚠️ AND IT IS RECORDED. The pause has to be checkable after the fact, or this rail
        # repeats the frozen-Oracle failure: refusing correctly, in private.
        row = conn.execute("SELECT status, reason FROM frontier_exec_runs "
                           "ORDER BY run_id DESC LIMIT 1").fetchone()
        assert row[0] == "budget_paused" and row[1]
    finally:
        conn.close()


def test_another_rails_spend_does_not_pause_frontier_execute(isolated_stats, kb_home, monkeypatch):
    """The same starvation the two gated rails suffered, checked in the other direction."""
    monkeypatch.setattr(frontier_execute, "_run", lambda conn, **kw: {"status": "ok"})
    with llm_client.rail(bookmark_catchup.RAIL):
        llm_client.record_external_cost("openrouter-embed", 5.00)
    assert frontier_execute.run_frontier_execute(conn=object())["status"] == "ok"


def test_hopper_refuses_when_its_runaway_guard_trips(isolated_stats):
    """`hopper` is user-initiated and cheap per call, so its ceiling is a RUNAWAY GUARD rather
    than a budget. It is also the one rail with a human reading its output, so the refusal is
    reported in the returned payload instead of a log nobody reads."""
    with llm_client.rail(hopper.RAIL):
        llm_client.record_external_cost("openrouter-embed", hopper.HOPPER_DAILY_USD)

    out = hopper.save(None, None, "https://example.com/post", confirm=True)
    assert out["status"] == "budget_paused"
    assert "message" in out and out["message"]


def test_hopper_refuses_the_PREVIEW_too(isolated_stats):
    """Both phases, one meaning. The preview is free for four of five kinds but PAYS for an X
    reference, and a preview that works while a confirm cannot is a confusing half-state — the
    user would read the preview and then be refused the save it exists to authorize."""
    with llm_client.rail(hopper.RAIL):
        llm_client.record_external_cost("openrouter-embed", hopper.HOPPER_DAILY_USD)
    assert hopper.save(None, None, "https://example.com/post")["status"] == "budget_paused"


def test_another_rails_spend_does_not_pause_hopper(isolated_stats, monkeypatch):
    monkeypatch.setattr(hopper, "preview",
                        lambda conn, reference, **kw: {"routable": False, "reference": reference})
    with llm_client.rail(frontier_execute.RAIL):
        llm_client.record_external_cost("openrouter-embed", 5.00)
    assert hopper.save(None, None, "not-a-url")["status"] == "unroutable"


# ── a pause has to be VISIBLE ────────────────────────────────────────────────────
#
# The pause itself was already correct before this work; it just happened in private, in
# ~/.opyt/bookmark_catchup.log, which nothing reads. That is the frozen-Oracle failure shape — a
# rail refusing correctly and looking broken — and it is what makes the wrong ceiling expensive to
# notice rather than merely wrong.


def _write_child_flush(stats_path, rail: str, cost_usd: float) -> None:
    """Stand in for a detached rail CHILD: write api_stats.json the way `flush_stats` would, from a
    process this one shares no memory with."""
    today = datetime.now(timezone.utc).date().isoformat()
    stats_path.write_text(json.dumps({
        "lifetime": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": cost_usd},
        "by_rail": {f"{rail}|{today}": {"calls": 0, "input_tokens": 0,
                                        "output_tokens": 0, "cost_usd": cost_usd}}}))


def _child_refused(rail: str):
    """Stand in for a detached rail CHILD hitting its start gate: the child's own
    `rail_budget_exhausted` returns True against the in-memory spend only IT can see, and stamps
    the marker. Written by hand here for the same reason `_write_child_flush` is — this process
    shares no memory with that child, so it cannot reach True through the real gate.

    Through `rail_runtime.refusal_marker`, never a hand-composed path: the autouse fixture in
    tests/conftest.py redirects that function so no test can stamp the real `~/.opyt`."""
    from pipeline.kb import rail_runtime
    marker = rail_runtime.refusal_marker(rail)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return marker


def test_a_pause_is_visible_to_a_process_that_did_not_spend(isolated_stats, kb_home):
    """⚠️ MEASURED 2026-08-16, AND IT DECIDES THIS DESIGN. `_load_stats_once` caches for the life
    of a process with no invalidation, and the four background rails each run as a detached CHILD
    that flushes at exit. So the long-lived MCP server's in-memory figure for a child's rail is
    0.0 FOREVER. A notice built on the GATE's reader would report every rail at $0 and never fire
    — silent success, which is the exact failure this whole change exists to remove.

    The refusal marker clears this the same way the flushed file does, and for the same reason:
    both are on disk, and disk is the only thing a parent and a detached child share."""
    llm_client.spend_today_for_rail(bookmark_catchup.RAIL)   # the server touches it → load cached
    _write_child_flush(isolated_stats, bookmark_catchup.RAIL,
                       bookmark_catchup.BOOKMARK_CATCHUP_DAILY_USD)
    _child_refused(bookmark_catchup.RAIL)

    assert llm_client.spend_today_for_rail(bookmark_catchup.RAIL) == 0.0, "blind, as measured"
    reported = rail_budgets.paused_today()
    assert [p["rail"] for p in reported] == [bookmark_catchup.RAIL]
    assert reported[0]["spent_usd"] == bookmark_catchup.BOOKMARK_CATCHUP_DAILY_USD, (
        "the dollar figure still comes off the merged meter, not the marker")


def test_hoppers_own_unflushed_spend_is_visible_too(isolated_stats, kb_home):
    """The mirror image. `hopper` runs IN the server, so its spend is in memory and NOT yet on
    disk — `flush_stats` runs at atexit, which for a long-lived server means never during the
    session. The reporting reader has to see both sides.

    This one reaches the pause through the REAL gate: `hopper.save` runs in this process, so its
    `_daily_budget_exhausted` sees the spend recorded two lines up and stamps the marker itself."""
    with llm_client.rail(hopper.RAIL):
        llm_client.record_external_cost("openrouter-embed", hopper.HOPPER_DAILY_USD)
    assert hopper.save(None, None, "https://example.com/post")["status"] == "budget_paused"
    assert [p["rail"] for p in rail_budgets.paused_today()] == [hopper.RAIL]


def test_spending_the_whole_ceiling_and_FINISHING_is_not_a_pause(isolated_stats, kb_home):
    """⚠️ THE CORRECTION OF 2026-08-30, and the reason `paused_today` stopped reading the meter.

    Every ceiling here is a start gate, so a rail that runs walks past its ceiling and stops only
    on the NEXT run. That makes "spent >= ceiling" the normal shape of a rail that DID ITS JOB.
    Reported as a pause, it told a brand-new user that the $8.03 bookmark backlog they had just
    consented to and watched drain completely was "paused ... missing recent material".

    No refusal has happened here, so there is nothing to report — however far over the ceiling
    the meter reads."""
    with llm_client.rail(bookmark_catchup.RAIL):
        llm_client.record_external_cost("openrouter-embed",
                                        bookmark_catchup.BOOKMARK_CATCHUP_DAILY_USD * 8)
    assert rail_budgets.paused_today() == []


def test_yesterdays_refusal_is_not_todays_pause(isolated_stats, kb_home, monkeypatch):
    """The marker is never cleaned up; it just stops matching. Pinned because "nothing deletes it"
    is only safe if a stale one reads as absent."""
    import os, time
    with llm_client.rail(hopper.RAIL):
        llm_client.record_external_cost("openrouter-embed", hopper.HOPPER_DAILY_USD)
    marker = _child_refused(hopper.RAIL)
    assert [p["rail"] for p in rail_budgets.paused_today()] == [hopper.RAIL]

    two_days = time.time() - 2 * 86400
    os.utime(marker, (two_days, two_days))
    assert rail_budgets.paused_today() == []


def test_a_rail_under_its_ceiling_is_not_reported(isolated_stats, kb_home):
    with llm_client.rail(hopper.RAIL):
        llm_client.record_external_cost("openrouter-embed", hopper.HOPPER_DAILY_USD - 0.01)
    assert rail_budgets.paused_today() == []


def test_curation_catchup_is_never_reported_as_paused(isolated_stats, kb_home):
    """It has no ceiling, so it can never be paused — even if something attributes spend to it.
    Reporting it would make `budget_paused` mean two different things."""
    with llm_client.rail(curation_catchup.RAIL):
        llm_client.record_external_cost("openrouter-embed", 99.0)
    assert rail_budgets.paused_today() == []


def test_the_candidate_probe_is_never_reported_as_paused_either(isolated_stats, kb_home):
    """Same absence from `_paid_rails`, for a different reason — this rail DOES spend, it is just
    bounded in candidates rather than dollars, so there is no dollar ceiling for a pause to be
    measured against. Its spend is attributed all the same, which is the whole point of labelling
    a rail that cannot be paused: visible, not gated."""
    with llm_client.rail(probe_catchup.RAIL):
        llm_client.record_external_cost("openrouter-embed", 99.0)
    assert rail_budgets.paused_today() == []
    assert llm_client.spend_today_by_rail().get(probe_catchup.RAIL) == 99.0, (
        "the dollars must still land under this rail's name — unpausable is not unattributed")


def test_a_reporting_hiccup_invents_no_pause(isolated_stats, kb_home, monkeypatch):
    """Fail-safe in the direction that matters: an unreadable meter must report NO pause, never a
    fabricated one. A false pause notice would send the user chasing a rail that is running fine."""
    monkeypatch.setattr(llm_client, "spend_today_by_rail",
                        lambda: (_ for _ in ()).throw(RuntimeError("stats file is a directory")))
    assert rail_budgets.paused_today() == []


def test_a_rails_spend_lands_under_its_own_name(isolated_stats, monkeypatch):
    """End to end through a real rail entry: the dollar recorded inside `run_bookmark_catchup`
    is readable through that rail's meter and no other."""
    def _spend(*args, **kwargs):
        llm_client.record_external_cost("openrouter-embed", 0.05)
        raise _Bail

    monkeypatch.setattr(bookmark_catchup, "load_rail_env", _spend)
    with pytest.raises(_Bail):
        bookmark_catchup.run_bookmark_catchup()

    assert llm_client.spend_today_for_rail("bookmark_catchup") == 0.05
    assert llm_client.spend_today_for_rail("oracle_refresh") == 0.0
    assert llm_client.spend_today_for_rail("unattributed") == 0.0
    assert llm_client.spend_today() == 0.05
