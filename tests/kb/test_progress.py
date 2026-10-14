"""`oracle(action='confirm' | 'ingest' | 'progress')` — the tools record what is owed and start
nothing; `progress` reads each pick's milestones.

The constant this file exercises is not a budget on work. `PROGRESS_WAIT_SECONDS` bounds how
long one question waits before answering; the scrapers never see it. A clock on the work itself
would be useless against a unit of work that cannot be interrupted, and an archive should not be.
"""
from __future__ import annotations

import threading
import time

import pytest

from mcp_server import oracle_tools
from pipeline.platforms import connections
from pipeline.worker import liveness
from pipeline.kb import schema
from pipeline.timeparse import utc_now
from pipeline.worker.quota_loop import loop_state
from pipeline.oracles import roster
from pipeline.oracles import registry
from pipeline.oracles import asks
from pipeline.oracles import windows
from pipeline.oracles import progress
from pipeline.oracles import admit


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    """These are about what is reported, not how long it is willing to wait for it. The tests
    that are about the wait set their own."""
    monkeypatch.setattr(progress, "PROGRESS_WAIT_SECONDS", 0.0)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    """X connected, the worker's word unknown, and every wake recorded instead of written."""
    monkeypatch.setattr(connections, "_connected_x", lambda: True)
    monkeypatch.setattr(liveness, "repair_or_report_cached",
                        lambda **kw: {"ok": None, "status": "unknown"})
    woken: list = []
    monkeypatch.setattr("pipeline.worker.job_queue.request_now",
                        lambda rail, **kw: woken.append(rail) or True)
    return woken


def _person(conn, uid, name, handle):
    schema.upsert_entity(conn, f"x:user:{uid}", name=name, profile={"handle": handle})
    return f"x:user:{uid}"


def _pulled(conn, handle, **cols):
    sets = ", ".join(f"{k}=?" for k in cols)
    conn.execute(f"UPDATE oracle_sources SET {sets} WHERE source_key=?", (*cols.values(), handle))
    conn.commit()


# ── confirm and ingest record work and start nothing ───────────────────────────

def test_confirm_records_each_picks_ask_and_wakes_the_scrapers(conn, _quiet):
    a = _person(conn, 1, "Andrej", "andrej")
    d = _person(conn, 2, "Dean", "dean")
    began = time.time()

    out = admit.confirm_picks(conn, canonical_ids=[a, d], add_handles=None, x_lookback="1yr")

    assert time.time() - began < 1.0
    assert [p["name"] for p in out["picks"]] == ["Andrej", "Dean"]
    assert all(p["newest_posts"] is False for p in out["picks"])
    assert asks.ask_for(conn, a).web_from == windows.ALL_TIME
    assert "scrapers" in _quiet
    assert "refresh_queued" not in out and "status" not in out


def test_confirm_refuses_a_bad_window_before_writing_anything(conn):
    a = _person(conn, 1, "Andrej", "andrej")
    out = admit.confirm_picks(conn, canonical_ids=[a], add_handles=None, x_lookback="1y")
    assert "error" in out
    assert not roster.on_roster(conn, a)


def test_a_venue_confirmed_without_topics_is_confirmed_and_says_it_needs_them(conn):
    schema.upsert_entity(conn, "openalex:S7", name="ChemRxiv")
    out = admit.confirm_picks(conn, canonical_ids=["openalex:S7"], add_handles=None)
    entry = out["confirmed"][0]
    assert entry["needs_topics"] is True and "add_oracle('openalex:S7')" in entry["note"]
    assert roster.on_roster(conn, "openalex:S7")
    assert asks.ask_for(conn, "openalex:S7").scholar_from is None


def test_ingest_widens_an_ask_and_never_narrows_it(conn):
    a = _person(conn, 1, "Andrej", "andrej")
    admit.confirm_picks(conn, canonical_ids=[a], add_handles=None, x_lookback="2yr")
    wide = asks.ask_for(conn, a).x_from

    out = admit.ask_again(conn, canonical_ids=[a], force=False, x_lookback="6mo",
                               web_lookback=None)

    assert asks.ask_for(conn, a).x_from == wide
    assert out["picks"][0]["canonical_id"] == a and "run_id" not in out



def test_ingest_takes_the_id_confirm_returned_after_the_head_moved(conn):
    """A footprint resolve can move the cluster head after confirm; `confirmed_oracles` then
    reports the new head, and ingest must still accept the id confirm returned."""
    a = _person(conn, 1, "Andrej", "andrej")
    admit.confirm_picks(conn, canonical_ids=[a], add_handles=None)
    b = _person(conn, 2, "Andrej", "andrej_alt")
    schema.set_canonical_ids(conn, {a: b, b: b})

    out = admit.ask_again(conn, canonical_ids=[a], force=False, x_lookback="6mo",
                          web_lookback=None)

    assert "error" not in out
    assert [p["canonical_id"] for p in out["picks"]] == [b]


def test_since_last_names_only_the_oracles_it_can_check(conn):
    """An Oracle with no X pair must not be listed under `checking`: nothing is stamped or woken
    for them."""
    schema.upsert_entity(conn, "substack:writer", name="Sub Writer")
    roster.upsert_oracle(conn, "substack:writer", name="Sub Writer")
    out = admit.ask_again(conn, canonical_ids=None, force=False, x_lookback="since_last",
                          web_lookback=None)
    assert "Sub Writer" not in (out.get("checking") or [])

def test_since_last_checks_the_newest_posts_rather_than_asking_again(conn):
    a = _person(conn, 1, "Andrej", "andrej")
    admit.confirm_picks(conn, canonical_ids=[a], add_handles=None)
    _pulled(conn, "andrej", last_pulled_at=utc_now().isoformat())
    asked = asks.ask_for(conn, a).asked_at

    out = admit.ask_again(conn, canonical_ids=[a], force=False, x_lookback="since_last",
                               web_lookback=None)

    assert out["checking"] == ["Andrej"]
    assert asks.ask_for(conn, a).asked_at == asked
    assert registry.list_sources(conn)[0].check_requested_at is not None


def test_force_asks_for_discovery_again(conn):
    a = _person(conn, 1, "Andrej", "andrej")
    admit.confirm_picks(conn, canonical_ids=[a], add_handles=None)
    admit.ask_again(conn, canonical_ids=[a], force=True, x_lookback=None, web_lookback=None)
    assert asks.ask_for(conn, a).rediscover


def test_topics_for_several_picks_are_refused(conn):
    a = _person(conn, 1, "Andrej", "andrej")
    d = _person(conn, 2, "Dean", "dean")
    out = admit.confirm_picks(conn, canonical_ids=[a, d], add_handles=None,
                                scholar_topics=["T1"])
    assert "ONE Oracle" in out["error"]


def test_the_promise_waits_for_the_workers_own_word(conn, monkeypatch):
    """Audit F6: "it fills in on its own" is said only on the one worker check's yes
    (`liveness.repair_or_report_cached`)."""
    a = _person(conn, 1, "Andrej", "andrej")
    monkeypatch.setattr(liveness, "repair_or_report_cached",
                        lambda **kw: {"ok": False, "cause": "the helper is off", "step": "turn it on"})
    out = admit.confirm_picks(conn, canonical_ids=[a], add_handles=None)
    assert "nothing is collecting it yet: the helper is off" in out["message"]
    assert out["picks"][0]["stuck"]["cause"] == "not_running"

    monkeypatch.setattr(liveness, "repair_or_report_cached", lambda **kw: {"ok": True})
    assert "on its own" in admit.ask_again(conn, canonical_ids=[a], force=False,
                                                 x_lookback=None, web_lookback=None)["message"]


# ── progress ───────────────────────────────────────────────────────────────────

def test_no_oracle_at_all_is_an_answer_not_an_error(conn):
    """A host handed an error reasons its way into a retry."""
    out = progress.progress(conn)
    assert out["picks"] == [] and "error" not in out


def test_every_pick_is_named_including_the_untouched_ones(conn):
    """An Oracle missing from a report reads as an Oracle who failed. `false` is what says
    "not yet" instead of letting silence say "nothing found"."""
    ids = [_person(conn, i, n, n.lower()) for i, n in enumerate(("Andrej", "Dean", "Soren"))]
    admit.confirm_picks(conn, canonical_ids=ids, add_handles=None)
    _pulled(conn, "andrej", last_pulled_at=utc_now().isoformat())

    picks = progress.progress(conn)["picks"]

    assert [(p["name"], p["newest_posts"]) for p in picks] == [
        ("Andrej", True), ("Dean", False), ("Soren", False)]


def test_there_is_no_complete_anywhere(conn):
    a = _person(conn, 1, "Andrej", "andrej")
    admit.confirm_picks(conn, canonical_ids=[a], add_handles=None)
    out = progress.progress(conn)
    assert "status" not in out and "complete" not in str(out)


def test_progress_can_be_narrowed_to_some_picks(conn):
    ids = [_person(conn, i, n, n.lower()) for i, n in enumerate(("Andrej", "Dean"))]
    admit.confirm_picks(conn, canonical_ids=ids, add_handles=None)
    assert [p["name"] for p in progress.progress(conn, canonical_ids=[ids[1]])["picks"]] == [
        "Dean"]



def test_asking_about_someone_who_is_not_an_oracle_names_them(conn):
    """With Oracles on the store, ids that match none of them must not be told "No Oracles are
    confirmed on this store yet"."""
    a = _person(conn, 1, "Andrej", "andrej")
    admit.confirm_picks(conn, canonical_ids=[a], add_handles=None)
    out = progress.progress(conn, canonical_ids=["x:user:999"])
    assert out["picks"] == [] and "No Oracles are confirmed" not in out["message"]
    assert "x:user:999" in out["message"]

def test_it_returns_early_the_moment_a_milestone_turns(conn, monkeypatch, kb_home):
    """The narration property: the call does not sit out its whole wait once it has news."""
    monkeypatch.setattr(progress, "PROGRESS_WAIT_SECONDS", 30.0)
    monkeypatch.setattr(progress, "_PROGRESS_POLL_SECONDS", 0.01)
    a = _person(conn, 1, "Andrej", "andrej")
    admit.confirm_picks(conn, canonical_ids=[a], add_handles=None)

    def land():
        c = schema.connect()
        c.execute("UPDATE oracle_sources SET last_pulled_at=?", (utc_now().isoformat(),))
        c.commit()
        c.close()
    threading.Timer(0.15, land).start()

    began = time.time()
    out = progress.progress(conn)
    assert time.time() - began < 5.0
    assert out["picks"][0]["newest_posts"] is True


def test_it_gives_up_waiting_and_still_answers(conn, monkeypatch):
    """A quiet stretch — one writer deep inside a blog archive — still gets an answer, because
    the caller has to be able to say something. Nothing is cut."""
    monkeypatch.setattr(progress, "PROGRESS_WAIT_SECONDS", 0.25)
    monkeypatch.setattr(progress, "_PROGRESS_POLL_SECONDS", 0.01)
    a = _person(conn, 1, "Andrej", "andrej")
    admit.confirm_picks(conn, canonical_ids=[a], add_handles=None)

    began = time.time()
    out = progress.progress(conn)
    assert 0.2 < time.time() - began < 5.0
    assert out["picks"][0]["newest_posts"] is False


def test_it_does_not_wait_when_nothing_can_move(conn, monkeypatch):
    """Everything in, or only stuck picks: there is no news to wait for."""
    monkeypatch.setattr(progress, "PROGRESS_WAIT_SECONDS", 30.0)
    a = _person(conn, 1, "Andrej", "andrej")
    admit.confirm_picks(conn, canonical_ids=[a], add_handles=None)
    loop_state.set_state("x:UserOriginalsTimeline", "paused", cause="x_session")

    began = time.time()
    out = progress.progress(conn)
    assert time.time() - began < 2.0
    assert out["picks"][0]["stuck"]["step"] == "onboard(source='x')"


def test_the_wait_is_well_inside_the_measured_client_wall():
    """61.2 / 61.3 / 62.3s measured. The headroom covers a slow first read on a cold store, not a
    longer wait — this constant must not creep upward."""
    assert progress.PROGRESS_WAIT_SECONDS <= 45.0


def test_an_unknown_action_names_progress(kb_home):
    """A host that cannot discover the action cannot use it."""
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
# A truncated call destroys its return value, but the tool description is
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


def test_the_host_is_told_the_work_needs_no_following():
    """The loop is gone with the pull thread. A host still told to loop until "not running" would
    poll forever: nothing is ever complete."""
    doc = _oracle_doc()
    assert 'THERE IS NO "COMPLETE"' in doc
    assert "NEVER HAVE TO CALL `progress` FOR THE WORK TO HAPPEN" in doc
    assert "LOOP ON `progress`" not in doc and "run_id" not in doc


def test_every_milestone_the_host_can_receive_is_explained():
    doc = _oracle_doc()
    for field in ("`newest_posts`", "`archives`", "`reach`", "`asked`", "`stuck`",
                  "`waiting_on`"):
        assert field in doc


def test_not_yet_is_never_to_be_called_missing():
    doc = _oracle_doc()
    assert "never that something is missing" in doc


def test_a_quiet_wait_is_explicitly_not_a_timeout():
    """A `progress` call that answers with no news is the system working. Left unsaid,
    "returned with nothing" becomes "timed out" becomes a diagnosis."""
    doc = _oracle_doc()
    assert 'NEVER say a pull "timed out"' in doc
    assert "neither sees that clock nor is cut by it" in doc


def test_the_error_rule_points_at_progress():
    doc = _oracle_doc()
    assert "Do not start another ingest" in doc
    assert "`oracle(action='progress')` — that is exactly what it is for" in doc


def test_a_stopped_worker_is_said_once_with_one_period(conn, monkeypatch):
    """The worker's cause already ends in a period, so the promise must not read "stopping.."."""
    monkeypatch.setattr(liveness, "repair_or_report_cached",
                        lambda **kw: {"ok": False, "cause": "the background helper is stopping."})
    a = _person(conn, 1, "Andrej", "andrej")
    out = admit.confirm_picks(conn, canonical_ids=[a], add_handles=None)
    assert "stopping." in out["message"] and ".." not in out["message"]


# ── what discovery found rides on progress ─────────────────────────────────

def _found(conn, cid, url, status, kind="source", reasons=("in 41 of their posts",
                                                          "links back to their X")):
    import json as _json
    conn.execute("INSERT INTO oracle_review_items (canonical_id, source_type, source_url, "
                 "source_key, kind, reasons, status) VALUES (?, 'substack', ?, ?, ?, ?, ?)",
                 (cid, url, url, kind, _json.dumps(list(reasons)), status))
    conn.commit()


def test_found_sources_are_said_once(conn):
    a = _person(conn, 1, "Alex", "alex")
    admit.confirm_picks(conn, canonical_ids=[a], add_handles=None)
    _found(conn, a, "https://alex.substack.com", "added")
    _found(conn, a, "https://tool.example", "added", kind="also", reasons=())
    _found(conn, a, "https://gone.example", "dismissed")

    first = progress.progress(conn)["picks"][0]["other_sources"]
    assert [e["source"] for e in first["added"]] == ["alex.substack.com"]
    assert first["also"] == ["tool.example"]
    assert first["message"] == (
        "Also pulling alex.substack.com (in 41 of their posts · links back to their X). "
        "Also: tool.example.")

    assert "other_sources" not in progress.progress(conn)["picks"][0]


def test_other_sources_shows_at_most_two_reasons_and_no_mechanics():
    items = [{"review_id": 1, "source_url": "https://a.dev/", "status": "added",
              "kind": "source", "reasons": ["one", "two", "three"]}]
    out = progress.other_sources(items, set())
    assert out["added"] == [{"source": "a.dev", "reasons": ["one", "two"]}]
    assert out["message"] == "Also pulling a.dev (one · two)."
    assert progress.other_sources([], set()) is None
