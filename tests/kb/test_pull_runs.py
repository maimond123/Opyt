"""The durable run record — `pipeline/kb/pull_runs.py`.

These prove the STORE, not the narrator: what a pull wrote while it was running is answerable
afterwards, from a connection that never saw the pull. That is the whole property the 60-second
wall destroyed when the report lived in a Python list.
"""
from __future__ import annotations

import time

import pytest

from pipeline.kb import pull_runs, schema


PICKS = [{"canonical_id": "c:andrej", "name": "Andrej"},
         {"canonical_id": "c:dean", "name": "Dean"},
         {"canonical_id": "c:soren", "name": "Soren"}]


@pytest.fixture()
def conn(kb_home):
    c = pull_runs.connect()
    yield c
    c.close()


def test_every_pick_has_a_row_before_anything_has_run(conn):
    """An Oracle absent from the report reads as an Oracle who failed. "Who is still waiting" has
    to be answerable one second in."""
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    rows = pull_runs.oracles_for(conn, run_id)
    assert [r.canonical_id for r in rows] == ["c:andrej", "c:dean", "c:soren"]
    assert [r.state for r in rows] == ["waiting", "waiting", "waiting"]


def test_the_roster_keeps_the_order_it_was_given(conn):
    """`_ordered_picks` is most-vouched-for-first, and that ordering is what breadth means. A
    reader that re-sorted would be a second opinion about who the user cares most about."""
    run_id = pull_runs.open_run(conn, kind="ingest", picks=list(reversed(PICKS)))
    assert [r.name for r in pull_runs.oracles_for(conn, run_id)] == ["Soren", "Dean", "Andrej"]


def test_a_finished_oracle_carries_what_it_yielded(conn):
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    pull_runs.start_oracle(conn, run_id, "c:andrej")
    pull_runs.finish_oracle(conn, run_id, "c:andrej", {"atoms_added": 24, "name": "Andrej"})

    andrej, dean, _ = pull_runs.oracles_for(conn, run_id)
    assert andrej.state == "done"
    assert andrej.result == {"atoms_added": 24, "name": "Andrej"}
    assert dean.state == "waiting"


def test_the_report_survives_a_connection_that_never_saw_the_pull(conn, kb_home):
    """THE POINT OF THE WHOLE MODULE. The call that started the run is gone; a later call, on its
    own connection, can still say what happened."""
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    pull_runs.finish_oracle(conn, run_id, "c:dean", {"atoms_added": 81})
    pull_runs.close_run(conn, run_id, lookback={"x": "6mo"})
    conn.close()

    fresh = pull_runs.connect()
    try:
        run = pull_runs.get_run(fresh, run_id)
        assert run is not None and run.finished_at is not None
        assert run.lookback == {"x": "6mo"}
        dean = [o for o in pull_runs.oracles_for(fresh, run_id) if o.canonical_id == "c:dean"][0]
        assert dean.result == {"atoms_added": 81}
    finally:
        fresh.close()


def test_start_is_stamped_once_so_two_passes_are_one_visit(conn):
    """Breadth and depth are two passes over one person. "Running since" means since breadth."""
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    pull_runs.start_oracle(conn, run_id, "c:andrej")
    first = pull_runs.oracles_for(conn, run_id)[0].started_at
    time.sleep(0.01)
    pull_runs.start_oracle(conn, run_id, "c:andrej")
    assert pull_runs.oracles_for(conn, run_id)[0].started_at == first


def test_a_second_ingest_appends_rather_than_being_refused(conn):
    """"Add one more person" mid-pull is the likely trigger, not a banned retry. Bouncing it
    would drop a writer the user just asked for."""
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS[:1])
    added = pull_runs.add_picks(conn, run_id, [{"canonical_id": "c:dean", "name": "Dean"}])
    assert added == 1
    assert [r.canonical_id for r in pull_runs.oracles_for(conn, run_id)] == ["c:andrej", "c:dean"]


def test_re_adding_somebody_already_on_the_roster_does_not_move_them(conn):
    """A no-op, not an error — and not a demotion to the back for having been named twice."""
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    added = pull_runs.add_picks(conn, run_id, [{"canonical_id": "c:andrej", "name": "Andrej"}])
    assert added == 0
    assert [r.canonical_id for r in pull_runs.oracles_for(conn, run_id)][0] == "c:andrej"


def test_every_row_done_is_not_complete_until_the_loop_says_so(conn):
    """A run whose rows are all done but whose parent is open is a loop that died between the
    last Oracle and the close. Those are different facts and must not read the same."""
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS[:1])
    pull_runs.finish_oracle(conn, run_id, "c:andrej", {"atoms_added": 1})
    run = pull_runs.get_run(conn, run_id)
    assert run.finished_at is None
    assert pull_runs.run_status(run, alive=False) == "stopped"


def test_a_dead_run_reads_as_stopped_and_a_slow_one_does_not(conn):
    """⚠️ `stopped` is a READ of liveness, never an inference from elapsed time. A pull is
    allowed to take twelve minutes."""
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    run = pull_runs.get_run(conn, run_id)
    assert pull_runs.run_status(run, alive=True) == "running"
    assert pull_runs.run_status(run, alive=False) == "stopped"

    pull_runs.close_run(conn, run_id)
    done = pull_runs.get_run(conn, run_id)
    assert pull_runs.run_status(done, alive=False) == "complete"


def test_the_rider_fires_once_and_then_never_again(conn):
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    assert pull_runs.unreported_finished(conn) is None      # still running

    pull_runs.close_run(conn, run_id)
    assert pull_runs.unreported_finished(conn).run_id == run_id

    pull_runs.mark_reported(conn, run_id)
    assert pull_runs.unreported_finished(conn) is None


def test_latest_run_is_the_newest_started_not_the_newest_finished(conn):
    """`progress` with no run id means "the one I just started" — a short run that finishes first
    must not displace the long one still going."""
    old = pull_runs.open_run(conn, kind="ingest", picks=PICKS[:1])
    time.sleep(0.01)
    new = pull_runs.open_run(conn, kind="ingest", picks=PICKS[1:])
    pull_runs.close_run(conn, old)
    assert pull_runs.latest_run(conn).run_id == new


def test_a_store_that_never_saw_these_tables_is_not_a_crash(conn, kb_home):
    """Fail-safe: `_ingest` hands this a plain `schema.connect()`. Every public writer runs the
    DDL, so there is exactly one place a store can be behind and it is never the caller."""
    plain = schema.connect()
    try:
        assert pull_runs.latest_run(plain) is None
        rid = pull_runs.open_run(plain, kind="ingest", picks=PICKS)
        assert len(pull_runs.oracles_for(plain, rid)) == 3
    finally:
        plain.close()


def test_unparseable_json_degrades_to_none_rather_than_raising(conn):
    """A report column is a convenience for the narrator. A store that cannot say what a pull
    yielded must still be able to say that it finished."""
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS[:1])
    conn.execute("UPDATE pull_run_oracles SET result = ? WHERE run_id = ?", ("{not json", run_id))
    conn.commit()
    assert pull_runs.oracles_for(conn, run_id)[0].result is None


# ── liveness ───────────────────────────────────────────────────────────────────
# A heartbeat on the RUN, not on the shared `oracle-refresh` lease. That lease is held in turn by
# the refresh rail, by footprint enrichment and by a pull, on purpose, because all three walk the
# same two x.com buckets — so it says "something in this family is working" and can never say
# which. Reading it here would report a dead pull as running whenever enrichment held it.

def test_a_run_with_a_fresh_beat_is_alive_and_a_stale_one_is_not(conn):
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    run = pull_runs.get_run(conn, run_id)
    assert pull_runs.is_alive(run)
    assert not pull_runs.is_alive(run, now=time.time() + pull_runs.LEASE_TTL + 1)


def test_a_run_from_before_the_column_existed_reads_as_dead(conn):
    """Fail-safe, and the direction is deliberate: it makes the report say "stopped at 4 of 7",
    never "still going" about nothing."""
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    conn.execute("UPDATE pull_runs SET heartbeat_at = NULL WHERE run_id = ?", (run_id,))
    conn.commit()
    assert not pull_runs.is_alive(pull_runs.get_run(conn, run_id))


def test_the_beat_keeps_a_long_pull_alive_without_the_work_touching_it(conn, kb_home):
    """⚠️ THE BEAT CANNOT RIDE ON THE WORK. One measured blog archive ran ~50s as a single
    uninterruptible unit, so stamping the row as each Oracle finishes would let a pull that is
    working hard read as a pull that died."""
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    conn.execute("UPDATE pull_runs SET heartbeat_at = ? WHERE run_id = ?", (0.0, run_id))
    conn.commit()
    assert not pull_runs.is_alive(pull_runs.get_run(conn, run_id))

    with pull_runs.Heartbeat(run_id, interval=0.01):
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if pull_runs.is_alive(pull_runs.get_run(conn, run_id)):
                break
            time.sleep(0.01)
    assert pull_runs.is_alive(pull_runs.get_run(conn, run_id))


def test_in_flight_is_the_live_unfinished_run_and_nothing_else(conn):
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    assert pull_runs.in_flight(conn).run_id == run_id

    # dead holder → not something to join, something to report as stopped
    assert pull_runs.in_flight(conn, now=time.time() + pull_runs.LEASE_TTL + 1) is None

    pull_runs.close_run(conn, run_id)
    assert pull_runs.in_flight(conn) is None


def test_a_queued_pick_carries_the_window_the_call_that_queued_it_was_given(conn):
    """`x_lookback` was an argument to a call that is gone by the time the loop reaches them.
    Re-deriving it would silently hand somebody the adapter's 183-day default after they asked
    for two years."""
    run_id = pull_runs.open_run(conn, kind="ingest", picks=[
        {"canonical_id": "c:andrej", "name": "Andrej", "window": "2024-01-01T00:00:00+00:00"}])
    assert pull_runs.oracles_for(conn, run_id)[0].window == "2024-01-01T00:00:00+00:00"

    pull_runs.add_picks(conn, run_id, [{"canonical_id": "c:dean", "name": "Dean",
                                        "window": "2023-05-05T00:00:00+00:00"}])
    assert pull_runs.oracles_for(conn, run_id)[1].window == "2023-05-05T00:00:00+00:00"


def test_unfinished_for_is_what_a_running_loop_re_reads(conn):
    run_id = pull_runs.open_run(conn, kind="ingest", picks=PICKS)
    pull_runs.finish_oracle(conn, run_id, "c:andrej", {"atoms_added": 1})
    assert [o.canonical_id for o in pull_runs.unfinished_for(conn, run_id)] == ["c:dean", "c:soren"]
