"""Concurrency bounds, host admission, and policy settlement on the owning thread."""
import sqlite3
from dataclasses import replace
from threading import Event, Lock, Thread, get_ident

import pytest

from pipeline.worker.quota_loop.retries import Outages, PLATFORM_PAUSES_S, Wait

from pipeline.host_access import Hosts, request, scope
from pipeline.kb import schema
from pipeline.worker.quota_loop.choose import Picked
from pipeline.worker.quota_loop.loop import QuotaLoop
from pipeline.worker.quota_loop.pool import WebPool, available, holds
from pipeline.worker.quota_loop.source import YOURS, NEW_POSTS, Outcome, Unit


def test_busy_host_does_not_hide_another_host():
    assert not available("a", ("source", "2"), {("source", "1"): "a"})
    assert available("b", ("source", "2"), {("source", "1"): "a"})
    assert not available("b", ("source", "1"), {("source", "1"): "a"})


@pytest.mark.parametrize("pause, held", [
    (None, False),                                  # no pause
    (Wait(100.0, "paused", "x_outage"), True),      # before its end
    (Wait(50.0, "paused", "x_outage"), False),      # its end has passed
    (Wait(None, "paused", "x_session"), True),      # until woken
])
def test_when_the_pools_pause_holds(pause, held):
    assert holds(pause, 50.0) is held


def test_four_hosts_run_and_same_host_waits_with_separate_connections(conn, tmp_path):
    path = tmp_path / "pool.sqlite"
    primary = schema.connect(path)
    primary.close()
    entered, release, done = Event(), Event(), Event()
    guard = Lock()
    running, peaks, connections, fetched_on, recorded_on = set(), [], set(), set(), set()
    finished = set()
    units = [Unit("a1", host="a"), Unit("a2", host="a"),
             Unit("b", host="b"), Unit("c", host="c"), Unit("d", host="d")]

    class Source:
        name, quota, klass, floor = "documents", "web", YOURS, 0

        def candidates(self, conn, now):
            return [u for u in units if u.key not in finished]

        def next_due(self, conn, now):
            return None

        def fetch(self, conn, unit, ctx):
            with guard:
                assert unit.host not in running
                running.add(unit.host)
                peaks.append(len(running))
                connections.add(id(conn))
                fetched_on.add(get_ident())
                if len(running) == 4:
                    entered.set()
            assert release.wait(5)
            with guard:
                running.remove(unit.host)
            return unit.key

        def record(self, conn, unit, fetched):
            recorded_on.add(get_ident())
            finished.add(fetched)
            if len(finished) == 5:
                pool.stop()
                done.set()
            return Outcome("saved")

    pool = QuotaLoop("web", [Source()], executor=WebPool(), connect=lambda: schema.connect(path), outages=Outages(PLATFORM_PAUSES_S))
    errors = []

    def execute():
        db = schema.connect(path)
        try:
            pool.executor.execute(pool, db)
        except BaseException as exc:
            errors.append(exc)
        finally:
            db.close()

    thread = Thread(target=execute)
    thread.start()
    try:
        assert entered.wait(5), "four independent hosts must start before any completes"
        assert peaks[:4] == [1, 2, 3, 4]
    finally:
        release.set()
        assert done.wait(5)
        pool.stop()
        thread.join(5)
    assert not errors and not thread.is_alive()
    assert len(connections) >= 4
    assert recorded_on == {thread.ident} and not (recorded_on & fetched_on)
    assert max(peaks) == 4 and len(finished) == 5


def test_pool_turns_count_at_admission_even_if_completion_order_differs(conn):
    class Source:
        quota, floor = "web", 0
        def __init__(self, name, klass):
            self.name, self.klass = name, klass
        def candidates(self, conn, now):
            return [Unit(str(i), host=f"{self.name}{i}") for i in range(20)]
    yours, new = Source("yours", YOURS), Source("new", NEW_POSTS)
    pool = QuotaLoop("web", [yours, new], executor=WebPool(), outages=Outages(PLATFORM_PAUSES_S), connect=schema.connect)
    choices = []
    for _ in range(20):
        source, unit = pool.rotation.pick(pool.sources, lambda s: pool.executor.units(s, conn, 0), None, pool.strikes, 0)
        choices.append(source.name)
        pool.executor._active[(source.name, unit.key)] = unit.host
        pool.rotation.turn(source)
    assert choices[9] == choices[19] == "new" and choices.count("new") == 2
    before = pool.rotation._since_new_posts
    pool._ran(Picked(yours, Unit("19")), "saved")
    assert pool.rotation._since_new_posts == before


def test_redirects_and_secondary_reads_share_destination_lock():
    hosts = Hosts()
    target_entered, release_target, redirect_seen, finished = Event(), Event(), Event(), Event()
    active = 0
    guard = Lock()

    class Response:
        def __init__(self, status, headers=None):
            self.status_code, self.headers = status, headers or {}
        def close(self):
            pass

    def send(url, **kw):
        nonlocal active
        assert kw["allow_redirects"] is False
        if "origin" in url:
            redirect_seen.set()
            return Response(302, {"Location": "https://target/page"})
        with guard:
            active += 1
            assert active == 1
        target_entered.set()
        assert release_target.wait(5)
        with guard:
            active -= 1
        return Response(200)

    def read(url):
        with scope(hosts):
            request(send, url)
        finished.set()

    first = Thread(target=read, args=("https://target/pdf",))
    second = Thread(target=read, args=("https://origin/post",))
    first.start()
    assert target_entered.wait(5)
    second.start()
    assert redirect_seen.wait(5)
    assert not finished.is_set()
    release_target.set()
    first.join(5)
    second.join(5)
    assert not first.is_alive() and not second.is_alive()


def test_streaming_probe_holds_host_until_response_closes():
    hosts = Hosts()
    class Response:
        status_code, headers = 200, {}
        def close(self):
            pass
    with scope(hosts):
        response = request(lambda *a, **kw: Response(), "https://a/pdf", stream=True)
        assert not hosts.lock("https://a/another").acquire(blocking=False)
        response.close()
        response.close()
        assert hosts.lock("https://a/another").acquire(blocking=False)
        hosts.lock("https://a/another").release()


def test_a_fetch_whose_connection_will_not_open_settles_with_its_unit(conn):
    """The connection is opened on the worker thread, after admission, so a failure there is the
    fetch's error and settles with its unit (struck, here) instead of escaping the pool."""
    class Source:
        name, quota, klass, floor = "documents", "web", YOURS, 0

        def candidates(self, _conn, _now):
            return [Unit("a", host="a")]

        def next_due(self, _conn, _now):
            return None

        def fetch(self, _conn, _unit, _ctx):
            raise AssertionError("fetched without a connection")

    def refuse():
        pool.stop()
        raise sqlite3.OperationalError("unable to open database file")

    pool = QuotaLoop("web", [Source()], executor=WebPool(), connect=refuse, outages=Outages(PLATFORM_PAUSES_S))
    pool.executor.execute(pool, conn)
    assert ("documents", "a") in pool.strikes
