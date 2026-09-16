"""
tests/gateway/test_children.py — the pool's two jobs: name a home safely, and end a child
without cutting a live request.

The reaper tests build `Child` objects around a stub process rather than spawning real ones.
That keeps the ordering rules (route deleted before the signal, in-flight requests spared)
testable in milliseconds, and it needs no seam in the production class — a dataclass is
constructible from a test by definition.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from gateway.children import BadSubject, Child, ChildPool, home_for


class StubProc:
    """The parts of `asyncio.subprocess.Process` the pool actually touches."""

    def __init__(self, pid: int = 1234) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.signals: list[str] = []

    def terminate(self) -> None:
        self.signals.append("TERM")
        self.returncode = -15

    def kill(self) -> None:
        self.signals.append("KILL")
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode if self.returncode is not None else 0


def _child(pool: ChildPool, subject: str, *, idle: float, inflight: int = 0) -> Child:
    child = Child(subject=subject, home=pool.homes_root / subject, port=1,
                  proc=StubProc(),  # type: ignore[arg-type]
                  last_seen=time.monotonic() - idle, inflight=inflight)
    pool._children[subject] = child
    return child


# ── Naming a home ───────────────────────────────────────────────────────────────────────────

def test_home_for_accepts_a_google_subject(tmp_path):
    assert home_for(tmp_path, "104729183746152938471") == tmp_path / "104729183746152938471"


@pytest.mark.parametrize("subject", ["..", ".", "a/b", "../etc", "a.b", "", "x" * 129,
                                     "a\x00b", "a b"])
def test_home_for_rejects_anything_that_could_escape_the_root(tmp_path, subject):
    """The subject becomes a directory name, so this is the trust boundary.

    Dots are rejected outright rather than filtered, which is what makes traversal
    unrepresentable instead of merely blocked.
    """
    with pytest.raises(BadSubject):
        home_for(tmp_path, subject)


# ── Reaping ─────────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_reap_ends_an_idle_child(tmp_path):
    pool = ChildPool(tmp_path, idle_seconds=60)
    child = _child(pool, "u1", idle=120)

    assert await pool.reap_once() == ["u1"]
    assert child.proc.signals == ["TERM"]        # type: ignore[attr-defined]
    assert "u1" not in pool._children


@pytest.mark.anyio
async def test_reap_spares_a_child_with_a_request_in_flight(tmp_path):
    """A streamed response can outlive the handler by minutes, so idleness alone is not
    grounds to kill. `inflight` is what the proxy holds open for the length of the stream."""
    pool = ChildPool(tmp_path, idle_seconds=60)
    child = _child(pool, "u1", idle=999, inflight=1)

    assert await pool.reap_once() == []
    assert child.proc.signals == []              # type: ignore[attr-defined]
    assert pool._children["u1"] is child


@pytest.mark.anyio
async def test_reap_spares_a_child_used_recently(tmp_path):
    pool = ChildPool(tmp_path, idle_seconds=60)
    _child(pool, "u1", idle=5)
    assert await pool.reap_once() == []


@pytest.mark.anyio
async def test_release_resets_the_idle_clock(tmp_path):
    pool = ChildPool(tmp_path, idle_seconds=60)
    child = _child(pool, "u1", idle=999, inflight=1)

    pool.release(child)

    assert child.inflight == 0
    assert await pool.reap_once() == []          # released, not stale

@pytest.mark.anyio
async def test_route_is_deleted_before_the_signal(tmp_path):
    """A request arriving between the signal and the exit must not be proxied into a dying
    process. The only way to guarantee that is to stop routing first."""
    pool = ChildPool(tmp_path, idle_seconds=60)
    order: list[str] = []

    class Recording(StubProc):
        def terminate(self) -> None:
            order.append("routable" if "u1" in pool._children else "unroutable")
            super().terminate()

    child = _child(pool, "u1", idle=120)
    child.proc = Recording()  # type: ignore[assignment]

    await pool.reap_once()

    assert order == ["unroutable"]


@pytest.mark.anyio
async def test_shutdown_ends_every_child(tmp_path):
    pool = ChildPool(tmp_path, idle_seconds=9999)
    a, b = _child(pool, "u1", idle=0), _child(pool, "u2", idle=0)

    await pool.shutdown()

    assert pool._children == {}
    assert a.proc.signals == ["TERM"] and b.proc.signals == ["TERM"]  # type: ignore[attr-defined]


# ── One real child ──────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_acquire_spawns_one_child_and_reuses_it(tmp_path):
    """The end-to-end spawn: a real `opyt-mcp --http` in a real (empty) home.

    Also pins the per-subject lock's job. A client's opening burst (`initialize`,
    `notifications/initialized`, `tools/list`) arrives within milliseconds, and without the
    lock each would miss the table and spawn its own child.
    """
    pool = ChildPool(tmp_path, idle_seconds=9999, spawn_timeout=60)
    try:
        children = await asyncio.gather(*(pool.acquire("42") for _ in range(3)))

        assert len({id(c) for c in children}) == 1, "the burst spawned more than one child"
        child = children[0]
        assert child.alive and child.inflight == 3
        assert child.home == tmp_path / "42"
        assert (child.home / "settings.yaml").exists(), "the child never bootstrapped its home"

        for c in children:
            pool.release(c)
        assert child.inflight == 0
    finally:
        await pool.shutdown()


@pytest.mark.anyio
async def test_a_crashed_child_is_replaced_rather_than_handed_out(tmp_path):
    pool = ChildPool(tmp_path, idle_seconds=9999, spawn_timeout=60)
    try:
        first = await pool.acquire("42")
        pool.release(first)
        first.proc.kill()
        await first.proc.wait()

        second = await pool.acquire("42")

        assert second is not first and second.alive
        pool.release(second)
    finally:
        await pool.shutdown()


@pytest.mark.anyio
async def test_a_child_leads_its_own_process_group(tmp_path):
    """`_terminate` signals a GROUP, and that group only exists if the child leads a session."""
    import os

    pool = ChildPool(tmp_path, idle_seconds=9999, spawn_timeout=60)
    try:
        child = await pool.acquire("111")
        assert os.getpgid(child.proc.pid) == child.proc.pid
        assert os.getpgid(child.proc.pid) != os.getpgid(0)
        pool.release(child)
    finally:
        await pool.shutdown()


@pytest.mark.anyio
async def test_terminating_a_child_takes_the_processes_it_started_with_it(tmp_path):
    """The measured defect: a hosted child's Chrome outliving the child that started it.

    SIGTERM's default disposition ends an interpreter without unwinding, so the context manager
    that owns a browser never reaches its `finally`. On 2026-09-08 that left two headless
    Chromes reparented to init, ten hours old, still holding the profile's `SingletonLock`, and
    the next real sign-in served a desktop with nothing drawn on it. A sleeper stands in for
    Chrome here; what is pinned is that ending a child ends what the child started.
    """
    import os
    import sys

    child_code = (
        "import subprocess, sys, time\n"
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        "print(g.pid, flush=True)\n"
        "time.sleep(300)\n"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", child_code,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    grandchild = int((await proc.stdout.readline()).strip())
    assert os.getpgid(grandchild) == proc.pid       # it inherited the child's group

    from gateway.children import _terminate
    await _terminate(proc)

    for _ in range(60):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.05)
    os.kill(grandchild, 9)
    pytest.fail("the grandchild outlived the child the gateway terminated")
