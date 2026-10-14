"""Several writers on one store at once, as threads in one process and as separate processes.

Each writer writes pages the way a scraper does: the page's atoms through `store_atom` (one
`write_tx` per atom and its body), then the page's done-state in a `write_tx` of its own. Every
page must land whole, with no lock error escaping, and each writer reports the longest page it
took, which under contention is mostly time spent waiting for the lock.

The last test is the starvation case: SQLite has no writer queue, so a writer
committing back to back keeps a polling writer out for longer than `busy_timeout`.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from pipeline.kb import schema, writes
from pipeline.kb.ingest_common import store_atom

REPO = Path(__file__).resolve().parents[2]
PAGES, ATOMS = 8, 6


def _atom(atom_id: str) -> dict:
    return dict(atom_id=atom_id, source_type="x", what_kind="opinion", who_id="x:u",
                when_ts="2024-01-01", when_precision="day", about_entities=[],
                source_url=f"https://x.com/u/status/{atom_id}", raw_hash="h",
                description="d", payload={}, entry_mode="user-saved")


def write_pages(path: str, writer: str, pages: int = PAGES, atoms: int = ATOMS) -> float:
    """One writer's whole run. Returns the longest page in seconds. Module-level so a child
    process can import it."""
    conn = schema.connect(path)
    worst = 0.0
    for page in range(pages):
        started = time.monotonic()
        for i in range(atoms):
            store_atom(conn, _atom(f"{writer}-{page}-{i}"), f"{writer} page {page} post {i} " * 40)
        with writes.write_tx(conn, site="page"):
            conn.execute("INSERT INTO pages_done VALUES (?, ?)", (writer, page))
        worst = max(worst, time.monotonic() - started)
    conn.close()
    return worst


@pytest.fixture()
def store(kb_home):
    """A store already through its migrations, so the writers' own `connect()` takes no lock and
    every write they make is a page."""
    path = kb_home / "opyt.db"
    c = schema.connect(path)
    c.execute("CREATE TABLE pages_done (writer TEXT, page INTEGER, PRIMARY KEY (writer, page))")
    c.commit()
    c.close()
    return path


def _run_threads(path, n: int) -> list[float]:
    go = threading.Barrier(n)
    worst: dict[int, float] = {}
    errors: list[BaseException] = []

    def run(k):
        try:
            go.wait()
            worst[k] = write_pages(str(path), f"t{k}")
        except BaseException as exc:          # surfaced below, with the writer that raised
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(k,)) for k in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    return list(worst.values())


_CHILD = """
import json, sys
from tests.kb.test_write_concurrency import write_pages
print("ready", flush=True)
sys.stdin.readline()
print(json.dumps(write_pages(sys.argv[1], sys.argv[2])), flush=True)
"""


def _run_processes(path, n: int) -> list[float]:
    procs = [subprocess.Popen([sys.executable, "-c", _CHILD, str(path), f"p{k}"], cwd=REPO,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
             for k in range(n)]
    for p in procs:                           # every child imported and waiting: start together
        assert p.stdout.readline().strip() == "ready", p.stderr.read()
    for p in procs:
        p.stdin.write("go\n")
        p.stdin.flush()
    worst = []
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
        worst.append(json.loads(out.strip().splitlines()[-1]))
    return worst


@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("kind", ["threads", "processes"])
def test_writers_at_once_all_land_whole(store, kind, n):
    worst = (_run_threads if kind == "threads" else _run_processes)(store, n)

    c = schema.connect(store, read_only=True)
    try:
        assert c.execute("SELECT COUNT(*) FROM pages_done").fetchone()[0] == n * PAGES
        assert c.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == n * PAGES * ATOMS
        # Every atom landed with its body, and every body in the index.
        assert c.execute("SELECT COUNT(*) FROM atoms WHERE atom_id NOT IN "
                         "(SELECT atom_id FROM bodies)").fetchone()[0] == 0
    finally:
        c.close()
    # The index check is an INSERT command, so it needs a writable handle; it writes nothing.
    check = schema.connect(store)
    try:
        check.execute("INSERT INTO bodies_fts(bodies_fts, rank) VALUES ('integrity-check', 1)")
    finally:
        check.close()
    assert len(worst) == n
    assert max(worst) < writes.LOCK_DEADLINE_S, f"longest page per writer: {worst}"


_BUSY = """
import sys, time
from pipeline.kb import schema, writes
c = schema.connect(sys.argv[1])
stop, n = time.monotonic() + float(sys.argv[2]), 0
while time.monotonic() < stop:
    with writes.write_tx(c, site="busy"):
        c.execute("INSERT INTO pages_done VALUES ('busy', ?)", (n,))
    n += 1
    if n == 20:
        print("writing", flush=True)
"""


def test_a_writer_kept_out_past_busy_timeout_still_lands(store):
    """A busy writer in another process commits back to back and leaves gaps of
    microseconds; a writer polling for the lock almost never lands in one, so with `busy_timeout`
    alone it fails. Here `busy_timeout` is 100 ms and the busy writer runs for 1.5 s (measured: the
    light writer is refused 3–4 times and waits up to ~1.2 s). `write_tx` keeps asking, and the
    light writer's write lands."""
    busy = subprocess.Popen([sys.executable, "-c", _BUSY, str(store), "1.5"], cwd=REPO,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    light = schema.connect(store)
    try:
        assert busy.stdout.readline().strip() == "writing", busy.stderr.read()
        light.execute("PRAGMA busy_timeout=100")
        with writes.write_tx(light, site="light"):
            light.execute("INSERT INTO pages_done VALUES ('light', 0)")
    finally:
        _, err = busy.communicate(timeout=30)
    assert busy.returncode == 0, err
    assert light.execute("SELECT COUNT(*) FROM pages_done WHERE writer='light'").fetchone()[0] == 1
    light.close()
