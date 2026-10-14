"""The concurrent ingest harness end to end.

`run_concurrent` in isolation — every item consumed exactly once, the bounded submission window
never balloons, and a producer or consumer that raises skips only its own item (the fail-safe
contract), never aborting the run. Also asserts the image-cache save survives concurrent writers
(the "dict changed size during iteration" crash the guard exists to prevent).
"""
from __future__ import annotations

import threading
import time

from pipeline.kb.ingest_common import run_concurrent


# ── run_concurrent in isolation ──────────────────────────────────────────────

def test_consumes_every_item_exactly_once():
    got: list[int] = []                                  # consumer is single-threaded → plain append ok
    run_concurrent(range(200), lambda i: i * 10, got.append, workers=8)
    assert sorted(got) == [i * 10 for i in range(200)]   # all present, none duplicated or dropped


def test_source_is_advanced_only_on_the_calling_thread():
    """A contract of run_concurrent, not an incidental property.

    Callers rely on it to keep rate-limited work OUT of the pool. A Python generator runs on
    whichever thread calls `next()` on it and pauses at each `yield`, so a caller that puts an HTTP
    fetch inside its generator gets a serial fetch for free — provided this loop keeps advancing the
    source on the calling thread, at both the priming loop and the refill point. The blog and
    Substack footprint ingesters depend on exactly that: their per-post fetch sits in the generator
    so the scraped host's request rate is unchanged while the paid LLM stages fan out.

    Move this advance onto a worker thread and that protection vanishes with no other symptom.
    Parallel scraping invites a Cloudflare soft-ban, which arrives as a challenge page rather than a
    clean 429 — so the AIMD gate cannot react to it and nothing fails until a host has already
    blocked us. Asserted HERE, where the guarantee lives, because a refactor of this function is
    exactly when it would break, and its four callers' own tests are the wrong place to find out."""
    main = threading.current_thread().name
    advanced_on: set[str] = set()
    worked_on: set[str] = set()

    def source():
        for i in range(20):
            advanced_on.add(threading.current_thread().name)
            yield i

    def work(i):
        worked_on.add(threading.current_thread().name)
        time.sleep(0.01)                                 # overlap, so the pool really is fanning out
        return i

    run_concurrent(source(), work, lambda r: None, workers=4)

    assert advanced_on == {main}, f"source advanced off the calling thread: {advanced_on}"
    # Negative control: without this, the assertion above would also pass if the pool had silently
    # collapsed to serial execution — in which case it proves nothing about thread placement.
    assert len(worked_on) > 1, "work never left the calling thread; the assertion above is vacuous"


def test_source_error_drains_produced_prefix_not_aborts():
    """A transient source error mid-iteration (e.g. an X scrape DependencyError on page 2) must stop
    intake but still consume everything already fetched — not discard the in-flight window."""
    def flaky_source():
        for i in range(30):
            yield i
        raise RuntimeError("scrape DependencyError on the next page")   # mid-pagination transient

    got: list[int] = []
    report = run_concurrent(flaky_source(), lambda i: i, got.append, workers=4)
    assert sorted(got) == list(range(30))            # the fetched prefix survived; run didn't abort
    # The caller can tell this apart from a source that finished: `source_error` is set, whereas
    # the counters alone would just look short.
    assert isinstance(report["source_error"], RuntimeError)


def test_a_finished_source_reports_no_error():
    """The other half of the pair, and the reason the first one matters. These two runs are
    indistinguishable in every counter a caller keeps; `source_error` is the only thing that
    separates them."""
    report = run_concurrent(range(30), lambda i: i, [].append, workers=4)
    assert report["source_error"] is None


def test_a_source_that_dies_before_yielding_is_still_reported():
    """The case no caller could detect from its own counters: `producer_failed = dispatched -
    consumed` is 0 when an iterator raises before its first yield, since nothing was dispatched,
    so the failure would be invisible. A dead X session has this shape."""
    def dead_source():
        raise RuntimeError("cookie expired")
        yield                                        # unreachable; makes this a generator

    report = run_concurrent(dead_source(), lambda i: i, [].append, workers=4)
    assert isinstance(report["source_error"], RuntimeError)
    assert report["producer_failed"] == 0            # nothing was ever dispatched to fail


def test_producer_error_skips_only_that_item():
    def work(i):
        if i == 42:
            raise ValueError("bad producer")
        return i
    got: list[int] = []
    report = run_concurrent(range(100), work, got.append, workers=8)
    assert 42 not in got and len(got) == 99              # one skipped, the run finished
    assert report["producer_failed"] == 1


def test_a_producer_returning_none_is_not_a_failure():
    """`dispatched - consumed` cannot tell these two apart: a `work_fn` that raised and one that legitimately returned nothing both leave the
    consumer un-called. Counting the raise where it happens can."""
    report = run_concurrent(range(10), lambda i: None, [].append, workers=4)
    assert report["producer_failed"] == 0


def test_consumer_error_skips_only_that_result():
    def consume(i):
        if i == 7:
            raise ValueError("bad consumer")
        got.append(i)
    got: list[int] = []
    report = run_concurrent(range(50), lambda i: i, consume, workers=8)
    assert 7 not in got and len(got) == 49
    assert report["consumer_failed"] == 1


def test_submission_window_stays_bounded():
    """Outstanding results (produced but not yet consumed) never exceed `inflight`, no matter how
    many items — this is the RAM bound (a rendered atom can be ~500 KB, so an unbounded backlog is a
    real hazard). A deliberately slow consumer maximizes the backlog pressure."""
    import time
    lock = threading.Lock()
    live = {"n": 0, "max": 0}

    def work(i):
        with lock:
            live["n"] += 1                               # result now outstanding
            live["max"] = max(live["max"], live["n"])
        return i

    def consume(i):
        time.sleep(0.001)                                # slow consumer → backlog wants to grow
        with lock:
            live["n"] -= 1

    run_concurrent(range(500), work, consume, workers=4, inflight=8)
    # Bound is inflight + 1: we top up before consuming the popped result, so one extra is briefly
    # outstanding. The point stands — it's bounded by the WINDOW, not by len(items)=500.
    assert live["max"] <= 9


def test_image_cache_save_survives_concurrent_writers(kb_home):
    """A producer adding a new key via cache_put while save_image_cache dumps the dict must not raise
    'dictionary changed size during iteration' — the exact crash the _CACHE_LOCK guards against."""
    import time
    from pipeline.image_cache import cache_put, save_image_cache

    cache: dict = {}
    stop = {"v": False}
    errors: list = []

    def writer():
        i = 0
        while not stop["v"]:
            cache_put(cache, f"u{i}", "x" * 40)           # keep adding new keys → dict resizes
            i += 1
            time.sleep(0.0003)                            # throttle: dict stays small, saves stay fast

    w = threading.Thread(target=writer)
    w.start()
    try:
        for _ in range(40):
            save_image_cache(kb_home, cache)              # snapshots under the lock, dumps outside it
    except Exception as e:                                 # pragma: no cover - the bug would land here
        errors.append(e)
    finally:
        stop["v"] = True
        w.join()
    assert not errors


def test_independent_workers_merge_image_results_without_losing_keys(kb_home):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from pipeline.image_cache import load_image_cache, save_image_cache
    barrier = Barrier(4)

    def finish(i):
        local = load_image_cache(kb_home)
        local[str(i)] = f"image {i}"
        barrier.wait(timeout=5)
        save_image_cache(kb_home, local)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(finish, range(4)))
    assert load_image_cache(kb_home) == {str(i): f"image {i}" for i in range(4)}
