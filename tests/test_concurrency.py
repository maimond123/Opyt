"""AdaptiveSemaphore — the AIMD concurrency gate (ARC-1 Phase 2).

Pins the two properties everything downstream trusts: (1) it NEVER admits more than `limit`
threads at once (the safety contract the transport seams rely on), and (2) the AIMD policy moves
the limit correctly — additive-increase up to max, multiplicative-decrease (halve) down to min —
with no permit leak on an exception. Pure in-process threads, no network.
"""
from __future__ import annotations

import threading
import time

from pipeline.concurrency import AdaptiveSemaphore


def _hammer(gate: AdaptiveSemaphore, n_threads: int, hold: float = 0.01) -> int:
    """Run `n_threads` through the gate concurrently; return the PEAK observed concurrency."""
    peak = {"cur": 0, "max": 0}
    lock = threading.Lock()

    def worker():
        with gate:
            with lock:
                peak["cur"] += 1
                peak["max"] = max(peak["max"], peak["cur"])
            time.sleep(hold)
            with lock:
                peak["cur"] -= 1

    ts = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return peak["max"]


def test_never_admits_more_than_limit():
    gate = AdaptiveSemaphore(4, max_permits=4)      # max==start → no increase; fixed at 4
    peak = _hammer(gate, n_threads=40)
    assert peak <= 4                                # the safety contract: never over-admit
    assert peak >= 2                                # ...and real concurrency actually happened
    assert gate.in_use == 0                         # every permit was returned


def test_decrease_halves_down_to_floor():
    gate = AdaptiveSemaphore(8, min_permits=2)
    assert gate.limit == 8
    gate.decrease(); assert gate.limit == 4          # 8 → 4
    gate.decrease(); assert gate.limit == 2          # 4 → 2
    gate.decrease(); assert gate.limit == 2          # floored at min, never below


def test_record_success_increases_up_to_max():
    gate = AdaptiveSemaphore(2, max_permits=4, increase_after=1)
    gate.record_success(); assert gate.limit == 3    # one clean wave → +1
    gate.record_success(); assert gate.limit == 4    # → +1
    gate.record_success(); assert gate.limit == 4    # capped at max, never above


def test_min_cannot_exceed_start():
    # A nonsensical min > start must be clamped, else decrease() would paradoxically RAISE the limit.
    gate = AdaptiveSemaphore(3, min_permits=10)
    gate.decrease()
    assert gate.limit == 3                            # max(min=3, 3//2=1) == 3, not 10


def test_no_permit_leak_on_exception():
    gate = AdaptiveSemaphore(1)
    try:
        with gate:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert gate.in_use == 0                           # __exit__ released despite the exception
    with gate:                                        # and the single permit is reusable (no deadlock)
        assert gate.in_use == 1


def test_limit_shrinks_live_and_throttles_new_acquires():
    """A decrease mid-flight must actually throttle: after halving 8→...→1, at most 1 runs at once."""
    gate = AdaptiveSemaphore(8, min_permits=1)
    for _ in range(3):
        gate.decrease()                               # 8 → 4 → 2 → 1
    assert gate.limit == 1
    assert _hammer(gate, n_threads=20) == 1           # now strictly serialized
