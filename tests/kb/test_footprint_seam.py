"""What is left of the merge seam between the footprint adapters and onboarding's callers.

A first pull does not sync a whole site: it begins the source's archive walk
(`first_pull`, `archive/run.py`), and a walk's failures are its units' own (`test_resumable_archives`).
"""
from __future__ import annotations


# ── a thin meter STARTS the walk ─────────────────────────────
#
# No reservation refuses an X walk because the bucket is below a fixed floor: durable partial walks
# removed its premise. The only refusal is
# `x_rate.refuse_if_spent`, which refuses a request x.com has already said it will 429.

def test_the_reserve_exists_in_neither_copy():
    """One bucket, one answer. `_x_budget_spent` and `oracle_refresh._budget_spent` were the same
    function written twice, sharing one justification — so removing one and keeping the other
    would have recreated the split. They went together."""
    from pipeline.oracles import admit, coverage

    assert not hasattr(admit, "_x_budget_spent")
    assert not hasattr(coverage, "_budget_spent")
    assert not hasattr(coverage, "BACKFILL_MIN_BUDGET")
