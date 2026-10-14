"""`admitting.settle` — the admissions' give-up rule, one test per row of its table."""
from __future__ import annotations

import pytest

from pipeline.frontier import admitting as fa


@pytest.mark.parametrize("attempts", [None, 0, 3])
def test_a_success_is_written_as_is_and_never_counts(attempts):
    assert fa.settle("materialized", attempts) == ("materialized", attempts or 0, False)


def test_a_failure_below_the_cap_stays_kept_and_counts_one():
    assert fa.settle("kept", fa.ADMIT_MAX_ATTEMPTS - 2) == ("kept", fa.ADMIT_MAX_ATTEMPTS - 1, False)


def test_the_failure_that_reaches_the_cap_is_rejected():
    assert fa.settle("kept", fa.ADMIT_MAX_ATTEMPTS - 1) == ("rejected", fa.ADMIT_MAX_ATTEMPTS, True)


def test_a_rejection_stays_rejected_and_counts_one():
    assert fa.settle("rejected", None) == ("rejected", 1, False)


@pytest.mark.parametrize("attempts", [None, 0, 4])
def test_a_busy_store_stays_kept_and_is_never_charged(attempts):
    assert fa.settle("locked", attempts) == ("kept", attempts or 0, False)
