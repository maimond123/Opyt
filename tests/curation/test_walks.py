"""`walks.next_step`: which kind of unit a walk owes next, one test per row of its table."""

import pytest

from pipeline.curation import listings, walks


@pytest.mark.parametrize("state, ends, step", [
    (None, True, walks.START),
    (listings.COMPLETE, True, walks.START),
    (None, False, None),
    (listings.COMPLETE, False, None),
    (listings.READY, True, walks.FINISH),
    (listings.READY, False, None),
    (listings.WALKING, True, walks.PAGES),
    (listings.WALKING, False, walks.PAGES),
])
def test_next_step_table(state, ends, step):
    assert walks.next_step(state, ends) == step

