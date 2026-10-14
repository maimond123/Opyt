"""`windows.first_pulls_done`: when an Oracle's starting pairs have all had their first pull, so
discovery's links stage may read their posts."""
from __future__ import annotations

import pytest

from pipeline.oracles import asks, registry, windows
from pipeline.worker.quota_loop import retries

from ._support import NOW, days

ASK = asks.Ask(canonical_id="x:user:1", x_from=days(90).isoformat(), web_from=windows.ALL_TIME,
               scholar_from=windows.ALL_TIME, scholar_topics=None,
               asked_at=days(1).isoformat(), position=1, discovered_at=None, rediscover=False)


def _x(**kw):
    return registry.SourceRow("x:user:1", "x", "carol", **kw)


def _web(stype="substack", **kw):
    return registry.SourceRow("x:user:1", stype, "https://carol.substack.com", **kw)


REACHED = dict(last_pulled_at=NOW.isoformat(), covered_from=days(400).isoformat())


@pytest.mark.parametrize("rows, done", [
    ([], True),                                                   # nothing to wait for
    ([_x()], False),                                              # never pulled
    ([_x(last_pulled_at=NOW.isoformat(), covered_from=days(10).isoformat())], False),  # short
    ([_x(**REACHED)], True),                                      # reached its window
    ([_x(failures=retries.GIVE_UP_AFTER)], True),                 # gave up: finished
    ([_x(last_status=retries.UNSAVED, failures=retries.STUCK_AFTER)], True),   # stuck: finished
    ([_x(**REACHED), _web()], False),                             # the archive is still owed
    ([_x(**REACHED), _web(last_pulled_at=NOW.isoformat())], True),  # pulled after the ask
    ([_x(**REACHED), _web(last_pulled_at=days(5).isoformat())], False),  # pulled before it
    ([_x(**REACHED), _web("openalex", last_pulled_at=NOW.isoformat())], True),
])
def test_first_pulls_done(rows, done):
    assert windows.first_pulls_done(rows, ASK, NOW) is done


def test_an_archive_given_up_on_every_post_it_has_left_is_finished():
    """One Substack post refused and given up on must not hold its pair open for good."""
    rows = [_x(**REACHED), _web()]
    assert windows.first_pulls_done(rows, ASK, NOW) is False
    exhausted = frozenset({("substack", "https://carol.substack.com")})
    assert windows.first_pulls_done(rows, ASK, NOW, exhausted) is True


def test_a_kind_the_ask_left_out_does_not_hold_discovery_back():
    ask = asks.Ask(**{**ASK.__dict__, "scholar_from": None})
    assert windows.first_pulls_done([_x(**REACHED), _web("openalex")], ask, NOW) is True
