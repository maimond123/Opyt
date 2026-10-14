"""One test per row of the GitHub search cache decision."""
from __future__ import annotations

import pytest

from pipeline.platforms.github.search import cache_action

_TTL = 7 * 24 * 3600
_NOW = 1_800_000_000.0


@pytest.mark.parametrize("entry", [None, {}])
def test_cache_action_fetches_without_an_entry(entry):
    assert cache_action(entry, _NOW, _TTL) is None


def test_cache_action_fetches_an_entry_at_least_ttl_old():
    assert cache_action({"at": _NOW - _TTL, "data": [{"full_name": "a/b"}]}, _NOW, _TTL) is None


@pytest.mark.parametrize("entry", [{"at": _NOW - 60, "data": None}, {"at": _NOW - 60}])
def test_cache_action_fetches_a_fresh_entry_holding_no_data(entry):
    assert cache_action(entry, _NOW, _TTL) is None


def test_cache_action_uses_a_fresh_entry_with_data():
    assert cache_action({"at": _NOW - 60, "data": []}, _NOW, _TTL) == []
