"""`windows.since`, one test per row of the decision table: values in, an instant (or a refusal) out."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pipeline.oracles import windows

NOW = datetime(2026, 10, 6, tzinfo=timezone.utc)


@pytest.mark.parametrize("kind", ["x", "web", "scholar"])
@pytest.mark.parametrize("preset", ["", None])
def test_no_preset_is_none(kind, preset):
    assert windows.since(kind, preset, NOW) is None


@pytest.mark.parametrize("preset,days", [("6mo", 183), ("1yr", 365), ("2yr", 730)])
def test_x_presets_are_days_before_now(preset, days):
    assert windows.since("x", preset, NOW) == NOW - timedelta(days=days)


def test_x_since_last_refuses():
    with pytest.raises(ValueError, match="per-Oracle"):
        windows.since("x", windows.X_SINCE_LAST, NOW)


@pytest.mark.parametrize("preset", ["1y", "all"])
def test_an_unknown_x_preset_refuses(preset):
    with pytest.raises(ValueError, match="unknown x_lookback"):
        windows.since("x", preset, NOW)


@pytest.mark.parametrize("kind,preset,days", [("web", "1yr", 365), ("web", "5yr", 1825),
                                              ("scholar", "2yr", 730), ("scholar", "10yr", 3653)])
def test_web_and_scholar_presets_are_days_before_now(kind, preset, days):
    assert windows.since(kind, preset, NOW) == NOW - timedelta(days=days)


@pytest.mark.parametrize("kind", ["web", "scholar"])
@pytest.mark.parametrize("preset", ["all", "nonsense"])
def test_web_and_scholar_all_or_unknown_widen_to_none(kind, preset):
    assert windows.since(kind, preset, NOW) is None
