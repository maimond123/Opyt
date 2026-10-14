import pytest

from pipeline.platforms.x.rate import budget_action, stricter_budget


@pytest.mark.parametrize(
    ("local", "durable", "expected"),
    [
        (None, None, None),
        ((4, 100.0), None, (4, 100.0)),
        (None, (3, 100.0), (3, 100.0)),
        ((1, 100.0), (9, 101.0), (9, 101.0)),
        ((9, 101.0), (1, 100.0), (9, 101.0)),
        ((3, 100.0), (1, 100.0), (1, 100.0)),
    ],
)
def test_stricter_budget_selects_current_more_pessimistic_observation(
        local, durable, expected):
    assert stricter_budget(local, durable) == expected


@pytest.mark.parametrize(
    ("observation", "now", "expected"),
    [
        (None, 100.0, "allow"),
        ((1, 200.0), 100.0, "allow"),
        ((0, 100.0), 100.0, "forget"),
        ((0, 200.0), 100.0, "refuse"),
        ((-1, 200.0), 100.0, "refuse"),
    ],
)
def test_budget_action(observation, now, expected):
    assert budget_action(observation, now) == expected
