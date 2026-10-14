import pytest

from pipeline.platforms.x.request import response_action


@pytest.mark.parametrize(
    ("status", "has_errors", "tolerate_errors", "expected"),
    [
        (429, False, False, "rate_limited"),
        (403, False, False, "unauthenticated"),
        (404, False, False, "drift"),
        (503, False, False, "http_error"),
        (200, True, False, "graphql_errors"),
        (200, True, True, "tolerated_errors"),
        (200, False, True, "return"),
    ],
)
def test_response_action(status, has_errors, tolerate_errors, expected):
    assert response_action(status, has_errors=has_errors,
                           tolerate_errors=tolerate_errors) == expected
