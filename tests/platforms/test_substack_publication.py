from datetime import datetime, timezone

import pytest

from pipeline.platforms.substack.publication import archive_progress

SINCE = datetime(2025, 6, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("page", "last_date", "since", "pages_read", "limit", "expected"),
    [
        ([], None, SINCE, 1, 500, "finish"),
        ([{}], "2025-05-31T23:59:59Z", SINCE, 1, 500, "finish"),
        ([{}], "not-a-date", SINCE, 1, 500, "continue"),
        ([{}], None, SINCE, 1, 500, "continue"),
        ([{}], "2025-06-02T00:00:00Z", SINCE, 1, 500, "continue"),
        ([{}], "2025-06-02T00:00:00", SINCE, 1, 500, "continue"),
        ([{}], None, None, 2, 2, "limit"),
        ([{}], "2025-05-31T00:00:00Z", SINCE, 2, 2, "finish"),
    ],
)
def test_archive_progress(page, last_date, since, pages_read, limit, expected):
    assert archive_progress(page, last_date, since, pages_read, limit) == expected
