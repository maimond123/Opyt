"""The saved-posts imports, read from their durable work: begun and not landed, or never begun."""

from datetime import datetime, timezone

import pytest

from pipeline.curation import (
    bookmark_listing,
    imports,
    listings,
    saved_items,
    substack_saved_listing,
)
from pipeline.kb import schema
from pipeline.worker.quota_loop.source import Retry

NOW = datetime(2026, 10, 5, tzinfo=timezone.utc)


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    c = schema.connect()
    listings.init_schema(c)
    saved_items.init_schema(c)
    yield c
    c.close()


def begin(conn, collector, account="me"):
    return listings.start(conn, collector, account, listings.Branch("Saved"), NOW)


def land(conn, walk):
    """The walk's one page accepted, then completed: the list was walked to the end."""
    from pipeline.platforms.collection_page import Page
    task = listings.pending(conn, walk)[0]
    listings.accept_page(conn, walk, task, Page((), None, (), ended=True))
    listings.complete(conn, walk, NOW)


def test_an_import_begun_and_not_walked_to_the_end_is_outstanding(conn):
    begin(conn, bookmark_listing.COLLECTOR)
    begin(conn, substack_saved_listing.COLLECTOR)

    assert imports.imports_outstanding() == ["bookmarks_import", "substack_saved_import"]


def test_an_import_never_begun_is_not_outstanding_but_never_queued(conn):
    begin(conn, substack_saved_listing.COLLECTOR)

    assert imports.imports_outstanding() == ["substack_saved_import"]
    assert imports.imports_never_queued({"x", "substack"}) == ["x"]


def test_a_walked_list_with_posts_still_owed_is_outstanding_until_they_land(conn):
    walk = begin(conn, bookmark_listing.COLLECTOR)
    land(conn, walk)
    saved_items.offer(conn, "x", "me", [("1", "r1", {"id": "1"})], NOW)

    assert imports.imports_outstanding() == ["bookmarks_import"]
    saved_items.settle(conn, saved_items.owed(conn, "x")[0])
    assert imports.imports_outstanding() == []


def test_a_post_the_loop_gave_up_on_is_not_waited_for(conn):
    walk = begin(conn, bookmark_listing.COLLECTOR)
    land(conn, walk)
    saved_items.offer(conn, "x", "me", [("1", "r1", {"id": "1"})], NOW)
    saved_items.ROWS.write(conn, ("x", "1", "r1"), Retry("blip", 10, None, "gone"))

    assert imports.imports_outstanding() == []


def test_a_platform_with_no_import_is_never_listed(conn):
    assert imports.imports_never_queued({"blog"}) == []


def test_an_unreadable_store_reports_nothing(conn, monkeypatch):
    def broken():
        raise RuntimeError("no store")

    monkeypatch.setattr(imports, "_states", broken)
    assert imports.imports_outstanding() == []
    assert imports.imports_never_queued({"x"}) == []


def test_an_import_in_progress_counts_its_posts_saved_of_those_found(conn):
    walk = begin(conn, bookmark_listing.COLLECTOR)
    land(conn, walk)
    saved_items.offer(conn, "x", "me", [(i, f"r{i}", {"id": i}) for i in ("1", "2", "3")], NOW)
    saved_items.settle(conn, saved_items.owed(conn, "x")[0])

    assert imports.in_progress(conn) == {"x": (1, 3)}
    for saved in saved_items.owed(conn, "x"):
        saved_items.settle(conn, saved)
    assert imports.in_progress(conn) == {}
