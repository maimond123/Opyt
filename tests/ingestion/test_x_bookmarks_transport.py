"""Bookmarks uses the shared X GraphQL transport boundary."""

import pytest

from pipeline.ingestion import x_graphql as xg
from pipeline.ingestion import x_graphql_core as core


def test_bookmark_iterator_uses_core_transport_and_propagates_its_rate_limit(monkeypatch):
    seen = {}
    monkeypatch.setattr(core, "read_x_cookies", lambda: {"ct0": "csrf"})

    def headers(cookies, referer):
        seen["referer"] = referer
        return {"Cookie": "ct0=csrf"}

    def resolve(op, cookies, **kwargs):
        seen["resolve"] = (op, kwargs)
        return "qid"

    def limited(*args, **kwargs):
        seen["request"] = (args, kwargs)
        raise core.XRateLimited("spent", op="Bookmarks")

    monkeypatch.setattr(core, "auth_headers", headers)
    monkeypatch.setattr(core, "resolve_query_id", resolve)
    monkeypatch.setattr(core, "graphql_get", limited)

    with pytest.raises(core.XRateLimited, match="spent"):
        next(xg.iterate_bookmarks())

    assert seen["referer"] == "https://x.com/i/bookmarks"
    assert seen["resolve"] == ("Bookmarks", {
        "env_var": "X_BOOKMARKS_QUERY_ID", "page_url": "https://x.com/i/bookmarks"})
    assert seen["request"][0][0] == "Bookmarks"
    assert seen["request"][1]["field_toggles"] is xg.BOOKMARKS_FIELD_TOGGLES
    assert "tolerate_errors" not in seen["request"][1]
