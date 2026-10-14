"""Bookmarks uses the shared X GraphQL transport boundary."""
import pytest

import pipeline.platforms.x.local as x_local
import pipeline.platforms.x.rate as x_rate
from pipeline.platforms.x import collections as x_collections
from pipeline.platforms.x.contracts import HOME_URL
from pipeline.platforms.x.request import XClient


def test_a_bookmarks_page_uses_core_transport_and_propagates_its_rate_limit(monkeypatch):
    """The Bookmarks walk reads its page through the session its collection names."""
    from pipeline.curation import bookmark_listing, listings
    from pipeline.curation.x_walks import XAccount
    from pipeline.worker.quota_loop.source import Context

    seen = {}
    monkeypatch.setattr(x_local, "read_x_cookies", lambda: {"ct0": "csrf"})

    def headers(cookies, referer):
        seen["referer"] = referer
        return {"Cookie": "ct0=csrf"}

    def resolve(_self, op, *, default_seed="", env_var=None, page_url=HOME_URL):
        seen["resolve"] = (op, {"default_seed": default_seed, "env_var": env_var,
                                "page_url": page_url})
        return "qid"

    def limited(_self, *args, **kwargs):
        seen["request"] = (args, kwargs)
        raise x_rate.XRateLimited("spent", op="Bookmarks")

    monkeypatch.setattr(x_local, "auth_headers", headers)
    monkeypatch.setattr(XClient, "resolve_query_id", resolve)
    monkeypatch.setattr(XClient, "graphql_get", limited)

    with pytest.raises(x_rate.XRateLimited, match="spent"):
        collection = bookmark_listing.Bookmarks()
        session = XAccount().session(Context("x:Bookmarks"), collection)
        collection.read(session, "me", listings.Branch(collection.operation))

    assert seen["referer"] == "https://x.com/i/bookmarks"
    assert seen["resolve"] == ("Bookmarks", {
        "default_seed": "", "env_var": "X_BOOKMARKS_QUERY_ID",
        "page_url": "https://x.com/i/bookmarks"})
    assert seen["request"][0][0] == "Bookmarks"
    assert seen["request"][1]["field_toggles"] is x_collections.BOOKMARKS_FIELD_TOGGLES
    assert "tolerate_errors" not in seen["request"][1]
