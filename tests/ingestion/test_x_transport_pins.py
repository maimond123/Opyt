"""Offline pins for the local X transport boundary."""

from __future__ import annotations

import sys
import types

import pipeline.platforms.x.local as x_local
import pipeline.platforms.x.query_ids as query_ids
from pipeline.platforms.x.request import XClient


def test_query_id_cache_put_and_clear_preserve_other_operations(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))

    query_ids.qid_cache_put("Bookmarks", "bookmark-id")
    query_ids.qid_cache_put("Viewer", "viewer-id")
    assert query_ids.qid_cache_get("Bookmarks") == "bookmark-id"
    assert query_ids.qid_cache_get("Viewer") == "viewer-id"

    query_ids.qid_cache_clear("Bookmarks")
    assert query_ids.qid_cache_get("Bookmarks") is None
    assert query_ids.qid_cache_get("Viewer") == "viewer-id"


def test_local_session_uses_the_local_query_graphql_and_viewer_jobs(monkeypatch):
    session = XClient(x_local.LocalXTransport({"twid": "u%3D77"}, {"Cookie": "managed"}))
    calls = []
    monkeypatch.setattr(query_ids, "qid_cache_get", lambda _op: None)
    monkeypatch.setattr(query_ids, "qid_cache_put", lambda *_args: None)
    monkeypatch.setattr(x_local, "discover_query_id",
                        lambda op, _page_url, cookies: calls.append(("qid", op, cookies)) or "q")

    class _Resp:
        status_code, headers, text = 200, {}, ""

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"ok": True}}

    cffi = types.ModuleType("curl_cffi")
    cffi.requests = types.SimpleNamespace(
        get=lambda url, **kw: calls.append(("graphql", url.rsplit("/", 2)[-1],
                                            url.rsplit("/", 2)[-2], kw["headers"])) or _Resp())
    monkeypatch.setitem(sys.modules, "curl_cffi", cffi)

    assert session.resolve_query_id("Bookmarks") == "q"
    assert session.graphql_get("Bookmarks", "q", {}, {}) == {"data": {"ok": True}}
    assert session.viewer_id() == "77"
    assert calls == [
        ("qid", "Bookmarks", {"twid": "u%3D77"}),
        ("graphql", "Bookmarks", "q", {"Cookie": "managed"}),
    ]
