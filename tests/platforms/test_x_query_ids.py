import pytest

import pipeline.platforms.x.query_ids as query_ids
from pipeline.platforms.x.query_ids import query_id_choice


@pytest.mark.parametrize(
    ("override", "seed", "cached", "expected"),
    [
        ("override-id", "seed-id", "cached-id", "override-id"),
        (None, "seed-id", "cached-id", "seed-id"),
        (None, None, "cached-id", "cached-id"),
        (None, None, None, None),
        ("", "", "", None),
    ],
)
def test_query_id_choice_preserves_precedence(override, seed, cached, expected):
    assert query_id_choice(override, seed, cached) == expected


def test_cached_query_id_uses_cache_without_discovery(monkeypatch):
    discovered = []
    monkeypatch.setattr(query_ids, "qid_cache_get", lambda _op: "cached-id")
    monkeypatch.setattr(query_ids, "qid_cache_put",
                        lambda *_args: pytest.fail("cached id should not be rewritten"))

    assert query_ids.cached_query_id("Bookmarks", lambda: discovered.append(True)) == "cached-id"
    assert discovered == []


def test_cached_query_id_stores_a_discovered_id(monkeypatch):
    stored = []
    monkeypatch.setattr(query_ids, "qid_cache_get", lambda _op: None)
    monkeypatch.setattr(query_ids, "qid_cache_put", lambda op, qid: stored.append((op, qid)))

    result = query_ids.cached_query_id("Bookmarks", lambda: "fresh-id")

    assert result == "fresh-id"
    assert stored == [("Bookmarks", "fresh-id")]


def test_cached_query_id_does_not_store_a_discovery_miss(monkeypatch):
    stored = []
    monkeypatch.setattr(query_ids, "qid_cache_get", lambda _op: None)
    monkeypatch.setattr(query_ids, "qid_cache_put", lambda *args: stored.append(args))

    assert query_ids.cached_query_id("Bookmarks", lambda: None) is None
    assert stored == []
