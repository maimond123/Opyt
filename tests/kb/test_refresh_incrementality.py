"""The two adapter incrementality seams: GitHub's `pushed_at` gate and blog's `known_urls` skip.

Modelled on tests/kb/test_bookmark_lookback.py: these assert that NO FETCH HAPPENED, not that a
count came out right. A count can be satisfied by a dedup skip further downstream — which is
exactly the bug these seams exist to remove, since the whole point is not paying for the call.

The MCP surface lives in test_oracle_refresh_surface.py: these prove an ADAPTER skips a call,
those prove the TOOL routes and reports.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pipeline.kb import ingest_github, link_discovery, schema

NOW = datetime(2026, 8, 8, tzinfo=timezone.utc)


# ── GitHub: `pushed_at` gate ────────────────────────────────────────────────────
def _repo(name, pushed, *, fork=False):
    return {"name": name, "pushed_at": pushed, "fork": fork, "stargazers_count": 1,
            "owner": {"login": "will"}, "html_url": f"https://github.com/will/{name}",
            "description": "d", "topics": [], "created_at": "2024-01-01T00:00:00Z"}


@pytest.fixture()
def gh(monkeypatch):
    """Spy on the two per-repo calls the crawl makes — the README fetch and the fork's upstream
    GET. Both are what `since` exists to avoid; both are AFTER the gate in the loop."""
    seen = {"readme": [], "repo_get": []}

    def fake_readme(owner, name):
        seen["readme"].append(name)
        return f"# {name}"

    def fake_repo_get(owner, name):
        seen["repo_get"].append(name)
        return {"source": {"full_name": "upstream/x"}}

    monkeypatch.setattr("pipeline.ingestion.sources.github._fetch_readme", fake_readme)
    monkeypatch.setattr(ingest_github, "_fetch_repo", fake_repo_get)
    monkeypatch.setattr(ingest_github, "_seed_owner_identity", lambda conn, handle: None)
    return seen


def test_stale_repo_costs_no_readme_call(kb_home, fake_embedder, monkeypatch, gh):
    monkeypatch.setattr(ingest_github, "_fetch_handle_repos", lambda h: [
        _repo("old", "2026-01-01T00:00:00Z"), _repo("new", "2026-08-07T00:00:00Z")])
    conn = schema.connect()
    try:
        summ = ingest_github.sync_github(conn, fake_embedder, handles=["will"],
                                         since=NOW - timedelta(days=14))
    finally:
        conn.close()
    assert gh["readme"] == ["new"]                 # the assertion that matters: `old` never fetched
    assert summ["stale"] == 1 and summ["added"] == 1


def test_stale_fork_costs_no_upstream_lookup(kb_home, fake_embedder, monkeypatch, gh):
    """The gate sits BEFORE the fork branch, so an untouched fork skips its extra GET too —
    14 of @willccbb's 26 repos are forks, and each one is a whole round-trip."""
    monkeypatch.setattr(ingest_github, "_fetch_handle_repos", lambda h: [
        _repo("oldfork", "2026-01-01T00:00:00Z", fork=True)])
    conn = schema.connect()
    try:
        summ = ingest_github.sync_github(conn, fake_embedder, handles=["will"],
                                         since=NOW - timedelta(days=14))
    finally:
        conn.close()
    assert gh["repo_get"] == [] and gh["readme"] == []
    assert summ["stale"] == 1 and summ["forked"] == 0


def test_without_since_nothing_changes(kb_home, fake_embedder, monkeypatch, gh):
    """Onboarding passes None and must behave exactly as before the seam existed."""
    monkeypatch.setattr(ingest_github, "_fetch_handle_repos", lambda h: [
        _repo("old", "2020-01-01T00:00:00Z"), _repo("new", "2026-08-07T00:00:00Z")])
    conn = schema.connect()
    try:
        summ = ingest_github.sync_github(conn, fake_embedder, handles=["will"])
    finally:
        conn.close()
    assert sorted(gh["readme"]) == ["new", "old"]
    assert summ["stale"] == 0 and summ["added"] == 2


def test_missing_pushed_at_is_processed_not_dropped():
    """Fail-safe: an absent or unparseable date must never silently drop a repo."""
    since = NOW - timedelta(days=14)
    assert ingest_github._pushed_before({"pushed_at": ""}, since) is False
    assert ingest_github._pushed_before({"pushed_at": "garbage"}, since) is False
    assert ingest_github._pushed_before({}, since) is False
    assert ingest_github._pushed_before({"pushed_at": "2026-01-01T00:00:00Z"}, None) is False


# ── blog: `known_urls` skips the paid gray triage ───────────────────────────────
@pytest.fixture()
def triage(monkeypatch):
    """The REAL `_triage_gray` is stubbed out globally by the autouse fixture; this replaces it
    with a spy so we can assert WHAT it was asked to decide, not just that it ran."""
    calls: list[list[str]] = []

    def spy(candidates, *, author_name=None, **kw):
        calls.append([c["url"] for c in candidates])
        return list(candidates)

    monkeypatch.setattr(link_discovery, "_triage_gray", spy)
    return calls


def _stub_discovery(monkeypatch, hub):
    monkeypatch.setattr("pipeline.ingestion.sources.blog._fetch_sitemap_urls", lambda b: [])
    monkeypatch.setattr("pipeline.ingestion.sources.blog.harvest_hub_links", lambda b: hub)
    monkeypatch.setattr("pipeline.ingestion.sources.blog.harvest_links_from", lambda p: [])


def test_known_gray_urls_never_reach_the_paid_triage(monkeypatch, triage):
    hub = [{"url": "https://a.com/already", "anchor": "old"},
           {"url": "https://a.com/brand-new", "anchor": "new"}]
    _stub_discovery(monkeypatch, hub)
    known = {link_discovery._canon_post_url("https://a.com/already")}

    out = link_discovery.discover_candidate_urls("https://a.com", known_urls=known)

    assert triage == [["https://a.com/brand-new"]]      # the known url was never asked about
    assert [e["url"] for e in out] == ["https://a.com/brand-new"]


def test_known_urls_omitted_means_todays_behavior(monkeypatch, triage):
    hub = [{"url": "https://a.com/already", "anchor": "old"}]
    _stub_discovery(monkeypatch, hub)
    link_discovery.discover_candidate_urls("https://a.com")
    assert triage == [["https://a.com/already"]]


def test_strong_candidates_are_not_filtered(monkeypatch, triage):
    """The filter is scoped to GRAY: a strong url costs no LLM call, and the adapter's policy-B
    hash skip already stops it before any fetch."""
    hub = [{"url": "https://a.com/blog/known-post", "anchor": "x"}]
    _stub_discovery(monkeypatch, hub)
    known = {link_discovery._canon_post_url("https://a.com/blog/known-post")}
    out = link_discovery.discover_candidate_urls("https://a.com", known_urls=known)
    assert [e["source"] for e in out] == ["strong"]


def test_blog_adapter_excludes_body_pending_from_known_urls(kb_home, fake_embedder, monkeypatch):
    """⚠️ The caller must pass `seen − body_pending`. An atom stored WITHOUT its body because a
    fetch was BLOCKED has to stay a discovery candidate, or the temporary block freezes into a
    permanent hole that never self-heals."""
    from pipeline.kb import ingest_blog

    conn = schema.connect()
    try:
        for atom_id, state in (("blog:a.com/done", "complete"), ("blog:a.com/blocked", "pending")):
            schema.upsert_atom(conn, {"atom_id": atom_id, "source_type": "blog",
                                      "who_id": "blog:a.com", "raw_hash": "h",
                                      "payload": {"body_state": state}, "description": "d"})
        captured = {}

        def spy_discover(base, *, handle=None, author_name=None, known_urls=None):
            captured["known"] = known_urls
            return []

        # `link_discovery` is imported lazily inside the adapter, so patch the module itself.
        monkeypatch.setattr(link_discovery, "discover_candidate_urls", spy_discover)
        monkeypatch.setattr(ingest_blog, "_feed_date_map", lambda b: {})
        ingest_blog.sync_blog_footprint(conn, fake_embedder, blog_url="https://a.com")
    finally:
        conn.close()

    assert captured["known"] == {"blog:a.com/done"}     # the blocked atom stays discoverable
