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
    """Fail-safe: an absent or unparseable date must never silently drop a repo. `None` from here
    means NEITHER window gate fires, so the repo is fetched."""
    assert ingest_github._pushed_at({"pushed_at": ""}) is None
    assert ingest_github._pushed_at({"pushed_at": "garbage"}) is None
    assert ingest_github._pushed_at({}) is None
    assert ingest_github._pushed_at({"pushed_at": "2026-01-01T00:00:00Z"}) == datetime(
        2026, 1, 1, tzinfo=timezone.utc)


# ── GitHub: the BACKWARD seam — `before`, the ceiling, and the frontier ─────────
# `since` bounds a refresh, but nothing bounded a FIRST crawl, and a first crawl of a 189-repo
# account is ~191 anonymous calls against a 60/hr per-IP limit. D1 made that crawl report BLOCKED
# instead of a healthy short list; it deliberately left the gap that a retry re-walks the same
# newest prefix and stops in the same place. These tests are what closes it: a bounded run
# that reports the frontier it REACHED, and a resume that starts from there.

def test_a_covered_repo_costs_no_readme_call(kb_home, fake_embedder, monkeypatch, gh):
    """`before` is the resume seam. Everything pushed at or after the frontier is already held,
    and skipping it for free is the whole saving — re-fetching that prefix is the spend the
    ceiling exists to bound."""
    monkeypatch.setattr(ingest_github, "_fetch_handle_repos", lambda h: [
        _repo("recent", "2026-08-07T00:00:00Z"), _repo("older", "2026-01-01T00:00:00Z")])
    conn = schema.connect()
    try:
        summ = ingest_github.sync_github(conn, fake_embedder, handles=["will"],
                                         before=NOW - timedelta(days=14))
    finally:
        conn.close()
    assert gh["readme"] == ["older"]
    assert summ["covered"] == 1 and summ["added"] == 1


def test_an_uncapped_sweep_reports_the_floor_it_reached_not_its_oldest_repo(
        kb_home, fake_embedder, monkeypatch, gh):
    """A run that read the WHOLE list reached the list's end, and saying so is what stops the
    resume livelocking. Report the oldest repo FETCHED instead and every instant between it and
    the floor is a permanent gap: the next run finds no repo there, reports nothing, and the
    frontier never moves again."""
    monkeypatch.setattr(ingest_github, "_fetch_handle_repos", lambda h: [
        _repo("new", "2026-08-07T00:00:00Z"), _repo("old", "2020-03-04T00:00:00Z")])
    conn = schema.connect()
    try:
        summ = ingest_github.sync_github(conn, fake_embedder, handles=["will"])
    finally:
        conn.close()
    assert "capped" not in summ
    assert summ["covered_from"] == "2020-03-04T00:00:00+00:00"


def test_a_capped_sweep_reports_the_oldest_repo_it_fetched_and_is_not_blocked(
        kb_home, fake_embedder, monkeypatch, gh):
    """Spending our OWN ceiling is not the host refusing us (`d7dbcfcf`). The run verified how
    far it got, so it carries no `error`, stays INGESTED, and hands back a frontier — which is
    exactly what a BLOCKED run must not do."""
    from pipeline.kb import ingest_common

    monkeypatch.setattr(ingest_github, "REPOS_PER_RUN", 1)
    monkeypatch.setattr(ingest_github, "_fetch_handle_repos", lambda h: [
        _repo("first", "2026-08-07T00:00:00Z"), _repo("second", "2026-06-06T00:00:00Z")])
    conn = schema.connect()
    try:
        summ = ingest_github.sync_github(conn, fake_embedder, handles=["will"])
    finally:
        conn.close()
    assert gh["readme"] == ["first"]                   # the second repo cost no call
    assert summ["capped"] == 1 and summ["added"] == 1
    assert summ["covered_from"] == "2026-08-07T00:00:00+00:00"
    assert "error" not in summ
    assert ingest_common.classify_run(summ) == ingest_common.RUN_INGESTED


def test_the_ceiling_leaves_a_later_handle_undetermined(kb_home, fake_embedder, monkeypatch, gh):
    """A multi-handle run whose ceiling goes in handle one must say it never opened handle two,
    reusing the same counter a rate limit reports through."""
    monkeypatch.setattr(ingest_github, "REPOS_PER_RUN", 1)
    monkeypatch.setattr(ingest_github, "_fetch_handle_repos",
                        lambda h: [_repo(f"{h}-only", "2026-08-07T00:00:00Z")])
    conn = schema.connect()
    try:
        summ = ingest_github.sync_github(conn, fake_embedder, handles=["will", "ada"])
    finally:
        conn.close()
    assert gh["readme"] == ["will-only"]
    assert summ["undetermined"] == 1


def test_a_resume_walks_the_next_batch_instead_of_the_same_prefix(
        kb_home, fake_embedder, monkeypatch, gh):
    """The gap D1 left, closed. Run one takes the newest repo and reports its frontier; run two
    hands that frontier back as `before` and gets the NEXT one — not the identical prefix at the
    identical cost."""
    monkeypatch.setattr(ingest_github, "REPOS_PER_RUN", 1)
    monkeypatch.setattr(ingest_github, "_fetch_handle_repos", lambda h: [
        _repo("newest", "2026-08-07T00:00:00Z"), _repo("middle", "2026-06-06T00:00:00Z"),
        _repo("oldest", "2026-02-02T00:00:00Z")])
    conn = schema.connect()
    try:
        first = ingest_github.sync_github(conn, fake_embedder, handles=["will"])
        second = ingest_github.sync_github(
            conn, fake_embedder, handles=["will"],
            before=datetime.fromisoformat(first["covered_from"]))
    finally:
        conn.close()
    assert gh["readme"] == ["newest", "middle"]
    assert second["covered"] == 1                      # `newest` skipped for free, not re-fetched
    assert second["covered_from"] == "2026-06-06T00:00:00+00:00"


def test_a_blocked_sweep_never_claims_the_floor(kb_home, fake_embedder, monkeypatch, gh):
    """The host stopped us mid-list, so the reach is the oldest repo we actually fetched. Claiming
    the floor here is the exact lie D1 removed from the run's STATUS, reappearing in its frontier:
    the list response arrives whole, so `oldest_listed` looks complete even when the README calls
    never got there."""
    from pipeline.ingestion.sources.github import GitHubRateLimited

    def _readme(owner, name):
        if name == "old":
            raise GitHubRateLimited("rate limit reached")
        gh["readme"].append(name)
        return f"# {name}"

    monkeypatch.setattr("pipeline.ingestion.sources.github._fetch_readme", _readme)
    monkeypatch.setattr(ingest_github, "_fetch_handle_repos", lambda h: [
        _repo("new", "2026-08-07T00:00:00Z"), _repo("old", "2020-03-04T00:00:00Z")])
    conn = schema.connect()
    try:
        summ = ingest_github.sync_github(conn, fake_embedder, handles=["will"])
    finally:
        conn.close()
    assert summ["error"] and summ["undetermined"] == 1
    assert summ["covered_from"] == "2026-08-07T00:00:00+00:00"


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
