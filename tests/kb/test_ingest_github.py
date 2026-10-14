"""pipeline.kb.ingest_github — the atom-KB GitHub adapter.

ONE repo→atom mapping (`_repo_atom`), for one repo an Oracle *referenced*
(`github_atom_from_url`, entry_mode='author_referenced').

These cover the URL parser and the single-repo entry offline (mocked fetch). The
key invariant is CANONICAL identity: the atom keys on the API's owner login, not the URL's casing,
so a footprint reference dedups against the archive walk instead of minting a twin.
"""
from __future__ import annotations

import json

import pytest

from pipeline.kb import ingest_github as gh
from pipeline.kb import schema


@pytest.mark.parametrize("url,expected", [
    ("https://github.com/ggerganov/llama.cpp", ("ggerganov", "llama.cpp")),
    ("https://github.com/Ggerganov/llama.cpp/blob/master/README.md", ("Ggerganov", "llama.cpp")),
    ("https://www.github.com/openai/whisper.git", ("openai", "whisper")),
    ("http://github.com/a/b?tab=readme", ("a", "b")),
    ("https://github.com/features/copilot", None),   # reserved product page, not an owner
    ("https://github.com/karpathy", None),           # bare profile (one segment)
    ("https://gist.github.com/x/abc", None),         # gist is a different host
    ("https://arxiv.org/abs/2401.1", None),          # not github at all
])
def test_github_owner_repo_parse(url, expected):
    assert gh.github_owner_repo(url) == expected


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _repo(owner="ggerganov", name="llama.cpp"):
    return {"name": name, "owner": {"login": owner}, "language": "C++",
            "stargazers_count": 60000, "forks_count": 8000, "description": "LLM inference in C/C++",
            "topics": ["llm"], "pushed_at": "2026-01-05T00:00:00Z", "license": {"spdx_id": "MIT"},
            "html_url": f"https://github.com/{owner}/{name}"}


def _patch_repo(monkeypatch, repo):
    from pipeline.platforms.github import api as gh_ing
    monkeypatch.setattr(gh, "_fetch_repo", lambda owner, name: repo)
    monkeypatch.setattr(gh_ing, "fetch_readme", lambda owner, name: "# readme\nbody")


def test_github_atom_from_url_mints_referenced_artifact(conn, monkeypatch):
    _patch_repo(monkeypatch, _repo())
    atom_id = gh.github_atom_from_url(conn, "https://github.com/ggerganov/llama.cpp")
    assert atom_id == "github:ggerganov/llama.cpp"
    row = conn.execute("SELECT who_id, what_kind, entry_mode, source_type FROM atoms "
                       "WHERE atom_id=?", (atom_id,)).fetchone()
    assert row["who_id"] == "github:ggerganov" and row["what_kind"] == "artifact"
    assert row["entry_mode"] == "author_referenced" and row["source_type"] == "github"


def test_github_atom_from_url_canonical_case(conn, monkeypatch):
    # URL casing differs from the API's canonical owner login; the atom keys on the API's, so a
    # footprint reference dedups against the tracked-handle crawl instead of minting a twin.
    _patch_repo(monkeypatch, _repo(owner="ggerganov"))
    atom_id = gh.github_atom_from_url(conn, "https://github.com/GGERGANOV/llama.cpp")
    assert atom_id == "github:ggerganov/llama.cpp"


def test_github_atom_from_url_idempotent(conn, monkeypatch):
    _patch_repo(monkeypatch, _repo())
    a = gh.github_atom_from_url(conn, "https://github.com/ggerganov/llama.cpp")
    b = gh.github_atom_from_url(conn, "https://github.com/ggerganov/llama.cpp")
    assert a == b == "github:ggerganov/llama.cpp"                       # unchanged snapshot → same id
    assert conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_id=?", (a,)).fetchone()[0] == 1


def test_github_atom_from_url_not_a_repo(conn):
    assert gh.github_atom_from_url(conn, "https://github.com/features/x") is None


def test_github_atom_from_url_fetch_failure_returns_none(conn, monkeypatch):
    monkeypatch.setattr(gh, "_fetch_repo", lambda owner, name: None)    # 404 / network failure
    assert gh.github_atom_from_url(conn, "https://github.com/ghost/missing") is None


# ── A rate limit is a refusal, not an empty archive ────────────────────────────────────
# The distinction is the whole of these two tests. If `_gh_get` logged a 403 and returned None,
# `_paginate` would break and `sync_github`'s summary would be byte-identical to "this person has no
# repos"; the run would read as `ingested`, the cursor would advance, `covered_from` would widen to
# a window the crawl never reached, and the TTL would be stamped. The gap would be permanent and
# invisible. Anonymous core REST is 60 req/hr and a 189-repo account is ~191
# calls, so this is the ordinary tokenless case, not an edge one.
#
# What a BLOCKED run must not do is claim coverage. What a run that spent its OWN ceiling must do
# is claim exactly the coverage it verified — the bounded-sweep half of the same decision, tested
# in tests/kb/test_refresh_incrementality.py.


def test_a_rate_limit_survives_the_fetch_helpers_catch_all(monkeypatch):
    """`_gh_get`'s `except Exception` must not erase the refusal it just raised.

    Pinned because the catch-all sits three lines below the raise and silently re-swallowed it
    the first time this was written — the failure mode returns the moment someone reorders them."""
    from pipeline.platforms.github import api as gh_ing
    from pipeline.platforms.github.api import GitHubRateLimited

    class _Limited:
        status_code = 403
        text = "API rate limit exceeded for 1.2.3.4"

    monkeypatch.setattr(gh_ing.requests, "get", lambda *a, **k: _Limited())
    with pytest.raises(GitHubRateLimited):
        gh_ing.gh_get("https://api.github.com/users/alice")


def test_a_rate_limited_single_repo_fetch_degrades_instead_of_raising(monkeypatch):
    """One repo claims no coverage, so the refusal is swallowed there — deliberately asymmetric.

    `_fetch_repo` feeds `github_atom_from_url` and the X-footprint prefetch, and both already read
    None as "could not fetch". Letting it raise would surface a traceback through `hopper.save`'s
    MCP boundary for a URL a user typed. Only the CRAWL, which stamps coverage, may not swallow."""
    from pipeline.platforms.github import api as gh_ing

    class _Limited:
        status_code = 403
        text = "API rate limit exceeded for 1.2.3.4"

    monkeypatch.setattr(gh_ing.requests, "get", lambda *a, **k: _Limited())
    assert gh._fetch_repo("acme", "memory") is None
