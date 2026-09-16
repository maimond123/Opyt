"""
pipeline/artifacts/github_client.py

The networked GitHub REST transport. Only the generic transport plus ONE artifact read survive
here: `search_repos` (topic → repos). The person verbs (`user`, `stargazers`, `contributors`)
were dropped along with the retired person-scoped `pipeline/github_scout` package; the per-repo
detail reads (`repo`, `readme`) went on 2026-09-05 with no caller left — the live per-repo
enrichment is `pipeline.ingestion.sources.github._fetch_repo` / `_fetch_readme`, which
`pipeline/kb/ingest_github.py` calls after discovery.

Every call goes through CircuitBreaker("github.com") so an outage / rate-limit trips ONCE
and fails fast instead of a retry storm, and results are cached to ~/.opyt so a re-run never
re-bills the rate limit. Token via get_credential("github") (env → ~/.opyt/.env → repo .env).
A 404 is NOT a failure (the repo just doesn't exist) — it returns None WITHOUT tripping the
breaker.

  GitHubClient      — the ABC (the injectable seam).
  GitHubApiClient   — live REST impl with breaker + on-disk cache.

The one live reuser is `pipeline/kb/frontier_sources.py`'s `GitHubAdapter` (Frontier stage 2).
A `FakeGitHubClient` test double lived here too until 2026-08-29 — deleted because nothing,
tests included, ever used it (the test suites build their own inline fakes against the ABC).
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

import requests

from opyt_core.paths import opyt_path
from pipeline.circuit_breaker import CircuitBreaker, CircuitOpenError
from pipeline.credentials import get_credential

API_BASE = "https://api.github.com"
_CACHE_TTL = 7 * 24 * 3600  # a topic's top repos are stable enough for a week


# Lazy module-level breaker so a GitHub outage trips ONCE and every caller (and every
# session — state is persisted in opyt.db) fails fast instead of re-hammering the API.
_gh_breaker: CircuitBreaker | None = None


def _github_breaker() -> CircuitBreaker:
    global _gh_breaker
    if _gh_breaker is None:
        _gh_breaker = CircuitBreaker("github.com")
    return _gh_breaker


class GitHubClient(ABC):
    """Injectable seam — the ONE method the discovery client owes its caller."""

    @abstractmethod
    def search_repos(self, query: str, limit: int = 10, sort: str = "stars") -> list[dict]:
        """Top repos matching `query`. `sort` is the GitHub search sort ("stars" default, or
        "updated" — the frontier wants RECENTLY-pushed, not most-starred). Returns
        [{full_name, owner, stars, description, pushed_at, html_url, language, topics, archived}]."""


class GitHubApiClient(GitHubClient):
    def __init__(self, token: str | None = None, cache_path: Path | None = None):
        self.token = token if token is not None else get_credential("github")
        self.cache_path = Path(cache_path) if cache_path else opyt_path("github_cache.json")
        self._cache = self._load_cache()

    # ── auth + cache ───────────────────────────────────────────────────────────
    def _headers(self) -> dict:
        h = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _load_cache(self) -> dict:
        try:
            return json.loads(self.cache_path.read_text())
        except Exception:
            return {}

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache))
        except Exception:
            pass  # a cache write failure must never break a call — fail-safe

    def _cache_get(self, key: str):
        ent = self._cache.get(key)
        if ent and (time.time() - ent.get("at", 0)) < _CACHE_TTL:
            return ent.get("data")
        return None

    def _cache_put(self, key: str, data) -> None:
        self._cache[key] = {"at": time.time(), "data": data}
        self._save_cache()

    # ── HTTP (breaker-guarded) ─────────────────────────────────────────────────
    def _get(self, path: str, params: dict | None = None):
        """One GET through the breaker. 404 → None (not a failure). Network/5xx are
        recorded by the breaker and degraded to None; CircuitOpenError propagates so the
        caller can report 'skipped, breaker open' instead of silently returning nothing."""
        def _do():
            resp = requests.get(f"{API_BASE}{path}", headers=self._headers(),
                                params=params, timeout=20)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        try:
            return _github_breaker().call(_do)
        except CircuitOpenError:
            raise
        except Exception:
            return None

    # ── endpoints ──────────────────────────────────────────────────────────────
    def search_repos(self, query: str, limit: int = 10, sort: str = "stars") -> list[dict]:
        key = f"search_repos:{query.lower()}:{limit}:{sort}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        data = self._get("/search/repositories",
                         {"q": query, "sort": sort, "order": "desc",
                          "per_page": min(limit, 50)})
        items = (data or {}).get("items", []) if isinstance(data, dict) else []
        out = [self._repo_row(r) for r in items[:limit] if isinstance(r, dict)]
        self._cache_put(key, out)
        return out

    @staticmethod
    def _repo_row(r: dict) -> dict:
        """Normalize one /search/repositories item to the frontier's repo dict. Every field is
        already present on the search item — no extra call — so a survey carries stars/pushed_at
        (the movement baseline) and description/language/topics (the shown quality signal)."""
        return {
            "full_name": r.get("full_name"),
            "owner": (r.get("owner") or {}).get("login"),
            "stars": int(r.get("stargazers_count") or 0),
            "description": r.get("description") or "",
            "pushed_at": (r.get("pushed_at") or "")[:10],
            "html_url": r.get("html_url") or "",
            "language": r.get("language") or "",
            "topics": r.get("topics") or [],
            "archived": bool(r.get("archived")),
        }

