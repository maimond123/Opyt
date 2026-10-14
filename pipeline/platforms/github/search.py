"""Search repositories on GitHub, with a persisted breaker and an on-disk response cache.

`search_repos` (topic → repos) is the one read; `search_since` runs one created-date window through
it and reports any failure as `SourceError`, so the pair stays unstamped and retries next pass.
Only successful results are cached; a 404 returns None without tripping the breaker.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime

import requests

from opyt_core.paths import opyt_path
from pipeline.circuit_breaker import CircuitBreaker
from pipeline.credentials import get_credential
from pipeline.platforms.github import api
from pipeline.platforms.errors import SourceError

API_BASE = "https://api.github.com"
_CACHE_TTL = 7 * 24 * 3600  # a topic's top repos are stable enough for a week


# Lazy module-level breaker, persisted in opyt.db, so an outage trips once for every caller.
_gh_breaker: CircuitBreaker | None = None


def _github_breaker() -> CircuitBreaker:
    global _gh_breaker
    if _gh_breaker is None:
        _gh_breaker = CircuitBreaker("github.com")
    return _gh_breaker


def search_since(get_client, name: str, *, since: datetime | None, limit: int) -> list[dict]:
    """Repos created since `since` whose name or description holds `name` as a phrase, ranked by
    stars; transport failure reported uniformly. Created, not pushed: an old repo with one fresh
    commit is not new."""
    phrase = " ".join((name or "").replace('"', " ").split())
    query = f'"{phrase}" in:name,description'
    if since:
        query += f" created:>{since.strftime('%Y-%m-%d')}"
    try:
        return get_client().search_repos(query, limit=limit, sort="stars")
    except Exception as exc:
        raise SourceError(f"github search failed: {type(exc).__name__}: {exc}") from None


def cache_action(entry, now: float, ttl: float):
    """The cached data to use for one entry, or None to fetch afresh.

    No entry, an entry at least `ttl` seconds old, and a fresh entry holding no data all fetch.
    """
    if not entry or now - entry.get("at", 0) >= ttl:
        return None
    return entry.get("data")


class GitHubApiClient:
    def __init__(self, token: str | None = None, cache_path: Path | None = None):
        self.token = token if token is not None else get_credential("github")
        self.cache_path = Path(cache_path) if cache_path else opyt_path("github_cache.json")
        self._cache = self._load_cache()

    # ── auth + cache ───────────────────────────────────────────────────────────
    def _headers(self) -> dict:
        return api.api_headers(self.token)

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
        return cache_action(self._cache.get(key), time.time(), _CACHE_TTL)

    def _cache_put(self, key: str, data) -> None:
        self._cache[key] = {"at": time.time(), "data": data}
        self._save_cache()

    # ── HTTP (breaker-guarded) ─────────────────────────────────────────────────
    def _get(self, path: str, params: dict | None = None):
        """One GET through the breaker. 404 → None. Other failures raise, as does
        CircuitOpenError: a failure must never read as an empty answer, or `search_repos` would
        cache it and the caller would stamp it."""
        def _do():
            resp = requests.get(f"{API_BASE}{path}", headers=self._headers(),
                                params=params, timeout=20)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        return _github_breaker().call(_do)

    # ── endpoints ──────────────────────────────────────────────────────────────
    def search_repos(self, query: str, limit: int = 10, sort: str = "stars") -> list[dict]:
        """Top repos matching `query`. `sort` is the GitHub search sort ("stars" default, or
        "updated"). Returns
        [{full_name, owner, stars, description, created_at, pushed_at, html_url, language, topics,
        archived}]."""
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
        """Normalize one /search/repositories item to the frontier's repo dict; every field is on
        the search item, so no extra call."""
        return {
            "full_name": r.get("full_name"),
            "owner": (r.get("owner") or {}).get("login"),
            "stars": int(r.get("stargazers_count") or 0),
            "description": r.get("description") or "",
            "created_at": (r.get("created_at") or "")[:10],
            "pushed_at": (r.get("pushed_at") or "")[:10],
            "html_url": r.get("html_url") or "",
            "language": r.get("language") or "",
            "topics": r.get("topics") or [],
            "archived": bool(r.get("archived")),
        }

