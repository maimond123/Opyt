"""
pipeline/frontier/github_candidates.py — GitHub as a Frontier source: repository search
(`pipeline/platforms/github/search.py`) mapped to `Candidate`s.
"""

from __future__ import annotations

from datetime import datetime

from pipeline.platforms.github.search import GitHubApiClient, search_since
from pipeline.frontier.candidates import MAX_SUMMARY_CHARS, Candidate


class GitHubAdapter:
    """GitHub repos created since the last pull whose name or description holds the watch's
    name (`created:>DATE`), sorted by stars, since the window already selects on recency. Stars
    reorder, never exclude: no minimum-star gate.
    """
    slug = "github"
    # 6.0 s = 10 requests/minute, GitHub's unauthenticated limit on `/search/repositories`, so an
    # install without a token stays inside it; a token only buys more throughput.
    host = "api.github.com"
    min_interval_s = 6.0
    min_lookback_days = 0              # `created:>DATE` windows on creation, which GitHub indexes

    def __init__(self, client=None):
        self._client = client

    def _get_client(self):
        if self._client is None:
            self._client = GitHubApiClient()
        return self._client

    def available(self) -> bool:
        """Always: `github_client` carries its own breaker on github.com and refuses inside
        `search`, which the loop records as an error."""
        return True

    def search(self, query: str, *, since: datetime | None, now: datetime, limit: int = 25,
               context=()) -> list[Candidate]:
        del now, context  # GitHub windows only on the lower created-date bound.
        rows = search_since(self._get_client, query, since=since, limit=limit)
        out = []
        for r in rows or []:
            full = r.get("full_name")
            if not full:
                continue
            out.append(Candidate(
                candidate_id=f"repo:{full}",
                source="github",
                kind="repo",
                title=full,
                url=r.get("html_url") or f"https://github.com/{full}",
                published=r.get("created_at"),
                summary=(r.get("description") or "")[:MAX_SUMMARY_CHARS],
                # Key names follow the payload contract, not GitHub's field names (`language`
                # means a natural language elsewhere). Pinned by tests/kb/test_payload_key_names.py.
                payload={"stars": r.get("stars"), "code_language": r.get("language"),
                         "source_tags": r.get("topics") or [],
                         "archived": bool(r.get("archived"))}))
        return out
