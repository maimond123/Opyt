"""Read repository resources from GitHub REST: a user's profile, a user's or an organization's
public repositories (paginated), and one repository's README. A rate-limit refusal raises
`GitHubRateLimited`; a repository list that could not be read to its end raises
`GitHubReadFailed`; any other failed single read reads as None.

Free API: 60 req/hr unauthenticated, 5000 req/hr with ``GITHUB_TOKEN``.
"""
from __future__ import annotations

import base64
import time
from enum import Enum

import requests

from pipeline.credentials import get_credential
from pipeline.host_access import request as host_request
from pipeline.ingestion.utils import log

FETCH_DELAY = 0.5  # seconds between API calls


class GitHubReadFailed(Exception):
    """A GitHub read did not complete: the request failed, the body was not JSON, or a list ran
    past the page cap. The list walk raises it, because a prefix returned as the whole list lets
    the crawl claim coverage it never reached; the single reads turn it into None."""


class GitHubRateLimited(GitHubReadFailed):
    """GitHub refused with a rate-limit 403.

    Raised, not returned as ``None``: a swallowed rate limit makes a truncated crawl look like
    "this account has few repos" and lets the pull advance its cursor and coverage. Per-IP and
    session-wide, so stopping the crawl loses nothing. Adapter boundaries convert it back into a
    returned summary carrying `error` + `undetermined`."""


# ── API helpers ──────────────────────────────────────────────────────────────

def api_headers(token: str | None) -> dict:
    """GitHub REST request headers, with the bearer token when there is one."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def gh_headers() -> dict:
    """Request headers for the REST reads here, with the user's token when one is stored."""
    return api_headers(get_credential("github"))


def _get_json(url: str, params: dict | None = None):
    """One GET's decoded JSON, or None when the resource does not exist (404). Raises
    `GitHubRateLimited` on a rate-limit 403 and `GitHubReadFailed` on any other failure."""
    try:
        resp = host_request(requests.get, url, headers=gh_headers(), params=params, timeout=15)
    except Exception as e:
        raise GitHubReadFailed(f"GitHub request failed: {e}") from e
    if resp.status_code == 404:
        return None
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        # The token is throughput, never capability: `onboard(start='github')` connects it.
        raise GitHubRateLimited(f"GitHub rate limit reached at {url}")
    if resp.status_code != 200:
        raise GitHubReadFailed(f"GitHub API {resp.status_code}: {url}")
    try:
        return resp.json()
    except ValueError as e:
        raise GitHubReadFailed(f"GitHub API returned a body that is not JSON: {url}") from e


def gh_get(url: str, params: dict | None = None):
    """One authenticated GET's decoded JSON; None on a 404 or any failure but a rate limit."""
    try:
        return _get_json(url, params)
    except GitHubRateLimited:
        raise                    # the refusal is the signal; the line below would erase it
    except GitHubReadFailed as e:
        log(f"  [warn] {e}")
        return None


class PageStep(str, Enum):
    END = "end"            # nothing to keep: the list is complete
    CAPPED = "capped"      # a full batch at the page cap: the list may go on, so refuse
    NEXT = "next"          # keep this batch, pace, and read the next page


def github_page_progress(batch: list | None, page_number: int, limit: int) -> PageStep:
    """What one fetched page means for the walk. `batch` is None when the resource is absent."""
    if not batch:
        return PageStep.END
    return PageStep.CAPPED if page_number >= limit else PageStep.NEXT


def _paginate(url: str, params: dict | None = None, max_pages: int = 20) -> list[dict]:
    """Every item of a GitHub list endpoint, newest page first.

    Raises `GitHubReadFailed` when a page fails or the list runs past `max_pages`: a prefix is
    never returned as the whole list. A 404 on the first page is an empty list."""
    params = dict(params or {})
    params.setdefault("per_page", 100)
    items = []
    for page in range(1, max_pages + 1):
        params["page"] = page
        batch = _get_json(url, params)
        step = github_page_progress(batch, page, max_pages)
        if step is PageStep.END:
            break
        if step is PageStep.CAPPED:
            raise GitHubReadFailed(f"GitHub list longer than {max_pages} pages: {url}")
        items.extend(batch)
        time.sleep(FETCH_DELAY)
    return items


def fetch_user(username: str) -> dict | None:
    """Fetch GitHub user profile."""
    return gh_get(f"https://api.github.com/users/{username}")


def fetch_repos(username: str) -> list[dict]:
    """Fetch all public repos for a GitHub user."""
    return _paginate(
        f"https://api.github.com/users/{username}/repos",
        {"sort": "pushed", "direction": "desc"},
    )


def fetch_org_repos(org: str) -> list[dict]:
    """Fetch all public repos for an organization."""
    return _paginate(
        f"https://api.github.com/orgs/{org}/repos",
        {"sort": "pushed", "direction": "desc"},
    )


def fetch_readme(owner: str, repo: str) -> str | None:
    """Fetch README content for a repo. Returns markdown text or None."""
    data = gh_get(f"https://api.github.com/repos/{owner}/{repo}/readme")
    if not data:
        return None
    content = data.get("content", "")
    encoding = data.get("encoding", "")
    if encoding == "base64" and content:
        try:
            return base64.b64decode(content).decode("utf-8", errors="replace")
        except Exception:
            pass
    return None
