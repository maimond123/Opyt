"""
pipeline/ingestion/sources/github.py
GitHub REST fetch helpers + repo→markdown rendering.

Free API: 60 req/hr unauthenticated, 5000 req/hr with ``GITHUB_TOKEN``.

Layer 1 only (see ``pipeline/ingestion/sources/__init__.py``); ``pipeline/kb/ingest_github.py``
imports its fetch/render helpers to land atoms.
"""

import base64
import os
import time

import requests

from pipeline.ingestion.utils import log

FETCH_DELAY = 0.5  # seconds between API calls


# ── API helpers ──────────────────────────────────────────────────────────────

def _gh_headers() -> dict:
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _gh_get(url: str, params: dict | None = None) -> requests.Response | None:
    """Make an authenticated GET request to GitHub API."""
    try:
        resp = requests.get(url, headers=_gh_headers(), params=params, timeout=15)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            log(f"  [warn] GitHub rate limit hit — set GITHUB_TOKEN for 5000 req/hr")
        elif resp.status_code not in (404,):
            log(f"  [warn] GitHub API {resp.status_code}: {url}")
    except Exception as e:
        log(f"  [warn] GitHub request failed: {e}")
    return None


def _paginate(url: str, params: dict | None = None, max_pages: int = 20) -> list[dict]:
    """Paginate through a GitHub API endpoint."""
    params = dict(params or {})
    params.setdefault("per_page", 100)
    items = []
    for page in range(1, max_pages + 1):
        params["page"] = page
        resp = _gh_get(url, params)
        if not resp:
            break
        batch = resp.json()
        if not batch:
            break
        items.extend(batch)
        time.sleep(FETCH_DELAY)
    return items


def _fetch_user(username: str) -> dict | None:
    """Fetch GitHub user profile."""
    resp = _gh_get(f"https://api.github.com/users/{username}")
    return resp.json() if resp else None


def _fetch_repos(username: str) -> list[dict]:
    """Fetch all public repos for a GitHub user."""
    return _paginate(
        f"https://api.github.com/users/{username}/repos",
        {"sort": "pushed", "direction": "desc"},
    )


def _fetch_org_repos(org: str) -> list[dict]:
    """Fetch all public repos for an organization."""
    return _paginate(
        f"https://api.github.com/orgs/{org}/repos",
        {"sort": "pushed", "direction": "desc"},
    )


def _fetch_readme(owner: str, repo: str) -> str | None:
    """Fetch README content for a repo. Returns markdown text or None."""
    resp = _gh_get(f"https://api.github.com/repos/{owner}/{repo}/readme")
    if not resp:
        return None
    data = resp.json()
    content = data.get("content", "")
    encoding = data.get("encoding", "")
    if encoding == "base64" and content:
        try:
            return base64.b64decode(content).decode("utf-8", errors="replace")
        except Exception:
            pass
    return None


# ── Markdown conversion ─────────────────────────────────────────────────────

def _repo_to_markdown(
    repo: dict,
    readme: str | None,
    author: str,
    author_name: str,
    category: str = "personal",
) -> str:
    name = repo.get("name", "")
    description = repo.get("description") or ""
    language = repo.get("language") or "Unknown"
    stars = repo.get("stargazers_count", 0)
    forks = repo.get("forks_count", 0)
    url = repo.get("html_url", "")
    topics = repo.get("topics", [])
    created = repo.get("created_at", "")[:10]
    updated = repo.get("pushed_at", "")[:10]
    is_fork = repo.get("fork", False)
    owner = repo.get("owner", {}).get("login", "")

    tags_str = "\n".join(f"  - {t}" for t in topics) if topics else ""
    tags_block = f"\ntags:\n{tags_str}" if tags_str else "\ntags: []"

    fm = (
        f"---\n"
        f"source: github\n"
        f"author: \"{author}\"\n"
        f"author_name: \"{author_name}\"\n"
        f"url: {url}\n"
        f"date: {updated}\n"
        f"created: {created}\n"
        f"type: github-repo\n"
        f"category: {category}\n"
        f"owner: {owner}\n"
        f"language: {language}\n"
        f"stars: {stars}\n"
        f"forks: {forks}\n"
        f"is_fork: {is_fork}"
        f"{tags_block}\n"
        f"---\n\n"
    )

    body = f"# {owner}/{name}\n\n"
    if description:
        body += f"> {description}\n\n"
    body += f"**Language:** {language} · **Stars:** {stars} · **Forks:** {forks}\n"
    if category != "personal":
        body += f"**Category:** {category}\n"
    body += "\n"
    if is_fork:
        body += f"*Forked repo*\n\n"

    if readme:
        max_len = 8000
        if len(readme) > max_len:
            readme = readme[:max_len] + "\n\n... *(README truncated)*"
        body += f"## README\n\n{readme}\n\n"

    body += f"---\n*GitHub · [View repo]({url})*\n"

    return fm + body
