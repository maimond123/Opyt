"""
pipeline/ingestion/sources/github.py

Render a fetched repository and its README as library Markdown.
"""


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
        body += "*Forked repo*\n\n"

    if readme:
        max_len = 8000
        if len(readme) > max_len:
            readme = readme[:max_len] + "\n\n... *(README truncated)*"
        body += f"## README\n\n{readme}\n\n"

    body += f"---\n*GitHub · [View repo]({url})*\n"

    return fm + body
