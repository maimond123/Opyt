"""
pipeline/ingestion/sources/blog.py

Render a fetched blog article as library Markdown.
"""


def _article_to_markdown(article: dict, author: str, author_name: str) -> str:
    """Convert extracted article to markdown with frontmatter."""
    title = article.get("title", "Untitled")
    date_str = article.get("date", "unknown")
    url = article.get("url", "")
    content = article.get("content", "")

    fm = (
        f"---\n"
        f"source: blog\n"
        f'author: "{author}"\n'
        f'author_name: "{author_name}"\n'
        f"url: {url}\n"
        f"date: {date_str}\n"
        f"type: article\n"
        f"tags: []\n"
        f"---\n\n"
    )

    body = f"# {title}\n\n"
    body += f"{content}\n\n"
    body += f"---\n*Blog · [Original post]({url})*\n"

    return fm + body
