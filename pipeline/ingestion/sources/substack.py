"""
pipeline/ingestion/sources/substack.py
A Substack archive post → markdown, for the library's footprint ingest.
"""

from datetime import datetime

import html2text

_h2t = html2text.HTML2Text()
_h2t.ignore_links      = False
_h2t.ignore_images     = False
_h2t.body_width        = 0
_h2t.unicode_snob      = True
_h2t.ignore_emphasis   = False


# ── Markdown ─────────────────────────────────────────────────────────────────

def _post_to_markdown(post: dict, author: str, author_name: str) -> str:
    """Convert a Substack archive post to markdown with frontmatter."""
    title = post.get("title", "Untitled")
    subtitle = post.get("subtitle", "")
    body_html = post.get("body_html", "")
    post_date = post.get("post_date", "")
    url = post.get("canonical_url", "")
    word_count = post.get("word_count", 0)

    date_str = ""
    if post_date:
        try:
            dt = datetime.fromisoformat(post_date.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            date_str = post_date[:10] if len(post_date) >= 10 else ""

    body_md = ""
    if body_html:
        body_md = _h2t.handle(body_html).strip()

    fm = (
        f"---\n"
        f"source: substack\n"
        f'author: "{author}"\n'
        f'author_name: "{author_name}"\n'
        f"url: {url}\n"
        f"date: {date_str}\n"
        f"type: article\n"
        f"tags: []\n"
        f"---\n\n"
    )

    body = f"# {title}\n\n"
    if subtitle:
        body += f"*{subtitle}*\n\n"
    if body_md:
        body += f"{body_md}\n\n"

    body += f"---\n*Substack · [Original post]({url})*\n"

    return fm + body
