"""A saved Substack post as snapshot markdown: its body cleaned from HTML, under provenance
frontmatter, or its preview when the body could not be fetched."""

from __future__ import annotations

from pipeline.ingestion.utils import log
from pipeline.kb import derive


def clean_body_html(body_html: str) -> str:
    """Converts a Substack `body_html` fragment (already de-boilerplated) to clean markdown for
    the stored body. Runs both trafilatura and html2text and keeps the longer result, since trafilatura
    can silently under-extract a bare fragment into a thin "invisible" atom. Empty in → "" out
    (caller falls back to the stub). Images are kept."""
    if not body_html:
        return ""
    # Imported here because both are heavy and only this function needs them. Only the parse
    # can fail, so each extractor gets its own guard.
    import html2text
    import trafilatura

    cands: list[str] = []
    try:
        md = trafilatura.extract(body_html, output_format="markdown",
                                 include_links=True, include_images=True, no_fallback=False)
        if md:
            cands.append(md)
    except Exception as e:
        log(f"[curation] trafilatura extract failed ({type(e).__name__}); html2text still runs")
    try:
        h = html2text.HTML2Text()
        h.ignore_links, h.ignore_images, h.body_width, h.unicode_snob = False, False, 0, True
        t = h.handle(body_html)
        if t:
            cands.append(t)
    except Exception as e:
        log(f"[curation] html2text extract failed ({type(e).__name__})")
    return max(cands, key=len).strip() if cands else ""


def _yaml_dq(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()


def saved_atom_markdown(rec: dict, body_md: str) -> str:
    """Builds snapshot markdown for a saved post: provenance frontmatter + the real body, or the
    stub preview when the full body couldn't be fetched. The whole snapshot is indexed, frontmatter
    included."""
    title = (rec.get("title") or "Untitled").strip()
    subtitle = (rec.get("subtitle") or "").strip()
    url = rec.get("url") or ""
    date_str = derive._iso_day(rec.get("post_date", ""))
    handle = (rec.get("author_handle") or "").strip()
    author = f"@{handle}" if handle else (rec.get("publication_name") or "substack")
    author_name = rec.get("author_name") or rec.get("publication_name") or ""
    body = (body_md or "").strip() or (rec.get("preview") or "").strip()

    fm = (
        "---\n"
        "source: substack\n"
        f'author: "{_yaml_dq(author)}"\n'
        f'author_name: "{_yaml_dq(author_name)}"\n'
        f'publication: "{_yaml_dq(rec.get("publication_name", ""))}"\n'
        f"url: {url}\n"
        f"date: {date_str}\n"
        f"saved_at: {rec.get('saved_at', '')}\n"
        f"audience: {rec.get('audience', '')}\n"
        "type: article\n"
        "---\n\n"
    )
    out = f"# {title}\n\n"
    if subtitle:
        out += f"*{subtitle}*\n\n"
    if body:
        out += f"{body}\n\n"
    out += f"---\n*Saved from Substack · [Original post]({url})*\n"
    return fm + out
