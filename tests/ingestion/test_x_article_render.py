"""X-Article rendering: the Draft.js body X ships on the tweet, through `_render_article`.

There used to be TWO shapes here — this one and twitterapi.io's `/twitter/article` response, with
`_article_shape` normalizing between them. The provider was removed on 2026-08-30 and that second
shape has no producer left, so the branch went with it. What remains is the distinction that still
bites: a node WITH `content_state.blocks` versus a node without."""
from __future__ import annotations

from pipeline.ingestion.x_render import _article_shape, _render_article, tweet_to_markdown

# GraphQL cookie-scrape shape: article.article_results.result.{title, content_state.blocks}
_GRAPHQL_ARTICLE = {
    "article_results": {"result": {
        "title": "Why Agents Win",
        "content_state": {"blocks": [
            {"type": "header-one", "text": "The thesis"},
            {"type": "unstyled", "text": "Autonomous agents compose small tools into systems."},
            {"type": "unordered-list-item", "text": "cheaper"},
            {"type": "unordered-list-item", "text": "faster"},
            {"type": "blockquote", "text": "the framework is the moat"},
            {"type": "atomic", "text": ""},          # media block → skipped in v1
        ]},
    }}
}

# What X ships when the request did NOT ask for `withArticleRichContentState`: cover, title and a
# preview, and no body at all. ~1.1 KB on the wire, and a 200 response like any other.
_TEASER_ARTICLE = {
    "article_results": {"result": {
        "title": "Why Agents Win",
        "preview_text": "Autonomous agents compose small tools…",
        "cover_media": {"media_info": {"original_img_url": "https://x/cover.jpg"}},
    }}
}


def test_article_shape_reads_the_title_and_blocks():
    title, blocks = _article_shape(_GRAPHQL_ARTICLE)
    assert title == "Why Agents Win" and len(blocks) == 6


def test_a_teaser_node_is_truthy_but_has_no_blocks():
    """⚠️ THE failure this shape exists to make visible. A teaser is a perfectly valid node on a
    200 response, so `bool(article)` reads it as "we have the article" and the body is lost in
    silence. The block list is the only thing that tells the truth, which is why callers that care
    (`ingest_x_footprint`, to set `article_incomplete`) ask for it rather than for the node."""
    assert bool(_TEASER_ARTICLE) is True
    title, blocks = _article_shape(_TEASER_ARTICLE)
    assert title == "Why Agents Win"      # the title arrives either way — also misleading
    assert blocks == []


def test_an_unknown_shape_renders_nothing_rather_than_crashing():
    assert _article_shape({}) == ("", [])
    assert _article_shape({"contents": [{"type": "unstyled", "text": "old provider"}]}) == ("", [])


def test_render_graphql_article_full_body():
    md = _render_article(_GRAPHQL_ARTICLE)
    assert "## Why Agents Win" in md
    assert "# The thesis" in md
    assert "Autonomous agents compose small tools into systems." in md
    assert "- cheaper" in md and "- faster" in md
    assert "> the framework is the moat" in md
    # atomic (media) block contributes no stray empty line content
    assert "atomic" not in md


def test_bookmarked_article_tweet_carries_full_body():
    # A bookmark-shaped norm with an article node → the rendered note contains the full body,
    # not just the teaser text, and is typed as an article.
    norm = {
        "id": "1", "author": {"userName": "carol", "name": "Carol", "id": "9"},
        "text": "new piece 👇", "createdAt": "", "likeCount": 3, "entities": {},
        "article": _GRAPHQL_ARTICLE,
    }
    md = tweet_to_markdown(norm, article=norm["article"], source="x-bookmark",
                           footer_label="Bookmarked")
    assert "type: article" in md
    assert "Autonomous agents compose small tools into systems." in md
    assert "new piece" in md   # the teaser is still there, the body is added below it


def test_runaway_guard_caps_pathological_article():
    huge = {"article_results": {"result": {"title": "x", "content_state": {"blocks": [
        {"type": "unstyled", "text": "z" * 100_000} for _ in range(10)  # ~1MB
    ]}}}}
    md = _render_article(huge)
    assert len(md) < 600_000   # runaway guard fired, well under the 1MB input
