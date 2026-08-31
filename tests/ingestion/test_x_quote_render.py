"""Quote-tweet rendering (Level 2): the quoted node's OWN link card, media (with VLM description),
and X-Article body are rendered — not just its text. The quoted node ships inside the bookmark
payload (x_graphql._normalize recurses), so this is a render change, not a fetch change.

One level only: a quote-of-a-quote renders the immediate quoted node's text but does not recurse."""
from __future__ import annotations

from pipeline.ingestion.x_render import _render_quoted_tweet, tweet_to_markdown

_GRAPHQL_ARTICLE = {
    "article_results": {"result": {
        "title": "Why Agents Win",
        "content_state": {"blocks": [
            {"type": "unstyled", "text": "Autonomous agents compose small tools into systems."},
        ]},
    }}
}


def _quoted(text="the quoted take", **extra):
    qt = {"id": "77", "author": {"userName": "quoted_guy", "name": "Quoted Guy"},
          "text": text, "entities": {}}
    qt.update(extra)
    return qt


# ── The renderer in isolation ────────────────────────────────────────────────────

def test_plain_text_quote_renders_only_text():
    md = _render_quoted_tweet(_quoted("just words"))
    assert "**Quoting** [@quoted_guy]" in md
    assert "> just words" in md
    assert "## Media" not in md and "## Links" not in md   # nothing to add → clean no-op


def test_quoted_media_with_description_is_rendered():
    # The quoted image + its VLM description (attached upstream by enrich_tweet_media) end up in the
    # atom — a quoted chart is now searchable, not a dropped CDN URL.
    qt = _quoted("look at this", extendedEntities={"media": [
        {"type": "photo", "media_url_https": "https://pbs/chart.jpg", "description": "CPI vs wages, 2020-2026"}
    ]})
    md = _render_quoted_tweet(qt)
    assert "https://pbs/chart.jpg" in md
    assert "*Image:* CPI vs wages, 2020-2026" in md


def test_quoted_link_card_is_rendered():
    qt = _quoted("read this",
                 card={"binding_values": [
                     {"key": "title", "value": {"string_value": "The Bitter Lesson"}},
                     {"key": "description", "value": {"string_value": "compute beats cleverness"}},
                 ]},
                 entities={"urls": [{"expanded_url": "https://example.com/bitter"}]})
    md = _render_quoted_tweet(qt)
    assert "The Bitter Lesson" in md and "https://example.com/bitter" in md


def test_quoted_article_body_is_rendered():
    md = _render_quoted_tweet(_quoted("their essay 👇", article=_GRAPHQL_ARTICLE))
    assert "Autonomous agents compose small tools into systems." in md


# ── End-to-end through tweet_to_markdown (the bookmark render path) ───────────────

def test_quote_tweet_note_carries_quoted_media():
    norm = {
        "id": "1", "author": {"userName": "alice", "name": "Alice", "id": "9"},
        "text": "this chart is wild", "createdAt": "", "likeCount": 5, "entities": {},
        "isQuote": True,
        "quoted_tweet": _quoted("data 👇", extendedEntities={"media": [
            {"type": "photo", "media_url_https": "https://pbs/d.jpg", "description": "a rising line chart"}
        ]}),
    }
    md = tweet_to_markdown(norm, source="x-bookmark", footer_label="Bookmarked")
    assert "type: quote" in md
    assert "this chart is wild" in md                   # the quoter's own take
    assert "a rising line chart" in md                  # the quoted image's description
