"""Links in post TEXT: a long post's own link list, and the media's own short link.

Both found from a ledger of real post links.
"""
from __future__ import annotations

import pipeline.platforms.x.records as x_records
from pipeline.ingestion.x_render import expand_urls, tweet_to_markdown


def _long_post(*, legacy_urls, note_urls, note_text, media=None):
    legacy = {"id_str": "9", "full_text": "short version https://t.co/SHORT1",
              "created_at": "Mon Sep 07 10:00:00 +0000 2026", "conversation_id_str": "9",
              "entities": {"urls": legacy_urls, **({"media": media} if media else {})}}
    return {"rest_id": "9", "legacy": legacy,
            "note_tweet": {"note_tweet_results": {"result": {
                "text": note_text, "entity_set": {"urls": note_urls}}}},
            "core": {"user_results": {"result": {"rest_id": "1", "core": {"screen_name": "a"}}}}}


def test_a_long_post_maps_its_own_short_codes():
    """The full text's codes are not the short version's — verified on 1890045180163572185."""
    n = x_records.normalize_tweet(_long_post(
        legacy_urls=[{"url": "https://t.co/SHORT1", "expanded_url": "https://arxiv.org/abs/1"}],
        note_urls=[{"url": "https://t.co/LONG1", "expanded_url": "https://arxiv.org/abs/1"},
                   {"url": "https://t.co/LONG2", "expanded_url": "https://example.com/past-280"}],
        note_text="two papers https://t.co/LONG1 and one past the cut https://t.co/LONG2"))

    assert [u["expanded_url"] for u in n["entities"]["urls"]] == [
        "https://arxiv.org/abs/1", "https://example.com/past-280"]     # the past-280 link exists
    rendered = expand_urls(n["text"], n["entities"])
    assert "t.co" not in rendered and "https://example.com/past-280" in rendered


def test_a_long_post_keeps_its_media_list():
    media = [{"url": "https://t.co/PIC", "type": "photo"}]
    n = x_records.normalize_tweet(_long_post(legacy_urls=[], note_urls=[], note_text="text", media=media))
    assert n["entities"]["media"] == media


def test_a_short_post_is_unchanged():
    r = _long_post(legacy_urls=[{"url": "https://t.co/S", "expanded_url": "https://x.y/z"}],
                   note_urls=[], note_text="")
    del r["note_tweet"]
    assert x_records.normalize_tweet(r)["entities"]["urls"][0]["expanded_url"] == "https://x.y/z"


def test_the_media_s_own_short_link_is_dropped_from_the_text():
    text = "wrote a blog on this last year https://t.co/CRXHG0U3Y6"
    ents = {"urls": [], "media": [{"url": "https://t.co/CRXHG0U3Y6", "type": "photo"}]}
    assert expand_urls(text, ents) == "wrote a blog on this last year"


def test_a_real_link_beside_media_is_still_expanded():
    text = "read https://t.co/A then look https://t.co/PIC"
    ents = {"urls": [{"url": "https://t.co/A", "expanded_url": "https://a.example/post"}],
            "media": [{"url": "https://t.co/PIC"}]}
    assert expand_urls(text, ents) == "read https://a.example/post then look"


def test_text_without_media_renders_byte_for_byte_as_before():
    """No media link removed → no trailing trim → no needless re-embed of every plain post."""
    text = "a plain post with trailing space  "
    assert expand_urls(text, {"urls": []}) == text


def test_the_quoted_post_s_media_link_goes_too():
    root = {"id": "1", "text": "my take", "author": {"userName": "a", "name": "A", "id": "1"},
            "createdAt": "Mon Sep 07 10:00:00 +0000 2026", "entities": {},
            "quoted_tweet": {"id": "2", "text": "their post https://t.co/QPIC",
                             "author": {"userName": "b", "name": "B", "id": "2"},
                             "entities": {"media": [{"url": "https://t.co/QPIC"}]}}}
    md = tweet_to_markdown(root, source="x-profile", footer_label="Profile extract")
    assert "t.co" not in md and "> their post" in md


# ── the quote's own t.co: undeclared, at the end of a quote post ─────────────────────

def test_a_quote_posts_trailing_undeclared_tco_is_dropped():
    """2105083603906695413: text ends `…9 months ago https://t.co/IddVclOKOL`, no entity declares
    it, and the quote renders on its own. Nothing could expand it."""
    out = expand_urls("was mine 9 months ago https://t.co/IddVclOKOL", {"urls": []},
                          quoting=True)
    assert out == "was mine 9 months ago"


def test_the_same_text_on_a_post_that_quotes_nothing_is_untouched():
    text = "was mine 9 months ago https://t.co/IddVclOKOL"
    assert expand_urls(text, {"urls": []}) == text


def test_a_declared_trailing_link_on_a_quote_post_is_expanded_not_dropped():
    out = expand_urls("read this https://t.co/abc", {"urls": [
        {"url": "https://t.co/abc", "expanded_url": "https://example.com/post"}]}, quoting=True)
    assert out == "read this https://example.com/post"


def test_an_undeclared_tco_mid_text_on_a_quote_post_is_untouched():
    text = "see https://t.co/xyz for more, and that is the point"
    assert expand_urls(text, {}, quoting=True) == text


def test_the_render_drops_it_from_a_quote_post():
    root = {"id": "1", "text": "mine 9 months ago https://t.co/IddVclOKOL",
            "author": {"userName": "s", "name": "S"}, "createdAt": "Mon Jan 06 10:00:00 +0000 2026",
            "entities": {}, "url": "https://x.com/s/status/1",
            "quoted_tweet": {"id": "2", "text": "the quoted post", "author": {"userName": "h", "name": "H"},
                             "entities": {}, "url": "https://x.com/h/status/2"}}
    md = tweet_to_markdown(root)
    assert "t.co" not in md and "mine 9 months ago" in md and "the quoted post" in md


def test_a_long_post_keeps_the_mentions_past_the_cut():
    raw = _long_post(legacy_urls=[], note_urls=[], note_text="thanks @early and @late")
    raw["legacy"]["entities"]["user_mentions"] = [{"id_str": "2", "screen_name": "early"}]
    raw["note_tweet"]["note_tweet_results"]["result"]["entity_set"]["user_mentions"] = [
        {"id_str": "2", "screen_name": "early"}, {"id_str": "3", "screen_name": "late"}]
    n = x_records.normalize_tweet(raw)
    assert [m["screen_name"] for m in n["entities"]["user_mentions"]] == ["early", "late"]
