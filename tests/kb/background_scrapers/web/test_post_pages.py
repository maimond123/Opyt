"""Which of a site's pages are its posts (pipeline/oracles/sources/archive/post_pages.py).

The cases are measured ones: the address shapes real feeds taught, the pages each signal caught
or wrongly kept, and the posts a naive address learner loses.
"""
from __future__ import annotations

import pytest

from pipeline.oracles.sources.archive import post_pages
from pipeline.platforms.web.article import declares

STRATECHERY = post_pages.learn(["https://stratechery.com/2026/its-not-you-its-me/",
                                "https://stratechery.com/2026/an-interview-with-katie-harbath/"],
                               "https://stratechery.com")
SIMON = post_pages.learn(["https://simonwillison.net/2026/Oct/9/gemini/",
                          "https://simonwillison.net/2026/Oct/11/uv/"], "https://simonwillison.net")
DUBACH = post_pages.learn(["https://philippdubach.com/posts/john-cochranes-lesson-on-inflation/",
                           "https://philippdubach.com/posts/future-of-knowledge/"],
                          "https://philippdubach.com")
SUBSTACK = post_pages.learn(["https://www.latent.space/p/ainews", "https://www.latent.space/p/rl"],
                            "https://www.latent.space")


@pytest.mark.parametrize("shapes,url,fits", [
    (STRATECHERY, "https://stratechery.com/2014/the-state-of-consumer-technology/", True),
    (STRATECHERY, "https://stratechery.com/company/apple/", False),     # a taxonomy page
    (STRATECHERY, "https://stratechery.com/concept/ai/", False),
    (STRATECHERY, "https://stratechery.com/2015/04", False),            # a month's archive
    (STRATECHERY, "https://stratechery.com/2010", False),
    (SIMON, "https://simonwillison.net/2007/Apr/3/extjs/", True),       # the feed is all October
    (SIMON, "https://simonwillison.net/tags/python/", False),
    (DUBACH, "https://philippdubach.com/posts/pozsars-bretton-woods-iii-the-framework-1/2/", True),
    (DUBACH, "https://philippdubach.com/writing/?topic=tech", False),
    (DUBACH, "https://philippdubach.com/categories/finance/", False),
    (SUBSTACK, "https://www.latent.space/p/2026", True),                # a post named by digits
    (SUBSTACK, "https://www.latent.space/archive", False),
    (SUBSTACK, "https://www.latent.space/podcast", False),
])
def test_an_address_fits_the_feeds_posts_or_does_not(shapes, url, fits):
    assert post_pages.fits(url, shapes) is fits


def test_a_feed_teaches_nothing_from_another_host_or_one_post():
    assert post_pages.learn(["https://feeds.example.net/a", "https://feeds.example.net/b"],
                            "https://writer.example") is None
    assert post_pages.learn(["https://writer.example/one"], "https://writer.example") is None


@pytest.mark.parametrize("name,nonpost", [
    ("https://stratechery.com/wp-sitemap-taxonomies-companies-1.xml", True),   # WordPress core
    ("https://stratechery.com/wp-sitemap-posts-page-1.xml", True),
    ("https://stratechery.com/wp-sitemap-users-1.xml", True),
    ("https://stratechery.com/wp-sitemap-posts-post-1.xml", False),
    ("https://a16z.com/page-sitemap.xml", True),                              # Yoast
    ("https://ifp.org/category-sitemap.xml", True),
    ("https://ifp.org/post-author-sitemap.xml", True),
    ("https://a16z.com/podcast-sitemap2.xml", False),                         # a custom post type
    ("https://www.thediff.co/sitemap-pages.xml", True),                       # Ghost
    ("https://www.thediff.co/sitemap-tags.xml", True),
    ("https://www.thediff.co/sitemap-posts.xml", False),
    ("https://reactionwheel.net/image-sitemap-1.xml", False),
])
def test_a_cms_sitemap_names_what_it_files(name, nonpost):
    assert post_pages.filed_as_nonpost(name) is nonpost


@pytest.mark.parametrize("fits_feed,said,post", [
    (True, {}, True),
    (None, {}, True),                                                # no feed: nothing says no
    (False, {"article": True, "published": True}, True),             # a post outside the feed
    (False, {"article": True, "published": False}, False),           # Substack's archive page
    (False, {"article": False, "published": True}, False),
    (False, {}, False),
])
def test_a_page_off_the_feed_is_a_post_only_if_it_says_it_is_a_dated_article(fits_feed, said, post):
    assert post_pages.is_post(fits_feed, said) is post


@pytest.mark.parametrize("html,said", [
    ('<meta property="og:type" content="article"><meta property="article:published_time" '
     'content="2026-01-02">', {"article": True, "published": True}),
    ('<meta content="website" property="og:type">', {"article": False, "published": False}),
    ('<script type="application/ld+json">{"@graph": [{"@type": ["BlogPosting"]}]}</script>'
     '<time itemprop="datePublished" datetime="2026-01-02">', {"article": True, "published": True}),
    # A CMS stamps a WebPage's datePublished on every page; it is not a post's publish time.
    ('<script type="application/ld+json">{"@type": "WebPage", "datePublished": "2026-01-02"}'
     '</script>', {"article": False, "published": False}),
    ('<script type="application/ld+json">{broken</script>', {"article": False, "published": False}),
])
def test_what_a_page_declares_itself_to_be(html, said):
    assert declares(html) == said
