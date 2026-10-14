"""Substack post search, signed out: the request it sends, the page read off a recorded response,
and a post's text for the mention check. Offline: `public_get_json` is replaced, the response a
fixture modeled on a live one."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.platforms.substack import search as substack_search
from pipeline.platforms.substack.contracts import SubstackFetchError, SubstackPublicReadError

RECORDED = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "substack"
                       / "post_search.json").read_text())


@pytest.fixture()
def asked(monkeypatch):
    """Every anonymous request, answered from `answers` by URL."""
    calls, answers = [], {}

    def get(url, params, *, referer=None):
        calls.append((url, params, referer))
        return answers[url]

    monkeypatch.setattr(substack_search.substack_http, "public_get_json", get)
    return calls, answers


def test_search_asks_signed_out_with_platform_results_and_a_date_window(asked):
    """Without `includePlatformResults` the route answers nothing signed out (measured)."""
    calls, answers = asked
    answers[substack_search.SEARCH_URL] = RECORDED
    substack_search.search_posts("rentosertib", date_range="week", page=1)
    assert calls == [(substack_search.SEARCH_URL,
                      {"query": "rentosertib", "page": 1, "includePlatformResults": "true",
                       "dateRange": "week"}, None)]


def test_a_page_reads_each_post_with_its_publication_and_byline():
    posts, more = substack_search.parse_page(RECORDED)
    assert more is True
    assert posts == [
        {"id": "219807469", "title": "CADD & AI in Drug Discovery",
         "subtitle": "Weekly digest · October 9, 2026",
         "description": "Weekly digest · October 9, 2026",
         "url": "https://reforgex.substack.com/p/cadd-and-ai-in-drug-discovery-e38",
         "published": "2026-10-11T01:35:27.809Z", "paid": False,
         "publication": "Weekly Drug Discovery & CADD Digest", "author": "Fixture Author"},
        {"id": "219900001", "title": "Rentosertib's aging-clock readout",
         "subtitle": "What the phase 2a data say", "description": "What the phase 2a data say",
         "url": "https://www.forever.ai/p/rentosertib-aging-clock",
         "published": "2026-10-12T09:00:00.000Z", "paid": True,
         "publication": "Forever", "author": ""}]          # the post with no address is dropped


def test_a_page_without_its_results_list_is_refused_not_read_as_a_quiet_week():
    with pytest.raises(SubstackPublicReadError):
        substack_search.parse_page({"error": "rate limited"})


@pytest.mark.parametrize("days, window", [(0.5, "day"), (1, "day"), (1.25, "week"), (7, "week"),
                                          (14, "month"), (30, "month")])
def test_the_window_is_the_narrowest_that_holds_the_days(days, window):
    assert substack_search.range_for(days) == window


def test_a_posts_text_is_the_one_post_read_anonymously_as_plain_text(monkeypatch):
    """Custom domains too: the post lives on the publication's own host. The read is the one
    admission uses (`fetch_full_post`), which retries Substack's intermittent 403 (seen on a
    live look)."""
    read = []
    monkeypatch.setattr(substack_search, "fetch_full_post", lambda base, slug, cookies: read.append(
        (base, slug, cookies)) or {"body_html": "<p>The <b>rentosertib</b> data &amp; the "
                                                "clock</p>\n<p>More</p>"})
    text = substack_search.post_text("https://www.forever.ai/p/rentosertib-aging-clock")
    assert text == "The rentosertib data & the clock More"
    assert read == [("https://www.forever.ai", "rentosertib-aging-clock", {})]


def test_a_url_that_is_not_a_post_is_refused_without_a_request(monkeypatch):
    monkeypatch.setattr(substack_search, "fetch_full_post",
                        lambda *a: pytest.fail("read a url that is not a post"))
    with pytest.raises(SubstackFetchError):
        substack_search.post_text("https://substack.com/@someone")
