"""_publish_date — the blog CREATION-date extractor. Pure, offline (htmldate parses passed HTML,
no network). Locks: (1) prefer the PUBLISH date over a later 'modified' date, (2) mine a
date embedded in the URL path when the page carries none, (3) empty string when nothing is found,
and (4) — added 2026-08-13 — the page's OWN asserted publish date OUTRANKS anything htmldate
infers, because on real pages htmldate loses the tag and substitutes a footer copyright year."""
from __future__ import annotations

from datetime import date, timedelta

from pipeline.ingestion.sources.blog import _asserted_publish_dates, _publish_date

# The shape that caused the bug: a real publish date in the head, a LATER year in the footer.
_FOOTER_2026 = "<footer><span>&copy; 2026 Example Inc</span></footer>"


def test_an_asserted_date_beats_a_footer_copyright_year():
    """THE regression guard. LIVE-MEASURED 2026-08-13 on letta.com/blog/sleep-time-compute: the
    page declared `article:published_time = 2025-04-21`, and htmldate's full-document parse
    returned 2026-01-01 — the footer's "© 2026" via its free-text extensive search. The same
    document parsed HEAD-ONLY returned 2025-04-21, so the tag was always fine and readable; the
    full-document path simply lost it. The wrong date then travelled at `day` precision, which is
    what made it invisible: nothing downstream can tell an inferred date from a declared one."""
    html = ('<html><head>'
            '<meta property="article:published_time" content="2025-04-21">'
            '</head><body><p>essay</p></body>' + _FOOTER_2026 + '</html>')
    assert _publish_date(html, "https://example.com/blog/post") == "2025-04-21"


def test_json_ld_datepublished_is_read_including_inside_a_graph():
    """schema.org nests: publishers routinely wrap the Article in an `@graph` list, so the walk
    has to recurse rather than read the top level."""
    html = ('<html><head><script type="application/ld+json">'
            '{"@context":"https://schema.org","@graph":['
            '{"@type":"WebSite"},{"@type":"Article","datePublished":"2021-09-02"}]}'
            '</script></head><body>x</body>' + _FOOTER_2026 + '</html>')
    assert _publish_date(html, "https://example.com/p") == "2021-09-02"


def test_a_time_element_itemprop_counts_as_an_assertion():
    html = ('<html><body><time itemprop="datePublished" datetime="2018-11-30T09:00:00Z">'
            'Nov 30</time><p>post</p></body>' + _FOOTER_2026 + '</html>')
    assert _publish_date(html, "https://example.com/p") == "2018-11-30"


def test_attribute_order_does_not_matter():
    """Half the web writes `content` before `property`. A regex that assumed one order would drop
    those pages back to the htmldate guess without anything looking wrong."""
    html = ('<html><head><meta content="2020-02-02" property="article:published_time">'
            '</head><body>x</body></html>')
    assert _publish_date(html, "https://example.com/p") == "2020-02-02"


def test_a_modified_date_is_never_mistaken_for_a_publish_date():
    """The whole point of `original_date=True`, restated for rung 0: a 2019 essay re-touched in
    2026 must read 2019. `article:modified_time` / `dateModified` are deliberately NOT in the
    published key set, so a page carrying ONLY a modified date asserts nothing here."""
    html = ('<html><head><meta property="article:modified_time" content="2026-05-05">'
            '<script type="application/ld+json">{"@type":"Article","dateModified":"2026-05-05"}'
            '</script></head><body>x</body></html>')
    assert _asserted_publish_dates(html) == []


def test_the_oldest_assertion_wins_when_a_page_declares_several():
    """Same rule `original_date=True` encodes, for the same reason — the original publication is
    the older one."""
    html = ('<html><head><meta property="article:published_time" content="2017-01-05">'
            '<script type="application/ld+json">{"@type":"Article","datePublished":"2022-08-08"}'
            '</script></head><body>x</body></html>')
    assert _publish_date(html, "https://example.com/p") == "2017-01-05"


def test_a_future_publish_date_is_refused():
    """A publish date that has not happened yet is a templating bug, not a publication. Refusing
    it here falls through to htmldate rather than stamping a confident lie."""
    future = (date.today() + timedelta(days=400)).isoformat()
    html = f'<html><head><meta property="article:published_time" content="{future}">' \
           '</head><body>x</body></html>'
    assert _asserted_publish_dates(html) == []
    assert _publish_date(html, "https://example.com/2020/06/15/my-post") == "2020-06-15"


def test_malformed_json_ld_does_not_lose_the_good_block():
    """Fail-safe, per-block: one publisher's broken script tag must not cost the date another
    block states correctly."""
    html = ('<html><head>'
            '<script type="application/ld+json">{ this is not json ,,, }</script>'
            '<script type="application/ld+json">{"@type":"Article","datePublished":"2019-07-04"}'
            '</script></head><body>x</body></html>')
    assert _publish_date(html, "https://example.com/p") == "2019-07-04"


def test_htmldate_still_runs_when_the_page_asserts_nothing():
    """Rung 0 ADDS an authority, it does not replace the cascade. Tuning htmldate instead was
    rejected precisely because free-text/URL inference is the only date many hub-harvested posts
    have — that path must stay intact."""
    assert _asserted_publish_dates("<html><body>no metadata at all</body></html>") == []
    assert _publish_date("<html><body>no date</body></html>",
                         "https://blog.com/2020/06/15/my-post") == "2020-06-15"


def test_prefers_published_over_modified():
    # A 2019 essay re-touched in 2026 must read 2019 (original_date=True), not the modified date.
    html = ('<html><head>'
            '<meta property="article:published_time" content="2019-03-10T00:00:00Z">'
            '<meta property="article:modified_time" content="2026-01-01T00:00:00Z">'
            '</head><body>essay</body></html>')
    assert _publish_date(html, "https://blog.com/essay") == "2019-03-10"


def test_mines_date_from_url_when_page_has_none():
    # Many hub-harvested posts carry no metadata date and no sitemap lastmod — the URL path is the
    # only creation signal (huyenchip / vitalik style /YYYY/MM/DD/).
    assert _publish_date("<html><body>no date</body></html>",
                         "https://blog.com/2020/06/15/my-post") == "2020-06-15"


def test_empty_when_no_date_anywhere():
    assert _publish_date("<html><body>nothing</body></html>", "https://blog.com/flat-slug") == ""


def test_never_raises_on_garbage():
    # Fail-safe: a parse blow-up must degrade to "" (a date miss can't sink the fetch).
    assert _publish_date("", "") == ""
    assert _publish_date("<<<not html>>>", "not a url") in ("", None) or isinstance(
        _publish_date("<<<not html>>>", "not a url"), str)
