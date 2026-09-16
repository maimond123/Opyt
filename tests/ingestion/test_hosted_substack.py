"""The hosted Substack transport: where the account read comes from, and what it may return.

The line this file defends is the 2026-09-06 session-token boundary. A hosted home reads the
user's own Substack account by asking Chrome to make the request inside its signed-in page —
never by reading a cookie into Python, which is what the local transport does.
"""
from __future__ import annotations

import pytest

from pipeline.ingestion import hosted_substack
from pipeline.ingestion.sources import substack as sub


@pytest.fixture
def hosted(monkeypatch):
    from pipeline.ingestion import hosted_browser
    monkeypatch.setattr(hosted_browser, "enabled", lambda: True)


class _Chrome:
    """A shared-Chrome stand-in that records the origin and URL of each request."""

    def __init__(self, *, user_id=4242, payload=None, payloads=None, status=None):
        from pipeline.ingestion.hosted_browser import ChromeRequestStatus
        self.user_id, self.payload = user_id, payload
        # `payloads` serves one body per request, in order, so a cursor walk gets DIFFERENT
        # pages — a single repeated payload would loop or dedupe to nothing and prove neither.
        self.payloads = list(payloads) if payloads is not None else None
        self.status = status or ChromeRequestStatus.OK
        self.evaluated: list[str] = []
        self.fetched: list[tuple[str, str]] = []

    def evaluate(self, home_url, expression):
        self.evaluated.append(home_url)
        return {"id": self.user_id}

    def fetch_json(self, home_url, url, headers_js="{}"):
        from pipeline.ingestion.hosted_browser import ChromeRequestResult
        self.fetched.append((home_url, url))
        body = self.payloads.pop(0) if self.payloads else self.payload
        return ChromeRequestResult(self.status, body)


@pytest.fixture
def hosted_chrome(monkeypatch):
    """Install one `_Chrome` behind `shared_chrome()` and hand it back."""
    import contextlib

    def install(**kwargs):
        fake = _Chrome(**kwargs)

        @contextlib.contextmanager
        def shared():
            yield fake

        monkeypatch.setattr(hosted_substack, "shared_chrome", shared)
        monkeypatch.setattr(hosted_substack, "profile_dir",
                            lambda: _AlwaysThere())
        return fake

    return install


class _AlwaysThere:
    def exists(self):
        return True


def test_the_hosted_reader_never_touches_a_cookie(hosted, hosted_chrome, monkeypatch):
    """`read_substack_cookies` returns cookie VALUES to Python. On a hosted child that is the
    forbidden transport, so the seam must not route through it — and it refuses if anything
    tries."""
    fake = hosted_chrome(payload={"subscriberLists": []})
    monkeypatch.setattr(sub, "read_opyt_cookies", lambda *a, **k: pytest.fail(
        "a hosted child read the local cookie jar"))

    assert sub.follow_source().follows() == []
    with pytest.raises(RuntimeError, match="never read_substack_cookies"):
        sub.read_substack_cookies()
    assert fake.fetched


def test_the_subscriber_list_request_runs_on_the_substack_origin(hosted, hosted_chrome):
    """A credentialed cross-origin fetch is subject to Substack's CORS policy, so this request
    has to be made from a substack.com page — not from whatever page X left open."""
    fake = hosted_chrome(payload={"subscriberLists": []})

    sub.follow_source().follows()

    home_url, url = fake.fetched[0]
    assert home_url == hosted_substack._HOME_URL == "https://substack.com/inbox"
    assert url == ("https://substack.com/api/v1/user/4242/subscriber-lists?lists=following")


def test_hosted_and_local_share_one_parser(hosted, hosted_chrome):
    """The transports differ; the payload's shape does not, so only one reader of it exists."""
    payload = {"subscriberLists": [{"groups": [{"users": [
        {"name": "Person", "primary_publication": {"name": "Pub", "subdomain": "pub"}},
        {"name": "No publication", "primary_publication": {}},
    ]}]}]}
    hosted_chrome(payload=payload)

    assert sub.follow_source().follows() == [
        {"name": "Pub", "url": "https://pub.substack.com"}]
    assert sub.parse_subscriber_lists(payload) == [
        {"name": "Pub", "url": "https://pub.substack.com"}]


def test_a_refused_subscriber_list_is_a_refusal_not_an_empty_follow_graph(hosted, hosted_chrome):
    """Cloudflare guards this endpoint, and a refusal must stay distinguishable from "this account
    follows nobody" — an empty list is a FACT about the user's graph and a 403 is not.

    Reverses this test's own earlier ruling ("a 403 must cost the pass its candidates, not the
    rail"). The rail-survives half was right and still holds: `sync_substack_follows` catches this and
    records `skipped: refused`, and `run_and_record` isolates one collector's failure from the
    other four. What was wrong was WHERE the swallow happened — down here it erased the
    distinction before anything could record it, so the clock stored `found=0, status=ok`. Measured
    2026-09-08: on a fresh install that made `onboard` tell the user their sign-in had gone stale
    and to reconnect, which reconnecting cannot fix."""
    from pipeline.ingestion.hosted_browser import ChromeRequestStatus

    hosted_chrome(status=ChromeRequestStatus.UNAUTHENTICATED, payload=None)

    with pytest.raises(sub.SubstackListingError):
        sub.follow_source().follows()


def test_no_account_id_means_no_request_at_all(hosted, hosted_chrome):
    """The id keys the request, so without it there is nothing to ask for — and asking anyway
    would spend a call against a Cloudflare-guarded endpoint to learn nothing.

    It raises rather than returning []: `readable()` already gated this collector on
    `has_connection()`, which IS the account-id probe, so an id that fails here means the read
    broke mid-pass, not that the user is signed out."""
    fake = hosted_chrome(user_id=None)

    with pytest.raises(sub.SubstackListingError):
        sub.follow_source().follows()
    assert fake.fetched == []


def test_the_account_id_is_read_inside_the_page_and_nothing_else_returns(hosted, hosted_chrome):
    """The extraction fetches the reader page, but only the id crosses back — the page does
    not. That is the same rule X's queryId discovery follows."""
    fake = hosted_chrome(user_id=99)

    assert hosted_substack.own_user_id() == 99
    assert fake.evaluated == [hosted_substack._HOME_URL]


def test_a_live_account_id_is_what_proves_the_hosted_session(hosted, hosted_chrome):
    hosted_chrome(user_id=99)
    assert hosted_substack.has_connection() is True

    hosted_chrome(user_id=None)
    assert hosted_substack.has_connection() is False


def test_the_phase_probe_asks_chrome_on_a_hosted_child(hosted, monkeypatch):
    """`has_managed_substack_session` answers "has the user connected Substack to OPYT". A
    hosted child has no local cookie jar to scan, so scanning one there would answer False
    forever and strand the user in the sources phase after a successful sign-in."""
    monkeypatch.setattr(hosted_substack, "has_connection", lambda: True)
    monkeypatch.setattr(sub, "list_opyt_logged_in", lambda *a, **k: pytest.fail(
        "the hosted probe scanned a local cookie jar"), raising=False)

    assert sub.has_managed_substack_session() is True


# ── the saved-posts list ────────────────────────────────────────────────────────

def _saved_item(post_id, cursor_note=""):
    return {"type": "post", "publication": {"name": "Pub", "base_url": "https://pub.substack.com"},
            "post": {"id": post_id, "title": f"Post {post_id}{cursor_note}", "slug": f"p{post_id}",
                     "canonical_url": f"https://pub.substack.com/p/p{post_id}",
                     "audience": "everyone"}}


def test_the_saved_list_request_runs_on_the_substack_origin(hosted, hosted_chrome):
    """Same CORS rule as `subscriber-lists`: a credentialed substack.com fetch has to be issued
    from a substack.com page."""
    fake = hosted_chrome(payload={"items": [_saved_item(1)], "nextCursor": None})

    hosted_substack.saved_page()

    home_url, url = fake.fetched[0]
    assert home_url == "https://substack.com/inbox"
    assert url == "https://substack.com/api/v1/reader/saved?filter=all"


def test_a_cursor_is_percent_encoded_into_the_query(hosted, hosted_chrome):
    """The cursor is a value Substack minted, not one this repo authored. Pasting it raw would
    let a `&` in it become a second query parameter."""
    fake = hosted_chrome(payload={"items": [], "nextCursor": None})

    hosted_substack.saved_page("a b&c=d")

    assert fake.fetched[0][1].endswith("&cursor=a%20b%26c%3Dd")


def test_the_hosted_walk_pages_through_the_cursor_and_lands_every_post(hosted, hosted_chrome):
    """The WALK lives in `sources.substack`, over whichever transport it is handed. This proves
    the hosted transport satisfies that contract end to end, not that a second walk exists."""
    hosted_chrome(payloads=[
        {"items": [_saved_item(1)], "nextCursor": "c1"},
        {"items": [_saved_item(2)], "nextCursor": None},
    ])

    recs, complete = sub.fetch_saved_posts(sub.saved_source())

    assert [r["id"] for r in recs] == [1, 2]
    # A cursor walked to its own null end is the ONE exit that may claim the list is whole. The
    # collector that counts saves hangs `set_signal` vs `ensure_signal` on exactly this.
    assert complete is True


def test_a_later_page_refused_keeps_what_arrived_and_says_the_list_is_partial(
        hosted, hosted_chrome, monkeypatch):
    """⚠️ THE SILENT-TRUNCATION HOLE, closed 2026-09-13. Three of this walk's exits truncate and
    return NORMALLY — a later page refused, a cursor that stops advancing, the page cap — and each
    announced itself with a log line and nothing else. A caller could not tell "you saved 1 post"
    from "we saw 1 of an unknown number", which is precisely the distinction
    `sync_substack_saved_signals` needs to decide whether it may write a count."""
    monkeypatch.setattr(sub.time, "sleep", lambda s: None)
    hosted_chrome(payloads=[{"items": [_saved_item(1)], "nextCursor": "c1"}, None])

    recs, complete = sub.fetch_saved_posts(sub.saved_source())

    assert [r["id"] for r in recs] == [1]      # page 1 answered; that is real and is kept
    assert complete is False                   # ...but no total may be claimed from it


def test_a_cursor_that_stops_advancing_also_reports_an_incomplete_list(
        hosted, hosted_chrome, monkeypatch):
    """The other normal-return truncation: a non-null cursor handing back nothing new. The walk
    already stopped rather than looping; what was missing was telling the caller it had."""
    monkeypatch.setattr(sub.time, "sleep", lambda s: None)
    hosted_chrome(payloads=[{"items": [_saved_item(1)], "nextCursor": "c1"},
                            {"items": [_saved_item(1)], "nextCursor": "c2"}])

    recs, complete = sub.fetch_saved_posts(sub.saved_source())

    assert [r["id"] for r in recs] == [1]
    assert complete is False


def test_the_page_cap_reports_an_incomplete_list(hosted, hosted_chrome, monkeypatch):
    """The third exit. A saved list longer than the cap is rare and the cap is a runaway
    backstop, not a budget — but a walk that stopped at it has seen a prefix, not a set."""
    monkeypatch.setattr(sub.time, "sleep", lambda s: None)
    hosted_chrome(payloads=[{"items": [_saved_item(1)], "nextCursor": "c1"},
                            {"items": [_saved_item(2)], "nextCursor": "c2"}])

    recs, complete = sub.fetch_saved_posts(sub.saved_source(), max_pages=2)

    assert [r["id"] for r in recs] == [1, 2]
    assert complete is False


def test_a_refused_first_page_is_a_listing_error_not_an_empty_list(hosted, hosted_chrome):
    """Cloudflare guards this endpoint too, and a refusal is a different fact from an empty
    saved list. Handed back as `[]` it arrives at the rail as `status: ok, added: 0` — the exact
    shape of a week in which the user saved nothing, on a rail that then never investigates.
    Measured live 2026-09-07 against the real endpoint, which refused four attempts in a row."""
    from pipeline.ingestion.hosted_browser import ChromeRequestStatus

    hosted_chrome(status=ChromeRequestStatus.UNAUTHENTICATED, payload=None)

    with pytest.raises(sub.SubstackListingError):
        sub.fetch_saved_posts(sub.saved_source())


def test_hosted_bodies_are_fetched_with_no_session(hosted, hosted_chrome, monkeypatch):
    """One Chrome tab per publication is hundreds of megabytes on a box sized for ~100 MB
    children, so a body comes from the publication's own public endpoint instead. The cookie
    argument is empty, and `sync_substack_saved` is what records the paid consequence."""
    hosted_chrome(payload={"items": [], "nextCursor": None})
    seen = []
    monkeypatch.setattr(sub, "_fetch_full_post",
                        lambda base, slug, cookies: seen.append(cookies) or {"body_html": "<p>x</p>"})

    source = sub.saved_source()
    source.full_post("https://pub.substack.com", "p1")

    assert seen == [{}]
    assert source.authenticated_body("https://pub.substack.com") is False


def test_the_local_transport_does_not_claim_a_session_a_custom_domain_refuses():
    """`build_cookie_header` writes the header by hand, so the session goes out to every host —
    but only `*.substack.com` honors it (measured 2026-09-08: a custom domain answered
    `confirmedLogin: False` with the cookie present).

    This was a class constant `authenticated_bodies = True` and it was false for every
    custom-domain publication, which made `sync_substack_saved` store a paid teaser as
    `body_state='complete'` — a claim `export` ships to shared KBs."""
    local = sub._LocalSaved()
    assert local.authenticated_body("https://pub.substack.com") is True
    assert local.authenticated_body("https://substack.com") is True
    assert local.authenticated_body("https://letters.example.com") is False
    # A host that merely CONTAINS the string is a different host.
    assert local.authenticated_body("https://substack.com.evil.example") is False
    assert local.authenticated_body("") is False
