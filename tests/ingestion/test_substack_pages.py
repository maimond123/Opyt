"""One page of a Substack account list, with the evidence a durable walk needs to continue or end.

The transports are faked; the payload shapes are the measured ones the drains already read.
"""

import pytest

from pipeline.platforms import sessions
from pipeline.platforms.substack import account as substack_account
from pipeline.platforms.substack import contracts as substack_contracts
from pipeline.platforms.substack import http as substack_http
from pipeline.platforms.substack import local as substack_local
import pipeline.platforms.collection_page as collection_page
from pipeline.platforms.collection_page import Incomplete


class Pages:
    """A transport serving canned pages by cursor; None is a refused read."""

    def __init__(self, pages):
        self.pages = pages
        self.asked = []

    def page(self, cursor):
        self.asked.append(cursor)
        return self.pages[cursor]


def saved_item(pid, kind="post"):
    if kind != "post":
        return {"type": kind, "comment": {"body": f"note {pid}"}}
    return {"type": "post", "post": {"id": pid, "title": f"T{pid}", "slug": f"p{pid}"},
            "publication": {"name": "Carol", "base_url": "https://carol.substack.com"}}


def test_a_saved_page_without_a_next_cursor_ends_the_list():
    page = substack_account.saved_page(Pages({None: {"items": [saved_item(1), saved_item(2)]}}))

    assert [key for key, _ in page.records] == ["1", "2"]
    assert page.records[0][1]["slug"] == "p1"
    assert page.ended and page.cursor is None


def test_a_saved_page_with_a_cursor_continues_and_sends_it_next():
    source = Pages({None: {"items": [saved_item(1)], "nextCursor": "c2"},
                    "c2": {"items": [saved_item(2)]}})

    first = substack_account.saved_page(source)
    second = substack_account.saved_page(source, first.cursor)

    assert not first.ended and first.cursor == "c2"
    assert second.ended and source.asked == [None, "c2"]


def test_a_note_occupies_its_place_in_the_list_without_being_a_record():
    """A page of only notes is not an empty list: it is observed, so a repeat of it cannot end
    the walk silently, and it carries no saved post."""
    page = substack_account.saved_page(
        Pages({None: {"items": [saved_item(7, "comment")], "nextCursor": "c"}}))

    assert page.records == ()
    assert len(page.observed) == 1 and page.observed[0].startswith("comment:")
    assert collection_page.continuation(page, set(), set()) == "c"


def test_a_refused_saved_page_raises_instead_of_reading_as_empty():
    with pytest.raises(substack_contracts.SubstackListingError):
        substack_account.saved_page(Pages({None: None}))


@pytest.mark.parametrize("data", [{}, {"items": None}, [], {"items": [], "nextCursor": 5}])
def test_a_saved_response_without_its_list_cannot_end_the_walk(data):
    with pytest.raises(Incomplete):
        substack_account.saved_page(Pages({None: data}))


SUBS = {
    "subscriptions": [
        {"id": 11, "publication_id": 101, "membership_state": "free_signup"},
        {"id": 13, "publication_id": 999, "membership_state": "free_signup"},   # no join
    ],
    "publications": [{"id": 101, "name": "Free Pub", "subdomain": "freepub"}],
}


def test_a_subscription_page_without_a_next_cursor_ends_the_list():
    page = substack_account.subscription_page(Pages({None: SUBS}))

    assert [(key, rec["url"]) for key, rec in page.records] == \
        [("11", "https://freepub.substack.com")]
    assert page.observed == ("11", "13")        # the unjoined row still holds its place
    assert page.ended


def test_a_subscription_cursor_that_repeats_page_one_is_incomplete_not_an_end():
    """The `cursor=` parameter name is inferred; a server that ignores it re-serves page 1. The
    drain stopped with what it had; the walk refuses to call that the whole list."""
    page = substack_account.subscription_page(Pages({None: dict(SUBS, nextCursor="again")}))

    with pytest.raises(Incomplete):
        collection_page.continuation(page, {"11", "13"}, set())


def test_a_refused_or_shapeless_subscription_page_never_reads_as_empty():
    with pytest.raises(substack_contracts.SubstackListingError):
        substack_account.subscription_page(Pages({None: None}))
    with pytest.raises(Incomplete):
        substack_account.subscription_page(Pages({None: {"publications": []}}))


def test_the_follow_list_is_one_page_read_for_the_known_account():
    class Follows:
        def follows(self, user_id=None):
            self.user_id = user_id
            return [{"name": "A", "url": "https://a.substack.com"},
                    {"name": "B", "url": "https://b.example.com"}]

    source = Follows()
    page = substack_account.follows_page(source, 42)

    assert source.user_id == 42
    assert [key for key, _ in page.records] == ["https://a.substack.com", "https://b.example.com"]
    assert page.ended and page.cursor is None


def test_the_local_follow_read_uses_the_account_it_is_given(monkeypatch):
    """No second reader-page request for an id the walk already holds."""
    monkeypatch.setattr(substack_local, "read_substack_cookies", lambda: {"substack.sid": "s"})
    monkeypatch.setattr(substack_local, "own_user_id", lambda cookies: pytest.fail("re-read the account"))
    monkeypatch.setattr(substack_http, "authed_get_json", lambda url, params, cookies: {"url": url})
    monkeypatch.setattr(substack_account, "parse_subscriber_lists", lambda data: [data])

    assert substack_local.LocalFollows().follows(42) == [
        {"url": "https://substack.com/api/v1/user/42/subscriber-lists"}]


def test_the_account_id_reads_the_local_session(monkeypatch):
    monkeypatch.setattr(substack_local, "read_substack_cookies", lambda: {"substack.sid": "s"})
    monkeypatch.setattr(substack_local, "own_user_id", lambda cookies: 42 if cookies else None)

    assert sessions.own_account_id() == 42


def test_the_parser_joins_a_subscription_to_its_publication():
    """The payload is two parallel arrays; the join is `subscriptions[].publication_id` ->
    `publications[].id` (measured 36/36). A subscription with no publication in the join has no
    URL, and the URL is the entity key, so it is dropped rather than keyed on `substack:unknown`.
    A custom domain wins over the subdomain: 20 of 36 measured subscriptions are on one."""
    page = {"subscriptions": [
                {"id": 11, "publication_id": 101, "membership_state": "free_signup",
                 "is_favorite": False},
                {"id": 12, "publication_id": 102, "membership_state": "subscribed",
                 "is_favorite": True},
                {"id": 13, "publication_id": 999, "membership_state": "free_signup"}],
            "publications": [
                {"id": 101, "name": "Free Pub", "subdomain": "freepub", "custom_domain": None},
                {"id": 102, "name": "Paid Pub", "subdomain": "paidpub",
                 "custom_domain": "letters.example.com"}]}

    assert substack_account.parse_subscription_page(page) == [
        {"id": 11, "name": "Free Pub", "url": "https://freepub.substack.com",
         "membership_state": "free_signup", "is_favorite": False},
        {"id": 12, "name": "Paid Pub", "url": "https://letters.example.com",
         "membership_state": "subscribed", "is_favorite": True},
    ]
