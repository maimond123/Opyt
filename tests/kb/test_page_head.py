"""A page's title and description, as a link card shows them (`platforms/web/page_head.py`)."""
from types import SimpleNamespace

import pytest

from pipeline.platforms.web import page_head

_PAGE = ("<html><head><title>Agent Use Cases &amp; more</title>"
         '<meta name="description" content="  A curated directory\n of agent use cases. ">'
         "</head><body><p>hello</p></body></html>")


@pytest.mark.parametrize("answer, want", [
    (SimpleNamespace(status_code=200, text=_PAGE, url="https://a.ai/"),
     page_head.Head("Agent Use Cases & more", "A curated directory of agent use cases.")),
    (SimpleNamespace(status_code=404, text="gone", url="https://a.ai/"), page_head.Head()),
    (SimpleNamespace(status_code=503, text="", url="https://a.ai/"), None),     # may pass
    (ConnectionError("down"), None),
])
def test_what_a_page_says_and_when_to_ask_again(monkeypatch, answer, want):
    def request(send, url, **kw):
        if isinstance(answer, Exception):
            raise answer
        return answer
    monkeypatch.setattr(page_head, "host_request", request)
    assert page_head.read_head("https://a.ai/") == want
