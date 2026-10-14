"""`reader_view` on the local server: the items an answer cited, read from this library and made into a
page by the real service in-process (`conftest.publisher`; only the socket is shimmed).

What has to hold: the page shows the items as the store keeps them, in the order given; an install
that never shared registers once and holds its token aside, so making a page never makes the
publishing pass upload the library; and every failure is a sentence.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from mcp_server.page_tools import register_page_tools
from opyt_core import keys
from pipeline.sharing import account, client, publishing
from tests.opyt_core.conftest import URL


def _tools():
    got = {}

    class _Fake:
        def tool(self, *a, **kw):
            def deco(fn):
                got[fn.__name__] = fn
                return fn
            return deco

    register_page_tools(_Fake())
    return SimpleNamespace(**got)


reader_view = _tools().reader_view


def _read(publisher, url: str):
    with publisher.on_service():
        return publisher.svc.client.get(url.removeprefix(URL))


def test_a_page_shows_the_cited_items_in_order(publisher):
    got = reader_view(atom_ids=["substack:post1", "x:1", "nope:1", "x:1"], title="Agents and rollups",
               note="What the user asked", answer="Agents settle on **rollups**.")
    assert got["status"] == "made" and got["url"].startswith(f"{URL}/p/")
    assert got["missing"] == ["nope:1"] and "1 of the ids" in got["message"]
    assert "deleted 90 days after it was last opened" in got["message"]
    html = _read(publisher, got["url"]).text
    assert "<h1>Agents and rollups</h1>" in html and "What the user asked" in html
    assert "<strong>rollups</strong>" in html
    assert '<article class="item item-substack" id="s1">' in html
    assert '<article class="item item-x" id="s2">' in html and 'id="s3"' not in html
    assert "2 sources" in html and "a react dashboard for the web" in html


def test_an_install_that_never_shared_registers_once_and_publishes_nothing(publisher,
                                                                          monkeypatch):
    monkeypatch.delenv("OPYT_SERVICE_TOKEN", raising=False)
    creds = {}
    monkeypatch.setattr("pipeline.credentials.get_credential",
                        lambda service: creds.get("opyt_service"))
    monkeypatch.setattr(keys, "set_key", lambda name, value: creds.__setitem__(name, value))
    registered = []
    real = client.register
    monkeypatch.setattr(client, "register",
                        lambda url, label: registered.append(label) or real(url, label))
    first, second = reader_view(atom_ids=["x:1"], title="a"), reader_view(atom_ids=["x:2"], title="b")
    assert first["status"] == second["status"] == "made" and registered == [None]
    # Held aside, never the account: the publishing pass is offered nothing to upload.
    assert account.load().token is None and account.pending_token() and creds == {}
    assert publishing.Publications()._due(None) is None
    # The held token made both, so it can take either down.
    assert reader_view(delete=first["url"])["status"] == "deleted"


def test_a_page_is_taken_down_by_its_link(publisher):
    url = reader_view(atom_ids=["x:1"], title="a")["url"]
    assert reader_view(delete=url + "#s1")["status"] == "deleted"
    assert _read(publisher, url).status_code == 404
    assert reader_view(delete=url)["status"] == "not_found"


def test_nothing_in_the_library_makes_no_page(publisher):
    got = reader_view(atom_ids=["nope:1"], title="a")
    assert got["status"] == "not_found" and got["missing"] == ["nope:1"]


def test_an_unreachable_service_is_a_sentence(publisher, monkeypatch):
    def down(*a, **kw):
        raise client.Failed("unreachable", f"could not reach {URL}")
    monkeypatch.setattr(client, "make_page", down)
    got = reader_view(atom_ids=["x:1"], title="a")
    assert got["status"] == "failed" and "could not reach" in got["message"]


@pytest.mark.parametrize("ids", [None, []])
def test_no_ids_is_no_page(publisher, ids):
    assert reader_view(atom_ids=ids, title="a")["status"] == "not_found"
