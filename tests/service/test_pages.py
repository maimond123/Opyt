"""Pages of sources on the service: made through `/v1/pages`, read at `/p/{id}`, gone 90 days
after they were last opened.

The service, its database and the routes are the real ones (`conftest.svc`). The network is the
one stand-in: `pdfs.fetch` is replaced where a test needs a paper's PDF to arrive, and `pdfs`'s
own `requests` where a test is about the fetch itself.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from service import db
from service.pages import make, pdfs, store

POST = {"atom_id": "x:1", "source_type": "x", "source_url": "https://x.com/alice/status/1",
        "raw": "---\nauthor: alice\n---\nrollups are cheap now", "payload": {}}
PAPER = {"atom_id": "paper:arXiv:2401.00001", "source_type": "paper",
         "source_url": "https://arxiv.org/abs/2401.00001",
         "raw": "# Cheap rollups\n\n## Abstract\n\nWe measure rollups.", "payload": {}}


def made(svc, *items: dict, headers=None) -> dict:
    r = svc.client.post("/v1/pages", headers=headers or svc.owner_hdr,
                        json={"title": "Rollups", "note": "What changed", "items": list(items)})
    assert r.status_code == 200, r.text
    return r.json()


def rows() -> int:
    conn = db.connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    finally:
        conn.close()


@pytest.fixture()
def pdf_arrives(monkeypatch):
    """Every PDF the service asks for arrives; the URLs asked for are kept."""
    asked = []

    def fetch(url, dest):
        asked.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.7 a paper")
        return True
    monkeypatch.setattr(pdfs, "fetch", fetch)
    return asked


# ── made and read ────────────────────────────────────────────────────────────────

def test_a_page_is_served_at_its_link_and_tells_nobody_else(svc):
    page = made(svc, POST)
    assert page["path"] == f"/p/{page['id']}" and len(page["id"]) == 22
    assert page["days"] == store.DAYS
    r = svc.client.get(page["path"])
    assert r.status_code == 200 and "rollups are cheap now" in r.text
    assert "<h1>Rollups</h1>" in r.text and "What changed" in r.text
    assert r.headers["content-security-policy"].startswith("default-src 'none'")
    assert r.headers["x-robots-tag"] == "noindex, nofollow"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["cache-control"] == "no-store"


def test_the_answer_a_page_was_made_with_is_drawn_above_its_sources(svc):
    r = svc.client.post("/v1/pages", headers=svc.owner_hdr,
                        json={"title": "Rollups", "answer": "Rollups got **cheap**.",
                              "items": [POST]})
    page = svc.client.get(r.json()["path"]).text
    assert "<strong>cheap</strong>" in page
    assert page.index('class="answer"') < page.index("rollups are cheap now")


def test_a_database_made_before_the_answer_gains_its_column(svc):
    """The box's `pages` table predates `answer`, and `CREATE TABLE IF NOT EXISTS` leaves it."""
    conn = db.connect()
    conn.executescript("DROP TABLE pages; CREATE TABLE pages (id TEXT PRIMARY KEY, owner TEXT "
                       "NOT NULL, title TEXT NOT NULL, note TEXT, items TEXT NOT NULL, made REAL "
                       "NOT NULL, expires REAL NOT NULL);")
    conn.close()
    page = made(svc, POST)
    assert "rollups are cheap now" in svc.client.get(page["path"]).text


def test_an_unknown_link_and_a_malformed_one_get_the_expired_page(svc):
    for path in ("/p/AAAAAAAAAAAAAAAAAAAAAA", "/p/x", "/p/" + "A" * 23, "/p/..service.db"):
        r = svc.client.get(path)
        assert r.status_code == 404 and "This page has expired" in r.text, path


def test_an_expired_page_is_gone_when_asked_for_and_its_row_with_it(svc, pdf_arrives):
    page = make.create(svc.owner, [PAPER], "Old", None, None, time.time() - store.LIFETIME - 1)
    assert store.pdf_path(page["id"], 1).exists()
    r = svc.client.get(page["path"])
    assert r.status_code == 404 and "This page has expired" in r.text
    assert rows() == 0 and not store.pdf_path(page["id"], 1).parent.exists()


def test_opening_a_page_starts_its_days_again(svc):
    """A page goes `LIFETIME` after it was last opened, not after it was made."""
    now = time.time()
    page = make.create(svc.owner, [POST], "Read lately", None, None, now - store.LIFETIME + 60)
    assert svc.client.get(page["path"]).status_code == 200      # a minute before it would go
    assert store.purge(now + store.LIFETIME - 60) == 0
    assert store.purge(now + store.LIFETIME + 60) == 1


def test_everything_that_states_the_lifetime_states_the_same_days():
    """`store.DAYS` is the one number. A docstring and a web page cannot read it, so each place
    that tells a user or a host how long a reader view lasts is checked against it here."""
    root = Path(__file__).resolve().parents[2]
    phrase = f"{store.DAYS} days after it was last opened"
    for path in ("mcp_server/page_tools.py", "service/connector.py", "TELEMETRY.md",
                 "README.md", "plugin/README.md"):
        assert phrase in " ".join(root.joinpath(path).read_text().split()), path


def test_the_nightly_purge_keeps_live_pages_and_clears_everything_else(svc, pdf_arrives):
    now = time.time()
    old = make.create(svc.owner, [PAPER], "Old", None, None, now - store.LIFETIME - 1)
    live = make.create(svc.owner, [PAPER], "Live", None, None, now)
    abandoned = store.pdf_path(store.new_id(), 1).parent   # a making that died before its row
    abandoned.mkdir(parents=True)
    os.utime(abandoned, (now - 7200, now - 7200))
    making = store.pdf_path(store.new_id(), 1).parent      # a making still running
    making.mkdir(parents=True)
    assert store.purge(now) == 1
    assert store.read(old["id"], now) is None and store.read(live["id"], now) is not None
    assert store.pdf_path(live["id"], 1).exists() and not store.pdf_path(old["id"], 1).exists()
    assert not abandoned.exists() and making.exists()


# ── what the service keeps of the items ───────────────────────────────────────────

def test_a_client_cannot_set_the_pdf_a_page_frames(svc, monkeypatch):
    monkeypatch.setattr(pdfs, "fetch", lambda url, dest: False)
    page = made(svc, {**PAPER, "pdf_url": "https://evil.test/x.pdf", "cite": "kept?"})
    [item] = store.read(page["id"], time.time()).items
    assert "pdf_url" not in item and "cite" not in item and item["raw"] == PAPER["raw"]


def test_nothing_to_show_and_too_much_are_refused_before_anything_is_written(svc):
    for items in ([], [POST] * (make.MAX_ITEMS + 1)):
        r = svc.client.post("/v1/pages", headers=svc.owner_hdr, json={"items": items})
        assert r.status_code == 400
    assert rows() == 0


def test_a_page_with_no_title_is_called_sources(svc):
    page = make.create(svc.owner, [POST], "  ", None, None, time.time())
    assert store.read(page["id"], time.time()).title == make.DEFAULT_TITLE


# ── PDFs ──────────────────────────────────────────────────────────────────────────

def test_a_papers_pdf_is_served_from_the_pages_own_address(svc, pdf_arrives):
    page = made(svc, POST, PAPER)
    assert pdf_arrives == ["https://arxiv.org/pdf/2401.00001"]
    html = svc.client.get(page["path"]).text
    assert f'<iframe src="{page["path"]}/pdf/2#view=FitH"' in html
    r = svc.client.get(f"{page['path']}/pdf/2")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content == b"%PDF-1.7 a paper"
    assert svc.client.get(f"{page['path']}/pdf/1").status_code == 404


def test_a_pdf_that_does_not_arrive_never_fails_the_page(svc, monkeypatch):
    monkeypatch.setattr(pdfs, "fetch", lambda url, dest: False)
    html = svc.client.get(made(svc, PAPER)["path"]).text
    assert "We measure rollups." in html and "no open PDF" in html


def test_a_pdf_lost_from_disk_leaves_the_abstract_and_link(svc, pdf_arrives):
    page = made(svc, PAPER)
    store.pdf_path(page["id"], 1).unlink()
    html = svc.client.get(page["path"]).text
    assert "<iframe" not in html and "no open PDF" in html


def test_a_full_disk_fetches_no_pdf(svc, pdf_arrives, monkeypatch):
    monkeypatch.setattr(store, "has_room", lambda: False)
    made(svc, PAPER)
    assert pdf_arrives == []


def test_where_a_papers_pdf_is_looked_for():
    stored = {**PAPER, "atom_id": "paper:DOI:10.1/x",
              "payload": {"external_ids": {"ArXiv": "2401.00002v3"},
                          "pdf_url": "https://journal.test/x.pdf"}}
    assert pdfs.urls(stored) == ["https://arxiv.org/pdf/2401.00002",
                                 "https://journal.test/x.pdf"]
    # A served copy keeps no `external_ids`: an arXiv paper's id is enough.
    assert pdfs.urls(PAPER) == ["https://arxiv.org/pdf/2401.00001"]
    assert pdfs.urls({**PAPER, "atom_id": "paper:DOI:10.1/x"}) == []
    assert pdfs.urls(POST) == []


# ── the fetch: HTTPS every hop, a PDF, not too big ────────────────────────────────

class _Response:
    def __init__(self, status=200, body=b"", location=None):
        self.status_code, self.body = status, body
        self.headers = {"location": location} if location else {}
        self.is_redirect = location is not None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def iter_content(self, size):
        for i in range(0, len(self.body), size):
            yield self.body[i:i + size]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def web(monkeypatch):
    """`pdfs`' view of the web: URL → response. Every URL asked for is kept."""
    pages, asked = {}, []

    def get(url, **kw):
        asked.append(url)
        return pages[url]
    monkeypatch.setattr(pdfs, "requests", SimpleNamespace(get=get))
    return SimpleNamespace(pages=pages, asked=asked)


def test_a_pdf_is_followed_across_https_redirects_into_its_file(web, tmp_path):
    web.pages["https://doi.test/1"] = _Response(301, location="https://pub.test/1.pdf")
    web.pages["https://pub.test/1.pdf"] = _Response(body=b"%PDF-1.4 " + b"x" * 200_000)
    dest = tmp_path / "1.pdf"
    assert pdfs.fetch("https://doi.test/1", dest) is True
    assert dest.read_bytes().startswith(b"%PDF-1.4") and dest.stat().st_size == 200_009
    assert not (tmp_path / "1.pdf.part").exists()


def test_a_plain_http_hop_is_never_requested(web, tmp_path):
    """The service's own loopback services answer plain HTTP; a page must not reach them."""
    web.pages["https://pub.test/a"] = _Response(302, location="http://127.0.0.1:8000/v1/stats")
    assert pdfs.fetch("http://127.0.0.1:9222/json/list", tmp_path / "1.pdf") is False
    assert pdfs.fetch("https://pub.test/a", tmp_path / "1.pdf") is False
    assert web.asked == ["https://pub.test/a"]


def test_what_is_not_a_pdf_or_is_too_big_is_not_kept(web, tmp_path, monkeypatch):
    web.pages["https://pub.test/page"] = _Response(body=b"<html>a landing page</html>")
    web.pages["https://pub.test/huge.pdf"] = _Response(body=b"%PDF-1.4" + b"x" * 300)
    web.pages["https://pub.test/gone.pdf"] = _Response(404)
    monkeypatch.setattr(pdfs, "MAX_BYTES", 200)
    for url in ("https://pub.test/page", "https://pub.test/huge.pdf", "https://pub.test/gone.pdf"):
        assert pdfs.fetch(url, tmp_path / "1.pdf") is False, url
    assert list(tmp_path.iterdir()) == []


def test_a_pdf_whose_robots_txt_closes_it_is_never_requested(web, tmp_path, monkeypatch):
    """The box follows the robots.txt rule the install's own reads follow, on every hop: a
    publisher that closes its PDFs to robots gets a link from the page, not a fetch."""
    from pipeline import robots
    closed = robots.parse("User-agent: *\nDisallow: /papers/\n")
    monkeypatch.setattr(robots, "_read", lambda origin, ua: closed)
    web.pages["https://doi.test/2"] = _Response(301, location="https://pub.test/papers/2.pdf")
    web.pages["https://pub.test/papers/2.pdf"] = _Response(body=b"%PDF-1.4 x")
    assert pdfs.fetch("https://pub.test/papers/2.pdf", tmp_path / "1.pdf") is False
    assert pdfs.fetch("https://doi.test/2", tmp_path / "1.pdf") is False
    assert web.asked == ["https://doi.test/2"]


# ── taking a page down ────────────────────────────────────────────────────────────

def test_only_the_library_that_made_a_page_takes_it_down(svc):
    page = made(svc, POST)
    other = svc.client.post("/v1/register", json={}).json()
    delete = f"/v1/pages/{page['id']}/delete"
    assert svc.client.post(delete, headers={"Authorization": f"Bearer {other['token']}"},
                           json={}).json() == {"deleted": False}
    assert svc.client.get(page["path"]).status_code == 200
    assert svc.client.post(delete, headers=svc.owner_hdr, json={}).json() == {"deleted": True}
    assert svc.client.get(page["path"]).status_code == 404
