"""Real listing adapters against saved transport responses; no whole-archive mocks."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from pipeline.kb.ingest_common import FETCH_OK, store_atom
from pipeline.worker.quota_loop.source import Context
from pipeline.oracles.sources.archive.adapter import Page
from pipeline.oracles.sources.archive.blog import Blog
from pipeline.oracles.sources.archive.github import Github
from pipeline.oracles.sources.archive.openalex import OpenAlex
from pipeline.oracles.sources.archive import state
from pipeline.oracles.sources.archive.substack import Substack

SPEC = {"url": "https://writer.example", "type": "blog", "since": None, "author": "Writer"}


class Response:
    headers = {}
    def __init__(self, value, status=200):
        self.value, self.status_code = value, status
        self.text = value if isinstance(value, str) else ""
        self.content = self.text.encode()
    def json(self):
        return self.value
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_substack_pages_follow_the_cursor_and_boundary_filters_before_admission():
    """The archive no longer pages by offset: it hands back `nextCursor`, and `offset` is refused
    with a 400. The adapter must send the cursor and never an offset."""
    responses = [
        Response({"posts": [{"id": 1, "post_date": "2026-09-01T00:00:00Z"}], "nextCursor": "c1"}),
        Response({"posts": [{"id": 2, "post_date": "2026-08-01T00:00:00Z"},
                            {"id": 3, "post_date": "2025-01-01T00:00:00Z"}]})]
    calls = []
    session = SimpleNamespace(get=lambda url, **kw: calls.append(kw["params"]) or responses.pop(0))
    ctx = Context("web", {"substack_sessions": {SPEC["url"]: session}})
    spec = {**SPEC, "since": "2026-01-01T00:00:00Z"}
    adapter = Substack()
    first = adapter.page(spec, adapter.initial(spec), ctx)
    assert first.cursors == [{"cursor": "c1"}]
    assert "offset" not in calls[0] and calls[0]["limit"] == 25
    second = adapter.page(spec, first.cursors[0], ctx)
    assert [key for key, _ in second.documents] == ["2"]
    assert second.cursors == [] and len(calls) == 2
    assert calls[1]["cursor"] == "c1" and "offset" not in calls[1]


def test_one_session_per_host_survives_every_page(monkeypatch):
    """Cloudflare's cookie lives on the session, so the pages of one host must share one."""
    made = []

    class FakeSession:
        def __init__(self):
            made.append(self)

        def get(self, url, **kw):
            return Response({"posts": [{"id": 1, "post_date": "2026-09-01T00:00:00Z"}],
                             "nextCursor": "c1"})

    monkeypatch.setattr("pipeline.oracles.sources.archive.substack.requests.Session", FakeSession)
    ctx = Context("web")
    adapter = Substack()
    adapter.page(SPEC, adapter.initial(SPEC), ctx)
    adapter.page(SPEC, {"cursor": "c1"}, ctx)
    assert len(made) == 1 and ctx.cache["substack_sessions"][SPEC["url"]] is made[0]


def test_substack_error_object_cannot_mean_empty_archive():
    ctx = Context("web", {"substack_sessions": {SPEC["url"]: SimpleNamespace(get=lambda *a, **kw: Response({"error": "blocked"}))}})
    with pytest.raises(Exception, match="list of posts"):
        Substack().page(SPEC, {"cursor": "c1"}, ctx)


def test_github_lists_one_page_and_filters_forks_and_untouched_repos(monkeypatch):
    records = [{"full_name": "a/new", "pushed_at": "2026-09-01T00:00:00Z"},
               {"full_name": "a/old", "pushed_at": "2020-01-01T00:00:00Z"},
               {"full_name": "a/fork", "fork": True}]
    calls = []
    monkeypatch.setattr("pipeline.oracles.sources.archive.github.get",
                        lambda path, **kw: calls.append((path, kw)) or Response(records))
    spec = {**SPEC, "url": "https://github.com/a", "since": "2026-01-01T00:00:00Z"}
    adapter = Github()
    page = adapter.page(spec, adapter.initial(spec), Context("web"))
    assert [key for key, _ in page.documents] == ["a/new"]
    assert page.cursors == [{"page": 2, "route": "users"}] and len(calls) == 1


def test_github_failed_readme_is_not_published_as_absent(conn, monkeypatch):
    monkeypatch.setattr("pipeline.oracles.sources.archive.github.request",
                        lambda *a, **kw: Response("unavailable", status=503))
    with pytest.raises(RuntimeError, match="503"):
        Github().document(conn, SPEC, {"full_name": "a/repo"}, Context("web"), None)


def test_openalex_resumes_exact_cursor_and_preserves_topic_filter():
    calls = []
    adapter = SimpleNamespace(available=lambda: True,
                              _call=lambda path, params: calls.append(params) or
                              {"results": [{"id": "W1"}], "meta": {"next_cursor": "next"}})
    spec = {**SPEC, "url": "S123", "topics": "T7|T9"}
    page = OpenAlex().page(spec, {"cursor": "saved"}, Context("web", {"openalex": adapter}))
    assert calls[0]["cursor"] == "saved"
    assert "T7" in calls[0]["filter"] and "T9" in calls[0]["filter"]
    assert page.cursors == [{"cursor": "next"}]


def test_openalex_document_holds_the_paper_for_the_run_to_write(conn):
    """The paper is held through the `hold` the archive run hands in, never written by the
    adapter itself: the run lands it with its task's completion in one transaction."""
    work = {"id": "https://openalex.org/W7", "doi": "https://doi.org/10.1234/w7",
            "title": "A Paper", "publication_date": "2026-03-01",
            "abstract_inverted_index": {"An": [0], "abstract": [1]},
            "authorships": [{"author": {"id": "https://openalex.org/A1", "display_name": "First Author"},
                             "author_position": "first"}]}
    held = []
    hold = lambda atom, md: held.append((atom, md))
    aid = OpenAlex().document(conn, {**SPEC, "url": "A1"}, work, Context("web"), hold)
    assert aid.startswith("paper:") and [atom["atom_id"] for atom, _ in held] == [aid]
    assert not conn.execute("SELECT 1 FROM atoms WHERE atom_id=?", (aid,)).fetchone()
    for atom, md in held:
        store_atom(conn, atom, md)
    assert conn.execute("SELECT 1 FROM atoms WHERE atom_id=?", (aid,)).fetchone()
    assert "A Paper" in conn.execute("SELECT text FROM bodies WHERE atom_id=?", (aid,)).fetchone()[0]
    assert OpenAlex().document(conn, {**SPEC, "url": "A1"}, work, Context("web"), hold) == "present"
    assert len(held) == 1


@pytest.mark.parametrize("payload", [None, {}, {"results": [{"id": "W1"}], "meta": {"next_cursor": "saved"}}])
def test_openalex_bad_or_repeated_listing_cannot_finish(payload):
    adapter = SimpleNamespace(available=lambda: True, _call=lambda *a: payload)
    with pytest.raises(Exception):
        OpenAlex().page({**SPEC, "url": "A123"}, {"cursor": "saved"}, Context("web", {"openalex": adapter}))


def test_blog_nested_sitemap_is_a_separate_durable_page(monkeypatch):
    responses = [Response('<sitemapindex><sitemap><loc>/posts.xml</loc></sitemap></sitemapindex>'),
                 Response('<urlset><url><loc>/essay</loc><lastmod>2026-07-01</lastmod></url></urlset>')]
    calls = []
    monkeypatch.setattr("pipeline.oracles.sources.archive.blog.request",
                        lambda send, url, **kw: calls.append(url) or responses.pop(0))
    adapter = Blog()
    first = adapter.page(SPEC, adapter.initial(SPEC), Context("web"))
    assert first.documents == [] and first.cursors[0]["urls"] == ["https://writer.example/posts.xml"]
    second = adapter.page(SPEC, first.cursors[0], Context("web"))
    assert second.cursors[0]["baseline"] == [{"url": "https://writer.example/essay", "lastmod": "2026-07-01"}]
    assert len(calls) == 2


@pytest.mark.parametrize("response", [Response("missing", 404), Response("<html>challenge</html>")])
def test_failed_child_sitemap_never_finishes_a_partial_listing(monkeypatch, response):
    monkeypatch.setattr("pipeline.oracles.sources.archive.blog.request", lambda *a, **kw: response)
    adapter = Blog()
    cursor = {**adapter.initial(SPEC), "indexed": True, "urls": ["https://writer.example/posts.xml"]}
    with pytest.raises(RuntimeError, match="sitemap"):
        adapter.page(SPEC, cursor, Context("web"))


def test_child_sitemaps_a_cms_files_under_non_post_types_are_never_read(monkeypatch):
    index = ('<sitemapindex><sitemap><loc>/wp-sitemap-posts-post-1.xml</loc></sitemap>'
             '<sitemap><loc>/wp-sitemap-posts-page-1.xml</loc></sitemap>'
             '<sitemap><loc>/wp-sitemap-taxonomies-companies-1.xml</loc></sitemap></sitemapindex>')
    monkeypatch.setattr("pipeline.oracles.sources.archive.blog.request", lambda *a, **kw: Response(index))
    adapter = Blog()
    first = adapter.page(SPEC, adapter.initial(SPEC), Context("web"))
    assert first.cursors[0]["urls"] == ["https://writer.example/wp-sitemap-posts-post-1.xml"]


def test_the_feed_marks_which_listed_pages_fit_its_posts(monkeypatch):
    feed = ('<rss><channel><item><link>https://writer.example/posts/one</link></item>'
            '<item><link>https://writer.example/posts/two</link></item></channel></rss>')
    monkeypatch.setattr("pipeline.oracles.sources.archive.blog.request", lambda *a, **kw: Response(feed))
    adapter = Blog()
    cursor = {**adapter.initial(SPEC), "phase": "feed", "urls": ["https://writer.example/feed"],
              "baseline": [{"url": "https://writer.example/posts/three", "lastmod": ""},
                           {"url": "https://writer.example/topics/ai", "lastmod": ""}]}
    after_feed = adapter.page(SPEC, cursor, Context("web")).cursors[0]
    selected = adapter.page(SPEC, {**after_feed, "phase": "select", "urls": []}, Context("web"))
    assert {d["url"]: d["fits_feed"] for _key, d in selected.documents} == {
        "https://writer.example/posts/three": True, "https://writer.example/topics/ai": False}


@pytest.mark.parametrize("said,saved", [({"article": False, "published": False}, False),
                                        ({"article": True, "published": True}, True)])
def test_a_page_off_the_feed_is_saved_only_if_it_says_it_is_a_dated_article(conn, monkeypatch,
                                                                            said, saved):
    url = "https://writer.example/topics/ai"
    monkeypatch.setattr("pipeline.oracles.sources.archive.blog.web_article.fetch_article",
                        lambda u: {"url": u, "content": "x", "declares": said})
    monkeypatch.setattr("pipeline.oracles.sources.archive.blog.ingest_blog._classify_article",
                        lambda article: FETCH_OK)
    monkeypatch.setattr("pipeline.oracles.sources.archive.blog.ingest_blog.build_article_atom",
                        lambda *a, **kw: {"outcome": "ok", "atom": {}, "md": ""})
    written = []
    ctx = SimpleNamespace(cache={"images": None})
    result = Blog().document(conn, {**SPEC, "entity": "blog:writer.example"},
                             {"url": url, "fits_feed": False}, ctx,
                             lambda atom, md: written.append(atom))
    assert (result != "skipped", bool(written)) == (saved, saved)


def test_cursor_cycle_rolls_back_documents_and_completion(conn):
    walk = state.start(conn, "a", "oracle", SPEC, {"page": 1}, datetime.now(timezone.utc))
    one = state.pending(conn, walk)[0]
    state.record_page(conn, walk, one, Page(cursors=[{"page": 2}]))
    two = state.pending(conn, walk)[0]
    with pytest.raises(RuntimeError, match="previous cursor"):
        state.record_page(conn, walk, two, Page([("bad", {"id": 99})], [{"page": 1}]))
    assert [r["id"] for r in state.pending(conn, walk)] == [two["id"]]
    assert conn.execute("SELECT count(*) FROM archive_units WHERE kind='document'").fetchone()[0] == 0


def test_a_link_to_one_repo_keeps_forks_and_ignores_the_account_window(monkeypatch):
    repo = {"full_name": "a/old-fork", "fork": True, "pushed_at": "2010-01-01T00:00:00Z"}
    monkeypatch.setattr("pipeline.oracles.sources.archive.github.get", lambda *a, **kw: Response(repo))
    spec = {**SPEC, "url": "https://github.com/a/old-fork", "since": "2026-01-01T00:00:00Z"}
    page = Github().page(spec, Github().initial(spec), Context("web"))
    assert page.documents == [("a/old-fork", repo)] and page.cursors == []
