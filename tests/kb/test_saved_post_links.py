"""Saved X posts follow their links: a bookmark or a
hand-saved post owes its paper, article, GitHub and Substack links to the referenced-links queue,
and the references worker captures them as it does an Oracle's."""
import json

from pipeline.curation import bookmarks
from pipeline.kb import ingest_x, link_cards, schema
from pipeline.kb.owed_links import saved_post_links
from pipeline.oracles.sources.references.cards import Cards
from pipeline.oracles.sources.references.documents import Documents
from pipeline.platforms.web import page_head
from pipeline.worker.quota_loop.loop import QuotaLoop
from pipeline.worker.quota_loop.retries import PLATFORM_PAUSES_S, Outages
from pipeline.worker.quota_loop.serial import Serial
from tests.kb.test_ingest_x_footprint import _NEWS_URL, _patch_news
from tests.kb.test_saved_x_posts import _norm, conn, offer, run  # noqa: F401  (conn: fixture)

_PAPER = "https://arxiv.org/abs/2401.00001"
_REPO = "https://github.com/bob/thing"
_SUBSTACK = "https://someone.substack.com/p/a-post"
_X = "https://x.com/dan/status/222"
_HOME = "https://agentusecases.ai/"
_CREATED = "Sat Aug 01 12:00:00 +0000 2026"      # X's own format, as the Bookmarks feed gives it


def _post(tid, *urls, author="1", handle="u", **extra):
    return {"id": tid, "text": f"post {tid}", "url": f"https://x.com/{handle}/status/{tid}",
            "createdAt": _CREATED, "author": {"id": author, "userName": handle},
            "entities": {"urls": [{"expanded_url": u} for u in urls]}, **extra}


def _urls(entries):
    return [e["url"] for e in entries]


def _payload(conn, atom_id):
    return json.loads(conn.execute("SELECT payload FROM atoms WHERE atom_id=?",
                                   (atom_id,)).fetchone()[0])


def test_a_saved_post_owes_what_an_oracle_post_would_and_not_its_authors_own_posts():
    own = "https://x.com/u/status/5"
    owed = saved_post_links(_post("100", _PAPER, _REPO, _SUBSTACK, _NEWS_URL, _X, own, _HOME), None,
                            "x:100")

    assert _urls(owed) == [_PAPER, _REPO, _SUBSTACK, _NEWS_URL, _X, _HOME]   # the home: its card
    assert owed[0]["ref"] == {"by": "x:user:1", "handle": "u", "post": "x:100",
                              "url": "https://x.com/u/status/100", "date": "2026-08-01"}


def test_only_the_saved_posts_own_thread_and_the_posts_it_quotes_owe_links():
    ancestor = _post("90", _REPO, author="2", handle="other")
    saved = _post("100", _PAPER, quoted_tweet=_post("80", _SUBSTACK, author="3", handle="q"))
    continuation = _post("101", _NEWS_URL)

    owed = saved_post_links(saved, [ancestor, saved, continuation], "x:100")

    assert _urls(owed) == [_PAPER, _NEWS_URL, _SUBSTACK]
    assert owed[1]["ref"]["url"] == "https://x.com/u/status/101"
    assert owed[2]["ref"]["via_quote"]["handle"] == "q"


def test_a_bookmark_the_listing_offered_is_saved_owing_its_links(conn):
    offer(conn, {**_norm(), "entities": {"urls": [{"expanded_url": u} for u in (_PAPER, _X)]}})

    assert run(conn).status == "saved"

    assert _urls(_payload(conn, "x:100")["links_owed"]) == [_PAPER, _X]


def test_a_post_an_oracle_linked_owes_only_its_paper_links(conn):
    post = _post("200", _REPO, _PAPER, _SUBSTACK, _NEWS_URL, _X, _HOME, author="2", handle="dan")
    status, aid = ingest_x.referenced_x_atom(conn, post,
                                             "# a linked post\n\npost 200", seen={})
    conn.commit()

    assert (status, aid) == ("saved", "x:200")
    assert _payload(conn, "x:200")["links_owed"] == [
        {"url": _PAPER, "ref": {"by": "x:user:2", "handle": "dan", "post": "x:200",
                                "url": "https://x.com/dan/status/200", "date": "2026-08-01"}}]


def test_the_references_worker_captures_a_bookmarks_link_with_its_referrer(conn, monkeypatch):
    _patch_news(monkeypatch)
    offer(conn, {**_norm(), "createdAt": _CREATED,
                 "entities": {"urls": [{"expanded_url": _NEWS_URL}]}})
    assert run(conn).status == "saved"
    health = Outages(PLATFORM_PAUSES_S)
    web = QuotaLoop("web", [Documents(health)], executor=Serial(), outages=health,
                    connect=schema.connect)
    web.ctx.cache.update(session={})

    assert web.step(conn).result == "captured"

    assert _payload(conn, "x:100")["links_owed"] == []
    mode, payload = conn.execute(
        "SELECT entry_mode, payload FROM atoms WHERE source_type='blog'").fetchone()
    assert mode == "author_referenced"
    assert json.loads(payload)["referenced_by"] == [
        {"by": "x:user:1", "handle": "u", "post": "x:100", "url": "https://x.com/u/status/100",
         "date": "2026-08-01"}]


def test_a_bookmarks_product_link_gets_its_card_and_a_re_render_keeps_it(conn, monkeypatch):
    post = {**_norm(), "createdAt": _CREATED, "entities": {"urls": [{"expanded_url": _HOME}]}}
    offer(conn, post)
    assert run(conn).status == "saved"
    monkeypatch.setattr(page_head, "read_head", lambda url: page_head.Head(
        "Agent Use Cases", "A directory of what people actually do with AI agents."))
    health = Outages(PLATFORM_PAUSES_S)
    web = QuotaLoop("web", [Cards(health)], executor=Serial(), outages=health,
                    connect=schema.connect)
    web.ctx.cache.update(session={})

    assert web.step(conn).result == "filled"

    assert f"> **[Agent Use Cases]({_HOME})**" in schema.read_body(conn, "x:100")
    assert _payload(conn, "x:100")["links_owed"] == []
    assert "unchanged" in bookmarks.render(post, bookmarks.stored(conn, "x:100"), None,
                                           link_cards.load(conn))
