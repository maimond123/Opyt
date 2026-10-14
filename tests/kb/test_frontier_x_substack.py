"""X and Substack as watch sources: what each asks, how
its results become candidates, and a watch's context words from the topic to the X request.
Offline: the platform reads are replaced, the X page is the recorded fixture."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pipeline.frontier import intake, substack_candidates, x_candidates
from pipeline.frontier import queries as fq
from pipeline.kb import ingest_x, schema, vision
from pipeline.platforms.substack.contracts import SubstackFetchError
from pipeline.platforms.x import conversation as x_conversation
from pipeline.platforms.x import local as x_local
from pipeline.platforms.x import search as x_search
from pipeline.topics import create

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
X_PAGE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "x"
                     / "search_timeline.json").read_text())


# ── X ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("context, says", [
    ((), '"Boltz-2" lang:en -filter:replies since:2026-10-06'),
    (["antibody"], '"Boltz-2" antibody lang:en -filter:replies since:2026-10-06'),
    (["AI", "jobs", "labor market"],
     '"Boltz-2" (AI OR jobs OR "labor market") lang:en -filter:replies since:2026-10-06'),
])
def test_x_asks_for_the_name_as_a_phrase_narrowed_by_any_context_word(context, says):
    """Measured: an ambiguous name floods with its other meanings, and one context word fixes
    it ("Chai-2" + antibody → only the on-topic post)."""
    assert x_candidates.raw_query("Boltz-2", context, datetime(2026, 10, 6, 9,
                                                               tzinfo=timezone.utc)) == says


def test_a_quote_in_a_name_cannot_end_xs_phrase_early():
    assert x_candidates.raw_query('the "x402" protocol', ['"agents"'], None) == \
        '"the x402 protocol" agents lang:en -filter:replies'


def test_x_posts_become_candidates_keyed_like_their_atoms(monkeypatch):
    """Shown by author (the title) and text (the summary), as the atom's link is labelled once
    admitted; X's escaped `&amp;` reads as `&`."""
    asked = {}
    monkeypatch.setattr(x_candidates.sessions, "x_session",
                        lambda referer: asked.setdefault("referer", referer) and "client")
    monkeypatch.setattr(x_search, "search_posts",
                        lambda client, raw: asked.update(raw=raw) or x_search.parse_posts(X_PAGE))
    found = x_candidates.XAdapter().search("Boltz-2", since=NOW - timedelta(days=7), now=NOW,
                                           limit=20, context=["protein"])
    assert asked["raw"] == '"Boltz-2" protein lang:en -filter:replies since:2026-10-06'
    assert asked["referer"].startswith("https://x.com/search?q=%22Boltz-2%22")
    first = found[0]
    assert (first.candidate_id, first.source, first.kind) == ("x:2108556992570638761", "x", "x")
    assert first.title == "Fixture Lab (@fixturelab)"
    assert first.url == "https://x.com/fixturelab/status/2108556992570638761"
    assert first.published == "2026-10-09T13:55:36+00:00"
    assert "inhibitor & it lands" in first.summary
    assert first.payload == {"handle": "fixturelab", "likes": 46}
    assert len(found) == 2 and intake.named(found, "Boltz-2")[1] == 0


def test_x_is_asked_only_where_search_can_run(monkeypatch):
    monkeypatch.setattr(x_search, "available", lambda: False)
    assert x_candidates.XAdapter().available() is False


# ── Substack ───────────────────────────────────────────────────────────────────────
def _post(pid, title, *, subtitle="", published="2026-10-12T09:00:00Z", paid=False,
          url=None, publication="Bio Weekly", author="A. Writer"):
    return {"id": pid, "title": title, "subtitle": subtitle, "description": subtitle,
            "url": url or f"https://bio.substack.com/p/{pid}", "published": published,
            "paid": paid, "publication": publication, "author": author}


@pytest.fixture()
def substack(monkeypatch):
    """Substack's search and post reads, answered from `pages` and `bodies`, and recorded."""
    state = {"pages": [], "bodies": {}, "asked": [], "read": []}

    def search_posts(query, *, date_range, page=0):
        state["asked"].append((query, date_range, page))
        return state["pages"][page]

    def post_text(url):
        state["read"].append(url)
        body = state["bodies"][url]
        if isinstance(body, Exception):
            raise body
        return body

    monkeypatch.setattr(substack_candidates.substack_search, "search_posts", search_posts)
    monkeypatch.setattr(substack_candidates.substack_search, "post_text", post_text)
    return state


def test_substack_reads_two_pages_of_the_window_that_holds_the_pulls(substack):
    """A daily pull's window (~30 hours) is held by `week`, the plan's background setting."""
    substack["pages"] = [([_post("1", "rentosertib news")], True),
                         ([_post("2", "More rentosertib")], True)]
    found = substack_candidates.SubstackAdapter().search(
        "rentosertib", since=NOW - timedelta(hours=30), now=NOW, limit=50)
    assert substack["asked"] == [("rentosertib", "week", 0), ("rentosertib", "week", 1)]
    assert [c.candidate_id for c in found] == ["substack:1", "substack:2"]


def test_x_and_substack_declare_the_month_past_which_they_cannot_be_asked():
    """Applied once in `lanes.window_days`, so a years-long `look` is narrowed to a month here
    instead of returning nothing: Substack's `dateRange` has no window wider than `month`, and
    X answers one page of 20 from `Top` however wide the window."""
    assert substack_candidates.SubstackAdapter().max_window_days == 30
    assert x_candidates.XAdapter().max_window_days == 30


def test_substack_keeps_only_posts_since_the_windows_start(substack):
    """`dateRange` is coarse: a week's results go back further than a day's pull."""
    substack["pages"] = [([_post("old", "rentosertib", published="2026-10-07T00:00:00Z"),
                           _post("new", "rentosertib", published="2026-10-13T01:00:00Z")],
                          False)]
    found = substack_candidates.SubstackAdapter().search(
        "rentosertib", since=NOW - timedelta(hours=30), now=NOW, limit=50)
    assert [c.candidate_id for c in found] == ["substack:new"]


def test_a_post_that_does_not_name_it_up_front_is_read_for_its_mention(substack):
    """Search matches word by word and carries no body: "Virtual Cell Challenge" brought 20
    results, 3 naming it. The passage that names it is what the check and the host read."""
    named = _post("1", "The Virtual Cell Challenge results")
    loose = _post("2", "This week in bio", subtitle="Cells, challenges, and more")
    absent = _post("3", "Virtual reality cells")
    substack["pages"] = [([named, loose, absent], False)]
    substack["bodies"] = {loose["url"]: "Lots of news. " + "x " * 300 +
                          "The Virtual Cell Challenge dataset is out. " + "y " * 300,
                          absent["url"]: "Nothing on it here."}
    found = substack_candidates.SubstackAdapter().search(
        "Virtual Cell Challenge", since=NOW - timedelta(days=7), now=NOW, limit=50)
    assert substack["read"] == [loose["url"], absent["url"]]       # never the post naming it
    kept, dropped = intake.named(found, "Virtual Cell Challenge")
    assert [c.candidate_id for c in kept] == ["substack:1", "substack:2"] and dropped == 1
    summary = kept[1].summary
    assert summary.startswith("This week in bio — Cells, challenges, and more\n…")
    assert "The Virtual Cell Challenge dataset is out." in summary and summary.endswith("…")


def test_reading_for_mentions_is_capped_per_search(substack, monkeypatch):
    monkeypatch.setattr(substack_candidates, "MENTIONS", 2)
    posts = [_post(str(i), f"Post {i}") for i in range(5)]
    substack["pages"] = [(posts, False)]
    substack["bodies"] = {p["url"]: "it names rentosertib" for p in posts}
    found = substack_candidates.SubstackAdapter().search(
        "rentosertib", since=NOW - timedelta(days=7), now=NOW, limit=50)
    assert len(substack["read"]) == 2
    assert len(intake.named(found, "rentosertib")[0]) == 2


def test_a_failed_post_read_fails_the_search_so_nothing_is_half_staged(substack):
    """A pull that staged the posts it could read and stamped its window would never see the
    rest again: the whole pull fails and is tried again."""
    substack["pages"] = [([_post("1", "News")], False)]
    substack["bodies"] = {_post("1", "News")["url"]: SubstackFetchError("HTTP 403")}
    with pytest.raises(SubstackFetchError):
        substack_candidates.SubstackAdapter().search(
            "rentosertib", since=NOW - timedelta(days=7), now=NOW, limit=50)


def test_a_substack_post_is_shown_by_its_publication_and_a_paid_one_says_so(substack):
    substack["pages"] = [([_post("9", "rentosertib, priced", paid=True,
                                 url="https://www.forever.ai/p/priced", publication="Forever")],
                          False)]
    [c] = substack_candidates.SubstackAdapter().search(
        "rentosertib", since=NOW - timedelta(days=7), now=NOW, limit=50)
    assert (c.candidate_id, c.source, c.kind) == ("substack:9", "substack", "substack")
    assert (c.title, c.url) == ("Forever (paid)", "https://www.forever.ai/p/priced")
    assert c.payload == {"publication": "Forever", "author": "A. Writer", "paid": True}


# ── the name check's passage ─────────────────────────────────────────────────────────
def test_a_mention_is_the_passage_around_the_first_place_the_name_appears():
    text = "Intro. Boltz-2 is out. More."
    assert intake.mention("Boltz 2", text, width=6) == "…Boltz-2 is…"           # whole words
    assert intake.mention("Boltz 2", text, width=9) == "Intro. Boltz-2 is out.…"
    assert intake.mention("Boltz-2", "Boltz-2 is out", width=50) == "Boltz-2 is out"
    assert intake.mention("Chai-2", text) is None


# ── context words, from the topic to the X request ─────────────────────────────────
@pytest.fixture()
def conn(kb_home, monkeypatch):
    monkeypatch.setattr(create.store.chain, "queries_added", lambda: True)
    c = schema.connect()
    yield c
    c.close()


def test_a_watchs_context_words_reach_every_spelling_and_the_crawler(conn):
    out = create.create(conn, name="Antibody design",
                        watches=[{"name": "Chai-2", "spellings": ["Chai 2"],
                                  "context": ["antibody", "binder"]}],
                        words=None, people=None)
    assert out["watches"] == [{"name": "Chai-2", "spellings": ["Chai-2", "Chai 2"],
                               "sources": ["arxiv", "github", "openalex", "x", "substack"],
                               "context": ["antibody", "binder"]}]
    running = {q["text"]: fq.context(q) for q in fq.active_queries(conn)}
    assert running == {"Chai-2": ["antibody", "binder"], "Chai 2": ["antibody", "binder"]}


def test_an_admitted_x_find_is_a_strangers_post_not_a_save(conn, monkeypatch):
    """Minted by hopper's path, but a watch found it: entry mode `frontier`, no `save` signal (it
    would make the stranger a curation candidate), only its papers owed, and its body says how
    it came."""
    norm = {"id": "100", "text": "Boltz-2 is out https://t.co/a https://t.co/b",
            "author": {"id": "7", "name": "U", "userName": "u"},
            "url": "https://x.com/u/status/100", "createdAt": "Fri Oct 09 13:55:36 +0000 2026",
            "likeCount": 3, "replyCount": 0, "isReply": False,
            "entities": {"urls": [
                {"url": "https://t.co/a", "expanded_url": "https://arxiv.org/abs/2501.00001"},
                {"url": "https://t.co/b", "expanded_url": "https://example.com/post"}]},
            "extendedEntities": {"media": []}}
    monkeypatch.setattr(x_local, "read_x_cookies", lambda: {"auth_token": "t", "ct0": "c"})
    monkeypatch.setattr(x_local, "auth_headers", lambda *a, **k: {})
    monkeypatch.setattr(x_conversation, "fetch_tweets_by_ids", lambda *a, **k: [norm])
    monkeypatch.setattr(x_conversation, "fetch_conversation", lambda *a, **k: [])
    monkeypatch.setattr(vision, "enrich_tweet_media", lambda *a, **k: 0)

    assert ingest_x.x_atom_from_url(conn, norm["url"],
                                    entry_mode="frontier") == ("saved", "x:100")
    atom = conn.execute("SELECT entry_mode, payload FROM atoms WHERE atom_id = 'x:100'"
                        ).fetchone()
    assert atom["entry_mode"] == "frontier"
    assert [e["url"] for e in json.loads(atom["payload"])["links_owed"]] == [
        "https://arxiv.org/abs/2501.00001"]
    assert conn.execute("SELECT COUNT(*) FROM curation_signals").fetchone()[0] == 0
    assert "*Found by a watch ·" in schema.read_body(conn, "x:100")


def test_a_name_watched_before_x_and_substack_gains_them_when_watched_again(conn):
    """The live store holds standing queries from before (`Virtual Cell Challenge` on OpenAlex
    only): the first asker's sources must not stand forever, or no topic could ever
    have the name searched on X or Substack."""
    fq.add(conn, "Virtual Cell Challenge", ["openalex"], now="2026-10-12T23:29:12+00:00")
    create.create(conn, name="Cells", watches=["Virtual Cell Challenge"], words=None,
                  people=None)
    [query] = fq.active_queries(conn)
    assert json.loads(query["target_sources"]) == ["openalex", "arxiv", "github", "x",
                                                   "substack"]
    assert query["created_at"] == "2026-10-12T23:29:12+00:00"      # the same watch
    create.create(conn, name="Narrow", watches=[{"name": "Virtual Cell Challenge",
                                                 "sources": ["github"]}],
                  words=None, people=None)
    assert len(json.loads(fq.active_queries(conn)[0]["target_sources"])) == 5   # never narrows


def test_a_watch_added_again_with_new_context_words_takes_them(conn):
    """The real path: X floods on a name, and the host adds the word that narrows it."""
    topic = create.create(conn, name="T", watches=["Chai-2"], words=None, people=None)
    assert {q["text"]: fq.context(q) for q in fq.active_queries(conn)} == {"Chai-2": []}
    create.edit(conn, topic_id=topic["topic_id"], watches=[{"name": "Chai-2",
                                                           "context": "antibody"}])
    assert {q["text"]: fq.context(q) for q in fq.active_queries(conn)} == {
        "Chai-2": ["antibody"]}
