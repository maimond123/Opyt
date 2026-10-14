"""post_people — who an Oracle's post quotes, mentions or links, with the Oracle's own words."""
from __future__ import annotations

from pipeline.kb.post_people import WORDS_LIMIT, post_people

_SELF = {"id": "1", "userName": "oracle"}


def _tw(tid, text="", *, mentions=(), urls=(), quoted=None):
    t = {"id": tid, "text": text, "author": _SELF,
         "entities": {"user_mentions": [{"id_str": i, "screen_name": h} for i, h in mentions],
                      "urls": [{"url": f"https://t.co/{n}", "expanded_url": u}
                               for n, u in enumerate(urls)]}}
    if quoted:
        t["quoted_tweet"] = quoted
    return t


def _quoted(handle="bob", uid="7", text="the quoted claim https://t.co/q"):
    return {"id": "500", "text": text, "author": {"id": uid, "userName": handle}}


def test_a_quote_names_its_author_and_keeps_only_the_oracles_words():
    rows = post_people([_tw("10", "This is the wrong take. https://t.co/ZZ", quoted=_quoted())])
    assert rows == [{"kind": "quote", "who": "x:user:7", "handle": "bob",
                     "words": "This is the wrong take.",
                     "url": "https://x.com/oracle/status/10"}]


def test_mentions_across_a_thread_each_with_the_tweet_that_names_them():
    rows = post_people([_tw("10", "1/ intro"),
                        _tw("11", "2/ @carol and I have something cooking",
                            mentions=[("8", "carol")])])
    assert [(r["kind"], r["who"], r["words"]) for r in rows] == [
        ("mention", "x:user:8", "2/ @carol and I have something cooking")]


def test_the_oracle_is_never_named_and_a_quoted_person_counts_once_as_the_quote():
    rows = post_people([_tw("10", "@bob replying to myself @oracle",
                            mentions=[("7", "bob"), ("1", "oracle")], quoted=_quoted())])
    assert [(r["handle"], r["kind"]) for r in rows] == [("bob", "quote")]


def test_a_quote_of_the_oracle_names_no_one():
    assert post_people([_tw("10", "still true", quoted=_quoted("oracle", "1"))]) == []


def test_a_link_to_someone_elses_post_names_them_by_handle():
    rows = post_people([_tw("10", "see", urls=["https://x.com/dana/status/42",
                                               "https://x.com/oracle/status/43",
                                               "https://x.com/i/status/44",
                                               "https://example.com/a"])])
    assert [(r["kind"], r["who"], r["handle"]) for r in rows] == [("link", None, "dana")]


def test_a_link_upgraded_to_a_mention_keeps_the_id():
    rows = post_people([_tw("10", "see", urls=["https://x.com/dana/status/42"]),
                        _tw("11", "@dana agreed", mentions=[("9", "dana")])])
    assert [(r["kind"], r["who"]) for r in rows] == [("link", "x:user:9")]


def test_words_are_capped():
    rows = post_people([_tw("10", "word " * 400, mentions=[("8", "carol")])])
    assert len(rows[0]["words"]) <= WORDS_LIMIT + 1 and rows[0]["words"].endswith("…")
