"""offers — at most one offer per answer, backed by a measurement.

The failures that matter are quiet ones: an offer the store cannot deliver (a count that matched
one common word), the same offer on every answer (the host drops it and the next-best never
shows), an offer for what the user is already doing, and an offer riding a call that is mid-setup
or irreversible. `pick` and the menus are plain values, so most of this needs no store.
"""
from __future__ import annotations

import pytest

from mcp_server import offers, rider_rules, riders
from mcp_server.offers import Author, Facts, History, Offer
from mcp_server.rider_rules import Call
from pipeline.kb import schema
from pipeline.oracles import probe_store
from pipeline.retrieval import corpus_census
from pipeline.retrieval.rank import fts_phrase
from tests.kb.test_candidate_search import _seed


def _offer(name, measure, bar=5, subject="rollups"):
    return Offer(name, subject, measure, bar, f"say {name}", f"do {name}")


# ── pick: the bar, a strong fit once used, rotation, what the call is doing ─────

def test_an_offer_below_its_bar_is_not_made():
    assert offers.pick([_offer("circle", 4)], None, History()) is None
    assert offers.pick([_offer("circle", 5)], None, History()).name == "circle"


def test_a_used_feature_needs_twice_its_bar():
    used = History(used=frozenset({"circle"}))
    assert offers.pick([_offer("circle", 9)], None, used) is None
    assert offers.pick([_offer("circle", 10)], None, used).name == "circle"


def test_offers_rotate_by_offer_whatever_the_subject():
    """Keyed on the subject too, the circle offer won all ten searches of a run, and the
    host, keeping each offer to once a conversation, would have shown nothing after the first."""
    menu = [_offer("circle", 9), _offer("topic", 20, bar=10), _offer("people", 9)]
    assert offers.pick(menu, None, History()).name == "circle"
    assert offers.pick(menu, None, History(shown=("circle",))).name == "topic"
    assert offers.pick(menu, None, History(shown=("circle", "topic"))).name == "people"
    assert offers.pick(menu, None,
                       History(shown=("topic", "people", "circle"))).name == "topic"
    assert offers.pick([_offer("circle", 9)], None, History(shown=("circle",))) is None


def test_no_offer_leads_to_what_the_call_is_already_doing():
    menu = [_offer("circle", 9), _offer("topic", 9)]
    assert offers.pick(menu, "circle", History()).name == "topic"


@pytest.mark.parametrize("tool, args, feature", [
    ("watch", {"action": "create"}, "topic"),
    ("watch", {"action": "checked"}, "topic"),
    ("oracle", {"action": "candidates", "query": "q", "oracle": None}, "circle"),
    ("oracle", {"action": "candidates", "query": None, "oracle": "imas"}, "engages"),
    ("oracle", {"action": "confirm"}, "people"),
    ("oracle", {"action": "map"}, "map"),
    ("add_oracle", {"confirm": True}, "people"),
    ("add_oracle", {"confirm": False}, None),
    ("hopper", {"confirm": True}, None),
    ("search", {"query": "q"}, None),
    ("reader_view", {"atom_ids": ["x:1"]}, "reader_view"),
])
def test_the_feature_a_call_uses(tool, args, feature):
    assert offers.feature_of(Call(tool, args)) == feature


# ── the menus: what fits each answer ──────────────────────────────────────

def test_a_search_leads_with_what_the_people_you_read_think():
    facts = Facts(subject="rollups", items=30, people=12)
    names = [o.name for o in offers.menu(Call("search", {"query": "rollups"}), facts)]
    assert names == ["circle", "topic", "people", "reader_view"]
    assert offers.pick(offers.menu(Call("search", {}), facts), None, History()).say.startswith(
        "12 people you read have posted about rollups")


def test_a_search_with_several_hits_offers_its_sources_in_reader_view():
    """A query of common words names no subject, so reader view is the one offer left to make."""
    search = Call("search", {"query": "the"})
    chosen = offers.pick(offers.menu(search, Facts(hits=3)), None, History())
    assert chosen.name == "reader_view" and chosen.if_yes.startswith("reader_view(atom_ids=")
    assert offers.pick(offers.menu(search, Facts(hits=2)), None, History()) is None
    made_one = History(used=frozenset({"reader_view"}))
    assert offers.pick(offers.menu(search, Facts(hits=5)), None, made_one) is None
    assert offers.pick(offers.menu(search, Facts(hits=6)), None, made_one).name == "reader_view"


def test_reader_view_rides_every_search_of_an_answer_when_no_other_offer_is_due():
    """A host runs several searches for one answer. Spent on the first and barred from the rest,
    the offer never reached the answer it was for (six searches
    behind one answer, none carrying it)."""
    search = Call("search", {"query": "value capture labs margins"})
    just_shown = History(shown=("people", "topic", "reader_view"))
    assert offers.pick(offers.menu(search, Facts(subject="value capture", hits=6)), None,
                       just_shown).name == "reader_view"
    # A subject offer that is due still takes the turn, and no other offer repeats.
    assert offers.pick(offers.menu(search, Facts(subject="value capture", hits=6, people=5)),
                       None, just_shown).name == "circle"
    assert offers.pick([_offer("circle", 9)], None, History(shown=("circle",))) is None


def test_a_subject_a_followed_topic_already_covers_is_not_offered_one():
    facts = Facts(subject="Rollups", items=30, followed=frozenset({"rollups"}))
    [topic] = [o for o in offers.menu(Call("search", {}), facts) if o.name == "topic"]
    assert topic.measure == 0


def test_a_subject_with_too_little_on_it_is_not_offered_a_topic():
    few = Facts(subject="rollups", items=9)
    assert offers.pick([o for o in offers.menu(Call("search", {}), few)
                        if o.name == "topic"], None, History()) is None


def test_a_store_that_follows_no_topic_is_offered_one_and_then_people():
    """A thin store and a deep one get the same offer: a topic's watches find what is not there
    yet, and its words catch what the user's people post on it."""
    thin = offers.pick(offers.store_offers(Facts(items=4)), None, History())
    deep = offers.pick(offers.store_offers(Facts(items=400)), None, History())
    following = offers.pick(offers.store_offers(Facts(items=400, followed=frozenset({"x"}),
                                                      waiting=30)), None, History())
    assert (thin.name, deep.name, following.name) == ("topic", "topic", "people")
    assert "watch(action='preview')" in deep.if_yes


def test_an_oracles_post_offers_who_they_engage_with_and_anyone_elses_offers_them_as_one():
    oracle = Author("Imas", "@imas", "imas", True, saved=0, named=40)
    other = Author("Bob", "@bob", None, False, saved=3, named=0)
    oracle_not_on_x = Author("Ann", None, None, True, saved=9, named=0)
    assert [o.name for o in offers.menu(Call("open", {}), Facts(author=oracle))] == ["engages"]
    [make] = offers.menu(Call("open", {}), Facts(author=other))
    assert make.name == "oracle" and "@bob" in make.if_yes and make.measure == 3
    assert offers.menu(Call("open", {}), Facts(author=oracle_not_on_x)) == []


def test_the_screen_offers_the_map_then_who_an_oracle_engages_with_then_the_circle():
    facts = Facts(people=200, top_oracle=("deanwball", 300))
    on_screen = offers.menu(Call("oracle", {"action": "screen"}), facts)
    assert [o.name for o in on_screen] == ["map", "engages", "circle"]
    by_oracle = Call("oracle", {"action": "candidates", "oracle": "deanwball"})
    after_map = History(shown=("map",))
    assert offers.pick(offers.menu(by_oracle, facts), offers.feature_of(by_oracle),
                       after_map).name == "circle"


def test_the_map_is_offered_only_once_enough_people_are_sampled():
    menu = offers.store_offers(Facts(followed=frozenset({"x"}), people=49))
    assert offers.pick(menu, None, History()) is None
    menu = offers.store_offers(Facts(followed=frozenset({"x"}), people=50))
    assert offers.pick(menu, None, History()).if_yes == "oracle(action='map')"


# ── the measurements: every word of the subject, or it is not about it ────────

def test_a_subject_counts_only_where_it_appears_as_a_phrase(kb_home):
    """Every word in one chunk counted "agent memory" in 180 people,
    about a third on topic; the phrase counted 25."""
    conn = schema.connect()
    _seed(conn,
          trusted=[("x:user:1", "rollup proofs are cheap"), ("x:user:2", "proofs for a rollup"),
                   ("x:user:3", "zero knowledge proofs")],
          probe=[("x:user:4", "rollup proofs again"), ("x:user:5", "a rollup and its proofs")])
    assert corpus_census.subject_count(conn, "rollup proofs") == 1
    assert probe_store.count_people_matching(conn, fts_phrase("rollup proofs")) == 1
    assert corpus_census.saved_count(conn, ["x:user:1", "x:user:2"]) == 0
    conn.close()


def test_a_search_of_only_common_words_offers_nothing(kb_home):
    """A host listed one person's posts by searching "the", and the
    answer offered to read "the" end to end (962 items). Common words name no subject."""
    conn = schema.connect()
    _seed(conn, trusted=[(f"x:user:{i}", f"the rollup proof number {i}") for i in range(12)])
    assert offers.offer(conn, Call("search", {"query": "the"}), {}) is None
    assert offers.offer(conn, Call("search", {"query": "of the"}), {}) is None
    made = offers.offer(conn, Call("search", {"query": "the rollup"}), {})
    assert made["say"].startswith("You have 12 items on the rollup")
    conn.close()


# ── the seam: use is stamped on every call, and the history survives the call ──

def test_every_call_stamps_its_feature_even_one_that_carries_no_offer(kb_home):
    riders.attach(Call("add_oracle", {"confirm": True}), {"status": "added"})
    riders.attach(Call("watch", {"action": "create"}), {"status": "ok"})
    assert offers._read_history().used == {"people", "topic"}


def test_a_handed_over_offer_is_stamped_so_the_next_answer_rotates(kb_home):
    facts_menu = offers.store_offers(Facts(items=400, waiting=30))
    first = offers._hand_over(offers.pick(facts_menu, None, offers._read_history()),
                              offers._read_history())
    second = offers.pick(facts_menu, None, offers._read_history())
    assert first["say"].startswith("You don't follow any topic yet") and second.name == "people"
    assert set(first) == {"say", "if_yes", "host_note"}


# ── the push: reader view on a new user's first answers ────────────────────────

def test_reader_view_is_pushed_until_the_user_has_made_one():
    """Said once, an offer can be missed, so it leads every answer's menu until a reader view is
    made; then the rotation is back and it needs a strong fit."""
    menu = offers.menu(Call("search", {"query": "rollups"}),
                       Facts(subject="rollups", hits=6, people=9))
    assert offers.push(menu, None, History()).name == "reader_view"      # ahead of the circle offer
    assert offers.push(menu, None, History(shown=("reader_view",))).name == "reader_view"
    made_one = History(used=frozenset({"reader_view"}))
    assert offers.push(menu, None, made_one) is None
    assert offers.pick(menu, None, made_one).name == "circle"


def test_a_pushed_offer_tells_the_host_to_say_it_on_its_first_three_turns(kb_home):
    """The host counts the turns: the server sees searches, and one answer runs several."""
    search, answer = Call("search", {"query": "the"}), {"hits": [{}] * 6}
    conn = schema.connect()
    pushed = offers.offer(conn, search, answer)
    again = offers.offer(conn, search, answer)
    assert pushed["host_note"] == again["host_note"] == offers.PUSH_NOTE
    assert "first three turns" in offers.PUSH_NOTE
    offers.note_use(Call("reader_view", {}))
    assert offers.offer(conn, search, answer)["host_note"] == offers.HOST_NOTE
    conn.close()


# ── which answers may carry one ───────────────────────────────

@pytest.mark.parametrize("call, answer", [
    (Call("add_oracle", {"confirm": True}), {"status": "added"}),
    (Call("forget", {}), {"forgotten": 1}),
    (Call("share", {}), {}),
    (Call("oracle", {"action": "confirm"}), {"confirmed": [{"canonical_id": "a"}]}),
    (Call("oracle", {"action": "progress"}), {"oracles": []}),
    (Call("open", {"kb": None}), {"error": "not found"}),
    (Call("open", {"kb": "peer"}), {"who_id": "x:user:1"}),
    (Call("hopper", {"confirm": False}), {"status": "preview"}),
    (Call("hopper", {"confirm": True}), {"status": "failed"}),
    (Call("oracle", {"action": "candidates"}), {"candidates": []}),
    (Call("watch", {"action": "preview"}), {"status": "preview"}),
    (Call("watch", {"action": "checked"}), {"status": "ok"}),
])
def test_mid_flow_irreversible_failed_or_empty_answers_carry_no_offer(call, answer):
    assert "offer" not in rider_rules.eligible(call, answer)


@pytest.mark.parametrize("call, answer", [
    (Call("open", {"kb": None}), {"who_id": "x:user:1"}),
    (Call("hopper", {"confirm": True}), {"status": "saved", "atom_id": "a"}),
    (Call("oracle", {"action": "candidates"}), {"candidates": [{"name": "c"}]}),
    (Call("oracle", {"action": "screen"}), {"groups": []}),
])
def test_answers_that_worked_carry_the_offer_last(call, answer):
    assert rider_rules.eligible(call, answer)[-1] == "offer"


def test_an_answer_that_already_asks_something_carries_no_offer():
    assert rider_rules.asks({"optional_keys": {"message": "m"}})
    assert rider_rules.asks({"notices": [{"code": "thin_coverage"}]})
    assert rider_rules.asks({"new_material": {"tour": {}}})
    assert not rider_rules.asks({"latest": {"arrived": 2}})
    assert not rider_rules.asks({"notices": [{"code": "results_truncated"}]})


# ── the optional-keys question: onboarding's, on a confirmation that asks nothing else ─────────

@pytest.mark.parametrize("call, answer, expected", [
    (Call("oracle", {"action": "confirm"}), {"confirmed": [{"canonical_id": "a"}]},
     ["optional_keys"]),
    (Call("oracle", {"action": "confirm"}), {"confirmed": [], "unresolved": ["@nobody"]}, []),
    # A venue's confirmation already asks which subjects to follow: one ask per answer.
    (Call("oracle", {"action": "confirm"}),
     {"confirmed": [{"canonical_id": "a"}, {"canonical_id": "v", "needs_topics": True}]}, []),
    (Call("oracle", {"action": "confirm"}), {"error": "confirm needs canonical_ids"}, []),
    (Call("add_oracle", {"confirm": True}), {"added": {"canonical_id": "a"}}, ["optional_keys"]),
    (Call("add_oracle", {"confirm": True}), {"error": "could not resolve"}, []),
    (Call("add_oracle", {"confirm": False}), {"canonical_id": "a"}, []),
    (Call("oracle", {"action": "ingest"}), {"picks": [{"canonical_id": "a"}]}, []),
])
def test_the_keys_question_rides_only_a_confirmation(call, answer, expected):
    assert rider_rules.eligible(call, answer) == expected


def test_the_picks_are_read_off_either_confirmation():
    assert rider_rules.picked({"confirmed": [{"canonical_id": "a"}, {"canonical_id": "b"}]}) == [
        "a", "b"]
    assert rider_rules.picked({"added": {"canonical_id": "c"}}) == ["c"]
