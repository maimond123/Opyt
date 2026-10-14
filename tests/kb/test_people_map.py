"""people_map — the people the user reads, grouped by their words, with each Oracle placed.

What would go wrong quietly: groups that mix subjects (the map then says nothing), an Oracle
placed by chance, a map that changes between two calls on the same store, and a "map" of a store
too thin to group.
"""
from __future__ import annotations

import random

from pipeline.kb import schema
from pipeline.oracles import people_map, probe_store
from tests import mcp_session

TOPICS = {
    "crypto": "rollup validator stablecoin ledger onchain wallet token bridge staking custody",
    "biology": "protein enzyme genome cell clinical assay antibody tissue receptor mutation",
    "policy": "regulation senate statute treasury tariff agency lawmakers subsidy export federal",
    "robots": "actuator gripper lidar locomotion servo humanoid manipulator torque odometry drone",
    "markets": "equity bond yield futures hedge dividend earnings liquidity volatility arbitrage",
}


def _people(per_topic=25, seed=1):
    """Synthetic sampled people: each writes a few posts in one topic's words."""
    rng = random.Random(seed)
    people, topic_of = [], []
    for topic, words in TOPICS.items():
        vocab = words.split()
        for _ in range(per_topic):
            people.append([" ".join(rng.sample(vocab, 4)) for _ in range(3)])
            topic_of.append(topic)
    return people, topic_of


def test_each_group_holds_one_subject_and_each_oracle_lands_in_its_own():
    people, topic_of = _people()
    oracles = [["new " + TOPICS["biology"]], ["the " + TOPICS["markets"]], ["nothing shared"]]
    groups, placed = people_map.group(people, oracles)

    assert len(groups) == len(TOPICS)
    by_id = {}
    for g in groups:
        subjects = {topic_of[i] for i in g["members"]}
        assert len(subjects) == 1, subjects
        by_id[g["id"]] = subjects.pop()
        assert set(g["words"]) <= set(TOPICS[by_id[g["id"]]].split())
    assert [by_id.get(p) for p in placed] == ["biology", "markets", None]


def test_the_same_store_gives_the_same_map():
    people, _ = _people()
    assert people_map.group(people, []) == people_map.group(people, [])


def test_links_mentions_and_stop_words_are_not_words():
    assert people_map.tokens(["The rollup at https://x.com/a/1 by @vitalik is [here](https://b)"]) \
        == ["rollup"]


def test_the_oracle_tool_answers_map_through_the_real_server(kb_home):
    out = mcp_session.session().call("oracle", action="map")
    assert out["groups"] == [] and out["people"] == 0


def test_a_store_too_thin_to_group_says_so_instead_of_inventing_groups(kb_home, monkeypatch):
    monkeypatch.setattr(probe_store, "texts_by_person",
                        lambda conn: {f"x:user:{i}": ["a post"] for i in range(10)})
    conn = schema.connect()
    out = people_map.people_map(conn)
    conn.close()
    assert out["groups"] == [] and out["people"] == 10 and "too few" in out["note"]
