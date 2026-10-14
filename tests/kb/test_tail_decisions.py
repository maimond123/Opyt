"""`tail_state.decide`: one test per row of its table, with plain values."""
from __future__ import annotations

from pipeline.oracles.sources.threads.tail_state import (TAIL_FETCHED, TAIL_GONE, TAIL_OWED,
                                                         TailDecision, decide)


def _prior(tail, ids):
    return {"thread_tail": tail, "thread_ids": ids}


def test_replies_to_a_stored_owed_thread_are_skipped():
    assert decide(_prior(TAIL_OWED, ["10"]), conv="10", ids=["11"], has_root=False, stored=True,
                  owed_convs=set()) == TailDecision(skip=True)


def test_replies_a_fetched_or_gone_thread_already_holds_are_skipped():
    for tail in (TAIL_FETCHED, TAIL_GONE):
        assert decide(_prior(tail, ["10", "11"]), conv="10", ids=["11"], has_root=False,
                      stored=True, owed_convs=set()) == TailDecision(skip=True)


def test_new_replies_to_a_stored_thread_mark_it_owed():
    for prior in (None, _prior(TAIL_FETCHED, ["10"])):
        d = decide(prior, conv="10", ids=["11"], has_root=False, stored=True, owed_convs=set())
        assert d == TailDecision(mark_owed=True, tail=TAIL_OWED) and d.owes


def test_a_stored_tail_holding_more_than_the_page_is_kept():
    assert decide(_prior(TAIL_FETCHED, ["10", "11", "12"]), conv="10", ids=["10", "11"],
                  has_root=True, stored=True, owed_convs={"10"}) == TailDecision(keep_stored=True)


def test_a_thread_is_owed_on_a_self_reply_a_missing_root_or_a_shorter_page():
    cases = [
        (None, ["10"], True, False, {"10"}),                       # a self-reply on this page
        (None, ["11"], False, False, set()),                       # no root, not stored
        (_prior(TAIL_OWED, ["10"]), ["10"], True, True, set()),    # still owed
        (_prior(TAIL_FETCHED, ["10", "11"]), ["10", "12"], True, True, set()),  # holds others
    ]
    for prior, ids, has_root, stored, owed in cases:
        assert decide(prior, conv="10", ids=ids, has_root=has_root, stored=stored,
                      owed_convs=owed) == TailDecision(tail=TAIL_OWED)


def test_a_solo_post_owes_nothing():
    d = decide(None, conv="10", ids=["10"], has_root=True, stored=False, owed_convs=set())
    assert d == TailDecision() and not d.owes
