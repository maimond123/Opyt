"""`peers.local_name` — what a reader calls an accepted knowledge base, one test per case."""
from __future__ import annotations

from pipeline.retrieval.peers import local_name


def test_the_name_asked_for_wins():
    assert local_name("leo", "David", "a3f9c2e1b704") == "leo"


def test_without_one_the_owners_label_is_used():
    assert local_name(None, "David", "a3f9c2e1b704") == "David"


def test_an_owner_labelled_with_the_reserved_name_falls_back_to_the_key():
    assert local_name(None, "me", "a3f9c2e1b704") == "a3f9c2e1b704"


def test_an_owner_with_no_label_is_named_by_the_key():
    assert local_name(None, None, "a3f9c2e1b704") == "a3f9c2e1b704"
    assert local_name("", "", "a3f9c2e1b704") == "a3f9c2e1b704"
