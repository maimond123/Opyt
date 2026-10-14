"""
tests/onboarding/test_topic_source.py

A topic the user follows is a source; one they stopped following is not. The old "user watch exists" read counted rows the user had retired, so a home
whose only watch was gone skipped the sources question and was told setup was done. The same holds
for a topic they dropped: it watches nothing more, so it is not something to read from.
"""
from __future__ import annotations

import pytest

from pipeline.kb import schema
from pipeline.onboarding import phase
from pipeline.platforms import connections
from pipeline.topics import store as topics_store


@pytest.fixture()
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setattr(connections, "_connected_x", lambda: False)
    monkeypatch.setattr(connections, "_connected_substack", lambda: False)
    return tmp_path


def _topic(*, dropped: bool = False) -> None:
    conn = schema.connect()
    try:
        topic_id = topics_store.new_topic(conn, name="Protein design", words=["protein design"],
                                          people=[], now="2026-10-11T00:00:00Z")
        if dropped:
            topics_store.update(conn, topic_id, now="2026-10-12T00:00:00Z",
                                status=topics_store.DROPPED)
        conn.commit()
    finally:
        conn.close()


@pytest.mark.usefixtures("home")
def test_a_followed_topic_is_a_source():
    _topic()
    assert phase.derive()["phase"] == "done"


@pytest.mark.usefixtures("home")
def test_a_dropped_topic_is_not_a_source():
    _topic(dropped=True)
    state = phase.derive()
    assert state["sources"]["topics"] is False
    assert state["phase"] == "sources"
