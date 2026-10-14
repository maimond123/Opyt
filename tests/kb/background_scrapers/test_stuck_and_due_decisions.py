"""`milestones.stuck` and `recommendations.pass_due`: one test per table row, with plain values."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pipeline.curation.clock import CollectorRun
from pipeline.oracles.asks import Ask
from pipeline.oracles.milestones import stuck
from pipeline.oracles.registry import SourceRow
from pipeline.timeparse import utc_iso
from pipeline.worker.quota_loop.retries import GIVE_UP_AFTER, STUCK_AFTER, UNSAVED
from pipeline.oracles.sources.timeline.walks import QUOTA as X
from pipeline.oracles.sources.web.discovery import QUOTA as WEB
from pipeline.oracles.sources.web.recommendations import pass_due

NOW = datetime(2026, 10, 7, tzinfo=timezone.utc)
HEAD = "x:user:1"


def _ask(**kw):
    return Ask(HEAD, x_from=None, web_from=utc_iso(NOW), scholar_from=None, scholar_topics=None,
               asked_at=utc_iso(NOW), position=0, discovered_at=utc_iso(NOW), rediscover=False,
               **kw)


def _stuck(rows=(), mine=(), scrapers=None, not_running=None, ask=None, owes_x=True,
           owes_web=True):
    return stuck(HEAD, ask, list(rows), list(mine), scrapers or {}, not_running,
                 owes_x=owes_x, owes_web=owes_web)


def test_a_liveness_report_that_says_not_running_is_stuck_first():
    paused = {X: {"state": "paused", "cause": "x_session"}}
    assert _stuck(scrapers=paused, not_running={"ok": False, "cause": "c", "step": "s"}) == {
        "cause": "not_running", "detail": "c", "step": "s"}


def test_an_owed_loop_paused_on_the_x_session_is_stuck():
    assert _stuck(scrapers={X: {"state": "paused", "cause": "x_session"}})["cause"] == "x_session"
    assert _stuck(scrapers={X: {"state": "paused", "cause": "x_session"}}, owes_x=False) is None


def test_an_owed_loop_paused_on_the_model_provider_is_stuck():
    assert _stuck(scrapers={WEB: {"state": "paused", "cause": "model:embed"}}) == {
        "cause": "model:embed"}


def test_a_venue_with_no_subjects_is_stuck():
    venue = SourceRow(HEAD, "openalex", "S1")
    assert _stuck(rows=[venue], ask=_ask())["cause"] == "needs_topics"


def test_discovery_given_up_is_stuck():
    ask = _ask(discovery_failures=GIVE_UP_AFTER, discovery_error="boom")
    ask = Ask(**{**vars(ask), "discovered_at": None})
    assert _stuck(ask=ask)["cause"] == "discovery_failed"


def test_a_page_that_would_not_save_is_stuck():
    row = SourceRow(HEAD, "x", "carol", last_status=UNSAVED, failures=STUCK_AFTER,
                    last_error="no")
    assert _stuck(rows=[row]) == {"cause": "unsaved", "detail": "no"}


def test_otherwise_nothing_is_stuck():
    assert _stuck(rows=[SourceRow(HEAD, "x", "carol")], not_running={"ok": True}) is None


def _run(days_ago):
    return CollectorRun("recommended", last_attempt_at=utc_iso(NOW - timedelta(days=days_ago)))


def test_a_pass_is_due_with_no_attempt_or_a_week_after_one():
    assert pass_due(None, [], NOW)
    assert pass_due(_run(7), [], NOW)


def test_a_pass_is_due_within_the_week_for_a_substack_oracle_added_after_it():
    added = SourceRow(HEAD, "substack", "a.substack.com", added_at=utc_iso(NOW - timedelta(days=1)))
    assert pass_due(_run(2), [added], NOW)


def test_otherwise_a_pass_is_not_due_within_the_week():
    older = SourceRow(HEAD, "substack", "a.substack.com", added_at=utc_iso(NOW - timedelta(days=3)))
    assert not pass_due(_run(2), [older], NOW)
