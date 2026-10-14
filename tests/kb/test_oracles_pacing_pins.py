"""Pins every pacing, request-budget and window value the oracles code sets.

X rate limits are real and shared, so a source's quota, class, floor and pacing are behavior. The
values must stay unchanged when the code that sets them moves; when a value changes where it
lives, only the import lines below change.
"""

from datetime import timedelta

from pipeline.oracles import windows
from pipeline.oracles import freshness
from pipeline.oracles.sources.threads import bookmarks
from pipeline.oracles.sources.threads import state as thread_state
from pipeline.oracles.sources.threads import tails
from pipeline.oracles.sources.references import documents
from pipeline.oracles.sources.references import x_posts
from pipeline.oracles.sources.timeline import walks
from pipeline.oracles.sources.timeline import scouting
from pipeline.oracles.sources.timeline import sources
from pipeline.oracles.sources.web import archives
from pipeline.oracles.sources.web import discovery
from pipeline.oracles.sources.web import recommendations


def test_windows():
    assert windows.X_LOOKBACK_PRESETS == {"6mo": 183, "1yr": 365, "2yr": 730}
    assert windows.WEB_LOOKBACK_PRESETS == {"1yr": 365, "2yr": 730, "5yr": 1825, "all": None}
    assert windows.SCHOLAR_LOOKBACK_PRESETS == {"2yr": 730, "5yr": 1825, "10yr": 3653,
                                                "all": None}
    assert windows.X_DEFAULT_DAYS == 183
    assert windows.X_CEILING_DAYS == 730


def test_refresh_pacing():
    assert windows.OVERLAP == timedelta(hours=6)
    assert windows.X_RESUME_OVERLAP == timedelta(minutes=10)
    assert freshness.STALE_FRACTION_ATTENTION == 0.5
    assert windows.FLAT_TTL_HOURS == {"x": 72.0, "substack": 168.0,
                                      "blog": 336.0, "github": 336.0}
    assert windows.DEFAULT_TTL_HOURS == 168.0
    assert windows.TTL_JITTER == 0.10


def test_candidate_and_discovery_pacing():
    assert scouting.PAGES == 3
    assert scouting.SPAN == timedelta(days=90)
    assert scouting.FLOOR == 50
    assert scouting._RANK_TTL_S == 300.0
    assert recommendations.FLOOR_HOURS == 168.0


def test_every_source_keeps_its_quota_class_and_floor():
    got = {c.__name__: (c.name, c.quota, c.klass, c.floor) for c in (
        sources.Picks, sources.Check, sources.NewPosts, scouting.Scouting,
        discovery.Discovery, archives.Archives, archives.WebNewPosts,
        recommendations.Recommendations, bookmarks.Bookmarks, tails.Tails,
        x_posts.XPosts, documents.Documents)}
    assert got == {
        "Picks": ("picks", "x:UserOriginalsTimeline", 0, 0),
        "Check": ("check", "x:UserOriginalsTimeline", 0, 0),
        "NewPosts": ("new_posts", "x:UserOriginalsTimeline", 1, 0),
        "Scouting": ("scouting", "x:UserOriginalsTimeline", 2, 50),
        "Discovery": ("discovery", "web", 0, 0),
        "Archives": ("archives", "web", 0, 0),
        "WebNewPosts": ("web_new_posts", "web", 1, 0),
        "Recommendations": ("recommendations", "web", 2, 0),
        "Bookmarks": ("bookmark_threads", "x:TweetDetail", 0, 0),
        "Tails": ("thread_tails", "x:TweetDetail", 0, 0),
        "XPosts": ("linked_x_posts", "x:TweetResultsByRestIds", 0, 0),
        "Documents": ("linked_documents", "web", 0, 0),
    }
    assert thread_state.QUOTA == "x:TweetDetail"
    assert discovery.QUOTA == "web"
    assert walks.QUOTA == "x:UserOriginalsTimeline"
