"""The X Bookmarks list, walked once for its two consumers.

A page's records land as observations, which the authors' `save` counts are built from, and as
saved-item offers, which content is saved from (`bookmarks.SavedXPosts`), in the page's own
transaction. The list is walked hourly so new saves reach content quickly, and its counts
publish on the six-hour signal clock. `walks.WalkSource` runs the walk.
"""

from datetime import timedelta

from pipeline.curation import listings, saved_items, walks, x_collectors, x_walks
from pipeline.platforms.x import collections as x_collections

COLLECTOR = "x_bookmarks"                 # the listing's walks
SIGNAL_COLLECTOR = "x_bookmark_signals"   # the clock the counts publish under (a durable name)
REFRESH = timedelta(hours=1)


class Bookmarks(walks.SavedList):
    collector, operation, platform = COLLECTOR, x_collections.BOOKMARKS_OP, x_walks.PLATFORM
    signal_collector, refresh = SIGNAL_COLLECTOR, REFRESH
    referer = "https://x.com/i/bookmarks"

    def revision(self, post):
        return saved_items.x_revision(post)

    def read(self, session, viewer, branch):
        return x_collections.bookmark_page(session, branch.cursor)

    def people(self, walk, branch, page):
        return [t.person for t in x_collectors.bookmark_authors(
            (post for _, post in page.records), walk.account)]

    def tally(self, conn, walk):
        return x_collectors.bookmark_authors(
            (post for _, _, post in listings.observations(conn, walk, self.operation)),
            walk.account)
