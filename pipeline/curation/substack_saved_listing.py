"""The Substack Saved Posts list, walked once for its two consumers.

A page's records land as observations, which the publications' `save` counts are built from, and
as saved-item offers, which content is saved from (`substack_saved.SavedSubstackPosts`), in the
page's own transaction. The list is walked six-hourly, and its counts publish on the `substack_saved_signals` clock. `walks.WalkSource` runs the
walk; `substack_walks.SubstackAccount` reaches the account.
"""

from datetime import timedelta

from pipeline.curation import listings, saved_items, substack_people, substack_walks, walks
from pipeline.platforms import sessions
from pipeline.platforms.substack import account as substack_account

COLLECTOR = "substack_saved"                  # the listing's walks
SIGNAL_COLLECTOR = "substack_saved_signals"   # the clock the counts publish under (a durable name)
REFRESH = timedelta(hours=6)


class SavedPosts(walks.SavedList):
    collector, operation, platform = COLLECTOR, "SubstackSaved", substack_walks.PLATFORM
    signal_collector, refresh = SIGNAL_COLLECTOR, REFRESH

    def revision(self, rec):
        return saved_items.revision(rec)

    def transport(self):
        return sessions.saved_source()

    def read(self, session, account, branch):
        return substack_account.saved_page(session, branch.cursor)

    def people(self, walk, branch, page):
        return [t.person for t in substack_people.saved_authors(rec for _, rec in page.records)]

    def tally(self, conn, walk):
        return substack_people.saved_authors(
            rec for _, _, rec in listings.observations(conn, walk, self.operation))
