"""The Substack people collectors: who the user follows, subscribes to and saves posts from.

Follows and subscriptions are two different graphs on two endpoints, so they are two collections
with two clocks, `substack_follows` and `substack_subscriptions`, each publishing from a complete walk on
the six-hour floor. The saved-post authors' rule lives here too; its walk is
`substack_saved_listing`'s, which also feeds saved content.

The account reads return publications, not people with handles, so follows and subscriptions key
on the publication (`derive.substack_entity_id(None, url)`), and Stage-3 merges them with the
handle-keyed saved authors through the publication URL. Absence from a walk proves nothing here:
only X's following is trusted for retirement.
"""

from typing import Iterable

from pipeline.curation import listings, substack_walks, walks
from pipeline.curation.signals import Person, Tally, author
from pipeline.kb import derive
from pipeline.platforms import sessions
from pipeline.platforms.substack import account as substack_account


def is_paid(membership_state: str) -> bool | None:
    """`membership_state` → the money claim `screen.reflect()` renders, or None for "unknown",
    so an unrecognised state can never invent a payment. The True arm is inferred from
    Substack's frontend, not yet observed; the raw state is kept on the signal for audit."""
    if membership_state == "subscribed":
        return True
    if membership_state == "free_signup":
        return False
    return None


def publication(rec: dict) -> Person:
    """A followed or subscribed publication as the person its signal is about."""
    url = rec.get("url") or ""
    return Person(derive.substack_entity_id(None, url), name=rec.get("name"), site=url or None)


def subscription(rec: dict) -> Tally:
    """One subscription, carrying whether the user pays for it. `is_favorite` is stored and
    nothing reads it yet."""
    state = rec.get("membership_state") or ""
    return Tally(publication(rec), 1, {"is_paid": is_paid(state), "membership_state": state,
                                       "is_favorite": rec.get("is_favorite")})


def saved_authors(records: Iterable[dict]) -> list[Tally]:
    """Each publication the user saved posts from, with how many. The same derivation saved
    content keys its atoms on, so both sides name one person identically. The user's own posts
    are not dropped: a saved record names a publication, not a user id to compare."""
    found: dict[str, Tally] = {}
    for rec in records:
        person = author(derive.derive_substack(rec))
        eid = person.entity_id
        if eid in found:
            found[eid] = Tally(found[eid].person, found[eid].count + 1)
        else:
            found[eid] = Tally(person, 1)
    return list(found.values())


class _AccountGraph(walks.Clocked):
    """The two account graphs: Substack's, and their people are the records' publications."""
    platform = substack_walks.PLATFORM

    def people(self, walk, branch, page):
        return [publication(rec) for _, rec in page.records]


class Follows(_AccountGraph):
    collector, operation, signal = "substack_follows", "SubstackFollows", "follow"

    def transport(self):
        return sessions.follow_source()

    def read(self, session, account, branch):
        return substack_account.follows_page(session, int(account))

    def tally(self, conn, walk):
        return [Tally(publication(rec), 1)
                for _, _, rec in listings.observations(conn, walk, self.operation)]


class Subscriptions(_AccountGraph):
    collector, operation, signal = "substack_subscriptions", "SubstackSubscriptions", "subscribe"

    def transport(self):
        return sessions.subscription_list_source()

    def read(self, session, account, branch):
        return substack_account.subscription_page(session, branch.cursor)

    def tally(self, conn, walk):
        return [subscription(rec)
                for _, _, rec in listings.observations(conn, walk, self.operation)]
