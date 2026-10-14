"""Construct the site readers a caller may use, over OPYT's managed local sessions.

Every collector gets its X client and its Substack readers here, so none builds a transport of its
own. A reader is built per call, not held as a singleton, so cookie reads stay lazy.
"""

from pipeline.platforms.substack import local as substack_local
from pipeline.platforms.substack.contracts import FollowSource, SavedSource, SubscriptionSource
from pipeline.platforms.x import local as x_local
from pipeline.platforms.x.request import XClient


def x_session(referer: str) -> XClient:
    """The one X reader a caller may use: a client over the managed cookie session, with headers
    for `referer`."""
    cookies = x_local.read_x_cookies()
    return XClient(x_local.LocalXTransport(cookies, x_local.auth_headers(cookies, referer)))


def follow_source() -> FollowSource:
    """The one transport a caller may use to read who the user follows on Substack."""
    return substack_local.LocalFollows()


def subscription_list_source() -> SubscriptionSource:
    """The one transport a caller may use to read the publications the user subscribes to.

    A separate function rather than a `kind=` argument on `follow_source`, so there is no argument
    a caller can get wrong.
    """
    return substack_local.LocalSubscriptionList()


def saved_source() -> SavedSource:
    """The one transport a caller may use to read the user's own saved posts."""
    return substack_local.LocalSaved()


def own_account_id() -> int | None:
    """The signed-in Substack account's numeric id, or None when the session carries no signed-in
    user. A refused reader page raises `SubstackListingError` (see `substack.local.own_user_id`)."""
    return substack_local.own_user_id(substack_local.read_substack_cookies())
