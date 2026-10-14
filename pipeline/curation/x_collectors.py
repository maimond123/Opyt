"""The X people collectors: who each X collection names, and how strongly.

Following, Likes and Lists are `walks.Collection`s, each publishing its counts from a complete
walk under its own collector clock, on the six-hour floor. Their totals are plain functions over
the walk's observations: distinct liked posts per author (`aggregate_authors`), distinct owned
lists per member (`aggregate_members`). The bookmark authors' rule lives here too; its walk is `bookmark_listing`'s.
"""

from typing import Iterable

from pipeline.curation import listings, walks, x_walks
from pipeline.curation.signals import Person, Tally
from pipeline.kb import derive
from pipeline.platforms.x import collections as x_collections


def person_profile(cand: dict) -> dict | None:
    """The kind-classify inputs from a normalized X user (bio, verified, followers, handle) for
    the entity's `profile` blob; None when none is present. Only truthy keys are kept, so
    `verified=False` and `followers_count=0` are dropped."""
    prof = {k: cand[k] for k in ("bio", "verified", "followers_count", "handle") if cand.get(k)}
    if "followers_count" in prof:                       # normalize the key the classifier reads
        prof["followers"] = prof.pop("followers_count")
    return prof or None


def x_person(user: dict) -> Person:
    """A normalized X user card (`x_records.normalize_user`) as the person a signal is about."""
    return Person(f"x:user:{user['user_id']}", name=user.get("display_name"),
                  site=user.get("site") or None, profile=person_profile(user))


def is_viewer(user_id: str, viewer: str) -> bool:
    """Is this user the signed-in viewer? The viewer's own card never makes them a candidate, in
    any X collection."""
    return user_id == viewer


def bookmark_author(post: dict, viewer: str) -> Person | None:
    """The person a saved post names as its author, for the listing's counts and for saving the
    post alike: the author its atom records (`derive.derive_x`). None for the viewer's own post."""
    if is_viewer(str((post.get("author") or {}).get("id") or ""), viewer):
        return None
    meta = derive.derive_x(post)
    return Person(meta["who_id"], name=meta["who_name"], site=meta.get("who_site") or None,
                  profile=person_profile({"handle": meta["who_handle"]}))


def bookmark_authors(posts: Iterable[dict], viewer: str) -> list[Tally]:
    """Each author of the saved posts, with how many of their posts were saved. The first post
    seen names its author."""
    found: dict[str, Tally] = {}
    for post in posts:
        person = bookmark_author(post, viewer)
        if person is None:
            continue
        eid = person.entity_id
        if eid in found:
            found[eid] = Tally(found[eid].person, found[eid].count + 1)
        else:
            found[eid] = Tally(person, 1)
    return list(found.values())


def aggregate_authors(authors: list[dict], viewer_id: str) -> list[dict]:
    """Fold the per-liked-tweet author list into deduped candidates with a `liked_count`,
    dropping the viewer. Sorted by liked_count then handle."""
    by_user: dict[str, dict] = {}
    for a in authors:
        if not a or is_viewer(a["user_id"], viewer_id):
            continue
        rec = by_user.get(a["user_id"])
        if rec is None:
            rec = {**a, "liked_count": 0}
            by_user[a["user_id"]] = rec
        rec["liked_count"] += 1
    return sorted(by_user.values(),
                  key=lambda c: (-c["liked_count"], c["handle"].lower()))


def aggregate_members(owned: list[dict], members_by_list: dict[str, list[dict]],
                      viewer_id: str) -> list[dict]:
    """Fold per-list member lists into deduped candidates, unioning the list names/ids
    (cross-list membership is a stronger signal, not a duplicate) and dropping the viewer.
    Sorted by breadth of membership then handle."""
    by_user: dict[str, dict] = {}
    for lst in owned:
        for m in members_by_list.get(lst["id"], []):
            if is_viewer(m["user_id"], viewer_id):
                continue
            rec = by_user.get(m["user_id"])
            if rec is None:
                rec = {**m, "list_names": [], "list_ids": []}
                by_user[m["user_id"]] = rec
            if lst["name"] not in rec["list_names"]:
                rec["list_names"].append(lst["name"])
                rec["list_ids"].append(lst["id"])
    return sorted(by_user.values(),
                  key=lambda c: (-len(c["list_names"]), c["handle"].lower()))


class Following(walks.Clocked):
    collector, operation, signal = "x_following", x_collections.FOLLOWING_OP, "follow"
    platform = x_walks.PLATFORM
    referer = "https://x.com/following"

    def read(self, session, viewer, branch):
        return x_collections.following_page(session, viewer, branch.cursor)

    def people(self, walk, branch, page):
        return [x_person(user) for _, user in page.records]

    def tally(self, conn, walk):
        return [Tally(x_person(user), 1)
                for _, _, user in listings.observations(conn, walk, self.operation)]


class Likes(walks.Clocked):
    collector, operation, signal = "x_likes", x_collections.LIKES_OP, "like"
    platform = x_walks.PLATFORM
    referer = "https://x.com/i/likes"

    def read(self, session, viewer, branch):
        return x_collections.likes_page(viewer, session, branch.cursor)

    def people(self, walk, branch, page):
        return [x_person(c) for c in aggregate_authors([a for _, a in page.records], walk.account)]

    def tally(self, conn, walk):
        liked = [a for _, _, a in listings.observations(conn, walk, self.operation)]
        return [Tally(x_person(c), c["liked_count"]) for c in aggregate_authors(liked, walk.account)]


class Lists(walks.Clocked):
    """The viewer's owned lists, then each list's members: one walk over two X operations, so
    it runs as two sources (`collectors.loops`). It finishes only once every member walk settles."""
    collector, operation, signal = "x_lists", x_collections.LISTS_MGMT_OP, "list"
    platform = x_walks.PLATFORM
    referer = "https://x.com/i/lists"
    members = x_collections.LIST_MEMBERS_OP

    def read(self, session, viewer, branch):
        if branch.operation == self.members:
            return x_collections.list_members_page(branch.partition, session, branch.cursor)
        return x_collections.owned_lists_page(session, viewer, branch.cursor)

    def children(self, branch, page):
        if branch.operation == self.members:
            return ()
        return tuple(listings.Branch(self.members, partition=list_id) for list_id, _ in page.records)

    def people(self, walk, branch, page):
        if branch.operation != self.members:
            return []
        return [x_person(m) for _, m in page.records if not is_viewer(m["user_id"], walk.account)]

    def tally(self, conn, walk):
        owned = [lst for _, _, lst in listings.observations(conn, walk, self.operation)]
        members: dict[str, list[dict]] = {}
        for list_id, _, member in listings.observations(conn, walk, self.members):
            members.setdefault(list_id, []).append(member)
        return [Tally(x_person(c), len(c["list_names"]), {"list_names": c["list_names"]})
                for c in aggregate_members(owned, members, walk.account)]
