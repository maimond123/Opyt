"""How a walk reaches the signed-in X account: one quota per X operation, the OPYT-managed
session, and the viewer read before every page, so a switch to another account mid-walk
supersedes the walk at once. A session that cannot name its viewer pauses the quota on the
session (`SyncAuthError`), as every X read does."""

from typing import Protocol

from pipeline.curation import walks
from pipeline.ingestion.utils import SyncAuthError
from pipeline.platforms import connections
from pipeline.platforms import sessions

PLATFORM = "x"


class XCollection(walks.Collection, Protocol):
    """An X collection: the page it says it came from, sent with each read."""
    referer: str


class XAccount:
    platform, host = PLATFORM, None

    def quota(self, operation):
        return f"x:{operation}"

    def connected(self):
        return connections._connected_x()

    def held(self, now):
        return None                 # the quota loop meters X's window itself

    def session(self, ctx, collection: XCollection):
        if "session" not in ctx.cache:
            ctx.cache["session"] = sessions.x_session(collection.referer)
        return ctx.cache["session"]

    def signed_in(self, ctx, collection: XCollection):
        viewer = self.session(ctx, collection).viewer_id()
        if not viewer:
            raise SyncAuthError("the X session has no viewer id")
        return viewer

    def switched(self, ctx, collection: XCollection):
        return self.session(ctx, collection).viewer_id()

    def failed(self, ctx, exc):
        if isinstance(exc, SyncAuthError):
            ctx.cache.pop("session", None)
        return exc

    def read(self):
        pass
