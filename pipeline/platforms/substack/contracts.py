"""Contracts for Substack's three distinct account-source reads, and the refusals Substack reads
raise."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class FollowSource(Protocol):
    def follows(self, user_id: int | None = None) -> list[dict]:
        """Return normalized publications associated with the followed people."""


@runtime_checkable
class SubscriptionSource(Protocol):
    def page(self, cursor: str | None) -> dict | None:
        """Return one raw subscriptions page, or None when the transport refused it."""


@runtime_checkable
class SavedSource(Protocol):
    def page(self, cursor: str | None) -> dict | None:
        """Return one raw saved-list page, or None when the transport refused it."""

    def full_post(self, base: str, slug: str) -> dict | None:
        """Return one full post body, or None when no body is available."""

    def authenticated_body(self, base: str) -> bool:
        """Say whether body reads from this publication host carry an accepted session."""


class SubstackListingError(RuntimeError):
    """The archive listing could not be completed — the walk stopped part-way.

    Distinct from `SubstackFetchError` because the fail-safe differs: a failed per-post fetch
    skips one post, while a failed listing means the inventory is a prefix, so continuing would
    ingest part of an archive as if it were the whole one."""


class SubstackFetchError(RuntimeError):
    """The per-post fetch could not be completed — a Cloudflare challenge, a transport failure,
    or a non-JSON body on a JSON endpoint.

    Distinct from "this post has no body": the caller skips the post without concluding anything
    about it (no atom, no `seen` mark, retried next run)."""


class SubstackPublicReadError(RuntimeError):
    """A public Substack read was refused or came back in a shape this module cannot use.

    Separate from `SubstackListingError` (an account read failed, usually a dead session): this
    one is a Cloudflare refusal or a moved publication, so the caller skips that publication and
    retries next pass.
    """
