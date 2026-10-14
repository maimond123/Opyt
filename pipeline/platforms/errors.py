"""The neutral failure a platform read raises when it cannot complete, shared by every client
that searches an outside source (the paper registries and GitHub)."""


class SourceError(RuntimeError):
    """A source client could not complete its request."""
