"""The archive adapter contract: one listing page or one document, never a whole crawl."""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Page:
    documents: list[tuple[str, dict]] = field(default_factory=list)
    cursors: list[dict] = field(default_factory=list)


class Adapter(Protocol):
    def initial(self, spec: dict) -> dict: ...
    def host(self, spec: dict, kind: str, data: dict) -> str | None: ...
    def page(self, spec: dict, cursor: dict, ctx) -> Page: ...
    # `hold(atom, snapshot)` keeps a document's atom for the write that lands it with its task.
    def document(self, conn, spec: dict, data: dict, ctx, hold) -> str: ...
