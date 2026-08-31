"""
pipeline/ingestion/trust_types.py

Pure data contracts for the owner-validation trust graph. No I/O, no deps.

An `Edge` is a directed claim "source links to target" discovered by some probe.
`TrustEvidence` is the verdict for a single canonical URL after propagation:
binary trusted/untrusted, plus the human-readable reasons and the supporting
edges so the decision is auditable rather than asserted.

See `trust_graph.propagate()` for the four reachability rules these feed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Edge:
    """A directed 'source → target' link between two canonical URLs.

    Frozen so edges are hashable and de-duplicate cleanly in sets.

    via:      how the link was found, e.g. "x_bio", "github_bio", "footer".
    found_by: which probe emitted it, e.g. "x", "github", "substack".
    """

    source: str
    target: str
    via: str = ""
    found_by: str = ""


@dataclass
class TrustEvidence:
    """The trust verdict for one canonical URL.

    trusted: binary outcome (this model does not score).
    reasons: why — e.g. ["X-attested (root)"] or ["Cited by 2 trusted sources"].
    edges:   the supporting edges as plain dicts, so the GUI / CLI can render
             "Linked from X bio" / "Bidirectional with someuser.ai" without
             re-deriving anything.
    """

    trusted: bool = False
    reasons: list = field(default_factory=list)
    edges: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"trusted": self.trusted, "reasons": list(self.reasons), "edges": list(self.edges)}
