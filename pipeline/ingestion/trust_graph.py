"""
pipeline/ingestion/trust_graph.py

`propagate(edges, x_attested)` — the binary trust model. Pure function, no I/O.

Trust is reachability over a directed edge graph rooted at the person's confirmed
profile. Five rules, evaluated to a fixed point:

  Rule 1  Confirmed profile → trusted (root).

  Rule 2  Bidirectional with ANY trusted node → trusted. Candidate C links to trusted node T
          and T links back to C (T may be a root or a node trusted earlier this run).

  Rule 3  ≥2 trusted nodes point at candidate → trusted. Corroboration: two independent
          already-trusted sources both linking C graduate C.

  Rule 4  Direction is asymmetric — only `trusted → candidate` edges count for Rule 3. A page
          that merely links OUT to trusted sites gains nothing from it.

  Rule 5  A single typed IDENTITY edge (`via` in IDENTITY_VIA) from a trusted node graduates
          its target directly, no bidirectional back-link required.

Fixed point: Rules 2/3/5 all read the trusted set, which they grow, so we iterate until no node
changes; termination is guaranteed since `trusted` only grows and is bounded by node count.

Input contract: `edges` and `x_attested` are already canonicalized
(`url_canon.canonical_identity`); this module does no URL parsing.

"""

from __future__ import annotations

from collections import defaultdict

from pipeline.ingestion.trust_types import Edge, TrustEvidence

# Edge `via` markers denoting a typed identity edge — a source profile declaring its OWN
# other account, rather than a generic outbound href. See Rule 5 in the module docstring.
IDENTITY_VIA = frozenset({
    "identity_verified",   # platform-verified connection (Substack is_connected_account=true)
    "identity_declared",   # typed self-declared account (userLinks entry, unverified)
})


def _edge_dict(index: dict, source: str, target: str) -> dict:
    """Render the (source→target) edge as a plain dict for evidence trails."""
    e = index[(source, target)]
    return {"source": e.source, "target": e.target, "via": e.via, "found_by": e.found_by}


def propagate(
    edges: list[Edge],
    x_attested: set[str],
    candidates: set[str] | None = None,
) -> dict[str, TrustEvidence]:
    """Compute a TrustEvidence verdict for every node in the graph.

    Args:
        edges:       directed canonical edges (source → target).
        x_attested:  canonical confirmed profiles = trust roots.
        candidates:  discovered sources to evaluate even if no edge touches them,
                     so a source nobody links still gets a "no corroboration"
                     verdict from here rather than a caller-side fallback.

    Returns:
        {canonical_url: TrustEvidence} for every node (roots, candidates, and
        anything appearing in an edge). Untrusted nodes are present with
        trusted=False and a near-miss reason so callers can render review.
    """
    x_attested = set(x_attested)

    # Adjacency. Sets dedupe duplicate edges natively (Rule-3 needs DISTINCT sources).
    out: dict[str, set[str]] = defaultdict(set)   # source → {targets}
    inc: dict[str, set[str]] = defaultdict(set)    # target → {sources}
    index: dict[tuple, Edge] = {}                   # (source, target) → Edge
    nodes: set[str] = set(x_attested) | set(candidates or ())

    priority = {"identity_verified": 2, "identity_declared": 1}
    for e in edges:
        if e.source == e.target:
            # Self-loops carry no trust signal; keep the node, drop the edge.
            nodes.add(e.source)
            continue
        out[e.source].add(e.target)
        inc[e.target].add(e.source)
        key = (e.source, e.target)
        # The graph owns duplicate evidence. Typed declarations outrank ordinary links.
        previous = index.get(key)
        if previous is None or (priority.get(e.via, 0), e.via, e.found_by) > (
                priority.get(previous.via, 0), previous.via, previous.found_by):
            index[key] = e
        nodes.add(e.source)
        nodes.add(e.target)

    result: dict[str, TrustEvidence] = {}
    trusted: set[str] = set()

    # Rule 1 — roots.
    for r in x_attested:
        trusted.add(r)
        result[r] = TrustEvidence(trusted=True, reasons=["Confirmed root"], edges=[])

    # Rules 2, 3 and 5 to a fixed point.
    changed = True
    while changed:
        changed = False
        for c in nodes:
            if c in trusted:
                continue

            # Rule 5 — typed IDENTITY edge from a trusted node graduates c directly, no
            # back-link needed. Checked before Rule 2 so identity provenance wins the reason
            # string. Sorted for a deterministic source.
            id_src = next(
                (t for t in sorted(inc[c])
                 if t in trusted
                 and index[(t, c)].via in IDENTITY_VIA),
                None,
            )
            if id_src is not None:
                trusted.add(c)
                result[c] = TrustEvidence(
                    trusted=True,
                    reasons=[f"Identity-attested by trusted {id_src} "
                             f"({index[(id_src, c)].via})"],
                    edges=[_edge_dict(index, id_src, c)],
                )
                changed = True
                continue

            # Rule 2 — bidirectional with ANY trusted node (sorted for a
            # deterministic partner in the evidence trail).
            partner = next((t for t in sorted(trusted) if t in out[c] and c in out[t]), None)
            if partner is not None:
                trusted.add(c)
                result[c] = TrustEvidence(
                    trusted=True,
                    reasons=[f"Bidirectional with trusted node {partner}"],
                    edges=[_edge_dict(index, c, partner), _edge_dict(index, partner, c)],
                )
                changed = True
                continue

            # Rule 3 (+ Rule 4) — ≥2 DISTINCT trusted nodes point AT c.
            pointers = sorted(t for t in inc[c] if t in trusted and t != c)
            if len(pointers) >= 2:
                shown = ", ".join(pointers[:3])
                trusted.add(c)
                result[c] = TrustEvidence(
                    trusted=True,
                    reasons=[f"Cited by {len(pointers)} trusted sources: {shown}"],
                    edges=[_edge_dict(index, t, c) for t in pointers],
                )
                changed = True
                continue

    # Everything else → needs review. Emit *why it fell short* (near-miss vs no
    # signal) so a human can confirm in seconds instead of investigating: this
    # is the evidence that makes strict mode's review step tolerable.
    for c in nodes:
        if c in result:
            continue
        trusted_pointers = sorted(t for t in inc[c] if t in trusted and t != c)
        ev_edges = [_edge_dict(index, s, c) for s in sorted(inc[c])]
        if len(trusted_pointers) == 1:
            reasons = [
                f"Near miss — linked by 1 trusted source ({trusted_pointers[0]}); "
                f"needs a 2nd trusted link (Rule 3) or a link back (Rule 2)"
            ]
        else:
            reasons = ["No trusted source links this — discovered independently, no corroboration"]
        result[c] = TrustEvidence(trusted=False, reasons=reasons, edges=ev_edges)

    return result
