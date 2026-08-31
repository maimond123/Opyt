"""Tests for trust_graph.propagate — the four reachability rules.

Inputs are already canonical (propagate does no URL parsing), so we use short
opaque node names. Each canonical case maps to one rule; the rest are the
adversarial / degenerate edges the design calls out.
"""

from pipeline.ingestion.trust_graph import propagate
from pipeline.ingestion.trust_types import Edge


def _trusted(result) -> set:
    return {k for k, v in result.items() if v.trusted}


# ── The 5 canonical cases (one per rule + the asymmetry it defends) ───────────

def test_rule1_root_is_trusted():
    r = propagate([], {"karpathy.ai"})
    assert _trusted(r) == {"karpathy.ai"}
    assert r["karpathy.ai"].reasons == ["X-attested (root)"]


def test_rule2_bidirectional_with_root():
    # X bio attests karpathy.ai; karpathy.ai links back to the X profile.
    edges = [Edge("x.com/karpathy", "karpathy.ai"), Edge("karpathy.ai", "x.com/karpathy")]
    r = propagate(edges, {"x.com/karpathy"})
    assert r["karpathy.ai"].trusted
    assert "Bidirectional" in r["karpathy.ai"].reasons[0]


def test_rule2_bidirectional_with_non_root_trusted_node_graduates():
    # Relaxed Rule 2 (2026-07-21): the bidirectional partner need not be a ROOT —
    # any trusted node counts. jane.blog earns trust via the root, THEN graduates
    # the github it mutually links, even though that github never touches the root.
    # Under the old root-only rule this github stayed needs-review.
    edges = [
        Edge("x.com/jane", "jane.blog"), Edge("jane.blog", "x.com/jane"),          # blog trusted via root
        Edge("jane.blog", "github.com/jane-dev"), Edge("github.com/jane-dev", "jane.blog"),
    ]
    r = propagate(edges, {"x.com/jane"})
    assert r["jane.blog"].trusted
    assert r["github.com/jane-dev"].trusted, "relaxed Rule 2 graduates a non-root bidirectional partner"
    assert "jane.blog" in r["github.com/jane-dev"].reasons[0]


def test_rule2_chains_through_the_fixed_point():
    # A→B→C mutual chain hanging off one root: each link graduates the next via
    # relaxed Rule 2 across iterations. C is two hops from the root and still lands.
    edges = [
        Edge("x.com/a", "b.com"), Edge("b.com", "x.com/a"),   # B trusted (bi w/ root)
        Edge("b.com", "c.com"), Edge("c.com", "b.com"),       # C trusted (bi w/ trusted B)
    ]
    r = propagate(edges, {"x.com/a"})
    assert r["b.com"].trusted and r["c.com"].trusted


def test_rule2_relaxation_still_requires_a_path_to_a_root():
    # The safety invariant the relaxation must NOT break: a mutual chain with NO
    # connection to any root stays untrusted — trust originates only at roots.
    edges = [
        Edge("p.com", "q.com"), Edge("q.com", "p.com"),   # p↔q, both rootless
        Edge("q.com", "s.com"), Edge("s.com", "q.com"),   # q↔s, still rootless
    ]
    r = propagate(edges, {"root.com"})
    assert not any(r[n].trusted for n in ("p.com", "q.com", "s.com"))


def test_rule3_two_trusted_pointers_graduate_candidate():
    # Two roots both link the YouTube channel → trusted by corroboration.
    edges = [Edge("karpathy.ai", "youtube.com/@k"), Edge("github.com/karpathy", "youtube.com/@k")]
    r = propagate(edges, {"karpathy.ai", "github.com/karpathy"})
    assert r["youtube.com/@k"].trusted
    assert "Cited by 2" in r["youtube.com/@k"].reasons[0]


def test_rule4_squatter_pointing_at_root_is_not_trusted():
    # Squatter name-drops the root in its footer (candidate → trusted).
    # Direction is asymmetric: this must NOT graduate the squatter.
    edges = [Edge("github.com/elonmusk", "karpathy.ai")]
    r = propagate(edges, {"karpathy.ai"})
    assert not r["github.com/elonmusk"].trusted


def test_single_trusted_pointer_is_insufficient():
    # One root links a candidate but it's not bidirectional and not corroborated.
    edges = [Edge("karpathy.ai", "someblog.com")]
    r = propagate(edges, {"karpathy.ai"})
    assert not r["someblog.com"].trusted


# ── Rule 5: typed identity edges (the de-X-rooting upgrade) ───────────────────

def test_rule5_identity_edge_graduates_without_bidirectional():
    # A Substack root declares its own X via a typed identity link (userLinks).
    # ONE such edge trusts the X — no back-link needed (Rule 2 would demand one).
    edges = [Edge("them.substack.com", "x.com/them", via="identity_verified")]
    r = propagate(edges, {"them.substack.com"})
    assert r["x.com/them"].trusted
    assert "Identity-attested" in r["x.com/them"].reasons[0]


def test_rule5_substack_only_person_reaches_t1_with_no_x():
    # The whole point: no X in the root set at all. Substack root's userLinks
    # declare their X, site, and YouTube → all T1 via identity edges.
    root = "them.substack.com"
    edges = [
        Edge(root, "x.com/them", via="identity_verified"),
        Edge(root, "them.com", via="identity_declared"),
        Edge(root, "youtube.com/@them", via="identity_declared"),
    ]
    r = propagate(edges, {root})
    assert all(r[t].trusted for t in ("x.com/them", "them.com", "youtube.com/@them"))


def test_identity_edge_only_fires_from_trusted_source():
    # A NON-root, non-trusted node claiming an identity edge grants nothing —
    # the source must already be trusted (a squatter can't self-declare into trust).
    edges = [Edge("squatter.com", "x.com/famous", via="identity_declared")]
    r = propagate(edges, {"root.com"})
    assert not r["x.com/famous"].trusted


def test_generic_edge_still_needs_corroboration_not_identity_shortcut():
    # A generic (untyped) single edge from a root does NOT get the Rule-5 shortcut,
    # and (no back-link, one pointer) also fails Rule 2 and Rule 3.
    edges = [Edge("them.substack.com", "random.com")]   # via="" (generic)
    r = propagate(edges, {"them.substack.com"})
    assert not r["random.com"].trusted


def test_rule5_declared_account_is_a_valid_rule2_partner():
    # The COLLAPSE decision (2026-07-21): a Rule-5-declared account is a FULL trusted
    # node, so relaxed Rule 2 accepts it as a bidirectional partner. The root declares
    # its site (identity edge); a candidate mutually linked with that SITE — not the
    # root itself — graduates via Rule 2. This is the recall the de-X-root contract
    # would otherwise have lost (under a keep-restriction form it stayed needs-review).
    edges = [
        Edge("x.com/me", "me.com", via="identity_declared"),                   # root → site (Rule 5)
        Edge("me.com", "me.substack.com"), Edge("me.substack.com", "me.com"),  # bi with the SITE
    ]
    r = propagate(edges, {"x.com/me"})
    assert r["me.com"].trusted                                    # Rule 5 (declared)
    assert r["me.substack.com"].trusted                           # Rule 2 off the declared partner
    assert "Bidirectional with trusted node me.com" in r["me.substack.com"].reasons[0]


# ── Degenerate / adversarial graphs ──────────────────────────────────────────

def test_empty_graph():
    assert propagate([], set()) == {}


def test_self_loop_does_not_self_graduate():
    # A non-root node linking to itself stays untrusted; the node still appears.
    r = propagate([Edge("a.com", "a.com")], {"root.com"})
    assert "a.com" in r and not r["a.com"].trusted


def test_cycle_without_root_stays_untrusted():
    # A↔B with no connection to any root: neither is reachable.
    r = propagate([Edge("a.com", "b.com"), Edge("b.com", "a.com")], {"root.com"})
    assert not r["a.com"].trusted and not r["b.com"].trusted


def test_duplicate_edges_do_not_count_as_two_pointers():
    # Same source linking C twice is ONE distinct pointer, not enough for Rule 3.
    edges = [Edge("karpathy.ai", "c.com", via="bio"), Edge("karpathy.ai", "c.com", via="footer")]
    r = propagate(edges, {"karpathy.ai"})
    assert not r["c.com"].trusted


def test_fixed_point_requires_iteration():
    # D can only graduate AFTER C does. Round 1: C trusted by two roots.
    # Round 2: C + a root both point at D → D trusted. A single pass would miss D.
    edges = [
        Edge("x.com/k", "c.com"), Edge("y.com", "c.com"),   # → C trusted (Rule 3)
        Edge("c.com", "d.com"), Edge("x.com/k", "d.com"),   # → D trusted once C is
    ]
    r = propagate(edges, {"x.com/k", "y.com"})
    assert r["c.com"].trusted
    assert r["d.com"].trusted, "fixed-point iteration must propagate trust to D"


def test_untrusted_no_corroboration_reason():
    # A node nothing trusted points at → "no corroboration", with inbound edges kept.
    edges = [Edge("squatter.com", "karpathy.ai")]
    r = propagate(edges, {"karpathy.ai"})
    ev = r["squatter.com"]
    assert not ev.trusted
    assert "no corroboration" in ev.reasons[0].lower()


def test_untrusted_near_miss_reason_names_the_single_pointer():
    # One trusted root links the candidate (no 2nd link, no back-edge) → near miss.
    # This is the naval.substack.com case: review must say WHY it's close.
    edges = [Edge("karpathy.ai", "blog.com")]
    r = propagate(edges, {"karpathy.ai"})
    ev = r["blog.com"]
    assert not ev.trusted
    assert "Near miss" in ev.reasons[0]
    assert "karpathy.ai" in ev.reasons[0]
