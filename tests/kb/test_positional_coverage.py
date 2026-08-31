"""D22's standing metric: did the model read the MIDDLE of the document?

Long-context models attend well to the start and end of a window and poorly to what is between.
A sitting renders chronologically, so the middle of the context window IS the middle of the time
period — and a model with that weakness produces a read weighted to the oldest and newest material
while every other counter stays perfectly healthy. Same silent-degradation shape as the 2026-08-01
OCR outage.

These lock the metric's ability to NAME each failure, and — just as important — its refusal to
manufacture one:

  • A uniform read is not flagged. A metric that always complains is a banner.
  • Lost-in-the-middle is flagged, by the number that shows it.
  • Undated atoms never enter the histogram. They trail the render with no position in time, so
    counting them fabricates recency bias in the exact shape of the real finding.
  • Too few citations to judge says so, rather than reporting a shape.
  • Degenerate inputs return a report, never an exception — this rides along on a PAID read and
    must never be what turns a success into a failure.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.kb import reader_core as core
from pipeline.kb import schema
from pipeline.kb import sitting_builder as sb
from pipeline.kb import sitting_render as sre

DIM = 32
ORDER = [f"a:{i:02d}" for i in range(40)]


def test_a_uniform_read_is_not_flagged():
    """THE NEGATIVE CONTROL. A note on every read trains the reader to skip the one that matters."""
    cov = core.positional_coverage(ORDER, ORDER[::4])          # every 4th atom, end to end
    assert cov["note"] is None
    assert cov["covered"] == 10
    assert 0.4 <= cov["middle_share"] <= 0.6


def test_lost_in_the_middle_is_named_with_the_number_that_shows_it():
    """Citations at both ends, nothing between — the exact signature, and the one that is invisible
    to every other signal on a read."""
    cited = ORDER[:6] + ORDER[-6:]
    cov = core.positional_coverage(ORDER, cited)
    assert cov["middle_share"] == 0.0
    assert "middle half" in cov["note"] and "skimming" in cov["note"]


def test_recency_bias_shows_as_a_tail_cluster():
    cov = core.positional_coverage(ORDER, ORDER[-10:])
    assert cov["covered"] <= 3
    assert cov["note"] and cov["middle_share"] < 0.25


def test_a_terse_lens_is_not_mistaken_for_a_degraded_one():
    """⚠️ THE FALSE FINDING THIS METRIC WOULD OTHERWISE MANUFACTURE, and it was caught by running
    the bake-off rather than by reading the code. Raw `covered` conflates citing SPARSELY with
    reading sparsely. Twelve citations thrown uniformly at ten sections only touch about eight of
    them, because some land together — so a lens that cites once per claim would be reported as
    lost-in-the-middle forever. The expectation correction is what separates the two."""
    cited = ORDER[::3][:12]                       # 12 citations, evenly spread end to end
    cov = core.positional_coverage(ORDER, cited)
    assert cov["cited"] == 12
    assert cov["covered"] < 10                    # cannot touch all ten with twelve citations...
    assert cov["covered_expected"] < 10           # ...and the metric knows that
    assert cov["note"] is None                    # so it says nothing


def test_clustering_is_still_caught_after_the_correction():
    """The correction must not be an amnesty. Twelve citations packed into two sections is real
    clustering, and the expectation for twelve is ~8."""
    cov = core.positional_coverage(ORDER, ORDER[:6] + ORDER[8:14])
    assert cov["covered"] <= 4 < cov["covered_expected"]
    assert cov["note"] is not None


def test_undated_atoms_are_excluded_rather_than_bucketed():
    """⚠️ THE TRAP. Undated atoms trail the render as a block with no position in time. Counting
    them piles every one into the last decile and reports recency bias — a fabricated finding
    wearing the exact shape of the real one this metric hunts."""
    undated = [f"u:{i}" for i in range(10)]
    order = ORDER + undated                       # as the render lays it out
    cov = core.positional_coverage(order, ORDER[::4] + undated, undated=undated)
    assert cov["atoms"] == 40                     # the undated block is not part of the document
    assert cov["undated_cited"] == 10             # reported, not silently dropped
    assert cov["note"] is None                    # the DATED citations are uniform, so: no finding


def test_too_few_citations_says_so_instead_of_reporting_a_shape():
    """Five citations land wherever they land. Naming a shape there is noise with a percentage
    sign on it."""
    cov = core.positional_coverage(ORDER, ORDER[:3])
    assert "too few" in cov["note"]
    assert cov["cited"] == 3


def test_unresolvable_citations_never_become_phantom_positions():
    """Citations must be passed in AFTER `_resolve_atom_ids`. An id that is not in the document has
    no position, so it is not counted — a hallucinated id must not look like coverage."""
    cov = core.positional_coverage(ORDER, ["nope:1", "nope:2"] + ORDER[:2])
    assert cov["cited"] == 2


def test_fewer_atoms_than_buckets_does_not_read_as_a_degraded_read():
    """With 9 atoms and 10 deciles, one decile is empty BY CONSTRUCTION — and with 5 atoms, five
    are. Reporting that as a partly-covered document would flag every small region forever, so the
    bucket count follows the document when the document is smaller."""
    cov = core.positional_coverage(ORDER[:9], ORDER[:9])
    assert len(cov["buckets"]) == 9 and cov["covered"] == 9
    assert cov["note"] is None


@pytest.mark.parametrize("order,cited", [([], []), ([], ["a:1"]), (ORDER, [])])
def test_degenerate_inputs_return_a_report_rather_than_raising(order, cited):
    """FAIL-SAFE. This is an observability signal riding on a paid read; it must never be the thing
    that turns a successful read into a failed one."""
    cov = core.positional_coverage(order, cited)
    assert cov["cited"] == 0 and isinstance(cov["buckets"], list)


def test_no_resolved_citations_reports_unknown_not_uniform():
    """"We could not tell" and "it read everything" are different facts and must not render the
    same — the second is the one that would let a broken read pass."""
    assert "unknown, not uniform" in core.positional_coverage(ORDER, [])["note"]


# ── the ordering the metric measures against ────────────────────────────────────
@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    from pipeline.kb.embed import ensure_kb_meta
    ensure_kb_meta(c, "fake", DIM, "local", "", storage_dtype="float32")
    yield c
    c.close()


def _at_cos(c: float, axis: int = 1) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[0], v[axis] = c, float(np.sqrt(max(0.0, 1.0 - c * c)))
    return v / (np.linalg.norm(v) + 1e-9)


def _atom(conn, atom_id, vec, *, when="2026-08-01"):
    conn.execute("INSERT INTO atoms (atom_id, source_type, who_id, when_ts, entry_mode) "
                 "VALUES (?,?,?,?,'user-saved')", (atom_id, "x", "x:u", when))
    text = f"{atom_id} " + ("word " * 60)
    conn.execute("INSERT INTO chunks (atom_id, seq, char_start, char_end, text, vector) "
                 "VALUES (?,0,0,?,?,?)", (atom_id, len(text), text, vec.tobytes()))
    conn.commit()


def test_the_metric_and_the_render_share_one_ordering(conn):
    """⚠️ THE POINT OF EXTRACTING `_chronological`. The metric asks "where in the document did the
    model look", which is a question about POSITION. A second, subtly different sort would
    mis-attribute every citation while both orderings still looked chronological."""
    _atom(conn, "a:seed", _at_cos(1.0), when="2026-05-01")
    _atom(conn, "a:old", _at_cos(0.85, axis=1), when="2026-01-01")
    _atom(conn, "a:new", _at_cos(0.85, axis=2), when="2026-09-01")
    _atom(conn, "a:undated", _at_cos(0.85, axis=3), when="")
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)

    chrono = sre.chronological_order(conn, rec["sitting_id"])
    assert chrono["order"] == ["a:old", "a:seed", "a:new", "a:undated"]
    assert chrono["undated"] == ["a:undated"]

    # The render is the ground truth: the metric's positions must be the ones a reader saw.
    md = sre.render_sitting(conn, rec["sitting_id"])
    rendered = [ln.rsplit("(", 1)[1].rstrip(")") for ln in md.splitlines() if ln.startswith("### ")]
    assert rendered == chrono["order"]
