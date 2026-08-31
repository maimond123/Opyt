"""Tests for handle_match — tight-fuzzy ownership matching.

The crisp contract: normalized-equality always matches; edit-distance ≤ 1 only at
≥ 6 chars; generic/short handles never match.
"""

from pipeline.ingestion.handle_match import (
    _edit_distance_le1,
    is_generic,
    match_known_handle,
    normalize_handle,
)


# ── normalize_handle ──────────────────────────────────────────────────────────

def test_normalize_strips_at_underscore_case():
    assert normalize_handle("@Gergely_Orosz") == "gergelyorosz"
    assert normalize_handle("Will.CB") == "willcb"
    assert normalize_handle(None) == ""
    assert normalize_handle("  ") == ""


# ── _edit_distance_le1 ────────────────────────────────────────────────────────

def test_edit_distance_equal_sub_indel():
    assert _edit_distance_le1("gergelyorosz", "gergelyorosz")   # equal
    assert _edit_distance_le1("gergelyorosz", "gergelyorosa")   # 1 substitution
    assert _edit_distance_le1("gergelyorosz", "gergelyoros")    # 1 deletion
    assert _edit_distance_le1("gergelyoros", "gergelyorosz")    # 1 insertion
    assert not _edit_distance_le1("gergely", "orosz")           # far apart
    assert not _edit_distance_le1("willcb", "willccbb")         # 2 insertions → NOT ≤1


# ── is_generic ────────────────────────────────────────────────────────────────

def test_generic_and_short_are_blocked():
    assert is_generic("blog")
    assert is_generic("admin")
    assert is_generic("abcd")          # len 4 < 5
    assert not is_generic("willcb")    # len 6, real
    assert not is_generic("gergelyorosz")



def test_willcb_domain_does_not_fuzzy_match_willccbb():
    # The plan's headline: willccbb auto-trusts via the EXACT username, NOT via a
    # fuzzy match on the 'willcb' blog label (edit distance 2). Guard that.
    assert match_known_handle("willccbb", {"willcb"}) is None
    assert match_known_handle("willccbb", {"willccbb", "willcb"}) == "willccbb"
