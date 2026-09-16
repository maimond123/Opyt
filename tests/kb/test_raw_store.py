"""raw_store — one atom, one snapshot file.

`raw_ref` is the ONLY pointer from an atom to its real text: `open()` serves it, `export` inlines
it, and `raw_hash` (computed on the string, before the write) is the change key. So the atom_id →
filename map has to be INJECTIVE. It was not: every separator collapsed to `_`, and blog atom_ids
preserve the URL path, so two real posts on one host could land on one file — the second write
replacing the body the first atom still points at, while its `raw_hash` kept describing the body
that was overwritten.
"""
from __future__ import annotations

from pipeline.kb.raw_store import read_snapshot, write_snapshot

# Two atom_ids `ingest_blog._canon_post_url` really produces for one host: a path separator and a
# literal colon in the path. Every non-alphanumeric character used to become the same `_`.
_A = "blog:example.com/a/b"
_B = "blog:example.com/a:b"


def test_two_path_shaped_blog_ids_get_their_own_snapshots(kb_home):
    ref_a, hash_a = write_snapshot("blog", _A, "body A")
    ref_b, hash_b = write_snapshot("blog", _B, "body B")

    assert ref_a != ref_b
    assert read_snapshot(ref_a) == "body A"       # not clobbered by the second write
    assert read_snapshot(ref_b) == "body B"
    assert hash_a != hash_b


def test_the_same_id_overwrites_its_own_snapshot(kb_home):
    """Deterministic, not merely unique — a re-ingest of an unchanged atom must reuse its file
    rather than accumulate a second one."""
    ref_first, _ = write_snapshot("blog", _A, "body A")
    ref_again, _ = write_snapshot("blog", _A, "body A revised")
    assert ref_again == ref_first
    assert read_snapshot(ref_first) == "body A revised"


def test_raw_ref_stays_relative_to_opyt_home(kb_home):
    """The Distributable invariant: a copied `~/.opyt` still finds its snapshots, so the stored
    ref may never carry an absolute path."""
    ref, _ = write_snapshot("blog", _A, "body A")
    assert not ref.startswith("/") and ref.startswith("kb_raw/blog/")
    assert (kb_home / ref).is_file()
