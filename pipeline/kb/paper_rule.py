"""
pipeline/kb/paper_rule.py — which linked papers become records, and which get a line. Pure.

A paper linked from
a post you saved on any platform, or from an Oracle's own X post, becomes its own record, abstract
only (`references/documents`). Every other linked paper gets one line in the post that links it,
title, authors and year (`references/cards`). A paper URL put in the hopper is a record too; that
path never reaches here.
"""

from __future__ import annotations


def earns_record(source_type: str | None, entry_mode: str | None) -> bool:
    """Does a paper linked from this post become its own record? Yes when you saved the post, or an
    Oracle posted it on X. No for an Oracle's blog or Substack post, a page an Oracle linked, or a
    GitHub README: there the post carries the meaning and its papers are citations."""
    return entry_mode == "user-saved" or (source_type == "x" and entry_mode == "oracle-footprint")
