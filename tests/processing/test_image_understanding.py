"""Money-safe failure skip for the paid vision path.

⚠️ MOST OF THIS FILE WENT WITH THE RADAR RAIL on 2026-08-13. It used to cover
`pipeline/parse_raw.py`'s image handling — `extract_media_urls`, `_is_chrome_image`,
`parse_file` — under headings A (source-agnostic extraction), B (chrome gate) and C
(tagged descriptions). `parse_raw` was deleted with `pipeline/radar/`; see
`docs/plans/2026-08-13-delete-the-radar-rail.md`.

THAT COVERAGE WAS NOT LOST, IT WAS ALREADY DUPLICATED. The atom rail never went through
`parse_raw` — it carries its own markdown image enricher at `pipeline/kb/vision.py` plus
`ingest_common.looks_like_image_url`, and those are covered by
`tests/kb/test_vision_markdown_images.py` (relative refs, non-image targets, cache keys,
concurrency). Verified two ways before deleting: no `pipeline/kb/*` module imported
`parse_raw` at the AST level, and the parallel implementation exists with its own tests.

WHAT SURVIVES HERE IS THE ONE TEST THAT NEVER TOUCHED `parse_raw`. It asserts on
`ocr_cascade.read_image`, which BOTH rails used and the atom rail still does
(`pipeline/kb/ingest_x.py` and `pipeline/kb/vision.py` import it directly). It pins a CLAUDE.md
invariant — a failed external call must SKIP, never write a partial or poison value — so it
outlives the rail that happened to host its test file.
"""

from __future__ import annotations

import pytest

from pipeline import llm_client


def _fake_raise(*, role, system, user, **kw):
    raise RuntimeError("HTTP 402: no credit")       # simulates a paid-call failure


# ── money-safe failure skip ──────────────────────────────────────────────────
def test_failure_returns_none_not_empty(monkeypatch):
    """None is the skip sentinel: never persist a poison value on a failed PAID call.

    The distinction that matters is None vs "" — an empty string is a value, it caches, and it
    then reads as "this image has no text in it" forever. None is not cached, so the next run
    retries. `$0-credit once dumped 2097 garbage rows` is the incident this pins against.
    """
    from pipeline import model_routing, ocr_cascade
    ocr_cascade._reset_stage_for_tests()     # clears the conftest pin → pin resolution explicitly
    monkeypatch.setattr(model_routing, "resolve_ocr_model",
                        lambda **kw: (model_routing.OCR_MODEL, "pinned"))
    monkeypatch.setattr(llm_client, "call", _fake_raise)
    assert ocr_cascade.read_image("https://x/y.jpg") is None
