"""body_state / body_basis — the atom-level answer to "is the body on disk the WHOLE thing?".

Two keys, mirroring `when_ts`/`when_precision`: the STATE, and how much to trust it. The pairing
matters because blog's `complete` ("nothing indicated otherwise") and Substack's `complete` ("the
API declared this post public") are not the same claim, and a single enum would make them read
as one.

The load-bearing test here is `test_every_adapter_writes_both_keys`. A constant value is an
ASSERTION ("this adapter only ever stores complete bodies") and is testable; an absent key is
indistinguishable from an unimplemented one, so encoding the invariant as a blank puts it in
tribal knowledge where it rots the moment a store-vs-skip policy changes.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pipeline.kb import schema
from pipeline.kb.ingest_common import (BASIS_ASSUMED, BASIS_OBSERVED, BASIS_STATED, BODY_ABSENT,
                                       BODY_COMPLETE, BODY_PARTIAL, BODY_PENDING, body_fields)


# ── the builder ───────────────────────────────────────────────────────────────────
def test_body_fields_returns_both_keys():
    assert body_fields(BODY_PARTIAL, BASIS_STATED) == {"body_state": "partial",
                                                       "body_basis": "stated"}


@pytest.mark.parametrize("state,basis", [("compleet", BASIS_STATED),      # typo'd state
                                         (BODY_COMPLETE, "guessed")])     # typo'd basis
def test_unknown_values_raise(state, basis):
    """A typo'd value would otherwise become a silent extra category that every consumer's
    branch misses — the failure mode is invisible, so it has to fail at the write."""
    with pytest.raises(ValueError):
        body_fields(state, basis)


# The verdict->state mapper `body_from_verdict` was deleted 2026-08-28: every adapter calls the
# sibling `body_fields` directly and nothing ever called it. Its tests went with it.

# ── the retry set ─────────────────────────────────────────────────────────────────
def _put(conn, atom_id: str, payload: dict) -> None:
    schema.upsert_atom(conn, {"atom_id": atom_id, "source_type": "substack", "payload": payload})


def test_load_body_pending_selects_only_pending(kb_home):
    """Only a blocked fetch (`pending`) earns a retry — `absent` and `complete` never re-fetch.
    (A legacy `body_pending` boolean arm was read here too until 2026-08-29; deleted after its
    removal condition — zero legacy-key atoms on any store the old writer touched — was met.)"""
    conn = schema.connect()
    _put(conn, "substack:new", body_fields(BODY_PENDING, BASIS_OBSERVED))
    _put(conn, "substack:done", body_fields(BODY_COMPLETE, BASIS_STATED))
    _put(conn, "substack:empty", body_fields(BODY_ABSENT, BASIS_OBSERVED))  # never retried

    assert schema.load_body_pending(conn, "substack") == {"substack:new"}


def test_rewriting_as_complete_clears_pending(kb_home):
    """Self-clearing — a successful re-fetch UPSERTs `complete` and it drops out of the retry set."""
    conn = schema.connect()
    _put(conn, "substack:99", body_fields(BODY_PENDING, BASIS_OBSERVED))
    assert schema.load_body_pending(conn, "substack") == {"substack:99"}
    _put(conn, "substack:99", body_fields(BODY_COMPLETE, BASIS_STATED))
    assert schema.load_body_pending(conn, "substack") == set()


# ── the universality property ─────────────────────────────────────────────────────
_ADAPTERS = ("ingest_x", "ingest_x_footprint", "ingest_substack", "ingest_curation",
             "ingest_blog", "ingest_github", "ingest_papers")


def _payload_dicts(path: Path) -> list[ast.Dict]:
    """Every `"payload": {...}` literal in one adapter, found structurally rather than by grep so
    a reformat or a multi-line dict can't slip past this guard."""
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (isinstance(key, ast.Constant) and key.value == "payload"
                    and isinstance(value, ast.Dict)):
                out.append(value)
    return out


def _writes_body_fields(payload: ast.Dict) -> bool:
    """True if this payload literal contributes both keys — either spelled out, or splatted from a
    `body_fields(...)` call (`**` shows up as a None key). The alternative spelling this used to
    accept, `body_from_verdict`, was deleted 2026-08-28 having never been an adapter's route."""
    keys = {k.value for k in payload.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    if {"body_state", "body_basis"} <= keys:
        return True
    return "body_fields" in ast.dump(payload)


def test_every_adapter_writes_both_keys():
    """THE load-bearing test. Every atom of every source must carry the completeness contract —
    a constant is an assertion, an absence is unfalsifiable. A new adapter that forgets fails
    here instead of quietly leaving a blind spot months later."""
    kb = Path(__file__).resolve().parents[2] / "pipeline" / "kb"
    inspected, missing = 0, []
    for name in _ADAPTERS:
        for payload in _payload_dicts(kb / f"{name}.py"):
            inspected += 1
            if not _writes_body_fields(payload):
                missing.append(name)
    assert inspected >= 8, f"guard inspected only {inspected} payloads — it has gone blind"
    assert not missing, f"adapters missing the body contract: {sorted(set(missing))}"


def test_blog_is_the_assumed_one():
    """Blog is the reason `body_basis` exists: a truncated 'read more' preview returns a healthy
    200 with real prose and no marker, so `complete` there means 'nothing indicated otherwise'."""
    src = (Path(__file__).resolve().parents[2] / "pipeline" / "kb" / "ingest_blog.py").read_text()
    assert "BASIS_ASSUMED" in src
    for name in ("ingest_substack", "ingest_papers", "ingest_github"):
        other = (Path(__file__).resolve().parents[2] / "pipeline" / "kb" / f"{name}.py").read_text()
        assert "BASIS_ASSUMED" not in other, f"{name} should not be assuming completeness"
