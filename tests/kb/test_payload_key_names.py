"""Payload key NAMES are a contract, and this pins the ones whose meaning is not self-evident.

`payload` is a free-form JSON blob, so nothing structural stops two adapters writing the same key
with incompatible meanings. Nothing catches it either: no schema, no migration, no type error — the
two values just land in one key and every reader downstream guesses. These tests are the only thing
standing between "the name says what it is" and "you have to know which adapter wrote it."

Checked with the AST rather than grep because the banned names are legitimate *source* field names.
GitHub's API really does call its field `language`; the ban is on writing it back out under that
name in `payload`, which grep cannot distinguish from reading it.
"""
from __future__ import annotations

import ast
from pathlib import Path

_KB = Path(__file__).resolve().parents[2] / "pipeline" / "kb"
# `frontier_sources` is here because its adapters write `frontier_candidates.payload`, a
# different table from `atoms.payload` but the same free-form blob with the same failure mode —
# and its GitHubAdapter shipped `language` and `topics` for months precisely because this list
# did not name it. A second writer for the same source is exactly when the names must agree.
_ADAPTERS = ("ingest_x", "ingest_x_footprint", "ingest_substack", "ingest_curation",
             "ingest_blog", "ingest_github", "ingest_papers", "frontier_sources")


def _literal_keys(d: ast.Dict) -> set[str]:
    return {k.value for k in d.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def _payload_keys(path: Path) -> set[str]:
    """Every literal key in every payload dict an adapter writes, in EITHER spelling.

    Two forms, because the atom writers and the frontier adapters build their payload
    differently and a check that saw only one form would pass the other vacuously — which is
    exactly what happened: `frontier_sources` writes `Candidate(..., payload={...})`, a keyword
    ARGUMENT, so the dict-entry walk returned an empty set for it and any ban would have been
    silently satisfied. A guard that cannot see its target is worse than an absent one.

      `"payload": {...}`   a dict entry   — the `ingest_*` atom writers
      `payload={...}`      a keyword arg  — `frontier_sources`' Candidate construction
    """
    keys = set()
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and key.value == "payload"
                        and isinstance(value, ast.Dict)):
                    keys |= _literal_keys(value)
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "payload" and isinstance(kw.value, ast.Dict):
                    keys |= _literal_keys(kw.value)
    return keys


def _all_payload_keys() -> dict[str, set[str]]:
    return {name: _payload_keys(_KB / f"{name}.py") for name in _ADAPTERS}


def test_no_adapter_writes_a_bare_language_key():
    """`language` is banned because two sources mean opposite things by it.

    GitHub's is a PROGRAMMING language ("C++"); Substack and blog both hand us a NATURAL language
    ("en"), which the field model captures as `content_lang`. Under one key they share an
    expression index, and a reader holding the value cannot tell which kind it has — "C++" and "en"
    are both just strings. Split at the NAME so the ambiguity cannot be constructed.
    """
    offenders = {n: ks for n, ks in _all_payload_keys().items() if "language" in ks}
    assert not offenders, (
        f"adapters writing a bare `language` payload key: {sorted(offenders)}. "
        "Use `code_language` (a programming language) or `content_lang` (a natural language).")


def test_github_declares_its_language_as_code():
    """The positive half — the ban is only meaningful if the replacement is actually present."""
    assert "code_language" in _payload_keys(_KB / "ingest_github.py")


# ── author-declared labels (§6) ─────────────────────────────────────────────────────
# Every source has its OWN word for "labels the author put on this": GitHub says `topics`,
# X says `hashtags`, Substack says `postTags`, blog says `tags`+`categories`. Storing each
# under its source's word makes "what did the author call this?" a per-source special case,
# which is the defect §6 exists to prevent. One key, every source.
_SOURCE_SPECIFIC_TAG_NAMES = {"topics", "hashtags", "postTags", "post_tags", "categories", "tags"}


def test_no_adapter_writes_a_source_specific_tag_key():
    offenders = {n: sorted(ks & _SOURCE_SPECIFIC_TAG_NAMES)
                 for n, ks in _all_payload_keys().items() if ks & _SOURCE_SPECIFIC_TAG_NAMES}
    assert not offenders, (
        f"adapters naming author-declared labels after their source: {offenders}. "
        "Author-declared labels go in `payload.source_tags` (§6) regardless of source.")


def test_the_sources_that_have_declared_labels_write_them():
    """The positive half. Only these three READ author labels today — Substack `postTags` and
    blog `meta.tags` are not captured yet.

    They must NOT write `source_tags: []` in the meantime. An empty list asserts "the author
    declared nothing," when the truth is "we never looked" — the same conflation the body
    contract just split apart (see `test_body_state.py`). An absent key is the honest shape
    until the read lands.
    """
    keys = _all_payload_keys()
    for name in ("ingest_github", "ingest_x", "ingest_x_footprint"):
        assert "source_tags" in keys[name], f"{name} reads author labels but does not store them"
    for name in ("ingest_substack", "ingest_blog"):
        assert "source_tags" not in keys[name], (
            f"{name} does not read author labels yet — writing the key would assert an "
            "emptiness we have not observed")


# ── the READ side: capture ⇒ visible, structurally ──────────────────────────────────
# The write-side tests above keep the names honest. This one keeps them REACHABLE.
#
# `payload` is returned to the host verbatim, so a field an adapter starts capturing is readable
# the day it lands — no second edit anywhere. The way that guarantee dies is quiet: someone needs
# one key on the read path, reaches for `payload.get("stars")`, and the next person copies the
# shape into a rebuilt dict. Now the returned payload is a fixed key list, every future field is
# invisible until someone remembers to extend it, and nothing fails while that is true. Pinned in
# the AST because there is no runtime moment where "an allowlist appeared" is observable.
_READ_PATH = {
    "pipeline/kb/retrieve.py": Path(__file__).resolve().parents[2] / "pipeline" / "kb" / "retrieve.py",
    "opyt_core/kb.py": Path(__file__).resolve().parents[2] / "opyt_core" / "kb.py",
}
# The ONLY literal keys the read path may name. Both are the documented promotion: moved to the
# top level of a hit because they qualify the snippet rather than describe the source.
_PROMOTED = {"body_state", "body_basis"}


def _literal_payload_keys(path: Path) -> set[str]:
    """Every string-literal key read off a local named `payload` — `payload["k"]`,
    `payload.get("k")`, `payload.pop("k")`."""
    keys = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "payload" and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            keys.add(node.slice.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "pop") and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "payload" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            keys.add(node.args[0].value)
    return keys


def test_the_read_path_names_no_payload_key_but_the_promoted_two():
    offenders = {name: sorted(_literal_payload_keys(p) - _PROMOTED)
                 for name, p in _READ_PATH.items()
                 if _literal_payload_keys(p) - _PROMOTED}
    assert not offenders, (
        f"read path naming individual payload keys: {offenders}. `payload` is returned VERBATIM "
        "so a newly captured field is visible with no read-side edit; naming keys here is the "
        "first step to an allowlist that silently hides every future one. Only "
        f"{sorted(_PROMOTED)} may be named — they are lifted to the top level of a hit.")


def test_the_promotion_itself_is_still_there():
    """The positive half. The ban above is satisfied trivially by a read path that stops
    extracting anything at all, which would put `body_state` back inside `payload` where a host
    about to quote a paywall teaser will not look for it."""
    for name, path in _READ_PATH.items():
        assert _literal_payload_keys(path) == _PROMOTED, (
            f"{name} no longer lifts {sorted(_PROMOTED)} out of payload")


def test_the_guard_can_actually_see_every_adapter_it_names():
    """The anti-vacuum check, and the reason this file grew a second AST form.

    A ban is only a ban if the extractor reaches the code. `frontier_sources` builds its payload
    as a keyword ARGUMENT, so the original dict-entry walk returned an empty set for it — every
    prohibition above would have "passed" while the module wrote two banned keys. An empty key
    set for a named adapter now fails here instead of quietly satisfying everything else.
    """
    empty = [n for n, ks in _all_payload_keys().items() if not ks]
    assert not empty, (
        f"adapters the extractor sees no payload keys for: {sorted(empty)}. Either they stopped "
        "writing a payload, or they build it in a form `_payload_keys` does not parse — in which "
        "case every ban in this file is passing vacuously for them.")
