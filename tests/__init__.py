"""Marks `tests/` as a package so pytest puts the REPO ROOT on `sys.path`, not `tests/`.

Without this file, pytest inserts the first `__init__.py`-less ancestor of each test module —
`tests/` — which makes `tests/service/` importable as a top-level `service` and SHADOWS the real
`service/` package. The existing `from tests.kb.conftest import ...` imports already assume the
repo root is what is on the path; this makes that true by construction instead of by luck.
"""
