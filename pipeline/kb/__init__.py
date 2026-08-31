"""pipeline.kb — the atom-KB layer: ingestion, the background rails, retrieval, and the
sitting/frontier machinery that read and write the store.

See docs/plans/2026-07-21-opyt-end-to-end-flow.md (master reference) for the current
architecture; individual subsystems are documented in their own modules (schema.py, retrieve.py,
rail_runtime.py, etc.) — per this repo's CLAUDE.md, this docstring is a pointer, not a state
description, and must not be trusted as a list of what the package holds.
"""
