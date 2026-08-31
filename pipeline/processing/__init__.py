"""pipeline/processing — the vault producer package. It is now EMPTY, and that is the finished
state, not a work-in-progress.

Every module that lived here wrote vault markdown, and all of them are deleted:

  • `run` (raw → vault-note orchestrator), `clean`, `fetch_urls`, `enrich_urls`, `output`
    (the note writer), `describe_images` — deleted with the vault-db migration.
  • `ocr_cascade`, `parse_raw`, `classify` — moved up to `pipeline/` (they were never
    vault-specific); the `moved-out-of-vault-producer` guard points at the new homes.
  • `process_braindump_articles` — retired with the braindump pipeline.
  • `process_academic_papers` — DELETED 2026-08-13, the last one out. It completed the `pending`
    stubs `save_paper` queued, and `save_paper` was deleted the same day. Its queue file had never
    been written on any machine.

The package survives as a package only so that a stray `from pipeline.processing.X import ...`
fails against a guard rule with an explanation, rather than as a bare ImportError that tells the
next session nothing. Do not add a module here. Nothing writes vault markdown any more; content
enters through the atom rail (`pipeline/kb/ingest_*`).
"""
