"""
pipeline/ingestion/sources/
Layer-1 source helpers: fetch a thing, render a thing. No vault, no notes, no state files.

These are the parts of the old ``pipeline/ingestion/ingest_{substack,blog,github}.py`` that the
ATOM path actually calls. They were fused into those modules with the vault ``sync_*``
note-writers, which meant "delete the vault adapters" also meant "break ``pipeline/kb/``". Splitting
them apart is what lets the note-writing half be deleted outright.

Nothing in here may import state machinery: no ``StatePaths``, no ``load_state``/``save_state``, no
note rendering that assumes a vault directory. A helper here answers "what does this URL contain",
never "where does it get filed".
"""
