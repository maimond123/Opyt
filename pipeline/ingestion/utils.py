"""
pipeline/ingestion/utils.py
Shared helpers used across all ingestion scripts.
"""

import sys
import yaml
from datetime import datetime
from pathlib import Path

from pipeline.config import ConfigNotFoundError

def load_yaml_config() -> dict:
    """Load the ACTIVE settings.yaml as a raw dict.

    Resolves through opyt_core.config.config_path() ($OPYT_CONFIG → ~/.opyt → repo default), so a
    distributed install reads the user's config, not the committed template. Raises
    ConfigNotFoundError rather than exiting. Prefer opyt_core.config.settings() for new code.
    """
    from opyt_core.config import config_path

    path = config_path()
    if not path.exists():
        raise ConfigNotFoundError(
            f"settings.yaml not found at {path}. "
            "Run `onboard`, or copy config/settings.example.yaml to ~/.opyt/settings.yaml."
        )
    with open(path) as f:
        return yaml.safe_load(f)


def log(msg: str) -> None:
    # stderr, not stdout: keeps `--json` output machine-parseable.
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True, file=sys.stderr)


# ── Failure handling ──────────────────────────────────────────────────────────

class SyncAuthError(Exception):
    """A source's credential is dead/expired (e.g. X OAuth refresh token revoked).

    Fatal to the whole source: must never be reported as "caught up / 0 new". A per-item guard
    that swallows this reports a dead credential as an empty inbox, which is the failure the
    class exists to name.
    """


# ── State tracking ────────────────────────────────────────────────────────────

def load_state(state_file: Path):
    """Return the dedup set for ``state_file``'s namespace.

    Dedup state lives in the ``sync_dedup`` SQLite table (path's stem = namespace), not the
    JSON blob; the JSON file is read at most once to seed the table, then ignored. See
    pipeline/dedup_store.py for why. The returned ``SyncSet`` quacks like the old ``set``.
    """
    from pipeline.dedup_store import SyncSet

    return SyncSet(namespace=state_file.stem, legacy_json=state_file)


def save_state(state_file: Path, ids) -> None:
    """Persist a plain set/iterable of dedup IDs — bulk-upserted, add-only. Dedup sets never
    shrink, to avoid re-pulling or re-billing an item. (A ``SyncSet`` needs no saving — each
    ``.add()`` is already durable — and no caller passes one.)
    """
    from pipeline.dedup_store import SyncSet

    s = SyncSet(namespace=state_file.stem, legacy_json=state_file)
    s.update(ids)
    s.close()
