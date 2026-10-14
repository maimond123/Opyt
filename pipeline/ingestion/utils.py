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
    """Load the active settings.yaml as a raw dict.

    Resolves through opyt_core.config.config_path() ($OPYT_CONFIG → ~/.opyt → repo default).
    Raises ConfigNotFoundError rather than exiting. Prefer opyt_core.config.settings() for new code.
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

    Fatal to the whole source: never report it as "caught up / 0 new", so per-item guards must
    not swallow it.
    """


# ── State tracking ────────────────────────────────────────────────────────────

def load_state(state_file: Path):
    """Return the dedup set for ``state_file``'s namespace (the path's stem).

    State lives in the ``sync_dedup`` SQLite table; the JSON file is read at most once to seed it.
    The returned ``SyncSet`` behaves like a ``set``.
    """
    from pipeline.dedup_store import SyncSet

    return SyncSet(namespace=state_file.stem, legacy_json=state_file)


def save_state(state_file: Path, ids) -> None:
    """Persist a plain set/iterable of dedup IDs, add-only: dedup sets never shrink, so an
    item is never re-pulled or re-billed.
    """
    from pipeline.dedup_store import SyncSet

    s = SyncSet(namespace=state_file.stem, legacy_json=state_file)
    s.update(ids)
    s.close()
