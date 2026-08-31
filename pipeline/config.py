"""
pipeline/config.py
Repo-anchored state paths for the ingesters, gated on setup having run.

The one live surface is `state_file()` (dedup/signal state for x_likes, x_lists,
discover_profile) plus the settings.yaml existence check — ConfigNotFoundError is how a
caller learns setup never ran. Settings VALUES are read through
`opyt_core.config.settings()` or `pipeline.ingestion.utils.load_yaml_config()`, never
through this class.

Named `VaultConfig` until 2026-08-29 — renamed with the deletion of its parsed-field
surface (8 YAML-parsed fields, a dotted-key `.get()`, a `from_paths()` constructor, none
with a production reader). There is no vault,
and after the collapse the class was config in name only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ConfigNotFoundError(Exception):
    """Raised when settings.yaml is not found and no config was provided."""
    pass


@dataclass
class StatePaths:
    """State-path anchor for the pipeline's ingesters.

    No `raw_path` and no vault fields — do not re-add them (see the `retired-raw-rail`
    and `retired-vault-path-resolver` guards).
    """

    state_dir: Path

    @classmethod
    def resolve(cls, path: Path | None = None) -> StatePaths:
        """Require the active settings.yaml to exist, then anchor state paths to the repo.

        Raises ConfigNotFoundError if the resolved file is missing. The file's CONTENT is
        not read here — no field of this class comes from it.
        """
        if path is None:
            # Resolve via the one resolver ($OPYT_CONFIG → ~/.opyt → repo default), same precedence every reader uses.
            from opyt_core.config import config_path
            path = config_path()

        if not path.exists():
            raise ConfigNotFoundError(
                f"settings.yaml not found at {path}. "
                "Run `onboard`, or copy config/settings.example.yaml to ~/.opyt/settings.yaml."
            )

        # state/ anchors to this module's location, not to `path` (which may resolve to ~/.opyt/settings.yaml).
        repo_root = Path(__file__).resolve().parent.parent
        return cls(state_dir=repo_root / "state")

    def state_file(self, name: str) -> Path:
        """Get state/{name}.json path."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        return self.state_dir / f"{name}.json"


# ── Global singleton ────────────────────────────────────────────────────────

_paths: StatePaths | None = None


def state_paths() -> StatePaths:
    """Get the StatePaths singleton. Lazy-resolves (and so setup-gates) on first use.

    Callers take an optional explicit `config`/`state_file` parameter and fall back
    here — pass explicitly in new code and tests, avoid the singleton.
    """
    global _paths
    if _paths is None:
        _paths = StatePaths.resolve()
    return _paths
