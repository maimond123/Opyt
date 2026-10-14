"""
pipeline/config.py — user-home state paths, gated on setup having run.

ConfigNotFoundError is how a caller learns setup never ran. Settings values are read through
`opyt_core.config.settings()` or `pipeline.ingestion.utils.load_yaml_config()`, not this class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from opyt_core.paths import opyt_path


class ConfigNotFoundError(Exception):
    """Raised when settings.yaml is not found and no config was provided."""
    pass


@dataclass
class StatePaths:
    """State-path anchor for the pipeline's ingesters."""

    state_dir: Path

    @classmethod
    def resolve(cls, path: Path | None = None) -> StatePaths:
        """Require the active settings.yaml to exist, then resolve state under OPYT_HOME.

        Raises ConfigNotFoundError if the resolved file is missing; its content is not read.
        """
        if path is None:
            from opyt_core.config import config_path
            path = config_path()

        if not path.exists():
            raise ConfigNotFoundError(
                f"settings.yaml not found at {path}. "
                "Run `onboard`, or copy config/settings.example.yaml to ~/.opyt/settings.yaml."
            )

        # Settings may live elsewhere; runtime state belongs to the current data home.
        return cls(state_dir=opyt_path("state"))

    def state_file(self, name: str) -> Path:
        """<OPYT_HOME>/state/{name}.json, creating the directory."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        return self.state_dir / f"{name}.json"


def state_paths() -> StatePaths:
    """Resolve state for the current OPYT_HOME on each call."""
    return StatePaths.resolve()
