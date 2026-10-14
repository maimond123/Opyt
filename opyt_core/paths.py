"""opyt_core/paths.py — the single source of truth for the OPYT data home.

Every path under ``~/.opyt`` resolves through here, so one ``OPYT_HOME`` relocates the whole
home in lockstep (database, ``.env``, settings, caches). There is deliberately no per-file
override: it would let a run write half its state to a sandbox and half to the real home.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_DEFAULT_HOME = Path.home() / ".opyt"

# A hosted home's id: a verified subject (Google's `sub`). It names the home's
# directory and keys its worker jobs. No dot is allowed, so traversal ("..") is unrepresentable.
HOME_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def validate_home_id(home_id: str) -> str:
    """Validate the identifier before it can become durable path-selection input."""
    if not HOME_ID.fullmatch(home_id):
        raise ValueError(f"home id is not a usable directory name: {home_id!r}")
    return home_id


def opyt_home() -> Path:
    """The OPYT data home. Honors ``$OPYT_HOME``; defaults to ``~/.opyt``.

    Pure: it does not create the directory; callers mkdir where they write.
    """
    env = os.environ.get("OPYT_HOME")
    return Path(env).expanduser() if env else _DEFAULT_HOME


def ensure_home(path: Path | str) -> Path:
    """Create a data home directory owner-only, or tighten an existing one to owner-only.

    The single owner of home creation; a home holds keys and user content, so a bare ``mkdir``
    would leave it world-readable. Only the leaf is chmodded, fail-safe.
    """
    path = Path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    owner_only(path, 0o700)  # mkdir's mode is masked by umask, and the dir may predate this
    return path


def owner_only(path: Path | str, mode: int = 0o600) -> None:
    """Make `path` readable by its owner only (0600 for a file, 0700 passed for a directory).

    Fail-safe: a filesystem that refuses chmod must not break the write that needed the file.
    """
    try:
        Path(path).chmod(mode)
    except OSError:
        pass


def opyt_path(*parts: str) -> Path:
    """A path under the OPYT home. ``opyt_path("opyt.db")`` -> ``<home>/opyt.db``."""
    return opyt_home().joinpath(*parts)


def opyt_db() -> Path:
    """The one SQLite DB (``<home>/opyt.db``)."""
    return opyt_path("opyt.db")
