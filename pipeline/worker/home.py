"""Which home and control database this process schedules for, read from its environment."""
from __future__ import annotations

import os
from pathlib import Path

from opyt_core.paths import opyt_path, validate_home_id

LOCAL_HOME_ID = "local"

# The two non-secret facts a hosted MCP child is given about the worker, and the only way it
# learns them. `OPYT_WORKER_HOME_ID` is the Google subject the gateway already validated;
# `OPYT_WORKER_DB` is the operator's one shared control database. No tool argument may set
# either, which is what stops one user's request from scheduling work against another's home.
WORKER_HOME_ID_ENV = "OPYT_WORKER_HOME_ID"
WORKER_DB_ENV = "OPYT_WORKER_DB"

class WorkerDbNotConfigured(RuntimeError):
    """A hosted process has a home id but no shared control database to write it into."""


def worker_db_path() -> Path:
    """The configured control database, or the local home's control database.

    A hosted process must be told the shared database explicitly: falling back to
    ``<OPYT_HOME>/rail_jobs.db`` would queue work where the worker never looks, so it raises.
    """
    configured = os.environ.get(WORKER_DB_ENV)
    if configured:
        return Path(configured).expanduser()
    if os.environ.get(WORKER_HOME_ID_ENV):
        raise WorkerDbNotConfigured(
            f"{WORKER_HOME_ID_ENV} is set without {WORKER_DB_ENV}; a hosted process cannot "
            f"queue rail work into a per-home database the worker never opens")
    return opyt_path("rail_jobs.db")


def queue_is_shared() -> bool:
    """Does this process queue into a shared control database that a separate resident worker
    claims from, rather than into its own home's? Not the same question as "can this platform
    install a LaunchAgent". Keyed on `WORKER_HOME_ID_ENV`, which only the gateway sets and which
    `worker_db_path` refuses without the shared database. It does not prove the worker is up.
    """
    return os.environ.get(WORKER_HOME_ID_ENV) is not None


def current_home_id() -> str:
    """The home id this process may schedule work for: its own, and only its own.

    Hosted, that is the subject the gateway validated before it named a directory. Locally
    there is one home and one fixed id. Validated again here, since an operator's service file
    can also set it.
    """
    configured = os.environ.get(WORKER_HOME_ID_ENV)
    return LOCAL_HOME_ID if configured is None else validate_home_id(configured)
