"""
opyt_core — the shared Python core for OPYT's front-ends.

Loads the .env files on import so provider keys resolve regardless of the caller's cwd: the MCP
server is spawned by its client with an arbitrary cwd.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from .paths import opyt_home

# load_dotenv never overrides a set var, so the repo .env (dev checkout) wins and the user-local
# <OPYT_HOME>/.env (written by `opyt-keys`) fills the gaps.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(opyt_home() / ".env")
