"""
opyt_core — the shared Python core for OPYT's front-ends.

The MCP server imports from here so the connect → load-embedder → query sequence
lives in ONE place instead of being re-implemented per surface. Wraps the
pipeline.kb atom-KB engine — it does not re-derive retrieval. The MCP server is
the only front-end left (the vault RAG rail and the FastAPI dashboard it used to
also serve are both deleted).

Loads the repo .env on import so provider keys resolve regardless of the caller's cwd
— the MCP server is spawned by its client with an arbitrary cwd.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from .paths import opyt_home

# Dev checkout: repo .env (loaded first → wins for the author). Distributed install: a
# user-local <OPYT_HOME>/.env (written by `opyt-keys`) supplies keys with no repo present.
# load_dotenv does not override already-set vars, so the repo .env (if any) takes precedence
# and the user .env fills the gaps. opyt_home() honors $OPYT_HOME (the one sandbox knob).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(opyt_home() / ".env")
