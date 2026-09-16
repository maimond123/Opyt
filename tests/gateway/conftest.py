"""
Async support for the gateway tests.

anyio's pytest plugin is already installed (fastmcp depends on anyio), so `@pytest.mark.anyio`
needs nothing but a backend fixture. Pinned to asyncio because the gateway runs under uvicorn,
which is asyncio, and `ChildPool` uses `asyncio.subprocess` directly — running these against
trio would test a stack that never ships.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
