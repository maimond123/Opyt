"""service/serving.py — an owner's served copy: received whole, patched, or not there yet.

The two flows that change what this service serves, and the check every read makes that there is
a copy to serve. In each flow the order of steps is the logic. Deleting the copy, with everything
else held for the library, is `deletion.py`.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from pipeline.sharing import delta
from service import publications, uploads
from service.access import agent_token, reader_of


async def receive(owner: str, label: str | None, chunks: AsyncIterator[bytes]) -> dict:
    """Make the bytes in `chunks` the export served under `owner`: `{owner, bytes, sha256, path}`.

    Raises `uploads.BadOwner`, `uploads.NoSpace` or `uploads.TooLarge`. On any failure, including
    a disconnect (hence `BaseException`), the previously served export is untouched."""
    rx = uploads.Receiver(owner)
    try:
        async for chunk in chunks:
            rx.write(chunk)
    except BaseException:
        rx.abort()
        raise
    result = rx.commit(label=label)
    # After the commit, so the recorded size always describes a file really being served.
    publications.record_upload(owner, result["bytes"])
    return result


async def receive_delta(owner: str, chunks: AsyncIterator[bytes]) -> dict:
    """Patch the export served under `owner` with the delta in `chunks`
    (`pipeline/sharing/delta.py`): `{owner, bytes, digest}`, `bytes` being the copy's size after.

    Raises `uploads.BadOwner`, `uploads.NoSpace`, `uploads.TooLarge` (the body), `delta.TooLarge`
    (the patched copy) or `delta.Stale` (a whole upload is needed instead). On any failure the
    served copy is as it was, and the delta file is gone either way."""
    rx = uploads.Receiver(owner, suffix=".delta")
    try:
        async for chunk in chunks:
            rx.write(chunk)
        landed = rx.close()
        # Off the event loop: a patch takes milliseconds, an import-sized one about a second.
        result = await run_in_threadpool(delta.apply, rx.dest, landed,
                                         max_bytes=uploads.MAX_EXPORT_BYTES)
    finally:
        rx.abort()
    size = rx.dest.stat().st_size
    # After the commit, so the recorded size always describes a file really being served.
    publications.record_upload(owner, size)
    return {"owner": owner, "bytes": size, "digest": result["digest"]}


def is_served(owner: str) -> bool:
    """Whether a copy of `owner`'s knowledge base is here to read.

    Every read checks it first: without it the local read entry points answer a missing `kb=` by
    listing every published knowledge base on this box."""
    return uploads.export_path(owner).exists()


def _arrived(owner: str) -> None:
    """404 when no copy of `owner`'s knowledge base is here yet."""
    if not is_served(owner):
        raise HTTPException(
            status_code=404,
            detail="this knowledge base has not arrived on the service yet. Its owner shared it, "
                   "and the copy usually lands within a minute or two of them doing so — nothing "
                   "is wrong with your install. Try again shortly.")


def served(owner: str, auth: dict = Depends(reader_of)) -> dict:
    """A reader or agent token and an export to serve. 404 when there is nothing here yet."""
    _arrived(owner)
    return auth


def acting(auth: dict = Depends(agent_token)) -> dict:
    """An agent token and its library's export to act on. 404 when there is nothing here yet."""
    _arrived(auth["owner"])
    return auth
