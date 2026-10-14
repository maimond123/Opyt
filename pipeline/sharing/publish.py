"""
pipeline/sharing/publish.py — makes the served copy hold what a fresh export of this store holds.

Only the sequencing: `build_export` projects the store and `fingerprint` hashes its rows; then
the service is sent what it lacks. Nothing, when it already holds those rows; a delta of the rows
that changed (`delta.py`, `POST /v1/upload/{owner}/delta`) when it holds an earlier push of
ours; the whole export (`POST /v1/upload/{owner}`) otherwise, or when the delta would be most of
the file, or when the service answers that its copy cannot take one. Its one caller in code is
the publishing pass (`publishing.py`); the operator fallback in `service/DEPLOY.md` §8 calls it
by hand.

What the service holds is remembered here, after it accepted a push (`push_gate.write_held`);
the pass keeps the watermark.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from opyt_core.paths import opyt_path
from pipeline.sharing import client, delta, fingerprint, push_gate
from pipeline.sharing.account import Account
from pipeline.sharing.export import build_export

# What the service answers when the whole export would fix it: 409 from `upload_delta`, and 404
# from a service older than that route.
_SEND_WHOLE = {404, 409}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def worth_a_delta(delta_bytes: int, export_bytes: int) -> bool:
    """Whether a delta is worth sending instead of the whole export: only while it is under half
    of it. Above that it saves little upload and journals most of the served file on the box."""
    return delta_bytes * 2 < export_bytes


def publish(acct: Account) -> dict:
    """Build this store's export and make the served copy hold what it holds.

    Every failure of the network or the service, and a store that has never ingested, is a
    status. A failure of the local build itself (a locked store, a full disk) raises.

    `{status: "ok", owner, atoms, sent, bytes, digest}`, where `sent` is `nothing`, `delta` or
    `whole` and `bytes` is what was uploaded (a whole upload adds its `sha256`); or a status and a
    `message` saying why not. A sha mismatch on a whole upload means the damaged copy was already
    swapped in, so it is reported as served."""
    try:
        owner = client.tokens(acct)["owner"]
    except client.Failed as e:
        return {"status": e.kind, "message": str(e)}

    out = opyt_path("tmp", "export-push.db")
    cut = opyt_path("tmp", "export-push.delta")
    try:
        manifest = build_export(out)
        now = fingerprint.of_copy(out)
        fingerprint.stamp(out, now.digest)
        res = _send(acct, owner, out, cut, push_gate.read_held(), now)
    except client.Failed as e:
        return {"status": "upload_failed" if e.kind == "refused" else e.kind,
                "message": str(e)}
    except ValueError as e:
        # `build_export` on a store nobody has ingested into. It already says the useful sentence.
        return {"status": "empty_store", "message": str(e)}
    finally:
        out.unlink(missing_ok=True)
        cut.unlink(missing_ok=True)

    if res["status"] != "ok":
        return res
    push_gate.write_held(now)
    return {"status": "ok", "owner": owner, "atoms": manifest["tables"]["atoms"],
            **{k: v for k, v in res.items() if k != "status"}}


def _send(acct: Account, owner: str, out: Path, cut: Path,
          held: fingerprint.Fingerprint | None, now: fingerprint.Fingerprint) -> dict:
    """Send the service what it lacks of the export at `out`, given what it `held` after our last
    push. Raises `client.Failed` for every refusal a whole upload would not fix."""
    if held is not None and held.digest == now.digest:
        return {"status": "ok", "sent": "nothing", "bytes": 0, "digest": now.digest}
    if held is not None:
        size = delta.build(out, fingerprint.diff(held, now), held.digest, now.digest, cut)
        if worth_a_delta(size, out.stat().st_size):
            try:
                client.upload_delta(acct, owner, cut)
                return {"status": "ok", "sent": "delta", "bytes": size, "digest": now.digest}
            except client.Failed as e:
                if e.status not in _SEND_WHOLE:
                    raise
    local_sha = _sha256(out)
    served = client.upload(acct, owner, out)
    if served["sha256"] != local_sha:
        return {"status": "corrupt",
                "message": (f"the service received different bytes than were sent — sent "
                            f"{local_sha}, received {served['sha256']}. That damaged copy is "
                            f"what it serves now; run this again.")}
    return {"status": "ok", "sent": "whole", "bytes": served["bytes"], "digest": now.digest,
            "sha256": local_sha}
