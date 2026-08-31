"""ARC-1 Phase 3 — concurrent embed dispatch.

`embed()` fans its 64-chunk slices across a bounded pool under an AIMD gate instead of a serial
loop. Three things must hold, and this file pins each:

  • ALIGNMENT — slices finish out of order, but the reassembled vector list stays in INPUT order.
    This is the one silent-corruption risk (a misalignment stores the wrong vector for a chunk with
    no crash), so the test forces earlier slices to complete LAST and asserts order survives.
  • THE GATE BINDS — no more than `_EMBED_CONCURRENCY` calls are ever in the provider call at once,
    and the fan-out really is concurrent (not accidentally serialized).
  • BACKOFF — a 429 that survives the transport retry halves the gate (multiplicative-decrease).
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest


def test_embed_preserves_order_under_out_of_order_completion(monkeypatch):
    """Adversarial timing: slice 0 sleeps the LONGEST, so a "consume as completed" bug would place
    it last. Collecting futures in submission order must still yield input order."""
    from pipeline.kb import embed as embed_mod

    cfg = dict(embed_mod._DEFAULTS)
    cfg["batch_size"] = 4
    emb = embed_mod.HostedEmbedder(cfg, use_breaker=False)
    emb._dim = 1

    def fake_batch(self, batch):
        first = int(batch[0])
        time.sleep(0.02 * (1.0 + (100 - first) / 100.0))   # earlier slices finish LATER
        return [np.array([float(int(t))], dtype=np.float32) for t in batch]

    monkeypatch.setattr(embed_mod.HostedEmbedder, "_embed_batch", fake_batch)

    texts = [str(i) for i in range(20)]                    # 5 slices of 4 → concurrent path
    out = emb.embed(texts)
    assert [float(v[0]) for v in out] == [float(i) for i in range(20)]


def test_embed_respects_gate_and_stays_aligned(monkeypatch):
    """Through the REAL _embed_batch (so the real gate wrap runs) with a mocked HTTP layer: never
    more than the ceiling in flight at once, and every vector lands at its input index."""
    from pipeline.kb import embed as embed_mod

    gate = embed_mod.AdaptiveSemaphore(4, min_permits=2, max_permits=8, increase_after=4)
    monkeypatch.setattr(embed_mod, "_EMBED_GATE", gate)     # fresh gate → isolated from the singleton
    monkeypatch.setattr("pipeline.credentials.get_credential", lambda name: "test-key")

    N = 40
    tracker = {"cur": 0, "max": 0, "lock": threading.Lock()}

    def fake_post(req, timeout):
        inp = json.loads(req.data)["input"]
        with tracker["lock"]:
            tracker["cur"] += 1
            tracker["max"] = max(tracker["max"], tracker["cur"])
        time.sleep(0.01)
        with tracker["lock"]:
            tracker["cur"] -= 1
        # one-hot on the GLOBAL index (a unit vector → survives L2-normalization) so alignment is
        # checkable: out[k] must be the one-hot at k.
        data = []
        for i, t in enumerate(inp):
            vec = [0.0] * N
            vec[int(t)] = 1.0
            data.append({"index": i, "embedding": vec})
        return {"data": data, "usage": {"total_tokens": 1}}

    monkeypatch.setattr(embed_mod, "_http_post_json", fake_post)

    cfg = dict(embed_mod._DEFAULTS)
    cfg["batch_size"] = 4
    emb = embed_mod.HostedEmbedder(cfg, use_breaker=False)
    out = emb.embed([str(i) for i in range(N)])            # 10 slices of 4

    assert len(out) == N
    for k, v in enumerate(out):
        assert int(np.argmax(v)) == k                      # aligned to input index
    assert tracker["max"] <= embed_mod._EMBED_CONCURRENCY  # gate never over-admitted
    assert tracker["max"] >= 2                             # ... and it was genuinely concurrent


def test_embed_gate_decreases_on_429(monkeypatch):
    """A 429 that survives the (single) transport attempt halves the gate and propagates."""
    from pipeline.kb import embed as embed_mod

    gate = embed_mod.AdaptiveSemaphore(8, min_permits=2, max_permits=8, increase_after=4)
    monkeypatch.setattr(embed_mod, "_EMBED_GATE", gate)
    monkeypatch.setattr("pipeline.credentials.get_credential", lambda name: "test-key")

    def fake_post(req, timeout):
        raise embed_mod.EmbedError("HTTP 429: slow down", retryable=True, status=429)

    monkeypatch.setattr(embed_mod, "_http_post_json", fake_post)

    cfg = dict(embed_mod._DEFAULTS)
    cfg["batch_size"] = 4
    cfg["max_attempts"] = 1                                # don't retry — surface the 429 immediately
    emb = embed_mod.HostedEmbedder(cfg, use_breaker=False)

    with pytest.raises(embed_mod.EmbedError):
        emb.embed(["a"])                                   # single slice → serial path, still gated
    assert gate.limit == 4                                 # halved 8 → 4
