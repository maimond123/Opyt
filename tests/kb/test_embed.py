"""Hermetic proof of the hosted-embedding seam (pipeline/kb/embed.py).

No network, no spend: the HTTP layer (`_http_post_json`) and the cost recorder are
monkeypatched. Proves the contracts the atom-KB relies on — normalization, input↔vector
alignment, the asymmetric query prefix, all-or-nothing failure, cost attribution, and the
kb_meta model-identity guard.
"""
import json
import sqlite3

import numpy as np
import pytest

from pipeline.kb.embed import (
    HostedEmbedder,
    EmbedError,
    SubspaceError,
    ensure_kb_meta,
    assert_model,
    _resolve_config,
    _QWEN_QUERY_INSTRUCTION,
)

QWEN = "qwen/qwen3-embedding-8b"


def _cfg(model=QWEN, batch_size=64, dim=None, price=0.01, query_instruction=None):
    """A resolved config dict (mirrors what _resolve_config produces), so tests can
    construct a HostedEmbedder without a settings.yaml."""
    qi = query_instruction
    if qi is None:
        qi = _QWEN_QUERY_INSTRUCTION if "qwen" in model else ""
    return {
        "provider": "openrouter", "model": model, "endpoint": "http://test/embeddings",
        "dim": dim, "batch_size": batch_size, "price_per_million": price,
        "query_instruction": qi, "timeout": 5.0,
    }


def _fake_http(dim=8, capture=None, usage_tokens=None):
    """A fake `_http_post_json`: reads the request body, returns one nonzero vector per
    input in OpenAI shape. `capture` (a list) collects each request's `input` array."""
    def fake(req, timeout):
        body = json.loads(req.data.decode())
        inputs = body["input"]
        if capture is not None:
            capture.append(inputs)
        data = [
            {"index": i, "embedding": [float((i + 1) * (j + 1)) for j in range(dim)]}
            for i in range(len(inputs))
        ]
        tok = usage_tokens if usage_tokens is not None else 10 * len(inputs)
        return {"data": data, "usage": {"total_tokens": tok}, "model": body["model"]}
    return fake


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    # get_credential("openrouter") checks the env first, so a dummy key satisfies it.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


# ── vectors: normalized, aligned, dim discovered ────────────────────────────────

def test_documents_are_normalized_and_aligned(monkeypatch):
    monkeypatch.setattr("pipeline.kb.embed._http_post_json", _fake_http(dim=8))
    emb = HostedEmbedder(_cfg(), use_breaker=False)
    vecs = emb.embed(["a", "b", "c"], role="document")
    assert len(vecs) == 3
    assert emb.dim == 8
    for v in vecs:
        assert v.dtype == np.float32
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_empty_input_returns_empty(monkeypatch):
    monkeypatch.setattr("pipeline.kb.embed._http_post_json", _fake_http())
    assert HostedEmbedder(_cfg(), use_breaker=False).embed([]) == []


# ── the Qwen wrinkle: query prefix, asymmetric ──────────────────────────────────

def test_query_prefix_applied_only_to_queries(monkeypatch):
    cap = []
    monkeypatch.setattr("pipeline.kb.embed._http_post_json", _fake_http(capture=cap))
    emb = HostedEmbedder(_cfg(), use_breaker=False)

    emb.embed(["what is attention?"], role="query")
    emb.embed(["Attention weights inputs."], role="document")

    sent_query, sent_doc = cap[0][0], cap[1][0]
    assert sent_query.startswith(_QWEN_QUERY_INSTRUCTION)
    assert sent_query.endswith("what is attention?")
    assert not sent_doc.startswith(_QWEN_QUERY_INSTRUCTION)  # documents go in raw


def test_openai_model_gets_no_prefix(monkeypatch):
    cap = []
    monkeypatch.setattr("pipeline.kb.embed._http_post_json", _fake_http(capture=cap))
    emb = HostedEmbedder(_cfg(model="openai/text-embedding-3-large"), use_breaker=False)
    emb.embed(["query text"], role="query")
    assert cap[0][0] == "query text"  # prefix-free model: role is a no-op


def test_bad_role_rejected(monkeypatch):
    monkeypatch.setattr("pipeline.kb.embed._http_post_json", _fake_http())
    with pytest.raises(ValueError):
        HostedEmbedder(_cfg(), use_breaker=False).embed(["x"], role="passage")


# ── batching: many inputs, still complete + aligned ─────────────────────────────

def test_batching_makes_multiple_requests_all_complete(monkeypatch):
    cap = []
    monkeypatch.setattr("pipeline.kb.embed._http_post_json", _fake_http(capture=cap))
    emb = HostedEmbedder(_cfg(batch_size=2), use_breaker=False)
    vecs = emb.embed([f"t{i}" for i in range(5)], role="document")
    assert len(vecs) == 5
    # `embed` fans slices across a thread pool, so `cap` records COMPLETION order, not submission
    # order — asserting [2, 2, 1] pinned the scheduler, not the contract. The real invariant is the
    # one the old comment already named: three requests that between them carry every input exactly
    # once. (Ordering of the returned VECTORS is a separate guarantee, pinned in
    # test_embed_concurrent.py::test_embed_preserves_order_under_out_of_order_completion.)
    assert sorted(len(c) for c in cap) == [1, 2, 2]     # three requests, sizes 2/2/1
    assert sorted(t for c in cap for t in c) == [f"t{i}" for i in range(5)]  # no drops, no dupes


# ── fail-safe: all-or-nothing, no cost on failure ───────────────────────────────

def test_http_failure_raises_and_records_no_cost(monkeypatch):
    def boom(req, timeout):
        raise EmbedError("HTTP 500: upstream down")
    monkeypatch.setattr("pipeline.kb.embed._http_post_json", boom)
    spy = []
    monkeypatch.setattr("pipeline.llm_client.record_external_cost",
                        lambda *a, **k: spy.append((a, k)))
    emb = HostedEmbedder(_cfg(), use_breaker=False)
    with pytest.raises(EmbedError):
        emb.embed(["a", "b"], role="document")
    assert spy == []  # a failed call must not bill (Fail-safe: no partial state)


def test_partial_response_raises(monkeypatch):
    def short(req, timeout):
        body = json.loads(req.data.decode())
        n = len(body["input"])
        data = [{"index": i, "embedding": [1.0, 2.0]} for i in range(n - 1)]  # one short
        return {"data": data, "usage": {"total_tokens": 5}}
    monkeypatch.setattr("pipeline.kb.embed._http_post_json", short)
    with pytest.raises(EmbedError):
        HostedEmbedder(_cfg(), use_breaker=False).embed(["a", "b", "c"])


def test_dim_drift_raises(monkeypatch):
    calls = {"n": 0}
    def drift(req, timeout):
        calls["n"] += 1
        d = 8 if calls["n"] == 1 else 16  # server silently switched -> must be caught
        body = json.loads(req.data.decode())
        data = [{"index": i, "embedding": [1.0] * d} for i in range(len(body["input"]))]
        return {"data": data, "usage": {"total_tokens": 5}}
    monkeypatch.setattr("pipeline.kb.embed._http_post_json", drift)
    emb = HostedEmbedder(_cfg(batch_size=1), use_breaker=False)
    with pytest.raises(EmbedError):
        emb.embed(["a", "b"], role="document")


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("pipeline.credentials.get_credential", lambda service: None)
    monkeypatch.setattr("pipeline.kb.embed._http_post_json", _fake_http())
    with pytest.raises(EmbedError):
        HostedEmbedder(_cfg(), use_breaker=False).embed(["a"])


# ── cost attribution ────────────────────────────────────────────────────────────

def test_cost_recorded_on_success(monkeypatch):
    monkeypatch.setattr("pipeline.kb.embed._http_post_json",
                        _fake_http(usage_tokens=1_000_000))
    spy = []
    monkeypatch.setattr("pipeline.llm_client.record_external_cost",
                        lambda provider, cost, **k: spy.append((provider, cost, k)))
    emb = HostedEmbedder(_cfg(price=0.01), use_breaker=False)
    emb.embed(["a"], role="document")
    assert len(spy) == 1
    provider, cost, kw = spy[0]
    assert provider == "openrouter-embed"
    assert abs(cost - 0.01) < 1e-9        # 1M tokens * $0.01/M
    assert kw.get("requests") == 1


# ── kb_meta: the model-identity guard ───────────────────────────────────────────

class _FakeEmb:
    def __init__(self, model, provider, dim, query_instruction=""):
        self.model, self.provider, self.dim = model, provider, dim
        self.query_instruction = query_instruction


def test_ensure_kb_meta_writes_then_enforces():
    conn = sqlite3.connect(":memory:")
    ensure_kb_meta(conn, QWEN, 4096, "openrouter", "Instruct: q\nQuery: ")
    # Same identity again -> no-op, returns the stored row.
    again = ensure_kb_meta(conn, QWEN, 4096, "openrouter", "Instruct: q\nQuery: ")
    assert again["dim"] == 4096 and again["query_instruction"] == "Instruct: q\nQuery: "


def test_ensure_kb_meta_mismatch_raises():
    conn = sqlite3.connect(":memory:")
    ensure_kb_meta(conn, QWEN, 4096, "openrouter")
    with pytest.raises(SubspaceError):
        ensure_kb_meta(conn, "openai/text-embedding-3-large", 3072, "openrouter")


def test_ensure_kb_meta_query_instruction_drift_raises():
    conn = sqlite3.connect(":memory:")
    ensure_kb_meta(conn, QWEN, 4096, "openrouter", "Instruct: A\nQuery: ")
    # Same model/dim/provider, but the query instruction changed -> guarded, must raise.
    with pytest.raises(SubspaceError):
        ensure_kb_meta(conn, QWEN, 4096, "openrouter", "Instruct: B\nQuery: ")


def test_assert_model_fresh_store_is_noop():
    conn = sqlite3.connect(":memory:")
    assert_model(conn, _FakeEmb(QWEN, "openrouter", 4096))  # no kb_meta yet -> no raise


def test_assert_model_catches_model_mismatch_even_without_dim():
    conn = sqlite3.connect(":memory:")
    ensure_kb_meta(conn, QWEN, 4096, "openrouter")
    # dim=None (embedder hasn't embedded yet) but the MODEL differs -> still raises.
    with pytest.raises(SubspaceError):
        assert_model(conn, _FakeEmb("openai/text-embedding-3-large", "openrouter", None))


def test_assert_model_catches_dim_mismatch_when_known():
    conn = sqlite3.connect(":memory:")
    ensure_kb_meta(conn, QWEN, 4096, "openrouter")
    with pytest.raises(SubspaceError):
        assert_model(conn, _FakeEmb(QWEN, "openrouter", 2560))


def test_assert_model_catches_query_instruction_drift():
    conn = sqlite3.connect(":memory:")
    ensure_kb_meta(conn, QWEN, 4096, "openrouter", "Instruct: A\nQuery: ")
    with pytest.raises(SubspaceError):
        assert_model(conn, _FakeEmb(QWEN, "openrouter", 4096, "Instruct: B\nQuery: "))


# ── config derivation ───────────────────────────────────────────────────────────

def test_config_derives_qwen_prefix_by_default(monkeypatch):
    monkeypatch.setattr("pipeline.ingestion.utils.load_yaml_config", lambda: {})
    cfg = _resolve_config()
    assert cfg["model"] == QWEN
    assert cfg["query_instruction"] == _QWEN_QUERY_INSTRUCTION


def test_config_no_prefix_for_non_qwen_override(monkeypatch):
    monkeypatch.setattr(
        "pipeline.ingestion.utils.load_yaml_config",
        lambda: {"embeddings": {"model": "openai/text-embedding-3-large"}},
    )
    cfg = _resolve_config()
    assert cfg["model"] == "openai/text-embedding-3-large"
    assert cfg["query_instruction"] == ""  # prefix travels with the model, not a stale line
