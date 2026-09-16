"""The product deliberately has no dollar-accounting surface."""

import json

from pipeline import llm_client


class _Breaker:
    def call(self, fn, *, ignore=()):
        return fn()


def test_llm_call_keeps_token_metadata_without_requesting_or_returning_cost(monkeypatch):
    """OpenRouter cost is neither requested nor represented by the client."""
    seen: dict = {}

    def fake_http_json(request, timeout=180.0):
        seen.update(json.loads(request.data))
        return ({"choices": [{"message": {"content": "ok"}}],
                 "usage": {"prompt_tokens": 3, "completion_tokens": 2}}, 0.1)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm_client, "_http_json", fake_http_json)
    monkeypatch.setattr(llm_client, "_openrouter_breaker", lambda: _Breaker())

    text, in_tokens, out_tokens, _elapsed, _raw = llm_client._call_openrouter_sync(
        "test/model", "system", "user", 10
    )

    assert (text, in_tokens, out_tokens) == ("ok", 3, 2)
    assert "usage" not in seen
    assert "cost_usd" not in llm_client.LLMResponse.__dataclass_fields__
