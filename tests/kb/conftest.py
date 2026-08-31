"""Shared fixtures for the atom-KB tests.

`kb_home` sandboxes the WHOLE store (DB + kb_raw snapshots) under a tmp dir via
`$OPYT_HOME`, so a test never touches the real `~/.opyt`. `fake_embedder` is a
deterministic bag-of-words embedder — it lets us prove the retrieval MECHANICS
(tag filter, BM25 arm, max-pool semantic arm, fusion, trust re-rank) offline, with no
paid API. The hosted embedder's own behavior (query prefix, cost, kb_meta guard) is
proven separately in test_embed.py against the live API.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest


@pytest.fixture()
def kb_home(tmp_path, monkeypatch):
    """Point $OPYT_HOME at a tmp dir so schema.connect()/opyt_db()/kb_raw all sandbox."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _deterministic_content_gate(request, monkeypatch):
    """Stub `content_gate.classify_page` to its keep-all fail-safe for every atom-KB test.

    WHY AUTOUSE (2026-08-02). The gate is an LLM call, and no long-form ingester harness stubbed
    it — so `test_ingest_blog`, `test_ingest_substack`, `test_ingest_curation` and
    `test_footprint_seam` were all reaching the REAL OpenRouter endpoint on every `pytest` run.
    They passed either way, which is what let it persist: a live verdict might drop units, or the
    call might fail into `_keep_all(degraded=True)`, and nothing asserted the difference. A test
    whose result depends on a live third party is not a test — it is a coin flip that also spends
    money. (Surfaced by the live-network guard in tests/conftest.py, added when the image migration
    made a lower seam stop intercepting.)

    Keep-all is the RIGHT default here because these tests are about ingest mechanics — atom
    identity, idempotency, embed batching, edges — not about which paragraphs survive grading. The
    gate's own behavior is proven in test_content_gate.py / test_content_gate_concurrency.py
    against a faked `llm_client`, and those modules opt out with `pytestmark = pytest.mark.real_gate`.
    """
    if request.node.get_closest_marker("real_gate"):
        return
    from pipeline.kb import content_gate as cg

    def _keep_all(md, **kw):
        units = cg._split_units(md)
        return cg.PageVerdict(units=units, keep=[True] * len(units), kept_text=md,
                              frontmatter="", degraded=True, n_calls=0)

    monkeypatch.setattr(cg, "classify_page", _keep_all)


@pytest.fixture(autouse=True)
def _deterministic_url_triage(request, monkeypatch):
    """Stub `link_discovery._triage_gray` to its approve-all fallback.

    The OTHER live seam on the long-form path — blog footprint reaches it via
    `discover_candidate_urls`. It already degrades to approve-all on ANY failure, and that is
    exactly how it stayed hidden: an unstubbed live call that errored produced the same visible
    result as a healthy degrade, so nothing ever looked wrong. `test_link_discovery` drives the
    REAL triage against a faked `llm_client.call` and opts out with `pytest.mark.real_triage`.

    Separate from the content-gate stub on purpose: one marker covering two unrelated seams means
    opting out of one silently opts you out of the other."""
    if request.node.get_closest_marker("real_triage"):
        return
    from pipeline.kb import link_discovery as ld
    monkeypatch.setattr(ld, "_triage_gray",
                        lambda candidates, *, author_name=None, **kw: list(candidates))


@pytest.fixture(autouse=True)
def ocr(monkeypatch):
    """Fake `ocr_cascade.read_image` — THE image seam since 2026-08-02.

    AUTOUSE, so no atom-KB test can reach a real image model by omission. Requesting `ocr` in a
    signature yields this same instance, so a test that wants to assert on the reads just names it;
    a test that merely happens to ingest a post containing an image gets a deterministic fake for
    free instead of a live call. That asymmetry is deliberate: the failure being prevented is a
    test that never mentions images at all silently paying for them.

    Before that date the X tweet-media path and the long-form markdown path both called
    `describe_images.describe_image`, and every test faked THAT. Both now run the OCR cascade, so
    patching the old name intercepts nothing and the call falls through to real HTTP (which is why
    `tests/conftest.py` blocks live `llm_client.call` — a fall-through must fail by name, not hang).

    `ocr.calls` is every URL read, in dispatch order. `ocr.respond(fn)` swaps the behavior, where
    `fn(url, context) -> MediaRead | None` and None means the read FAILED (the poison-value rule:
    caller must not cache it). The default returns document-substance text echoing the URL."""
    from pipeline import ocr_cascade

    class _Ocr:
        def __init__(self):
            self.calls: list[str] = []
            self._fn = lambda url, context: ocr_cascade.MediaRead(f"desc({url})", "document", True)

        def respond(self, fn):
            self._fn = fn
            return self

        def _read(self, url, *, context=""):
            self.calls.append(url)
            return self._fn(url, context)

    spy = _Ocr()
    monkeypatch.setattr(ocr_cascade, "read_image", spy._read)
    return spy


class FakeEmbedder:
    """Deterministic bag-of-words embedder over a fixed vocabulary. A text maps to a
    binary presence vector, L2-normalized, so cosine == fraction of shared vocab words.
    Ignores `role` (like the local fallback) — the query-prefix ritual is tested elsewhere."""

    provider = "local"
    model = "fake-bow"
    query_instruction = ""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.dim = len(vocab)

    def embed(self, texts, *, role: str = "document"):
        out = []
        for t in texts:
            low = (t or "").lower()
            v = np.array([1.0 if w in low else 0.0 for w in self.vocab], dtype=np.float32)
            n = float(np.linalg.norm(v))
            out.append(v / n if n else v)
        return out


@pytest.fixture()
def fake_embedder():
    # Vocabulary spanning the test corpus' distinguishing words.
    return FakeEmbedder([
        "agent", "framework", "autonomous", "tools", "library",
        "crypto", "rollup", "proof", "react", "dashboard", "web",
    ])


class RecordingEmbedder:
    """Deterministic text→vector embedder that RECORDS each `embed()` call it receives — one entry
    in `.calls` per invocation = the batch of texts it saw. An `AtomSink` calls `embed()` ONCE per
    FLUSH (with that flush's whole flat chunk list), so `len(.calls)` counts FLUSHES: a batched
    ingester feeding N atoms into one flush records ONE call, where the old per-atom `store_atom`
    recorded N. That's the ARC-1 Job A proof. `poison`: any text containing the marker fails the
    whole batch (all-or-nothing, like the hosted embedder) to exercise the sink's per-atom isolation."""

    provider = "local"
    model = "fake-rec"
    query_instruction = ""

    def __init__(self, dim: int = 6, batch_size: int = 64, poison: str | None = None):
        self.dim = dim
        self.batch_size = batch_size
        self.poison = poison
        self.calls: list[list[str]] = []

    def embed(self, texts, *, role: str = "document"):
        from pipeline.kb.embed import EmbedError
        self.calls.append(list(texts))
        if self.poison and any(self.poison in t for t in texts):
            raise EmbedError("poison chunk in batch", retryable=False)
        out = []
        for t in texts:
            h = hashlib.sha256((t or "").encode()).digest()
            v = np.frombuffer(h[: self.dim], dtype=np.uint8).astype(np.float32)
            n = float(np.linalg.norm(v))
            out.append(v / n if n else v)
        return out


@pytest.fixture()
def recording_embedder():
    """A call-recording embedder for the batching tests (`len(emb.calls)` == number of flushes)."""
    return RecordingEmbedder()


def last_run(conn, *, status: str | None = None, generator: str | None = None):
    """Most recent `frontier_reader_runs` row, optionally scoped to a status and a generator.

    Lived in `frontier_queries` until 2026-08-28, when it was deleted for having no production
    caller — its last one went with the bookmark reader on 2026-08-16, and `sitting_scheduler`
    reads the table inline. The assertions in these tests are its only readers, so it lives with
    them. `generator IS NULL` counts as a match: rows predating the column were all bookmark reads.
    """
    where, args = [], []
    if status:
        where.append("status=?")
        args.append(status)
    if generator:
        where.append("(generator=? OR generator IS NULL)")
        args.append(generator)
    sql = "SELECT * FROM frontier_reader_runs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY ran_at DESC, run_id DESC LIMIT 1"
    return conn.execute(sql, tuple(args)).fetchone()
