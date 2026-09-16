"""The query-id fast path follows the order X put bundles in the page."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

REPO = Path(__file__).resolve().parents[2]

_CHILD = textwrap.dedent("""
    import importlib
    import sys
    import types

    module_name = sys.argv[1]
    scenario = sys.argv[2]
    if scenario == "runtime":
        urls = [
            "https://abs.twimg.com/responsive-web/client-web/vendor.abcdef.js",
            "https://abs.twimg.com/responsive-web/client-web/main.abcdef.js",
        ]
        page = (
            'p.u=e=>""+(({10:"bundle.Bookmarks",20:"shared~bundle.Bookmarks"}'
            ')[e]||e)+"."+({10:"abc123",20:"def456"})[e]+"a.js"'
            + "".join(f'<script src="{url}"></script>' for url in urls)
        )
    else:
        urls = [
            f"https://abs.twimg.com/responsive-web/client-web/entry-{i}.abcdef.js"
            for i in range(7)
        ]
        page = "".join(f'<script src="{url}"></script>' for url in urls)

    class Response:
        def __init__(self, text):
            self.text = text

    def get(url, **kwargs):
        if url == "https://x.com/i/bookmarks":
            return Response(page)
        if scenario == "runtime":
            if url.endswith("bundle.Bookmarks.abc123a.js"):
                return Response("")
            if url.endswith("shared~bundle.Bookmarks.def456a.js"):
                return Response('queryId:"qid-runtime",operationName:"Bookmarks"')
        for index, entry in enumerate(urls):
            if url == entry:
                if scenario == "runtime":
                    return Response("")
                return Response(
                    f'queryId:"qid-{index}",operationName:"Bookmarks"')
        raise AssertionError(f"unexpected URL: {url}")

    cffi = types.ModuleType("curl_cffi")
    requests = types.ModuleType("curl_cffi.requests")
    requests.get = get
    cffi.requests = requests
    sys.modules["curl_cffi"] = cffi
    sys.modules["curl_cffi.requests"] = requests

    module = importlib.import_module(module_name)
    if module_name == "pipeline.ingestion.x_graphql":
        result = module.discover_query_id({})
    else:
        result = module.discover_query_id("Bookmarks", "https://x.com/i/bookmarks", {})
    print(result)
""")


def _run_discovery(module_name: str, seed: str, tmp_path: Path, scenario: str = "ordered") -> str:
    home = tmp_path / f"opyt-home-{scenario}-{seed}"
    home.mkdir()
    env = {
        **os.environ,
        "OPYT_HOME": str(home),
        "PYTHONHASHSEED": seed,
        "PYTHONPATH": str(REPO),
    }
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, module_name, scenario],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_discovery_fast_path_keeps_page_entry_order_across_hash_seeds(tmp_path):
    results = [
        _run_discovery("pipeline.ingestion.x_graphql_core", seed, tmp_path)
        for seed in ("0", "1", "2")
    ]
    assert results == ["qid-0", "qid-0", "qid-0"]


def test_discovery_uses_bookmarks_chunks_from_the_inline_runtime(tmp_path):
    result = _run_discovery("pipeline.ingestion.x_graphql_core", "0", tmp_path, "runtime")
    assert result == "qid-runtime"
