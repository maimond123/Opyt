"""The query-id fast path follows the order X put bundles in the page."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_CHILD = textwrap.dedent("""
    import importlib
    import sys
    import types

    module_name = sys.argv[1]
    scenario = sys.argv[2]
    if scenario == "unnamed_chunk":
        # The operation lives in a chunk whose NAME has nothing to do with it — as
        # `ListsManagementPageTimeline` does in `shared~JetfuelWrapper~bundle.AudioSpaceDetail~…`.
        urls = ["https://abs.twimg.com/responsive-web/client-web/main.abcdef.js"]
        page = (
            'p.u=e=>(({10:"bundle.Unrelated",20:"shared~JetfuelWrapper~bundle.AudioSpaceDetail"}'
            ')[e]||e)+"."+({10:"abc123",20:"def456"})[e]+"a.js"'
            + "".join(f'<script src="{url}"></script>' for url in urls)
        )
    elif scenario in ("runtime", "runtime_2026_09_30"):
        urls = [
            "https://abs.twimg.com/responsive-web/client-web/vendor.abcdef.js",
            "https://abs.twimg.com/responsive-web/client-web/main.abcdef.js",
        ]
        # X has shipped the chunk map both ways: with a leading `""+` (the original sample) and,
        # later, without it. The second form is what broke discovery in production.
        prefix = '""+' if scenario == "runtime" else ""
        page = (
            'p.u=e=>' + prefix + '(({10:"bundle.Bookmarks",20:"shared~bundle.Bookmarks"}'
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
        if scenario == "unnamed_chunk":
            if url.endswith("AudioSpaceDetail.def456a.js"):
                return Response('queryId:"qid-unnamed",operationName:"Bookmarks"')
            if url.endswith(".js"):
                return Response("")
        if scenario.startswith("runtime"):
            if url.endswith("bundle.Bookmarks.abc123a.js"):
                return Response("")
            if url.endswith("shared~bundle.Bookmarks.def456a.js"):
                return Response('queryId:"qid-runtime",operationName:"Bookmarks"')
        for index, entry in enumerate(urls):
            if url == entry:
                if scenario.startswith("runtime"):
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
        _run_discovery("pipeline.platforms.x.local", seed, tmp_path)
        for seed in ("0", "1", "2")
    ]
    assert results == ["qid-0", "qid-0", "qid-0"]


def test_discovery_uses_bookmarks_chunks_from_the_inline_runtime(tmp_path):
    result = _run_discovery("pipeline.platforms.x.local", "0", tmp_path, "runtime")
    assert result == "qid-runtime"


def test_discovery_reads_the_chunk_map_x_ships_today(tmp_path):
    """X dropped the leading `""+` from the chunk map
    (`.u=e=>(({…` instead of `.u=e=>""+(({…`). The parser required it, matched nothing, and fell
    back to the entry bundles — so every operation in a lazy chunk, `Bookmarks` and
    `ListsManagementPageTimeline` among them, became undiscoverable while the only test, pinned to
    the old sample, stayed green."""
    result = _run_discovery("pipeline.platforms.x.local", "0", tmp_path,
                            "runtime_2026_09_30")
    assert result == "qid-runtime"


def test_discovery_finds_an_operation_in_a_chunk_not_named_for_it(tmp_path):
    """The second gap, found the same day. Discovery used to open only the entry bundles and the
    chunks whose NAME contains the operation, and `ListsManagementPageTimeline` lives in
    `shared~JetfuelWrapper~bundle.AudioSpaceDetail~…` — so even with the chunk map parsed, it was
    unreachable. The last-resort scan opens every remaining chunk."""
    result = _run_discovery("pipeline.platforms.x.local", "0", tmp_path, "unnamed_chunk")
    assert result == "qid-unnamed"
