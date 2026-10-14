"""Behavior pins for the GitHub REST read helpers (_paginate, _fetch_user, _fetch_repos,
_fetch_org_repos, _fetch_readme). Network is stubbed at ``requests.get``; pacing at ``time.sleep``.
These freeze today's output, bugs included, so a pure move that changes it fails here. Three
behaviors were fixed on purpose; their rows assert the refusal."""
from __future__ import annotations

import base64

import pytest

from pipeline.platforms.github import api as gh


class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture()
def calls(monkeypatch):
    """Record every GET (url, params snapshot, headers, timeout) and no-op the pacing sleep."""
    log: list[dict] = []
    monkeypatch.setattr(gh.time, "sleep", lambda s: log.append({"sleep": s}))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    return log


def _serve(monkeypatch, calls, responses):
    """Serve ``responses`` in order (an Exception instance is raised); record each request."""
    queue = list(responses)

    def fake_get(url, **kw):
        calls.append({"url": url, "params": dict(kw.get("params") or {}),
                      "headers": dict(kw.get("headers") or {}), "timeout": kw.get("timeout")})
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(gh.requests, "get", fake_get)


def _gets(calls):
    return [c for c in calls if "url" in c]


def _sleeps(calls):
    return [c["sleep"] for c in calls if "sleep" in c]


# ── _paginate ────────────────────────────────────────────────────────────────

def test_paginate_walks_pages_until_empty_batch(monkeypatch, calls):
    _serve(monkeypatch, calls, [_Resp(payload=[{"n": 1}, {"n": 2}]), _Resp(payload=[{"n": 3}]),
                                _Resp(payload=[])])
    out = gh._paginate("https://api.github.com/x", {"sort": "pushed"})
    assert out == [{"n": 1}, {"n": 2}, {"n": 3}]
    assert [c["params"] for c in _gets(calls)] == [
        {"sort": "pushed", "per_page": 100, "page": 1},
        {"sort": "pushed", "per_page": 100, "page": 2},
        {"sort": "pushed", "per_page": 100, "page": 3},
    ]
    assert all(c["url"] == "https://api.github.com/x" and c["timeout"] == 15 for c in _gets(calls))
    assert _sleeps(calls) == [gh.FETCH_DELAY, gh.FETCH_DELAY]       # one per non-empty page


def test_paginate_does_not_mutate_caller_params_and_keeps_explicit_per_page(monkeypatch, calls):
    _serve(monkeypatch, calls, [_Resp(payload=[])])
    params = {"per_page": 30}
    assert gh._paginate("https://api.github.com/x", params) == []
    assert params == {"per_page": 30}
    assert _gets(calls)[0]["params"] == {"per_page": 30, "page": 1}


def test_paginate_no_params_defaults_per_page(monkeypatch, calls):
    _serve(monkeypatch, calls, [_Resp(payload=[])])
    gh._paginate("https://api.github.com/x")
    assert _gets(calls)[0]["params"] == {"per_page": 100, "page": 1}


def test_paginate_failed_page_refuses(monkeypatch, calls):
    # B03: a mid-crawl failure (500 here) refuses instead of returning the prefix as complete.
    _serve(monkeypatch, calls, [_Resp(payload=[{"n": 1}]), _Resp(status=500)])
    with pytest.raises(gh.GitHubReadFailed, match="GitHub API 500"):
        gh._paginate("https://api.github.com/x")
    assert len(_gets(calls)) == 2


def test_paginate_transport_error_refuses(monkeypatch, calls):
    # B03: a raised transport error refuses instead of returning the prefix as complete.
    _serve(monkeypatch, calls, [_Resp(payload=[{"n": 1}]), ConnectionError("boom")])
    with pytest.raises(gh.GitHubReadFailed, match="boom"):
        gh._paginate("https://api.github.com/x")


def test_paginate_403_without_rate_limit_text_refuses(monkeypatch, calls):
    # A plain 403 must not end the walk as if the list had.
    _serve(monkeypatch, calls, [_Resp(payload=[{"n": 1}]), _Resp(status=403, text="Forbidden")])
    with pytest.raises(gh.GitHubReadFailed, match="GitHub API 403"):
        gh._paginate("https://api.github.com/x")


def test_paginate_first_page_404_is_an_empty_list(monkeypatch, calls):
    # A user handle 404s the org endpoint and vice versa: absence is an answer, not a failure.
    _serve(monkeypatch, calls, [_Resp(status=404)])
    assert gh._paginate("https://api.github.com/x") == []


def test_paginate_refuses_a_full_page_at_the_cap(monkeypatch, calls):
    # B03: hitting the page cap refuses; the list may go on past it.
    _serve(monkeypatch, calls, [_Resp(payload=[{"p": i}]) for i in range(1, 4)])
    with pytest.raises(gh.GitHubReadFailed, match="longer than 3 pages"):
        gh._paginate("https://api.github.com/x", max_pages=3)
    assert [c["params"]["page"] for c in _gets(calls)] == [1, 2, 3]


def test_paginate_default_cap_is_20_pages(monkeypatch, calls):
    _serve(monkeypatch, calls, [_Resp(payload=[{"p": i}]) for i in range(1, 25)])
    with pytest.raises(gh.GitHubReadFailed, match="longer than 20 pages"):
        gh._paginate("https://api.github.com/x")
    assert len(_gets(calls)) == 20


class _BadJson(_Resp):
    def json(self):
        raise ValueError("not json")


def test_paginate_non_json_200_refuses(monkeypatch, calls):
    # B18: a 200 with a non-JSON body is a failed read, not a ValueError escaping.
    _serve(monkeypatch, calls, [_BadJson()])
    with pytest.raises(gh.GitHubReadFailed, match="not JSON"):
        gh._paginate("https://api.github.com/x")


def test_fetch_user_non_json_200_is_none(monkeypatch, calls):
    # B18
    _serve(monkeypatch, calls, [_BadJson()])
    assert gh.fetch_user("a") is None


def test_fetch_readme_non_json_200_is_none(monkeypatch, calls):
    # B18
    _serve(monkeypatch, calls, [_BadJson()])
    assert gh.fetch_readme("o", "r") is None


def test_paginate_rate_limit_raises_out(monkeypatch, calls):
    _serve(monkeypatch, calls, [_Resp(payload=[{"n": 1}]),
                                _Resp(status=403, text="API rate limit exceeded for 1.2.3.4")])
    with pytest.raises(gh.GitHubRateLimited):
        gh._paginate("https://api.github.com/x")


# ── _fetch_user ──────────────────────────────────────────────────────────────

def test_fetch_user_returns_json_and_hits_users_endpoint(monkeypatch, calls):
    _serve(monkeypatch, calls, [_Resp(payload={"login": "karpathy", "name": "Andrej"})])
    assert gh.fetch_user("karpathy") == {"login": "karpathy", "name": "Andrej"}
    (c,) = _gets(calls)
    assert c["url"] == "https://api.github.com/users/karpathy"
    assert c["params"] == {}
    assert c["headers"] == {"Accept": "application/vnd.github.v3+json"}
    assert _sleeps(calls) == []


def test_fetch_user_sends_bearer_token_when_set(monkeypatch, calls):
    monkeypatch.setenv("GITHUB_TOKEN", "tok123")
    _serve(monkeypatch, calls, [_Resp(payload={"login": "a"})])
    gh.fetch_user("a")
    assert _gets(calls)[0]["headers"] == {"Accept": "application/vnd.github.v3+json",
                                          "Authorization": "Bearer tok123"}


def test_fetch_user_404_is_none_without_warning(monkeypatch, calls, capsys):
    _serve(monkeypatch, calls, [_Resp(status=404)])
    assert gh.fetch_user("ghost") is None
    assert "[warn]" not in capsys.readouterr().err


def test_fetch_user_other_status_is_none_with_warning(monkeypatch, calls, capsys):
    _serve(monkeypatch, calls, [_Resp(status=500)])
    assert gh.fetch_user("a") is None
    assert "[warn] GitHub API 500: https://api.github.com/users/a" in capsys.readouterr().err


def test_fetch_user_403_without_rate_limit_text_is_none(monkeypatch, calls):
    _serve(monkeypatch, calls, [_Resp(status=403, text="Forbidden")])
    assert gh.fetch_user("a") is None


def test_fetch_user_transport_error_is_none(monkeypatch, calls, capsys):
    _serve(monkeypatch, calls, [TimeoutError("slow")])
    assert gh.fetch_user("a") is None
    assert "[warn] GitHub request failed: slow" in capsys.readouterr().err


def test_fetch_user_rate_limit_raises(monkeypatch, calls):
    _serve(monkeypatch, calls, [_Resp(status=403, text="You have exceeded a secondary RATE LIMIT")])
    with pytest.raises(gh.GitHubRateLimited, match="https://api.github.com/users/a"):
        gh.fetch_user("a")


# ── _fetch_repos / _fetch_org_repos ──────────────────────────────────────────

def test_fetch_repos_paginates_user_repos_sorted_by_push(monkeypatch, calls):
    _serve(monkeypatch, calls, [_Resp(payload=[{"name": "r1"}]), _Resp(payload=[])])
    assert gh.fetch_repos("karpathy") == [{"name": "r1"}]
    assert [(c["url"], c["params"]) for c in _gets(calls)] == [
        ("https://api.github.com/users/karpathy/repos",
         {"sort": "pushed", "direction": "desc", "per_page": 100, "page": 1}),
        ("https://api.github.com/users/karpathy/repos",
         {"sort": "pushed", "direction": "desc", "per_page": 100, "page": 2}),
    ]


def test_fetch_org_repos_paginates_org_repos_sorted_by_push(monkeypatch, calls):
    _serve(monkeypatch, calls, [_Resp(payload=[{"name": "r1"}, {"name": "r2"}]), _Resp(payload=[])])
    assert gh.fetch_org_repos("anthropics") == [{"name": "r1"}, {"name": "r2"}]
    assert [(c["url"], c["params"]) for c in _gets(calls)] == [
        ("https://api.github.com/orgs/anthropics/repos",
         {"sort": "pushed", "direction": "desc", "per_page": 100, "page": 1}),
        ("https://api.github.com/orgs/anthropics/repos",
         {"sort": "pushed", "direction": "desc", "per_page": 100, "page": 2}),
    ]


def test_fetch_repos_first_page_failure_refuses(monkeypatch, calls):
    # An unreachable endpoint must not read the same as "no repos".
    _serve(monkeypatch, calls, [_Resp(status=500)])
    with pytest.raises(gh.GitHubReadFailed):
        gh.fetch_repos("a")


def test_fetch_org_repos_rate_limit_raises(monkeypatch, calls):
    _serve(monkeypatch, calls, [_Resp(status=403, text="rate limit")])
    with pytest.raises(gh.GitHubRateLimited):
        gh.fetch_org_repos("a")


# ── _fetch_readme ────────────────────────────────────────────────────────────

def _b64(s: bytes) -> str:
    return base64.b64encode(s).decode()


def test_fetch_readme_decodes_base64(monkeypatch, calls):
    _serve(monkeypatch, calls, [_Resp(payload={"content": _b64("# Hi\nbody".encode()),
                                               "encoding": "base64"})])
    assert gh.fetch_readme("o", "r") == "# Hi\nbody"
    assert _gets(calls)[0]["url"] == "https://api.github.com/repos/o/r/readme"


def test_fetch_readme_decodes_github_wrapped_base64(monkeypatch, calls):
    # GitHub wraps the base64 payload with newlines; b64decode's non-validating mode skips them.
    raw = _b64(b"line one\nline two\n" * 10)
    wrapped = "\n".join(raw[i:i + 60] for i in range(0, len(raw), 60))
    _serve(monkeypatch, calls, [_Resp(payload={"content": wrapped, "encoding": "base64"})])
    assert gh.fetch_readme("o", "r") == "line one\nline two\n" * 10


def test_fetch_readme_invalid_utf8_is_replaced(monkeypatch, calls):
    _serve(monkeypatch, calls, [_Resp(payload={"content": _b64(b"ok \xff end"), "encoding": "base64"})])
    assert gh.fetch_readme("o", "r") == "ok � end"


@pytest.mark.parametrize("payload", [
    {"content": "aGk=", "encoding": "utf-8"},        # non-base64 encoding
    {"content": "", "encoding": "base64"},           # empty content
    {"encoding": "base64"},                          # no content key
    {"content": "aGk="},                             # no encoding key
    {"content": "a", "encoding": "base64"},          # undecodable base64 (bad padding)
])
def test_fetch_readme_unusable_payload_is_none(monkeypatch, calls, payload):
    _serve(monkeypatch, calls, [_Resp(payload=payload)])
    assert gh.fetch_readme("o", "r") is None


def test_fetch_readme_404_is_none(monkeypatch, calls):
    _serve(monkeypatch, calls, [_Resp(status=404)])
    assert gh.fetch_readme("o", "r") is None


def test_fetch_readme_rate_limit_raises(monkeypatch, calls):
    _serve(monkeypatch, calls, [_Resp(status=403, text="rate limit exceeded")])
    with pytest.raises(gh.GitHubRateLimited):
        gh.fetch_readme("o", "r")


# ── github_page_progress: one row per case ──────────────────────────────────────

def test_github_page_progress_ends_on_no_response():
    assert gh.github_page_progress(None, 1, 20) is gh.PageStep.END


def test_github_page_progress_ends_on_an_empty_batch():
    assert gh.github_page_progress([], 2, 20) is gh.PageStep.END


def test_github_page_progress_refuses_a_batch_at_the_cap():
    assert gh.github_page_progress([{"n": 1}], 20, 20) is gh.PageStep.CAPPED


def test_github_page_progress_continues_after_any_batch_below_the_cap():
    assert gh.github_page_progress([{"n": 1}], 3, 20) is gh.PageStep.NEXT
