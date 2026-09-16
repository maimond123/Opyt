"""The identity-token rule that replaced two host rules on the blog path.

Both admission gates asked "is this the person's own thing?" and answered with a HOST rule, so an
author's work on any other host was invisible. Measured on karpathy.ai, 2026-09-09: the host rule
kept 3 of his 55 own links.
"""
from __future__ import annotations

import pytest

from pipeline.ingestion.identity_tokens import (host_carries_token, tokens_for, url_carries_token)


_TOKENS = tokens_for("https://karpathy.ai/", "Andrej Karpathy")


def test_tokens_come_from_the_first_host_label_and_the_name():
    assert _TOKENS == {"karpathy", "andrej"}


def test_the_first_host_label_wins_not_the_one_before_the_tld():
    """`_openalex_root` takes the label before the TLD because it asks who PUBLISHES a venue. This
    asks who a site BELONGS to. Taking the other label would read `karpathy.github.io` as "github"
    and hand every GitHub URL on the page a token match."""
    assert "karpathy" in tokens_for("https://karpathy.github.io", None)
    assert "github" not in tokens_for("https://karpathy.github.io", None)


def test_the_name_carries_an_institutional_root_the_host_cannot():
    """A live Oracle is rooted at `blog:meche.mit.edu`, whose host says "meche" and whose name says
    "Buehler". Short labels drop out: `mit` is 3 characters, and so is the `J.` initial."""
    tokens = tokens_for("https://meche.mit.edu/people/faculty/", "Markus J. Buehler")

    assert tokens == {"meche", "markus", "buehler"}


def test_a_role_label_never_stands_for_a_person():
    """`blog.example.com` would otherwise yield the token "blog" and then match `/blog/` on every
    other site linked from the page."""
    assert tokens_for("https://blog.example.com", None) == frozenset()


# ── what the two gates ask ───────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "https://karpathy.github.io/2015/05/21/rnn-effectiveness/",
    "https://karpathy.medium.com",
    "https://karpathy.bearblog.dev/blog/",
])
def test_another_home_is_recognised_by_its_host(url):
    """The gate-1 question. A host match may promote a link to its own SOURCE, so its feed yields
    the posts rather than the hub crawl guessing at them."""
    assert host_carries_token(url, _TOKENS)


@pytest.mark.parametrize("url", [
    "https://cs.stanford.edu/people/karpathy/advice.html",
    "https://cs.stanford.edu/~karpathy/discovery/",
    "https://medium.com/@karpathy",
])
def test_work_on_someone_elses_host_is_recognised_by_its_path(url):
    """The gate-2 question, and the reason a path match must NOT promote to a source: promoting
    `cs.stanford.edu` would register all of Stanford CS as his site."""
    assert url_carries_token(url, _TOKENS)
    assert not host_carries_token(url, _TOKENS)


@pytest.mark.parametrize("url", [
    "https://www.wired.com/2015/01/karpathy/",              # press ABOUT him
    "https://theblock.co/post/gajesh",                      # the same shape, another person
    "https://techcrunch.com/2017/06/20/tesla-hires-andrej-karpathy-to-lead-ai",
])
def test_writing_about_the_person_is_not_writing_by_them(url):
    """The whole point of requiring a PREFIX. Their own area has content beneath it; an article
    about them ends on their name. Substring matching admitted 6 of these on karpathy.ai;
    whole-segment matching still admitted 1; requiring a non-final segment admits none."""
    tokens = _TOKENS | {"gajesh"}

    assert not url_carries_token(url, tokens)


@pytest.mark.parametrize("url", [
    "https://github.com/jcjohnson/densecap",                # a DIFFERENT person's repo
    "https://www.cs.toronto.edu/~hinton/",                  # another researcher's home
    "https://vision.stanford.edu/",
])
def test_another_persons_site_is_never_theirs(url):
    assert not url_carries_token(url, _TOKENS)


def test_no_tokens_matches_nothing():
    """An empty set must never mean "everything" — `_is_owned` falls back to the host rule alone
    for a caller with no name and no origin, and that fallback is what keeps it safe."""
    assert not url_carries_token("https://karpathy.github.io/2015/11/14/ai/", frozenset())
    assert not host_carries_token("https://karpathy.github.io", frozenset())
