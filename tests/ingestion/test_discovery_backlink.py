"""`backlink` parsers on saved responses, and `facts_for`'s dispatch with the network stubbed.
Plus the ORCID root probe's parser. No live call."""

from types import SimpleNamespace

from pipeline.oracles.discovery import backlink
from pipeline.oracles.discovery import probes

# Trimmed live shapes.
SUBSTACK_PROFILE = {
    "name": "Alex Imas", "handle": "aleximas",
    "twitterAccount": {"screen_name": "alexolegimas", "is_connected_account": True},
    "userLinks": [{"url": "https://aleximas.com"},
                  {"url": "https://www.youtube.com/@someone"}],
}
GITHUB_USER = {"login": "rhyssullivan", "twitter_username": "RhysSullivan",
               "blog": "rhys.dev", "public_repos": 61}
HOME_WITH_FEED = """<html><head>
  <link rel="alternate" type="application/rss+xml" href="/feed.xml">
</head><body>
  <a href="https://x.com/naval">X</a> <a href="https://naval.substack.com">Newsletter</a>
  <a href="/about">About</a> <a href="https://youtube.com/@naval">video</a>
</body></html>"""
ORCID_PERSON = {
    "name": {"given-names": {"value": "Frances H."}, "family-name": {"value": "Arnold"},
             "credit-name": None},
    "researcher-urls": {"researcher-url": [
        {"url-name": "Frances H. Arnold Research Group",
         "url": {"value": "http://cheme.che.caltech.edu/groups/fha/index.html"}}]},
}


def test_substack_profile_names_its_x_and_declares_its_links():
    f = backlink.facts_from_substack_profile("aleximas.substack.com", SUBSTACK_PROFILE)
    assert f.reached and f.has_feed is True and f.final_key is None
    assert f.links_to == {"x.com/alexolegimas", "aleximas.com"}     # video links have no identity
    assert f.declared[0] == "https://x.com/alexolegimas"


def test_github_user_declares_its_x_and_blog():
    f = backlink.facts_from_github_user("github.com/rhyssullivan", GITHUB_USER)
    assert f.links_to == {"x.com/rhyssullivan", "rhys.dev"}
    assert f.declared == ("https://x.com/RhysSullivan", "https://rhys.dev")
    assert f.has_feed is True
    assert backlink.facts_from_github_user("github.com/e", {"public_repos": 0}).has_feed is False


def test_a_home_page_with_a_feed_and_its_links():
    f = backlink.facts_from_html("nav.al", HOME_WITH_FEED)
    assert f.links_to == {"x.com/naval", "naval.substack.com"}
    assert f.has_feed is True and f.declared == ()


def test_no_feed_tag_is_unknown_not_none():
    assert backlink.facts_from_html("chrisbarber.co", "<a href='https://x.com/c'>x</a>").has_feed is None


def test_a_redirect_is_recorded_only_when_it_lands_elsewhere():
    assert backlink.facts_from_html("herbiebradley.substack.com", "",
                                    "https://pathwaysai.org/").final_key == "pathwaysai.org"
    assert backlink.facts_from_html("nav.al", "", "https://nav.al/").final_key is None


def test_a_failed_read_is_unknown(monkeypatch):
    def boom(*a, **k):
        raise backlink.requests.ConnectionError("down")
    monkeypatch.setattr(backlink.requests, "get", boom)
    f = backlink.facts_for("nav.al", "https://nav.al")
    assert not f.reached and f.links_to == frozenset()


def test_a_github_rate_limit_is_unknown(monkeypatch):
    def limited(_login):
        raise backlink.github_api.GitHubRateLimited("403")
    monkeypatch.setattr(backlink.github_api, "fetch_user", limited)
    assert not backlink.facts_for("github.com/x", "https://github.com/x").reached


def test_a_custom_domain_served_by_substack_gets_the_author_profile(monkeypatch):
    def get(url, **kw):
        if url == "https://pathwaysai.org":
            return SimpleNamespace(status_code=200, url="https://pathwaysai.org/",
                                   text='<script src="https://substackcdn.com/x.js"></script>')
        if "users/ranked" in url:
            return SimpleNamespace(status_code=200, url=url, json=lambda: [{"handle": "herbie"}])
        if "user/herbie/public_profile" in url:
            return SimpleNamespace(status_code=200, url=url, json=lambda: {
                "twitterAccount": {"screen_name": "herbiebradley"},
                "userLinks": [{"url": "https://herbiebradley.com"}]})
        return SimpleNamespace(status_code=404, url=url, text="")
    monkeypatch.setattr(backlink.requests, "get", get)
    f = backlink.facts_for("pathwaysai.org", "https://pathwaysai.org")
    assert "x.com/herbiebradley" in f.links_to and f.has_feed is True
    assert "https://herbiebradley.com" in f.declared


def test_orcid_person_gives_the_name_and_listed_sites():
    name, urls = probes.orcid_person(ORCID_PERSON)
    assert name == "Frances H. Arnold"
    assert urls == ["http://cheme.che.caltech.edu/groups/fha/index.html"]
    assert probes.orcid_person({}) == ("", [])


def test_a_refusal_is_an_answer_not_unknown(monkeypatch):
    """nytimes.com refuses a bot. That is an answer with no link back, not an unknown that would
    ask the user about every newspaper an Oracle cites."""
    monkeypatch.setattr(backlink.requests, "get",
                        lambda url, **k: SimpleNamespace(status_code=403, url=url, text=""))
    f = backlink.facts_for("nytimes.com", "https://nytimes.com")
    assert f.reached and f.links_to == frozenset()


def test_a_github_profile_declares_its_x_and_its_website(monkeypatch):
    """The GitHub profile probe, read beside the root at the links stage."""
    monkeypatch.setattr(probes.github_api, "fetch_user", lambda owner: GITHUB_USER)
    info, sources, targets = probes.probe_github_profile("RhysSullivan")
    assert info["display_name"] == "rhyssullivan"
    assert info["root_url"] == "https://github.com/rhyssullivan"
    assert [s.url for s in sources] == ["https://github.com/rhyssullivan",
                                        "https://x.com/RhysSullivan", "https://rhys.dev"]
    assert targets == [("x.com/rhyssullivan", False), ("rhys.dev", False)]


def test_a_github_profile_that_could_not_be_read_is_empty(monkeypatch):
    """A rate limit or a failed read answers nothing, as a root that does not resolve: the
    caller retries rather than losing what it declares."""
    def limited(_owner):
        raise probes.github_api.GitHubRateLimited("403")
    monkeypatch.setattr(probes.github_api, "fetch_user", limited)
    assert probes.probe_github_profile("rhyssullivan") == ({}, [], [])
    monkeypatch.setattr(probes.github_api, "fetch_user", lambda owner: None)
    assert probes.probe_github_profile("rhyssullivan") == ({}, [], [])


def test_an_organizations_github_profile_declares_nothing_as_theirs(monkeypatch):
    """Gajesh's cluster holds github:Layr-Labs, his employer's account (live). It
    names Eigen Labs' X and site; read as his profile, both would be added as his without a
    link back, and an employer's site is never pulled as their writing."""
    org = {"login": "Layr-Labs", "type": "Organization", "name": "Eigen Labs",
           "blog": "https://www.eigenlabs.org", "twitter_username": "eigenlabs",
           "public_repos": 184}
    monkeypatch.setattr(probes.github_api, "fetch_user", lambda owner: org)
    info, sources, targets = probes.probe_github_profile("Layr-Labs")
    assert info["display_name"] == "Eigen Labs"                 # it answered: nothing to retry
    assert [s.url for s in sources] == ["https://github.com/Layr-Labs"] and targets == []
