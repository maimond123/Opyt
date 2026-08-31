"""Tests for source_classify — typed (type, is_profile, handle, shape) classification.

Where _classify_url returns a bare type string, classify_source adds the three
facts hub expansion + atom onboarding need: profile-vs-artifact, org-vs-personal,
and the account handle.
"""

from pipeline.ingestion.source_classify import classify_source


# ── GitHub: user profile vs repo artifact vs org ──────────────────────────────

def test_github_user_is_a_personal_profile():
    pl = classify_source("https://github.com/willccbb")
    assert (pl.type, pl.is_profile, pl.handle, pl.shape) == ("github", True, "willccbb", "personal")


def test_github_repo_is_an_artifact_not_a_profile():
    pl = classify_source("https://github.com/willccbb/verifiers")
    assert pl.type == "github"
    assert pl.is_profile is False
    assert pl.handle == "willccbb"   # still know whose repo it is


def test_github_org_is_org_shaped():
    pl = classify_source("https://github.com/orgs/anthropics")
    assert (pl.type, pl.is_profile, pl.shape) == ("github", True, "org")
    assert pl.handle == "anthropics"


def test_github_feature_page_is_not_a_profile():
    assert classify_source("https://github.com/features/copilot").is_profile is False


# ── Substack: home vs post ────────────────────────────────────────────────────

def test_substack_home_is_a_profile():
    pl = classify_source("https://willcb.substack.com")
    assert (pl.type, pl.is_profile, pl.handle) == ("substack", True, "willcb")


def test_substack_post_is_an_artifact():
    pl = classify_source("https://willcb.substack.com/p/some-essay")
    assert pl.type == "substack"
    assert pl.is_profile is False
    assert pl.handle == "willcb"


# ── Papers: arxiv / pdf / doi → never a profile ───────────────────────────────

def test_arxiv_abs_is_a_paper_artifact():
    pl = classify_source("https://arxiv.org/abs/2310.12345")
    assert (pl.type, pl.is_profile, pl.handle) == ("paper", False, None)


def test_pdf_url_is_a_paper():
    assert classify_source("https://example.com/papers/thesis.pdf").type == "paper"


def test_doi_is_a_paper():
    assert classify_source("https://doi.org/10.1000/xyz").type == "paper"


# ── Scholar / ORCID ───────────────────────────────────────────────────────────

def test_orcid_profile():
    pl = classify_source("https://orcid.org/0000-0002-1825-0097")
    assert (pl.type, pl.is_profile, pl.handle) == ("orcid", True, "0000-0002-1825-0097")


def test_google_scholar_citations_profile():
    pl = classify_source("https://scholar.google.com/citations?user=abcDEF123")
    assert (pl.type, pl.is_profile, pl.handle) == ("scholar", True, "abcDEF123")


def test_semantic_scholar_author_profile():
    pl = classify_source("https://www.semanticscholar.org/author/Jane-Doe/1234567")
    assert pl.type == "scholar" and pl.is_profile is True


def test_dblp_author_is_a_scholar_profile():
    pl = classify_source("https://dblp.org/pid/12/3456.html")
    assert (pl.type, pl.is_profile, pl.handle) == ("scholar", True, "3456")


def test_arxiv_author_page_is_a_profile_not_a_paper():
    # arxiv.org/a/name is an author listing — must beat the paper-host bucket.
    pl = classify_source("https://arxiv.org/a/brown_w_1")
    assert (pl.type, pl.is_profile, pl.handle) == ("scholar", True, "brown_w_1")
    # but an actual paper on the same host is still an artifact:
    assert classify_source("https://arxiv.org/abs/2310.12345").type == "paper"


def test_researchgate_and_academia_profiles():
    assert classify_source("https://www.researchgate.net/profile/Jane-Doe").type == "scholar"
    assert classify_source("https://janedoe.academia.edu").is_profile is True
    assert classify_source("https://www.academia.edu/12345").type == "scholar"


# ── YouTube: channel vs video ─────────────────────────────────────────────────

def test_youtube_handle_channel_is_a_profile():
    pl = classify_source("https://www.youtube.com/@AndrejKarpathy")
    assert (pl.type, pl.is_profile, pl.handle) == ("youtube", True, "andrejkarpathy")


def test_youtube_channel_id_is_a_profile():
    pl = classify_source("https://youtube.com/channel/UC1234")
    assert pl.is_profile is True and pl.handle == "uc1234"


def test_youtube_watch_is_an_artifact():
    assert classify_source("https://www.youtube.com/watch?v=dQw4w9WgXcQ").is_profile is False


# ── LinkedIn: personal vs company ─────────────────────────────────────────────

def test_linkedin_in_is_personal():
    pl = classify_source("https://www.linkedin.com/in/someone")
    assert (pl.type, pl.is_profile, pl.shape) == ("linkedin", True, "personal")


def test_linkedin_company_is_org():
    pl = classify_source("https://linkedin.com/company/anthropic")
    assert (pl.type, pl.shape) == ("linkedin", "org")


# ── X / Twitter ───────────────────────────────────────────────────────────────

def test_x_handle_is_a_profile_and_twitter_aliases():
    pl = classify_source("https://twitter.com/willccbb")   # aliases to x
    assert (pl.type, pl.is_profile, pl.handle) == ("x", True, "willccbb")


def test_x_status_is_an_artifact():
    assert classify_source("https://x.com/willccbb/status/123").is_profile is False


# ── Bare personal domain ──────────────────────────────────────────────────────

def test_bare_personal_domain_is_a_blog_profile_with_domain_label_handle():
    pl = classify_source("https://willcb.com")
    assert (pl.type, pl.is_profile, pl.handle) == ("blog", True, "willcb")


def test_subdomain_blog_uses_registrable_label():
    assert classify_source("https://blog.willcb.com").handle == "willcb"


def test_ccTLD_domain_label():
    assert classify_source("https://foo.co.uk").handle == "foo"


def test_deep_blog_page_is_not_a_profile():
    assert classify_source("https://simonwillison.net/2024/some-post").is_profile is False


# ── Garbage ───────────────────────────────────────────────────────────────────

def test_empty_and_garbage_return_none():
    assert classify_source("") is None
    assert classify_source("   ") is None
    assert classify_source(None) is None


def test_non_web_hrefs_return_none():
    # Caught live: real blog HTML links these, and they must not become 'profiles'.
    assert classify_source("mailto:w.brown@columbia.edu") is None
    assert classify_source("tel:+15551234567") is None
    assert classify_source("javascript:void(0)") is None
    assert classify_source("#section") is None


def test_url_field_preserved():
    assert classify_source("https://github.com/willccbb").url == "https://github.com/willccbb"
