"""A report's landing page → what it declares about its paper (`pipeline/platforms/papers/landing.py`).

The pages are cut down from the real ones: `eprint.iacr.org/2025/2041` and
`eccc.weizmann.ac.il/report/2026/237/`, markup kept as served.
"""
from __future__ import annotations

from pipeline.platforms.papers import landing

_EPRINT = """<html><head>
<meta name="tdm-reservation" content="1">
<meta name="citation_title" content="Sum-check Is All You Need: An Opinionated Survey on Fast Provers in SNARK Design">
<meta name="citation_author" content="Justin Thaler">
<meta name="citation_journal_title" content="Cryptology ePrint Archive">
<meta name="citation_publication_date" content="2025">
<meta name="citation_pdf_url" content="https://eprint.iacr.org/2025/2041.pdf">
<style>
  .paper-abstract {
     white-space: pre-wrap;
  }
</style></head><body>
<div class="col-md-7 col-lg-8 pr-md-5">
    <h5 class="mt-3">Abstract</h5>
    <p style="white-space: pre-wrap;">SNARKs work by having a prover commit to a witness.
The prover’s work is dominated by two tasks.

But not all uses of sum-check are equally effective.</p>
  </div>
  <div id="metadata" class="col-md-5 col-lg-4 ps-md-5 mt-4 mt-md-0">
    <p>Category: Cryptographic protocols</p>
  </div>
</body></html>"""

_ECCC = """<html><head>
<meta name="Description" content="Homepage of the Electronic Colloquium on Computational Complexity" />
<meta name="citation_title" content="Counterexamples to Beyond-Johnson Proximity Gaps over Binary Fields" />
<meta name="citation_publication_date" content='2026-10-09' />
<meta name="citation_technical_report_institution" content="Electronic Colloquium on Computational Complexity (ECCC)" />
<meta name="citation_author" content="Quang Dao" />
<meta name="citation_author" content="Scott Duke Kominers" />
<meta name="citation_author" content="Justin Thaler" />
<meta name="citation_author" content="Kai Zhe Zheng" />
<meta name="citation_pdf_url" content="https://eccc.weizmann.ac.il/report/2026/237/download" />
</head><body>
		Authors: <a href="/author/345">Justin Thaler</a>
	<br /><b>Abstract:</b><br />
	<p>Such gaps are known above the Johnson threshold, about $\\sqrt{\\rho}$ for rate $\\rho$.</p>

<p>On Binius64&#39;s domain, the combination reaches 49.4% agreement.</p>
</div>
        <div style="clear:both;">
        	<br />
        	<hr />
			<small>ISSN 1433-8092 | <a href="/static/website/imprint">Imprint</a></small>
			<p>Not the abstract.</p>
		</div>
</body></html>"""


def test_an_eprint_page_names_its_paper():
    assert landing.declared(_EPRINT) == {
        "title": "Sum-check Is All You Need: An Opinionated Survey on Fast Provers in SNARK Design",
        "authors": [{"name": "Justin Thaler"}],
        "abstract": ("SNARKs work by having a prover commit to a witness. The prover’s work is "
                     "dominated by two tasks.\n\nBut not all uses of sum-check are equally effective."),
        "year": 2025,                                     # ePrint dates a report by its year alone
        "venue": "Cryptology ePrint Archive",
        "openAccessPdf": {"url": "https://eprint.iacr.org/2025/2041.pdf"},
    }


def test_an_eccc_page_names_its_paper():
    assert landing.declared(_ECCC) == {
        "title": "Counterexamples to Beyond-Johnson Proximity Gaps over Binary Fields",
        "authors": [{"name": "Quang Dao"}, {"name": "Scott Duke Kominers"},
                    {"name": "Justin Thaler"}, {"name": "Kai Zhe Zheng"}],
        "abstract": ("Such gaps are known above the Johnson threshold, about $\\sqrt{\\rho}$ for "
                     "rate $\\rho$.\n\nOn Binius64's domain, the combination reaches 49.4% agreement."),
        "year": 2026,
        "publicationDate": "2026-10-09",
        "venue": "Electronic Colloquium on Computational Complexity (ECCC)",
        "openAccessPdf": {"url": "https://eccc.weizmann.ac.il/report/2026/237/download"},
    }


def test_a_page_that_declares_nothing_names_nothing():
    assert landing.declared("<html><head><title>Just a moment...</title></head></html>") == {}


class _Resp:
    def __init__(self, status, text=""):
        self.status_code, self.text = status, text


def test_a_refused_page_reads_as_nothing(monkeypatch):
    monkeypatch.setattr(landing, "host_request", lambda send, url, **kw: _Resp(403, _EPRINT))
    assert landing.metadata("https://eprint.iacr.org/2025/2041") is None


def test_a_page_is_read_once(monkeypatch):
    calls = []
    monkeypatch.setattr(landing, "host_request",
                        lambda send, url, **kw: calls.append(url) or _Resp(200, _ECCC))
    assert landing.metadata("https://eccc.weizmann.ac.il/report/2026/237")["year"] == 2026
    assert calls == ["https://eccc.weizmann.ac.il/report/2026/237"]
