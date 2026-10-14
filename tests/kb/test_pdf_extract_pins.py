"""Behavior pins for `platforms/papers/pdf._extract_pypdf`.

Real PDFs built in-test with pypdf's writer, so these exercise pypdf itself rather than a mock.
"""
from __future__ import annotations

import io
import sys

import pytest
from pypdf import PdfWriter
from pypdf.errors import EmptyFileError, PdfStreamError
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from pipeline.platforms.papers import pdf as paper_pdf


def _pdf(pages):
    """A real PDF; `pages` is a list where each item is a list of text lines, or None for a blank
    page."""
    w = PdfWriter()
    for lines in pages:
        page = w.add_blank_page(612, 792)
        if lines is None:
            continue
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): w._add_object(font)})})
        ops, y = [], 740
        for text in lines:
            ops.append(f"BT /F1 12 Tf 72 {y} Td ({text}) Tj ET")
            y -= 20
        stream = DecodedStreamObject()
        stream.set_data("\n".join(ops).encode())
        page[NameObject("/Contents")] = w._add_object(stream)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def test_pages_are_joined_by_a_blank_line_and_counted():
    data = _pdf([["Hello world", "second line"], None, ["Page three"]])
    # A blank middle page still contributes its "" between two "\n\n" separators.
    assert paper_pdf._extract_pypdf(data) == ("Hello world\nsecond line\n\n\n\nPage three", 3)


def test_a_single_blank_page_is_empty_text_and_one_page():
    assert paper_pdf._extract_pypdf(_pdf([None])) == ("", 1)


def test_text_is_returned_raw_with_no_repair():
    """Ligature folding and hyphen rejoining are `_repair_extraction`'s job, not this one's."""
    text, pages = paper_pdf._extract_pypdf(_pdf([["seg-", "mentation"]]))
    assert (text, pages) == ("seg-\nmentation", 1)


@pytest.mark.parametrize("data, exc", [
    (b"", EmptyFileError),
    (b"not a pdf", PdfStreamError),
    (b"%PDF-1.4\n", PdfStreamError),
])
def test_unreadable_bytes_raise_pypdfs_own_error(data, exc):
    """It never catches: the fail-safe lives in its caller `_pdf_bytes_to_text`."""
    with pytest.raises(exc):
        paper_pdf._extract_pypdf(data)


def test_a_missing_pypdf_raises_import_error_at_call_time(monkeypatch):
    """The import is lazy, inside the function, so the module imports without pypdf and the
    absence surfaces only as an ImportError from this call."""
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(ImportError):
        paper_pdf._extract_pypdf(b"%PDF-1.4\n")


def test_the_caller_turns_an_extractor_failure_into_none(monkeypatch):
    """Pins the seam the two tests above rely on: `_pdf_bytes_to_text` swallows both."""
    assert paper_pdf._pdf_bytes_to_text(b"not a pdf") is None
    monkeypatch.setitem(sys.modules, "pypdf", None)
    assert paper_pdf._pdf_bytes_to_text(b"%PDF-1.4\n") is None


# ── usable_text: one row per case ───────────────────────────────────────────────

def test_usable_text_none_below_the_body_floor():
    assert paper_pdf.usable_text("x" * 499, 1) is None


def test_usable_text_none_below_the_per_page_floor():
    assert paper_pdf.usable_text("x" * 1999, 10) is None


def test_usable_text_is_the_repaired_text_when_whole():
    raw = "seg-\nmentation " + "y" * 600
    assert paper_pdf.usable_text(raw, 1) == "segmentation " + "y" * 600


# ── created_year: one row per case ──────────────────────────────────────────────

def test_created_year_none_without_a_d_year():
    assert paper_pdf.created_year("2021-03-04", 2026) is None


def test_created_year_none_before_1990():
    assert paper_pdf.created_year("D:19891231", 2026) is None


def test_created_year_none_after_the_current_year():
    assert paper_pdf.created_year("D:2027", 2026) is None


def test_created_year_is_the_year_otherwise():
    assert paper_pdf.created_year("D:20260101120000Z", 2026) == 2026


# ── choose_title: one row per case ──────────────────────────────────────────────

def test_choose_title_keeps_metadata_when_both_agree():
    assert paper_pdf.choose_title("Selling Information in Markets", "Selling information in markets") == (
        "Selling Information in Markets", "pdf_metadata_and_first_page")


def test_choose_title_keeps_neither_when_they_disagree():
    assert paper_pdf.choose_title("Selling Information in Markets", "A Journal of Economics") == (
        None, None)


def test_choose_title_metadata_alone():
    assert paper_pdf.choose_title("Selling Information", None) == ("Selling Information", "pdf_metadata")


def test_choose_title_first_page_alone():
    assert paper_pdf.choose_title(None, "Selling Information") == ("Selling Information",
                                                            "pdf_first_page")


def test_choose_title_neither():
    assert paper_pdf.choose_title(None, None) == (None, None)


# ── page_title: one row per case, in the gate's order ───────────────────────────

_BODY = ("body text " * 20, 10.0)


def test_page_title_none_without_positive_runs():
    assert paper_pdf.page_title([("", 0.0), ("   ", 12.0)]) is None


def test_page_title_none_when_nothing_stands_out():
    assert paper_pdf.page_title([("A Slightly Larger Line", 11.0), _BODY]) is None


@pytest.mark.parametrize("title", ["Two Words", " ".join(["word"] * 41)])
def test_page_title_none_outside_three_to_forty_words(title):
    assert paper_pdf.page_title([(title, 20.0), _BODY]) is None


def test_page_title_none_for_a_masthead():
    assert paper_pdf.page_title([("International Journal of Creative Research", 20.0), _BODY]) is None


def test_page_title_none_with_a_leading_digit():
    assert paper_pdf.page_title([("2 Selling Information Markets", 20.0), _BODY]) is None


def test_page_title_none_when_digits_pass_a_fifth():
    assert paper_pdf.page_title([("Run 12345 of 67890 set", 20.0), _BODY]) is None


def test_page_title_is_the_first_large_block():
    runs = [("Selling Information", 20.0), ("", 0.0), ("in Competitive Markets", 18.0),
            _BODY, ("Later Big Heading Here", 20.0)]
    assert paper_pdf.page_title(runs) == "Selling Information in Competitive Markets"
