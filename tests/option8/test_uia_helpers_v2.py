"""Tests for the v2 UiA helpers: latest-year filter, field extractor, header builder."""

from __future__ import annotations

from option8_rag.ingest.uia import (
    _keep_latest_year_per_course,
    extract_course_fields,
    synthesize_header,
)

_SAMPLE_PAGE = """IKT450 Deep Neural Networks (Autumn 2024)
ECTS Credits:
7.5
Responsible department:
Faculty of Engineering and Science
Course Leader:
Morten Goodwin
Lecture Semester:
Autumn
Teaching language:
English
Duration:
1/2 year
Some long descriptive paragraph that follows.
Examinations
Portfolio assessment. Information about the content of the portfolio.
"""


def test_extract_course_fields_full() -> None:
    fields = extract_course_fields(_SAMPLE_PAGE)
    assert fields["code"] == "IKT450"
    assert fields["title"] == "Deep Neural Networks"
    assert fields["semester"] == "Autumn"
    assert fields["year"] == "2024"
    assert fields["ects"] == "7.5"
    assert fields["course_leader"] == "Morten Goodwin"
    assert fields["teaching_language"] == "English"
    assert fields["examination"] == "Portfolio assessment"


def test_extract_course_fields_handles_missing() -> None:
    fields = extract_course_fields("Random page with no structure.")
    assert fields == {}


def test_synthesize_header_includes_key_facts() -> None:
    header = synthesize_header(extract_course_fields(_SAMPLE_PAGE))
    assert header.startswith("[course-header] ")
    assert "code: IKT450" in header
    assert "title: Deep Neural Networks" in header
    assert "course_leader: Morten Goodwin" in header
    assert "ects: 7.5" in header


def test_synthesize_header_empty() -> None:
    assert synthesize_header({}) == ""


def test_latest_year_keeps_most_recent_per_course() -> None:
    urls = [
        "https://www.uia.no/english/studies/courses/2020/autumn/ikt450.html",
        "https://www.uia.no/english/studies/courses/2024/autumn/ikt450.html",
        "https://www.uia.no/english/studies/courses/2022/autumn/ikt450.html",
        "https://www.uia.no/english/studies/courses/2023/spring/ikt100.html",
        "https://www.uia.no/english/studies/courses/2024/autumn/ikt100.html",
        "https://www.uia.no/english/studies/courses/2024/spring/ikt100.html",
    ]
    kept = _keep_latest_year_per_course(urls)
    assert "/2024/autumn/ikt450.html" in kept[0] or "/2024/autumn/ikt450.html" in kept[1]
    # IKT450 should appear exactly once and be the 2024 edition.
    ikt450 = [u for u in kept if "/ikt450.html" in u]
    assert len(ikt450) == 1
    assert "/2024/autumn/" in ikt450[0]
    # IKT100: 2024-autumn beats 2024-spring (autumn > spring rank).
    ikt100 = [u for u in kept if "/ikt100.html" in u]
    assert len(ikt100) == 1
    assert "/2024/autumn/" in ikt100[0]


def test_latest_year_passes_through_non_course_urls() -> None:
    urls = [
        "https://www.uia.no/english/about/contact",
        "https://www.uia.no/english/studies/courses/2024/autumn/ikt100.html",
    ]
    kept = _keep_latest_year_per_course(urls)
    assert "https://www.uia.no/english/about/contact" in kept
    assert any("ikt100" in u for u in kept)
