"""Tests for UiA crawler helpers (sitemap parsing, doc-id derivation)."""

from __future__ import annotations

from option8_rag.ingest.uia import _doc_id_from_url, _extract_sitemap_urls


def test_extract_sitemap_urls_basic() -> None:
    body = b"""<?xml version='1.0' encoding='UTF-8'?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
   <url><loc>https://www.uia.no/english/studies/courses/2026/autumn/ikt450.html</loc></url>
   <url><loc>https://www.uia.no/english/studies/courses/2026/spring/ikt451.html</loc></url>
</urlset>
"""
    urls = _extract_sitemap_urls(body)
    assert urls == [
        "https://www.uia.no/english/studies/courses/2026/autumn/ikt450.html",
        "https://www.uia.no/english/studies/courses/2026/spring/ikt451.html",
    ]


def test_doc_id_from_url_replaces_slashes() -> None:
    url = "https://www.uia.no/english/studies/courses/2026/autumn/ikt450.html"
    doc_id = _doc_id_from_url(url)
    assert doc_id == "english__studies__courses__2026__autumn__ikt450"
    assert "/" not in doc_id
    assert ".html" not in doc_id


def test_doc_id_stable_across_repeated_calls() -> None:
    url = "https://www.uia.no/path/to/page.html"
    assert _doc_id_from_url(url) == _doc_id_from_url(url)
