"""Polite, sitemap-driven static crawler for the UiA IKT corpus.

The crawler does three things:

1. Fetches ``sitemap.xml`` and filters URLs by a configurable regex.
2. Optionally appends extra hand-picked URLs (programme overviews etc.).
3. Fetches each URL with Scrapling's ``Fetcher``, respecting ``robots.txt``
   via ``urllib.robotparser`` and a configurable per-host delay with
   jitter.

Raw HTML and a cleaned plain-text view are persisted side-by-side per page.
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger

from option8_rag.types import Document, ensure_dir


@dataclass(frozen=True, slots=True)
class CrawlConfig:
    """Configuration for the UiA crawler.

    Attributes:
        sitemap_url: URL of the sitemap.xml index.
        url_regex: Regex applied to each sitemap URL; only matches are kept.
        extra_urls: Hand-picked URLs to add on top of sitemap matches.
        user_agent: User-Agent string used for requests *and* robots.txt
            evaluation.
        delay_seconds: Base delay between successive requests to the same
            host.
        jitter_seconds: Maximum random jitter added to ``delay_seconds``.
        max_pages: Hard cap on number of pages fetched (safety belt).
        obey_robots: When true, evaluate robots.txt and skip disallowed URLs.
        latest_year_only: When true, keep only the most recent year-edition
            of each course. Without this, the corpus accumulates 5+ near-
            identical copies per course (one per academic year), and dense
            retrieval ends up returning the same boilerplate paragraph
            five times in the top-k. Strongly recommended for the UiA
            corpus.
    """

    sitemap_url: str
    url_regex: str
    extra_urls: tuple[str, ...] = ()
    user_agent: str = "option8-rag/0.1 (academic research)"
    delay_seconds: float = 1.0
    jitter_seconds: float = 0.5
    max_pages: int = 1000
    obey_robots: bool = True
    latest_year_only: bool = False


class UiaCrawler:
    """Sitemap-driven static crawler.

    Args:
        config: Crawl configuration.
        out_dir: Directory under which raw HTML and cleaned text are
            persisted. ``raw/`` and ``text/`` subdirectories are created.
    """

    def __init__(self, *, config: CrawlConfig, out_dir: Path) -> None:
        self.config = config
        self.out_dir = ensure_dir(out_dir)
        self.raw_dir = ensure_dir(out_dir / "raw")
        self.text_dir = ensure_dir(out_dir / "text")
        self.fields_dir = ensure_dir(out_dir / "fields")
        self.url_list_path = out_dir / "urls.json"
        self._rp_cache: dict[str, urllib.robotparser.RobotFileParser] = {}

    # -- discovery -----------------------------------------------------

    def discover_urls(self) -> list[str]:
        """Return the deduplicated, regex-filtered URL list to crawl.

        The result is persisted to ``urls.json`` for reproducibility.
        """

        from scrapling.fetchers import Fetcher  # local import keeps optional dep optional

        logger.info("fetching sitemap {url}", url=self.config.sitemap_url)
        page = Fetcher.get(self.config.sitemap_url, timeout=30)
        if page.status not in (200, 304):
            raise RuntimeError(
                f"sitemap fetch failed: {self.config.sitemap_url} -> HTTP {page.status}",
            )

        urls = _extract_sitemap_urls(page.body)
        pattern = re.compile(self.config.url_regex)
        matched = [u for u in urls if pattern.match(u)]

        if self.config.latest_year_only:
            before = len(matched)
            matched = _keep_latest_year_per_course(matched)
            logger.info(
                "latest-year-only filter: {before} -> {after} URLs",
                before=before,
                after=len(matched),
            )

        # Deduplicate while preserving order; extras appended last.
        seen: set[str] = set()
        ordered: list[str] = []
        for u in (*matched, *self.config.extra_urls):
            if u in seen:
                continue
            seen.add(u)
            ordered.append(u)

        if len(ordered) > self.config.max_pages:
            logger.warning(
                "discovered {n} URLs, capping to max_pages={cap}",
                n=len(ordered),
                cap=self.config.max_pages,
            )
            ordered = ordered[: self.config.max_pages]

        self.url_list_path.write_text(json.dumps(ordered, indent=2))
        logger.info(
            "discovered {n} URLs (regex={regex!r}); written to {path}",
            n=len(ordered),
            regex=self.config.url_regex,
            path=str(self.url_list_path),
        )
        return ordered

    # -- crawling ------------------------------------------------------

    def crawl(self, urls: Iterable[str]) -> list[Document]:
        """Fetch each URL politely; return a list of cleaned :class:`Document`.

        Pages disallowed by ``robots.txt`` (when ``obey_robots`` is true)
        are skipped with a warning. Failed fetches are logged and skipped;
        they do not abort the crawl.
        """

        from scrapling.fetchers import FetcherSession

        documents: list[Document] = []
        with FetcherSession(timeout=30, retries=3, retry_delay=1) as session:
            for url in urls:
                if self.config.obey_robots and not self._allowed(url):
                    logger.warning("robots.txt disallows {url}; skipping", url=url)
                    continue
                try:
                    doc = self._fetch_one(session, url)
                except Exception:  # pragma: no cover — exercised by integration tests only
                    logger.exception("fetch failed for {url}", url=url)
                    continue
                if doc is not None:
                    documents.append(doc)
                self._sleep()

        logger.info("crawl finished: {n} pages persisted", n=len(documents))
        return documents

    # -- internals -----------------------------------------------------

    def _fetch_one(self, session, url: str) -> Document | None:
        page = session.get(url, headers={"User-Agent": self.config.user_agent})
        if page.status != 200:
            logger.warning("HTTP {status} for {url}", status=page.status, url=url)
            return None

        doc_id = _doc_id_from_url(url)
        raw_path = self.raw_dir / f"{doc_id}.html"
        text_path = self.text_dir / f"{doc_id}.txt"
        fields_path = self.fields_dir / f"{doc_id}.json"
        raw_path.write_bytes(page.body)

        title = _extract_title(page)
        text = _extract_text(page)
        text_path.write_text(text)

        fields = extract_course_fields(text)
        fields["url"] = url
        fields_path.write_text(json.dumps(fields, indent=2))

        return Document(
            doc_id=doc_id,
            text=text,
            title=title,
            source=url,
            metadata={
                "raw_path": str(raw_path),
                "text_path": str(text_path),
                "fields_path": str(fields_path),
                **{k: v for k, v in fields.items() if isinstance(v, str | int | float | bool)},
            },
        )

    def _allowed(self, url: str) -> bool:
        host = urlparse(url).netloc
        rp = self._rp_cache.get(host)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"https://{host}/robots.txt")
            try:
                rp.read()
            except Exception:  # pragma: no cover — unlikely
                logger.warning("robots.txt fetch failed for {host}; allowing", host=host)
                # Returning a permissive parser here keeps the cache
                # populated so we don't retry repeatedly.
                rp = urllib.robotparser.RobotFileParser()
                rp.parse([])
            self._rp_cache[host] = rp
        return rp.can_fetch(self.config.user_agent, url)

    def _sleep(self) -> None:
        delay = self.config.delay_seconds + random.random() * self.config.jitter_seconds
        time.sleep(delay)


# -- helpers ----------------------------------------------------------


_NAMESPACES = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def _extract_sitemap_urls(body: bytes) -> list[str]:
    """Extract ``<loc>`` URLs from a sitemap XML body."""

    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise RuntimeError(f"failed to parse sitemap XML: {exc}") from exc

    return [elem.text.strip() for elem in root.findall(".//sm:loc", _NAMESPACES) if elem.text]


def _doc_id_from_url(url: str) -> str:
    """Stable, filesystem-safe document id derived from the URL path."""

    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        path = parsed.netloc.replace(".", "_")
    return path.replace("/", "__").replace(".html", "").replace(".", "_")


_COURSE_URL_RE = re.compile(
    r"/english/studies/courses/(?P<year>\d{4})/(?P<season>autumn|spring)/"
    r"(?P<code>[\w-]+)\.html$",
)
_SEASON_RANK = {"spring": 0, "autumn": 1}

# Field labels we extract from a UiA course page. Each maps to the
# canonical key we store under in the fields JSON. Matching is done on
# whole, trimmed lines because the page format is "Label:\n<value>\n...".
_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("ECTS Credits:", "ects"),
    ("Responsible department:", "department"),
    ("Course Leader:", "course_leader"),
    ("Lecture Semester:", "lecture_semester"),
    ("Teaching language:", "teaching_language"),
    ("Duration:", "duration"),
)

_TITLE_RE = re.compile(
    r"^(?P<code>IKT[\w-]+)\s+(?P<title>.+?)\s+\((?P<season>Autumn|Spring)\s+(?P<year>\d{4})\)\s*$",
)


def extract_course_fields(text: str) -> dict[str, str]:
    """Pull the structured field block from a UiA course page.

    Returns a dict with whatever subset of keys was present. Missing
    fields are simply absent (not empty strings) so downstream callers
    can use ``dict.get`` with a default.
    """

    fields: dict[str, str] = {}
    lines = [line.rstrip() for line in text.split("\n")]

    if lines:
        m = _TITLE_RE.match(lines[0]) or (_TITLE_RE.match(lines[1]) if len(lines) > 1 else None)
        if m:
            fields["code"] = m.group("code")
            fields["title"] = m.group("title")
            fields["semester"] = m.group("season")
            fields["year"] = m.group("year")

    for i, line in enumerate(lines):
        for label, key in _FIELD_LABELS:
            if line.strip() == label and i + 1 < len(lines):
                value = lines[i + 1].strip()
                if value:
                    fields[key] = value

    # Best-effort examination type — appears further down with no label.
    for keyword in (
        "Portfolio examination",
        "Portfolio assessment",
        "Written examination",
        "Oral examination",
        "Home examination",
        "Project assignment",
    ):
        if keyword in text:
            fields["examination"] = keyword
            break
    return fields


def synthesize_header(fields: dict[str, str]) -> str:
    """Render a one-line header summarising a course page for chunk prefixing.

    The output is a compact ``key=value`` line that exposes every fact a
    typical lookup question can target (course code, title, leader,
    ECTS, language, semester, examination). Prepending this to every
    chunk of a page means retrieval can pull *any* chunk for *any* fact
    question and the answer will be in the chunk text — no longer
    dependent on which paragraph the SentenceSplitter put the leader
    name into.
    """

    if not fields:
        return ""
    order = (
        "code",
        "title",
        "course_leader",
        "ects",
        "teaching_language",
        "semester",
        "year",
        "examination",
        "duration",
        "department",
    )
    parts = [f"{k}: {fields[k]}" for k in order if fields.get(k)]
    if not parts:
        return ""
    return "[course-header] " + " | ".join(parts)


def _keep_latest_year_per_course(urls: list[str]) -> list[str]:
    """Collapse each (course-code) to its single most-recent edition.

    The UiA sitemap exposes every yearly edition of every course (e.g.
    IKT450 appears for 2020, 2021, 2022, 2023, 2024). Indexing all of
    them produces 5+ near-identical chunks per course that flood the
    top-k for every retrieval. This helper keeps only the latest year,
    breaking ties in favour of autumn over spring (autumn is the
    canonical first-half-of-the-academic-year semester).
    """

    best: dict[str, tuple[tuple[int, int], str]] = {}
    pass_through: list[str] = []
    for url in urls:
        m = _COURSE_URL_RE.search(url)
        if not m:
            pass_through.append(url)
            continue
        year = int(m.group("year"))
        season = m.group("season")
        code = m.group("code").lower()
        rank = (year, _SEASON_RANK.get(season, 0))
        prev = best.get(code)
        if prev is None or rank > prev[0]:
            best[code] = (rank, url)

    # Preserve original sitemap order for stable hashing of the resolved
    # URL list. We sort the kept-URL set by the order they appeared.
    kept_urls = {entry[1] for entry in best.values()}
    ordered_kept = [u for u in urls if u in kept_urls]
    return [*ordered_kept, *pass_through]


def _extract_title(page) -> str:
    title = page.css_first("title::text") if hasattr(page, "css_first") else None
    if title is None:
        # Fallback for older API surface.
        title_list = page.css("title::text").getall() if hasattr(page, "css") else []
        return title_list[0].strip() if title_list else ""
    return title.strip() if isinstance(title, str) else str(title).strip()


def _extract_text(page) -> str:
    """Extract a readable plain-text view from a Scrapling page.

    Scrapes paragraph- and heading-like elements; collapses whitespace.
    """

    if hasattr(page, "css"):
        parts = page.css("h1::text, h2::text, h3::text, p::text, li::text").getall()
    else:  # pragma: no cover — defensive
        parts = []
    cleaned: list[str] = []
    for p in parts:
        t = " ".join((p or "").split())
        if t:
            cleaned.append(t)
    return "\n".join(cleaned)
