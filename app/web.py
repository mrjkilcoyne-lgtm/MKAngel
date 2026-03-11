"""
Web capability — the Angel's eyes on the world.

Provides web search, page fetching, content extraction, and
summarization using only Python stdlib (no requests, no beautifulsoup).

The Angel sees the web through grammar-coloured glasses:
every page is a derivation, every link a production rule,
every search result a path through the Borges Library.

Capabilities:
  - search: Query DuckDuckGo and extract results
  - fetch: Download and clean a web page to text
  - extract: Pull structured data from HTML
  - summarize: Condense fetched content (via GLM or provider)
  - monitor: Watch a URL for changes
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any


# ---------------------------------------------------------------------------
# HTML text extractor (no external deps)
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Extracts visible text from HTML, stripping tags and scripts."""

    _SKIP_TAGS = frozenset({
        "script", "style", "noscript", "svg", "path",
        "meta", "link", "head",
    })

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag.lower() in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        raw = " ".join(self._parts)
        # Collapse whitespace
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


class _LinkExtractor(HTMLParser):
    """Extracts links from HTML."""

    def __init__(self):
        super().__init__()
        self.links: list[dict[str, str]] = []
        self._current_href = ""
        self._current_text_parts: list[str] = []
        self._in_a = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self._in_a = True
            self._current_text_parts = []
            for name, value in attrs:
                if name.lower() == "href" and value:
                    self._current_href = value

    def handle_data(self, data):
        if self._in_a:
            self._current_text_parts.append(data.strip())

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._in_a:
            self._in_a = False
            text = " ".join(self._current_text_parts).strip()
            if self._current_href and text:
                self.links.append({
                    "href": self._current_href,
                    "text": text,
                })
            self._current_href = ""
            self._current_text_parts = []


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    rank: int = 0

    def __str__(self):
        return f"{self.rank}. {self.title}\n   {self.snippet}\n   {self.url}"


@dataclass
class FetchResult:
    """Result of fetching a URL."""
    url: str
    text: str
    title: str = ""
    links: list[dict[str, str]] = field(default_factory=list)
    status_code: int = 0
    content_type: str = ""
    fetch_time: float = 0.0
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error and self.status_code < 400

    def summary(self, max_chars: int = 500) -> str:
        """Return a short summary of the fetched content."""
        if self.error:
            return f"[Error] {self.error}"
        text = self.text[:max_chars]
        if len(self.text) > max_chars:
            text += "..."
        return f"[{self.title or self.url}]\n{text}"


@dataclass
class WebMonitor:
    """Monitor a URL for changes."""
    url: str
    last_content: str = ""
    last_check: float = 0.0
    change_detected: bool = False
    check_count: int = 0


# ---------------------------------------------------------------------------
# Core web functions
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": "MKAngel/1.0 (Grammar Language Model; +https://github.com/mrjkilcoyne-lgtm/MKAngel)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def search(query: str, num_results: int = 5) -> list[SearchResult]:
    """Search the web using DuckDuckGo HTML (no API key needed).

    Args:
        query: Search query string.
        num_results: Maximum results to return.

    Returns:
        List of SearchResult objects.
    """
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers=_HEADERS)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return [SearchResult(
            title="Search Error",
            url="",
            snippet=f"Could not reach search engine: {exc.reason}",
            rank=0,
        )]
    except Exception as exc:
        return [SearchResult(
            title="Search Error", url="",
            snippet=str(exc), rank=0,
        )]

    results: list[SearchResult] = []

    # Extract result blocks
    # DuckDuckGo HTML results have class="result"
    title_pattern = re.compile(
        r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'class="result__snippet"[^>]*>(.*?)</(?:a|span|td)',
        re.DOTALL,
    )

    titles = title_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, ((href, title), snippet) in enumerate(
        zip(titles[:num_results], snippets[:num_results])
    ):
        clean_title = re.sub(r"<[^>]+>", "", title).strip()
        clean_snippet = re.sub(r"<[^>]+>", "", snippet).strip()
        # DuckDuckGo wraps URLs in a redirect
        actual_url = href
        if "uddg=" in href:
            match = re.search(r"uddg=([^&]+)", href)
            if match:
                actual_url = urllib.parse.unquote(match.group(1))

        results.append(SearchResult(
            title=clean_title,
            url=actual_url,
            snippet=clean_snippet,
            rank=i + 1,
        ))

    return results


def fetch(url: str, max_chars: int = 20000) -> FetchResult:
    """Fetch a URL and extract clean text content.

    Args:
        url: The URL to fetch.
        max_chars: Maximum characters of text to return.

    Returns:
        FetchResult with extracted text, title, and links.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    start = time.time()
    req = urllib.request.Request(url, headers=_HEADERS)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
            content_type = resp.headers.get("Content-Type", "")
            status = resp.status
    except urllib.error.HTTPError as exc:
        return FetchResult(
            url=url, text="", status_code=exc.code,
            error=f"HTTP {exc.code}: {exc.reason}",
            fetch_time=time.time() - start,
        )
    except urllib.error.URLError as exc:
        return FetchResult(
            url=url, text="",
            error=f"Network error: {exc.reason}",
            fetch_time=time.time() - start,
        )
    except Exception as exc:
        return FetchResult(
            url=url, text="",
            error=str(exc),
            fetch_time=time.time() - start,
        )

    # Decode
    encoding = "utf-8"
    if "charset=" in content_type:
        match = re.search(r"charset=([^\s;]+)", content_type)
        if match:
            encoding = match.group(1)

    try:
        html = raw.decode(encoding, errors="replace")
    except (UnicodeDecodeError, LookupError):
        html = raw.decode("utf-8", errors="replace")

    # Extract title
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
    title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip() if title_match else ""

    # Extract text
    extractor = _TextExtractor()
    try:
        extractor.feed(html)
    except Exception:
        pass
    text = extractor.get_text()

    if len(text) > max_chars:
        text = text[:max_chars]

    # Extract links
    link_extractor = _LinkExtractor()
    try:
        link_extractor.feed(html)
    except Exception:
        pass

    return FetchResult(
        url=url,
        text=text,
        title=title,
        links=link_extractor.links[:50],
        status_code=status,
        content_type=content_type,
        fetch_time=time.time() - start,
    )


def fetch_json(url: str) -> dict[str, Any]:
    """Fetch a URL and parse as JSON.

    Args:
        url: The URL to fetch.

    Returns:
        Parsed JSON as dict, or error dict.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    req = urllib.request.Request(url, headers={
        **_HEADERS,
        "Accept": "application/json",
    })

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data
    except Exception as exc:
        return {"error": str(exc)}


def check_connectivity() -> bool:
    """Quick connectivity test."""
    try:
        req = urllib.request.Request("https://httpbin.org/get", method="HEAD")
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Web monitor
# ---------------------------------------------------------------------------

class WebWatcher:
    """Watch URLs for changes — the Angel's vigilance.

    Checks periodically and reports when content differs
    from the last known state.
    """

    def __init__(self):
        self._monitors: dict[str, WebMonitor] = {}

    def watch(self, url: str) -> WebMonitor:
        """Start watching a URL."""
        if url not in self._monitors:
            self._monitors[url] = WebMonitor(url=url)
        return self._monitors[url]

    def check(self, url: str) -> WebMonitor:
        """Check a watched URL for changes."""
        monitor = self._monitors.get(url)
        if not monitor:
            monitor = self.watch(url)

        result = fetch(url, max_chars=5000)
        monitor.check_count += 1
        monitor.last_check = time.time()

        if result.success:
            new_content = result.text[:2000]
            if monitor.last_content and new_content != monitor.last_content:
                monitor.change_detected = True
            monitor.last_content = new_content
        else:
            monitor.change_detected = False

        return monitor

    def check_all(self) -> list[WebMonitor]:
        """Check all watched URLs."""
        return [self.check(url) for url in list(self._monitors)]

    def unwatch(self, url: str) -> bool:
        if url in self._monitors:
            del self._monitors[url]
            return True
        return False

    @property
    def watched_urls(self) -> list[str]:
        return list(self._monitors.keys())


# ---------------------------------------------------------------------------
# Content extraction helpers
# ---------------------------------------------------------------------------

def extract_headings(html: str) -> list[dict[str, str]]:
    """Extract headings (h1-h6) from HTML."""
    headings = []
    pattern = re.compile(r"<(h[1-6])[^>]*>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
    for match in pattern.finditer(html):
        level = match.group(1).lower()
        text = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        if text:
            headings.append({"level": level, "text": text})
    return headings


def extract_meta(html: str) -> dict[str, str]:
    """Extract meta tags from HTML."""
    meta = {}
    pattern = re.compile(
        r'<meta\s+(?:name|property)="([^"]+)"\s+content="([^"]*)"',
        re.IGNORECASE,
    )
    for match in pattern.finditer(html):
        meta[match.group(1)] = match.group(2)
    return meta
