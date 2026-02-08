"""
Management command to scrape ALL important pages from the
RAIL (Responsible AI Lab) website and save structured Markdown to docs/rail.md.

Run:
    python manage.py scrape_rail

Requires:
    requests
    beautifulsoup4
"""

import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from django.conf import settings
from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)

BASE_URL = "https://rail.knust.edu.gh"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
}

SKIP_PATTERNS = [
    "/login",
    "/account",
    "/wp-admin",
    "/wp-login",
    "/feed",
    "/author",
    "/tag",
    "/category",
    "javascript:",
    "#",
]

MAX_CONTENT_LENGTH = 50_000


# ---------------------------
# Utility functions
# ---------------------------

def fetch_page(url: str, timeout: int = 10) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def should_skip_url(url: str) -> bool:
    u = url.lower()
    return any(p in u for p in SKIP_PATTERNS)


def extract_links(soup: BeautifulSoup, current_url: str) -> set[str]:
    links = set()
    base_domain = urlparse(BASE_URL).netloc

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue

        full_url = urljoin(current_url, href)
        parsed = urlparse(full_url)

        if parsed.netloc != base_domain:
            continue
        if should_skip_url(full_url):
            continue

        links.add(normalize_url(full_url))

    return links


# ---------------------------
# Markdown extraction
# ---------------------------

def clean_soup(soup: BeautifulSoup):
    for tag in soup([
        "script", "style", "nav", "footer",
        "header", "aside", "form", "noscript"
    ]):
        tag.decompose()


def element_to_markdown(el: Tag) -> str:
    if isinstance(el, NavigableString):
        return el.strip()

    if el.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        level = int(el.name[1])
        return f"\n{'#' * level} {el.get_text(strip=True)}\n"

    if el.name == "p":
        text = el.get_text(" ", strip=True)
        return f"\n{text}\n" if text else ""

    if el.name == "ul":
        lines = []
        for li in el.find_all("li", recursive=False):
            t = li.get_text(" ", strip=True)
            if t:
                lines.append(f"- {t}")
        return "\n".join(lines) + "\n" if lines else ""

    if el.name == "ol":
        lines = []
        for i, li in enumerate(el.find_all("li", recursive=False), 1):
            t = li.get_text(" ", strip=True)
            if t:
                lines.append(f"{i}. {t}")
        return "\n".join(lines) + "\n" if lines else ""

    if el.name == "a" and el.get("href"):
        text = el.get_text(strip=True)
        href = el["href"]
        if text and href.startswith("http"):
            return f"[{text}]({href})"
        return text

    return ""


def extract_markdown(soup: BeautifulSoup) -> str:
    clean_soup(soup)

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_="content")
        or soup.body
    )

    parts = []
    for el in main.descendants:
        if isinstance(el, Tag):
            md = element_to_markdown(el)
            if md:
                parts.append(md)

    text = "\n".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------
# Scraping logic
# ---------------------------

def scrape_url(url: str, timeout: int = 10):
    html = fetch_page(url, timeout)
    if not html:
        return None, set()

    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else urlparse(url).path
    content = extract_markdown(soup)

    links = extract_links(soup, url)

    if len(content) < 100:
        return None, links

    content = content[:MAX_CONTENT_LENGTH]

    return {
        "url": url,
        "title": title,
        "content": content,
    }, links


# ---------------------------
# Django Command
# ---------------------------

class Command(BaseCommand):
    help = "Scrape RAIL website into structured Markdown"

    def add_arguments(self, parser):
        parser.add_argument("--output", type=str, default=None)
        parser.add_argument("--max-pages", type=int, default=50)
        parser.add_argument("--timeout", type=int, default=10)

    def handle(self, *args, **options):
        output = options["output"]
        max_pages = options["max_pages"]
        timeout = options["timeout"]

        output_path = (
            Path(output)
            if output
            else Path(settings.BASE_DIR) / "docs" / "rail.md"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        to_visit = {normalize_url(BASE_URL + "/")}
        visited = set()
        results = []

        self.stdout.write("Scraping RAIL website...")

        while to_visit and len(visited) < max_pages:
            url = to_visit.pop()
            if url in visited:
                continue

            visited.add(url)
            self.stdout.write(f"[{len(visited)}/{max_pages}] {url}")

            data, new_links = scrape_url(url, timeout)
            if data:
                results.append(data)

            for link in new_links:
                if link not in visited:
                    to_visit.add(link)

        # Build Markdown
        sections = [
            "# RAIL — Responsible Artificial Intelligence Lab\n",
            f"> Scraped {len(results)} pages from {BASE_URL}\n",
        ]

        for page in results:
            sections.append("\n---\n")
            sections.append(f"## {page['title']}\n")
            sections.append(f"**Source:** {page['url']}\n\n")
            sections.append(page["content"])
            sections.append("\n")

        final_text = "\n".join(sections)
        final_text = re.sub(r"\n{4,}", "\n\n\n", final_text)

        output_path.write_text(final_text, encoding="utf-8")

        self.stdout.write(
            self.style.SUCCESS(f"Saved {len(results)} pages to {output_path}")
        )
