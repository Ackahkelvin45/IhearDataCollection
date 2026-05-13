"""
Management command to scrape a website into markdown for RAG ingestion.

Run: python manage.py scrape_rail

Uses the shared web_fetcher module for text extraction and link discovery.
"""

from pathlib import Path
from urllib.parse import urlparse

from django.core.management.base import BaseCommand

from chatbot.services.web_fetcher import (
    fetch_page_html,
    clean_text,
    extract_links,
    is_internal_url,
)

BASE_URL = "https://rail.knust.edu.gh"
MAX_PAGES = 50


class Command(BaseCommand):
    help = "Scrape RAIL website into markdown for chatbot RAG ingestion"

    def add_arguments(self, parser):
        parser.add_argument(
            "--base-url",
            default=BASE_URL,
            help=f"Base URL to start crawling from (default: {BASE_URL})",
        )
        parser.add_argument(
            "--max-pages",
            type=int,
            default=MAX_PAGES,
            help=f"Maximum pages to scrape (default: {MAX_PAGES})",
        )
        parser.add_argument(
            "--output",
            default="docs/rail.md",
            help="Output file path (default: docs/rail.md)",
        )

    def handle(self, *args, **options):
        base_url = options["base_url"]
        max_pages = options["max_pages"]
        output_path = Path(options["output"])

        self.stdout.write(f"Scraping {base_url} (max {max_pages} pages)...")

        visited = set()
        to_visit = [base_url]
        pages = []

        while to_visit and len(pages) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue

            self.stdout.write(f"[{len(visited) + 1}/{max_pages}] {url}")
            visited.add(url)

            html = fetch_page_html(url)
            if not html:
                continue

            text = clean_text(html)
            if len(text) < 100:
                continue

            pages.append(f"## {url}\n\n{text}\n\n---\n")

            links = extract_links(html, url)
            for link in links:
                if link not in visited and link not in to_visit:
                    to_visit.append(link)

        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text("\n".join(pages), encoding="utf-8")

        self.stdout.write(
            self.style.SUCCESS(f"Saved {len(pages)} pages to {output_path.resolve()}")
        )
