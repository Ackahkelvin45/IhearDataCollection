import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from django.core.management.base import BaseCommand
from pathlib import Path
import logging
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

BASE_URL = "https://rail.knust.edu.gh"
MAX_PAGES = 50

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAILScraper/1.0)"
}


def is_internal(url: str) -> bool:
    return urlparse(url).netloc == urlparse(BASE_URL).netloc


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_links(html: str, current_url: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#") or href.startswith("mailto"):
            continue

        full_url = urljoin(current_url, href)
        full_url = full_url.split("#")[0]

        if is_internal(full_url):
            links.add(full_url)

    return links


def fetch_page(url: str) -> str | None:
    try:
        r = requests.get(
            url,
            headers=HEADERS,
            timeout=15,
            verify=False
        )
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None


class Command(BaseCommand):
    help = "Scrape RAIL website into markdown"

    def handle(self, *args, **kwargs):
        self.stdout.write("Scraping RAIL website…")

        visited = set()
        to_visit = [BASE_URL]
        pages = []

        while to_visit and len(pages) < MAX_PAGES:
            url = to_visit.pop(0)
            if url in visited:
                continue

            self.stdout.write(f"[{len(visited)+1}/{MAX_PAGES}] {url}")
            visited.add(url)

            html = fetch_page(url)
            if not html:
                continue

            text = clean_text(html)
            if len(text) < 100:
                continue  # still skip *empty* pages

            pages.append(f"## {url}\n\n{text}\n\n---\n")

            links = extract_links(html, url)
            for link in links:
                if link not in visited and link not in to_visit:
                    to_visit.append(link)

        output_path = Path("docs/rail.md")
        output_path.parent.mkdir(exist_ok=True)

        output_path.write_text("\n".join(pages), encoding="utf-8")

        self.stdout.write(
            self.style.SUCCESS(f"Saved {len(pages)} pages to {output_path.resolve()}")
        )
